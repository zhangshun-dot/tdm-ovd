# -*- coding: utf-8 -*-
"""
DOTA v1.5 独立评估（held-out）：train 选 prompt（冻结），val 盲评。

协议与 DIOR 一致（400张校准 / 完整测试集盲评）：
  calibration = train 每类 20 张（dota_suite / dota_tiled 的选词来源，prompt 已冻结）
  held-out    = val 全集（458 张，含 10 个重叠类者）只做一次最终评估

配置 = 5 前缀 + adaptive(冻结自 train top-2) + fixed(FIXED_MODS)。
缓存独立目录 _det_cache/dota-val-gdino-tiny/（与 train 的 dota-tiled 缓存不撞）。
GT = val HBB 标签，difficult=1 排除（与 train 口径一致），large/small-vehicle 合并 vehicle。
输出 = results/comparison/dota_val_gdino-tiny_<ts>.json（含逐类 + 汇总 + 关键差异 bootstrap CI）。
"""
import os, sys, json, time, datetime
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, GRANULARITY
from run_dota_suite import DOTA_TO_DIOR, FIXED_MODS, tp_fp_fn_dota, compute_ap_dota, summarize
from run_dota_tiled import detect_tiled_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KEY = "dota-val-gdino-tiny"
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
FROZEN_JSON = r"<REPO_ROOT>\results\comparison\dota_calibration_frozen_prompts.json"

VAL_IMG_DIR = r"<REPO_ROOT>\dota_data\DOTA_V1.5\val\images"
VAL_HBB_DIR = r"<REPO_ROOT>\dota_data\DOTA_V1.5\val\labelTxt-v1.5\DOTA-v1.5_val_hbb"

N_BOOT = 10000
SEED = 20260816


def load_val_gt():
    """返回 (gt, gt_img)：与 train load_dota_gt 同口径，但读 val。"""
    gt = defaultdict(lambda: defaultdict(list))
    gt_img = defaultdict(set)
    for fn in sorted(os.listdir(VAL_HBB_DIR)):
        if not fn.endswith(".txt"):
            continue
        fname = fn[:-4] + ".png"
        if not os.path.exists(os.path.join(VAL_IMG_DIR, fname)):
            continue
        with open(os.path.join(VAL_HBB_DIR, fn), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                    cls = parts[8]
                    difficult = int(parts[9]) if len(parts) >= 10 else 0
                except ValueError:
                    continue
                if cls not in DOTA_TO_DIOR:
                    continue
                dior_cls = DOTA_TO_DIOR[cls]
                gt_img[dior_cls].add(fname)
                if difficult:
                    continue
                xs = [x1, x2, x3, x4]; ys = [y1, y2, y3, y4]
                gt[dior_cls][fname].append([min(xs), min(ys), max(xs), max(ys)])
    return gt, gt_img


def val_images(cls, gt_img):
    """返回该类 val 全部 (fname, path)，按文件名定序。"""
    return [(f, os.path.join(VAL_IMG_DIR, f)) for f in sorted(gt_img[cls])]


def detect_cached(processor, model, prompt, entries, device):
    dets = u.load_detections(MODEL_KEY, prompt)
    if dets is not None:
        return dets
    out = {fname: detect_tiled_image(processor, model, prompt, path, device)
           for fname, path in entries}
    u.save_detections(MODEL_KEY, prompt, out)
    return out


def evaluate_class(processor, model, device, cls, gt, gt_img, prompt):
    entries = val_images(cls, gt_img)
    dets = detect_cached(processor, model, prompt, entries, device)
    pred_bpi, pred_spi, gt_bpi = [], [], []
    for fname, _ in entries:
        gt_bpi.append(gt[cls][fname])
        d = dets.get(fname, {"boxes": [], "scores": []})
        pred_bpi.append(d["boxes"])
        pred_spi.append(d["scores"])
    tp, fp, fn = tp_fp_fn_dota(pred_bpi, pred_spi, gt_bpi, u.IOU_F1)
    ap = compute_ap_dota(pred_bpi, pred_spi, gt_bpi, u.IOU_MAP)
    return {"DR": u.det_rate(dets), "AP": ap,
            "tp": tp, "fp": fp, "fn": fn,
            "n_gt": sum(len(g) for g in gt_bpi), "n_img": len(entries)}


def boot_mean_diff(a, b, n=N_BOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float)
    k = len(a); rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_f1_diff(ta, fa, na, tb, fb, nb, n=N_BOOT, seed=SEED):
    ta = np.asarray(ta, float); fa = np.asarray(fa, float); na = np.asarray(na, float)
    tb = np.asarray(tb, float); fb = np.asarray(fb, float); nb = np.asarray(nb, float)
    k = len(ta); rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    f1 = lambda t, f, nn, i: np.array([u.micro_f1(t[r].sum(), f[r].sum(), nn[r].sum()) for r in i])
    d = f1(ta, fa, na, idx) - f1(tb, fb, nb, idx)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    gt, gt_img = load_val_gt()
    classes = [c for c in DIOR_CLASSES if c in gt]
    n_img = {c: len(gt_img[c]) for c in classes}
    print(f"val overlap classes: {len(classes)}   val images/class: {n_img}", flush=True)

    # 冻结 adaptive prompt（calibration=train, tiled 协议选词，冻结 JSON 含哈希与全文）
    froz = json.load(open(FROZEN_JSON, encoding="utf-8"))
    tdm_prompts = {c: froz["per_class"][c]["prompt"] for c in classes}

    configs = {}
    for pname in u.PREFIXES:
        configs[pname] = lambda cls, p=pname: u.PREFIXES[p](cls)
    configs["adaptive"] = lambda cls: tdm_prompts[cls]
    configs["fixed"] = lambda cls: f"an overhead image of {GRANULARITY[cls]['standard']}, {FIXED_MODS}"

    proc, model = u.load_model("gdino-tiny", DEVICE)
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)

    out = {"model": "gdino-tiny", "dataset": "DOTA-v1.5-val", "mode": "tiled",
           "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
           "conf_thresh": u.CONF_THRESH, "iou_map": u.IOU_MAP, "iou_f1": u.IOU_F1,
           "n_classes": len(classes), "classes": classes, "n_img": n_img,
           "frozen_source": os.path.basename(FROZEN_JSON)}

    P = {}
    for name, builder in configs.items():
        per_class = {}
        for ci, cls in enumerate(classes):
            per_class[cls] = evaluate_class(proc, model, DEVICE, cls, gt, gt_img, builder(cls))
            if ci % 2 == 0:
                print(f"  [{name}] {cls} done ({time.time()-t0:.0f}s)", flush=True)
        P[name] = per_class
        out[name] = {"summary": summarize(per_class), "per_class": {
            c: {"DR": r["DR"], "AP": r["AP"], "tp": r["tp"], "fp": r["fp"],
                "fn": r["fn"], "n_gt": r["n_gt"], "n_img": r["n_img"]}
            for c, r in per_class.items()}}

    # ── 汇总打印 ─────────────────────────────────────────────────────
    print("\n=== DOTA val (held-out) 汇总 ===", flush=True)
    for name in configs:
        s = out[name]["summary"]
        print(f"  {name:15s} IRR={s['DR']:.4f} mAP={s['mAP']:.4f} P={s['P']:.4f} R={s['R']:.4f} F1={s['F1']:.4f}")

    # ── 关键差异 bootstrap CI（over classes）─────────────────────────
    def vec(name):
        pc = P[name]
        cl = classes
        return ([pc[c]["DR"] for c in cl],
                [pc[c]["AP"] for c in cl],
                [pc[c]["tp"] for c in cl], [pc[c]["fp"] for c in cl], [pc[c]["fn"] for c in cl])

    print("\n=== 关键差异 bootstrap CI (held-out) ===", flush=True)
    out["diffs"] = {}
    for a, b in [("overhead", "bare"), ("adaptive", "overhead"), ("fixed", "overhead")]:
        da, apa, tpa, fpa, fna = vec(a)
        db, apb, tpb, fpb, fnb = vec(b)
        dr = boot_mean_diff(da, db)
        ap = boot_mean_diff(apa, apb)
        f1 = boot_f1_diff(tpa, fpa, fna, tpb, fpb, fnb)
        out["diffs"][f"{a}-vs-{b}"] = {"IRR": dr, "mAP": ap, "F1": f1}
        print(f"  {a:9s} - {b:9s} ΔIRR={dr[0]:+.4f} [{dr[1]:+.4f},{dr[2]:+.4f}]  "
              f"ΔmAP={ap[0]:+.4f} [{ap[1]:+.4f},{ap[2]:+.4f}]  ΔF1={f1[0]:+.4f} [{f1[1]:+.4f},{f1[2]:+.4f}]")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"dota_val_gdino-tiny_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
