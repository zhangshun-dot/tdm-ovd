# -*- coding: utf-8 -*-
"""
heldout probe v2: dump per-image detections once (GPU), then evaluate MANY
calibration/test splits offline (CPU, free).

Why: v1 stored only per-class metrics, so every new split needed a re-run.
Here detections are dumped per (model, class, prompt, image), so:
  - Protocol A (paper): select top-2 by DR on all 20, eval on all 20  [sanity: ~0.1967 gdino-tiny]
  - Protocol B (held-out): N random K=10/10 splits; select on calib, eval on disjoint test
    -> mean/std/min/max pooled micro-F1, plus the overhead baseline on the same test sets
       so we can ask "does TDM still beat overhead under held-out?"

Usage: python heldout_probe_v2.py gdino-tiny gdino-base owlv2
"""
import os, sys, json, time, datetime
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, AXIS_CONFIG, build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = u.IMG_DIR
CONF, IOU_MAP, IOU_F1 = u.CONF_THRESH, u.IOU_MAP, u.IOU_F1
K = 100
N_SPLITS = 50
BASE_SEED = 20260920
MODIFIER_AXES = [a for a in AXIS_CONFIG if a not in ("grammar", "negation")]
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
DUMP_DIR = os.path.join(RESULTS_DIR, "heldout_dets")
BATCH = 8  # images per GPU forward in detect(); auto-halved on OOM


def truncate_prompt(text, proc, max_len=16):
    try:
        nspec = len(proc.tokenizer.encode("x", add_special_tokens=True)) \
              - len(proc.tokenizer.encode("x", add_special_tokens=False))
    except Exception:
        nspec = 2
    words = text.split(); kept = ""
    for w in words:
        cand = (kept + " " + w).strip() if kept else w
        if len(proc.tokenizer.encode(cand, add_special_tokens=True)) <= max_len:
            kept = cand
        else:
            break
    return kept if kept else text


def variant_prompt(cls, axis=None, level=None, mods=None):
    kw = {a: "none" for a in ["granularity", "spatial", "geometric", "scale", "spectral", "texture"]}
    kw["granularity"] = "standard"
    if axis is not None:
        kw[axis] = level
    if mods:
        kw.update(mods)
    return f"an overhead image of {build_prompt(cls, grammar='bare', **kw)}"


def class_imgs(cls, gt):
    out = []
    for f in sorted(os.listdir(IMG_DIR)):
        if not f.endswith(".jpg"):
            continue
        c, iid = u.parse_filename(f)
        if c == cls and iid in gt:
            out.append((f, iid))
            if len(out) >= 200:
                break
    return out


def gt_boxes_for(cls, gt, fnames):
    boxes = []
    for fname, iid in fnames:
        g = gt[iid]
        for b, c in zip(g["bboxes"], g["cats"]):
            if u._cat_name(c) == cls:
                x, y, w, h = b
                boxes.append([x, y, x + w, y + h])
    return boxes


def _detect_batch(proc, model, prompt, imgs, sizes):
    """One batched forward over `imgs` (list of PIL) sharing the same `prompt`."""
    inputs = proc(images=imgs, text=[prompt] * len(imgs), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        o = model(**inputs)
    targets = torch.tensor(sizes, device=DEVICE)
    return proc.post_process_grounded_object_detection(o, threshold=CONF, target_sizes=targets)


def detect(proc, model, prompt, fnames, trunc=None, batch_size=BATCH):
    p = trunc(prompt) if trunc else prompt
    out = {}
    if not fnames:
        return out
    imgs = [Image.open(os.path.join(IMG_DIR, f)).convert("RGB") for f in fnames]
    sizes = [list(img.size[::-1]) for img in imgs]  # (H, W) per image
    i, n = 0, len(fnames)
    while i < n:
        bs = min(batch_size, n - i)
        sub_f, sub_imgs, sub_sizes = fnames[i:i + bs], imgs[i:i + bs], sizes[i:i + bs]
        try:
            res = _detect_batch(proc, model, p, sub_imgs, sub_sizes)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if bs > 1:
                half = bs // 2
                res = []
                for j in range(i, i + bs, half):
                    bs2 = min(half, i + bs - j)
                    res += _detect_batch(proc, model, p, imgs[j:j + bs2], sizes[j:j + bs2])
            else:
                torch.cuda.empty_cache()
                res = _detect_batch(proc, model, p, sub_imgs, sub_sizes)
        for f, r in zip(sub_f, res):
            out[f] = {"boxes": r["boxes"].cpu().numpy().tolist(),
                      "scores": [float(s) for s in r["scores"].cpu().numpy()]}
        i += bs
    return out


def metrics_on(det_map, cls, gt, fnames):
    pb, ps = [], []
    for fname, _ in fnames:
        d = det_map[fname]
        pb += d["boxes"]; ps += d["scores"]
    gb = gt_boxes_for(cls, gt, fnames)
    tp, fp, fn = u.tp_fp_fn(pb, gb, IOU_F1)
    ap = u.compute_ap(pb, ps, gb, IOU_MAP)
    dr = float(np.mean([len(det_map[f]["boxes"]) > 0 for f, _ in fnames]))
    return {"tp": tp, "fp": fp, "fn": fn, "AP": ap, "DR": dr}


def load_dump(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def save_dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def run_model(MK, gt, classes=None, do_eval=True):
    if classes is None:
        classes = DIOR_CLASSES
    t0 = time.time()
    proc, model = u.load_model(MK, DEVICE)
    trunc = (lambda t: truncate_prompt(t, proc, 16)) if MK == "owlv2" else None
    mdir = os.path.join(DUMP_DIR, MK)
    os.makedirs(mdir, exist_ok=True)
    print(f"\n===== {MK} ({DEVICE}) =====", flush=True)

    prompts_needed = {}   # cls -> {tag: prompt}
    for cls in classes:
        imgs = class_imgs(cls, gt)
        if len(imgs) < 200:
            continue
        fnames = [f for f, _ in imgs]
        dpath = os.path.join(mdir, f"{cls}.json")
        dump = load_dump(dpath)
        changed = False
        plist = {"__overhead__": u.PREFIXES["overhead"](cls)}
        for axis in MODIFIER_AXES:
            for level in AXIS_CONFIG[axis]["perturbations"]:
                plist[f"{axis}::{level}"] = variant_prompt(cls, axis, level)
        for tag, pr in plist.items():
            if pr not in dump or any(f not in dump[pr] for f in fnames):
                # gdino-base has a transformers-5.17 batch>1 forward bug (fusion
                # cross-attn reshape fails); fall back to single-image (verified 0.82s/img).
                _bs = 1 if MK == "gdino-base" else BATCH
                got = detect(proc, model, pr, fnames, trunc, batch_size=_bs)
                dump.setdefault(pr, {}).update(got)
                changed = True
        if changed:
            save_dump(dpath, dump)
        torch.cuda.empty_cache()          # keep fragmentation in check across classes
        prompts_needed[cls] = plist
        print(f"  [dump] {cls} ({time.time()-t0:.0f}s)", flush=True)

    if not do_eval:
        print(f"[{MK}] dump-only (--no-eval) complete, {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
        return

    # ---- offline multi-split evaluation (CPU) + on-demand combined-prompt detection ----
    seeds = [BASE_SEED + i for i in range(N_SPLITS)]
    per_seed = []
    for si, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        T = F = N = 0                       # TDM pooled over test images
        To = Fo = No = 0                    # overhead pooled over test images
        dr_t, dr_o, ap_t = [], [], []
        for cls in classes:
            imgs = class_imgs(cls, gt)
            if len(imgs) < 200:
                continue
            fnames = [f for f, _ in imgs]
            dump = load_dump(os.path.join(mdir, f"{cls}.json"))
            oh_p = u.PREFIXES["overhead"](cls)
            idx = rng.permutation(len(imgs))
            calib = [imgs[i] for i in idx[:K]]
            test = [imgs[i] for i in idx[K:]]
            oh_dr_cal = metrics_on(dump[oh_p], cls, gt, calib)["DR"]
            pos = []
            for tag, pr in prompts_needed[cls].items():
                if tag == "__overhead__":
                    continue
                d = metrics_on(dump[pr], cls, gt, calib)["DR"] - oh_dr_cal
                if d > 0:
                    pos.append((tag, pr, d))
            pos.sort(key=lambda x: -x[2])
            top2 = pos[:2]
            tdm_p = oh_p
            if len(top2) == 1:
                tdm_p = top2[0][1]
            elif len(top2) == 2:
                a1, l1 = top2[0][0].split("::"); a2, l2 = top2[1][0].split("::")
                tdm_p = variant_prompt(cls, mods={a1: l1, a2: l2})
            # ensure detections exist for tdm_p on test fnames
            tfn = [f for f, _ in test]
            if tdm_p not in dump or any(f not in dump[tdm_p] for f in tfn):
                miss = [f for f in tfn if tdm_p not in dump or f not in dump[tdm_p]]
                got = detect(proc, model, tdm_p, miss, trunc)
                dump.setdefault(tdm_p, {}).update(got)
                save_dump(os.path.join(mdir, f"{cls}.json"), dump)
            mt = metrics_on(dump[tdm_p], cls, gt, test)
            mo = metrics_on(dump[oh_p], cls, gt, test)
            T += mt["tp"]; F += mt["fp"]; N += mt["fn"]
            To += mo["tp"]; Fo += mo["fp"]; No += mo["fn"]
            dr_t.append(mt["DR"]); dr_o.append(mo["DR"]); ap_t.append(mt["AP"])
        per_seed.append({
            "seed": sd,
            "tdm_microF1": u.micro_f1(T, F, N), "oh_microF1": u.micro_f1(To, Fo, No),
            "tdm_DR": float(np.mean(dr_t)), "oh_DR": float(np.mean(dr_o)), "tdm_mAP": float(np.mean(ap_t)),
        })
        print(f"  [split {si}] seed={sd} TDM={per_seed[-1]['tdm_microF1']:.4f}  OH={per_seed[-1]['oh_microF1']:.4f}", flush=True)

    # ---- Protocol A (paper): select on all 20, eval on all 20 ----
    Ta = Fa = Na = 0; dr_a = []; ap_a = []
    Toa = Foa = Noa = 0
    for cls in classes:
        imgs = class_imgs(cls, gt)
        if len(imgs) < 200:
            continue
        dump = load_dump(os.path.join(mdir, f"{cls}.json"))
        oh_p = u.PREFIXES["overhead"](cls)
        oh_dr = metrics_on(dump[oh_p], cls, gt, imgs)["DR"]
        pos = []
        for tag, pr in prompts_needed[cls].items():
            if tag == "__overhead__":
                continue
            d = metrics_on(dump[pr], cls, gt, imgs)["DR"] - oh_dr
            if d > 0:
                pos.append((tag, pr, d))
        pos.sort(key=lambda x: -x[2]); top2 = pos[:2]
        tdm_p = oh_p
        if len(top2) == 1:
            tdm_p = top2[0][1]
        elif len(top2) == 2:
            a1, l1 = top2[0][0].split("::"); a2, l2 = top2[1][0].split("::")
            tdm_p = variant_prompt(cls, mods={a1: l1, a2: l2})
        fnames = [f for f, _ in imgs]
        if tdm_p not in dump or any(f not in dump[tdm_p] for f in fnames):
            miss = [f for f in fnames if tdm_p not in dump or f not in dump[tdm_p]]
            got = detect(proc, model, tdm_p, miss, trunc)
            dump.setdefault(tdm_p, {}).update(got)
            save_dump(os.path.join(mdir, f"{cls}.json"), dump)
        m = metrics_on(dump[tdm_p], cls, gt, imgs)
        mo = metrics_on(dump[oh_p], cls, gt, imgs)
        Ta += m["tp"]; Fa += m["fp"]; Na += m["fn"]; dr_a.append(m["DR"]); ap_a.append(m["AP"])
        Toa += mo["tp"]; Foa += mo["fp"]; Noa += mo["fn"]

    tdm_arr = np.array([s["tdm_microF1"] for s in per_seed])
    oh_arr = np.array([s["oh_microF1"] for s in per_seed])
    out = {
        "model": MK, "K": K, "n_splits": N_SPLITS, "base_seed": BASE_SEED,
        "iou_f1": IOU_F1, "iou_map": IOU_MAP, "conf": CONF,
        "protocolA_select20_eval20": {
            "tdm_microF1": u.micro_f1(Ta, Fa, Na), "oh_microF1": u.micro_f1(Toa, Foa, Noa),
            "tdm_DR": float(np.mean(dr_a)), "tdm_mAP": float(np.mean(ap_a))},
        "heldout_splits": per_seed,
        "heldout_summary": {
            "tdm_microF1_mean": float(tdm_arr.mean()), "tdm_microF1_std": float(tdm_arr.std()),
            "tdm_microF1_min": float(tdm_arr.min()), "tdm_microF1_max": float(tdm_arr.max()),
            "oh_microF1_mean": float(oh_arr.mean()), "oh_microF1_std": float(oh_arr.std()),
            "tdm_minus_oh_mean": float((tdm_arr - oh_arr).mean()),
            "frac_splits_tdm_gt_oh": float((tdm_arr > oh_arr).mean()),
        },
    }
    print(f"\n=== {MK} ===", flush=True)
    print(f"  ProtocolA (select20/eval20): TDM F1={out['protocolA_select20_eval20']['tdm_microF1']:.4f} "
          f"OH F1={out['protocolA_select20_eval20']['oh_microF1']:.4f}", flush=True)
    hs = out["heldout_summary"]
    print(f"  Held-out ({N_SPLITS} splits): TDM F1={hs['tdm_microF1_mean']:.4f} +/- {hs['tdm_microF1_std']:.4f} "
          f"[{hs['tdm_microF1_min']:.4f},{hs['tdm_microF1_max']:.4f}]", flush=True)
    print(f"                        OH  F1={hs['oh_microF1_mean']:.4f} +/- {hs['oh_microF1_std']:.4f}  "
          f"TDM-OH={hs['tdm_minus_oh_mean']:+.4f}  splits TDM>OH: {hs['frac_splits_tdm_gt_oh']:.0%}", flush=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"heldout_v2_{MK}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}  ({time.time()-t0:.0f}s)", flush=True)
    del model, proc
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    models, classes, do_eval = [], None, True
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--no-eval":
            do_eval = False
        elif a == "--classes":
            lo, hi = args[i + 1].split("-")
            # convert int range to the actual class-name slice so the dump
            # loop iterates real class names (NOT ints 0..hi)
            classes = DIOR_CLASSES[int(lo):int(hi) + 1]; i += 1
        elif not a.startswith("--"):
            models.append(a)
        i += 1
    if not models:
        models = ["gdino-tiny", "gdino-base", "owlv2"]
    gt = u.load_gt()
    for MK in models:
        run_model(MK, gt, classes=classes, do_eval=do_eval)
