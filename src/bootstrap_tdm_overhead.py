# -*- coding: utf-8 -*-
"""
Bootstrap significance test: TDM vs overhead vs fixed on tiled DOTA (gdino-tiny).

Pure cache read (no forward re-run). Resamples the 10 DOTA classes with
replacement (N=10000) and computes the 95% percentile CI of the paired
differences in mAP, F1, and DR between the three configs, mirroring the
aggregation in run_dota_suite.summarize (mAP = mean of per-class AP;
F1 = micro over pooled tp/fp/fn; DR = mean of per-class DR).
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, GRANULARITY
from run_dota_suite import (
    load_dota_gt, class_images_dota, FIXED_MODS,
    tp_fp_fn_dota, compute_ap_dota,
)

MODEL_KEY = "dota-tiled-gdino-tiny"
RESULT_JSON = r"<REPO_ROOT>\results\comparison\dota_tiled_gdino-tiny_20260816_012427.json"
N_BOOT = 10000
SEED = 0


def per_class_metrics(cls, gt, gt_img, prompt):
    entries = class_images_dota(cls, gt_img)
    dets = u.load_detections(MODEL_KEY, prompt)
    if dets is None:
        raise RuntimeError(f"missing cache for {cls}: {prompt!r}")
    pred_bpi, pred_spi, gt_bpi = [], [], []
    for fname, _ in entries:
        gt_bpi.append(gt[cls][fname])
        d = dets.get(fname, {"boxes": [], "scores": []})
        pred_bpi.append(d["boxes"])
        pred_spi.append(d["scores"])
    tp, fp, fn = tp_fp_fn_dota(pred_bpi, pred_spi, gt_bpi, u.IOU_F1)
    ap = compute_ap_dota(pred_bpi, pred_spi, gt_bpi, u.IOU_MAP)
    dr = u.det_rate(dets)
    return {"DR": dr, "AP": ap, "tp": tp, "fp": fp, "fn": fn}


def main():
    gt, gt_img = load_dota_gt()
    classes = [c for c in DIOR_CLASSES if c in gt]
    with open(RESULT_JSON, encoding="utf-8") as f:
        res = json.load(f)
    tdm_prompts = {c: res["tdm"]["per_class"][c]["prompt"] for c in classes}

    configs = {
        "overhead": lambda c: u.PREFIXES["overhead"](c),
        "tdm": lambda c: tdm_prompts[c],
        "fixed": lambda c: f"an overhead image of {GRANULARITY[c]['standard']}, {FIXED_MODS}",
    }

    # per-class metrics for each config
    P = {name: {c: per_class_metrics(c, gt, gt_img, builder(c)) for c in classes}
         for name, builder in configs.items()}

    # sanity: reproduce the reported JSON summary numbers
    print("=== sanity check (reproduce JSON summary) ===")
    for name in configs:
        aps = [P[name][c]["AP"] for c in classes]
        tp = sum(P[name][c]["tp"] for c in classes)
        fp = sum(P[name][c]["fp"] for c in classes)
        fn = sum(P[name][c]["fn"] for c in classes)
        print(f"{name:10s} mAP={np.mean(aps):.6f} F1={u.micro_f1(tp, fp, fn):.6f} "
              f"DR={np.mean([P[name][c]['DR'] for c in classes]):.6f}")

    # bootstrap over classes
    rng = np.random.default_rng(SEED)
    n = len(classes)
    pairs = [("tdm", "overhead"), ("fixed", "overhead"), ("fixed", "tdm")]
    diffs = {(a, b): {"mAP": [], "F1": [], "DR": []} for a, b in pairs}
    for _ in range(N_BOOT):
        sample = rng.integers(0, n, n)
        stats = {}
        for name in configs:
            idxs = [classes[i] for i in sample]
            aps = [P[name][c]["AP"] for c in idxs]
            tp = sum(P[name][c]["tp"] for c in idxs)
            fp = sum(P[name][c]["fp"] for c in idxs)
            fn = sum(P[name][c]["fn"] for c in idxs)
            stats[name] = {
                "mAP": np.mean(aps),
                "F1": u.micro_f1(tp, fp, fn),
                "DR": np.mean([P[name][c]["DR"] for c in idxs]),
            }
        for a, b in pairs:
            for m in ("mAP", "F1", "DR"):
                diffs[(a, b)][m].append(stats[a][m] - stats[b][m])

    print(f"\n=== bootstrap {N_BOOT} iters over {n} classes ===")
    for (a, b), dd in diffs.items():
        print(f"\n{a} - {b}:")
        for m in ("mAP", "F1", "DR"):
            arr = np.array(dd[m])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            print(f"  {m:4s} mean={arr.mean():+.5f}  95%CI=[{lo:+.5f}, {hi:+.5f}]  "
                  f"crosses0={lo <= 0 <= hi}  P(>0)={np.mean(arr > 0):.3f}")


if __name__ == "__main__":
    main()
