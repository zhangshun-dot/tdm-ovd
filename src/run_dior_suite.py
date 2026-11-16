# -*- coding: utf-8 -*-
"""
GDINO-tiny DIOR full suite (rewrite口径, 2026-08-15).

Runs and caches all forward passes, then computes:
  1. prefix ablation  (5 prefixes x 20 classes)            -> DR / mAP / F1
  2. OFAT             (6 modifier axes x 2 levels x 20 cls) -> per-class delta vs overhead
  3. TDM              (overhead + top-2 positive modifiers) -> DR / mAP / F1
  4. modifier quantity (uniform 0..6 stacking)             -> DR / mAP / F1

TDM selection (matches verify_tdm_overhead.py): per class, measure each
(axis, level) delta-DR vs overhead; keep positive, take top-2, combine.

Modifier stack (deterministic, documented): spatial.context, geometric.geometry,
scale.comparative, spectral.spectral, texture.pattern, granularity.fine.
"""
import os, sys, json, time, datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, AXIS_CONFIG, build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KEY = "gdino-tiny"
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"

MODIFIER_AXES = [a for a in AXIS_CONFIG if a not in ("grammar", "negation")]
MODIFIER_STACK = [
    ("spatial", "context"), ("geometric", "geometry"), ("scale", "comparative"),
    ("spectral", "spectral"), ("texture", "pattern"), ("granularity", "fine"),
]


def overhead_with_mods(cls, mods):
    kw = {"granularity": "standard", "spatial": "none", "geometric": "none",
          "scale": "none", "spectral": "none", "texture": "none"}
    kw.update(mods)
    core = build_prompt(cls, grammar="bare", **kw)
    return f"an overhead image of {core}"


def main():
    t0 = time.time()
    gt = u.load_gt()
    proc, model = u.load_model(MODEL_KEY, DEVICE)
    print(f"model loaded ({time.time()-t0:.0f}s)")

    out = {"model": MODEL_KEY, "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
           "conf_thresh": u.CONF_THRESH, "iou_map": u.IOU_MAP, "iou_f1": u.IOU_F1}

    # ── 1. prefix ablation ──────────────────────────────────────────
    print("\n=== [1/4] prefix ablation ===")
    out["prefix"] = {}
    for pname in u.PREFIXES:
        r = u.evaluate_builder(MODEL_KEY, proc, model, DEVICE, gt,
                               builder=lambda cls, p=pname: u.PREFIXES[p](cls))
        out["prefix"][pname] = r["summary"]
        print(f"  {pname:15s} DR={r['summary']['DR']:.4f} mAP={r['summary']['mAP']:.4f} F1={r['summary']['F1']:.4f}")

    overhead_pc = {cls: u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, cls,
                                         u.PREFIXES["overhead"](cls))
                   for cls in DIOR_CLASSES}

    # ── 2. OFAT ─────────────────────────────────────────────────────
    print("\n=== [2/4] OFAT (12 modifiers) ===")
    out["ofat"] = {}   # cls -> {axis: {level: {dr_delta, ap_delta, dr, ap}}}
    for ci, cls in enumerate(DIOR_CLASSES):
        oh = overhead_pc[cls]
        out["ofat"][cls] = {}
        for axis in MODIFIER_AXES:
            out["ofat"][cls][axis] = {}
            for lvl in AXIS_CONFIG[axis]["perturbations"]:
                p = overhead_with_mods(cls, {axis: lvl})
                r = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, cls, p)
                out["ofat"][cls][axis][lvl] = {
                    "dr": r["DR"], "ap": r["AP"],
                    "dr_delta": round(r["DR"] - oh["DR"], 4),
                    "ap_delta": round(r["AP"] - oh["AP"], 4),
                }
        print(f"  [{ci+1:2d}/{len(DIOR_CLASSES)}] {cls} ({time.time()-t0:.0f}s)")

    # ── 3. TDM ──────────────────────────────────────────────────────
    print("\n=== [3/4] TDM (overhead + top-2 positive) ===")
    out["tdm"] = {"per_class": {}, "summary": None}
    tp = fp = fn = 0
    for cls in DIOR_CLASSES:
        oh = overhead_pc[cls]
        # gather positive-delta (axis, level) by DR delta
        pos = []
        for axis in MODIFIER_AXES:
            for lvl in AXIS_CONFIG[axis]["perturbations"]:
                d = out["ofat"][cls][axis][lvl]["dr_delta"]
                if d > 0:
                    pos.append((axis, lvl, d))
        pos.sort(key=lambda x: -x[2])
        top2 = pos[:2]
        mods = {a: l for a, l, _ in top2}
        p = overhead_with_mods(cls, mods)
        r = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, cls, p)
        out["tdm"]["per_class"][cls] = {
            "DR": r["DR"], "AP": r["AP"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "n_gt": r["n_gt"], "prompt": p, "selected": [(a, l, round(d, 3)) for a, l, d in top2],
            "dr_delta": round(r["DR"] - oh["DR"], 4),
            "ap_delta": round(r["AP"] - oh["AP"], 4),
        }
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
    out["tdm"]["summary"] = {
        "DR": float(np.mean([r["DR"] for r in out["tdm"]["per_class"].values()])),
        "mAP": float(np.mean([r["AP"] for r in out["tdm"]["per_class"].values()])),
        "F1": u.micro_f1(tp, fp, fn),
        "n_tp": tp, "n_fp": fp, "n_fn": fn,
        "n_up": sum(1 for r in out["tdm"]["per_class"].values() if r["dr_delta"] > 0.01),
        "n_down": sum(1 for r in out["tdm"]["per_class"].values() if r["dr_delta"] < -0.01),
    }
    s = out["tdm"]["summary"]
    print(f"  TDM summary: DR={s['DR']:.4f} mAP={s['mAP']:.4f} F1={s['F1']:.4f} "
          f"(up={s['n_up']} down={s['n_down']})")

    # ── 4. modifier quantity ablation ───────────────────────────────
    print("\n=== [4/4] modifier quantity ablation ===")
    out["modq"] = {}
    for n in range(0, 7):
        mods = dict(MODIFIER_STACK[:n])
        r = u.evaluate_builder(MODEL_KEY, proc, model, DEVICE, gt,
                               builder=lambda cls, m=mods: overhead_with_mods(cls, m))
        out["modq"][n] = r["summary"]
        print(f"  {n} mods: DR={r['summary']['DR']:.4f} mAP={r['summary']['mAP']:.4f} F1={r['summary']['F1']:.4f}")

    # ── save ────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"dior_suite_gdino-tiny_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
