# -*- coding: utf-8 -*-
"""
意见十 — Leave-one-class-out CV for TDM (NEW caliber, unified_eval).

Selection (OFAT top-2 positive modifiers) done ONLY on 17 TRAIN classes.
3 HOLDOUT classes (harbor/overpass/ground_track_field) evaluated with modifiers
TRANSFERRED from the training distribution (global consensus top-2), NO holdout GT used.

Answers reviewer: does TDM's gain survive when selection never sees the test class's labels?
  - transferred-TDM gain on holdout  vs  full-TDM gain on holdout (peeks at holdout GT)
  - transferred-TDM gain on holdout  vs  in-sample TDM gain on train classes
"""
import os, sys, json, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, AXIS_CONFIG, build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KEY = "gdino-tiny"
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
HOLDOUT = ["harbor", "overpass", "ground_track_field"]
MODIFIER_AXES = [a for a in AXIS_CONFIG if a not in ("grammar", "negation")]


def overhead_with_mods(cls, mods):
    kw = {"granularity": "standard", "spatial": "none", "geometric": "none",
          "scale": "none", "spectral": "none", "texture": "none"}
    kw.update(mods)
    core = build_prompt(cls, grammar="bare", **kw)
    return f"an overhead image of {core}"


def global_consensus_top2(suite):
    """Top-2 (axis,level) by #train classes with positive DR delta (tie-break mean delta)."""
    train = [c for c in DIOR_CLASSES if c not in HOLDOUT]
    cnt = {}
    for cls in train:
        for axis in MODIFIER_AXES:
            for lvl in AXIS_CONFIG[axis]["perturbations"]:
                d = suite["ofat"][cls][axis][lvl]["dr_delta"]
                if d > 0:
                    cnt.setdefault((axis, lvl), []).append(d)
    # score = frequency, then mean delta
    ranked = sorted(cnt.items(), key=lambda kv: (-len(kv[1]), -np.mean(kv[1])))
    top2 = ranked[:2]
    return {a: l for (a, l), _ in top2}


def main():
    gt = u.load_gt()
    proc, model = u.load_model(MODEL_KEY, DEVICE)
    js = sorted(glob.glob(os.path.join(RESULTS_DIR, "dior_suite_gdino-tiny_*.json")))
    suite = json.load(open(js[-1], encoding="utf-8"))
    print(f"device={DEVICE}  suite={os.path.basename(js[-1])}", flush=True)

    consensus = global_consensus_top2(suite)
    print(f"global consensus transferred modifiers: {consensus}", flush=True)

    out = {"suite": os.path.basename(js[-1]), "holdout": HOLDOUT,
           "transferred_modifiers": consensus, "per_holdout": {}, "train_gain": {}}

    # ── HOLDOUT classes: overhead vs transferred-TDM vs full-TDM ──
    for h in HOLDOUT:
        oh = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, h,
                              u.PREFIXES["overhead"](h), cache_only=False)
        tprompt = overhead_with_mods(h, consensus)
        tr = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, h, tprompt, cache_only=False)
        full = suite["tdm"]["per_class"][h]
        # full-TDM metrics from cache
        full_r = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, h,
                                 full["prompt"], cache_only=True)
        out["per_holdout"][h] = {
            "overhead": {"DR": oh["DR"], "AP": oh["AP"], "F1": u.micro_f1(oh["tp"], oh["fp"], oh["fn"])},
            "transferred": {"DR": tr["DR"], "AP": tr["AP"], "F1": u.micro_f1(tr["tp"], tr["fp"], tr["fn"]), "prompt": tprompt},
            "full_tdm": {"DR": full_r["DR"], "AP": full_r["AP"], "F1": u.micro_f1(full_r["tp"], full_r["fp"], full_r["fn"]), "prompt": full["prompt"]},
        }
        print(f"\n  HOLDOUT {h}:", flush=True)
        print(f"    overhead   DR={oh['DR']:.3f} AP={oh['AP']:.3f} F1={u.micro_f1(oh['tp'],oh['fp'],oh['fn']):.3f}", flush=True)
        print(f"    transferred DR={tr['DR']:.3f} AP={tr['AP']:.3f} F1={u.micro_f1(tr['tp'],tr['fp'],tr['fn']):.3f}", flush=True)
        print(f"    full-TDM   DR={full_r['DR']:.3f} AP={full_r['AP']:.3f} F1={u.micro_f1(full_r['tp'],full_r['fp'],full_r['fn']):.3f}", flush=True)
        d_trans = tr["DR"] - oh["DR"]
        d_full = full_r["DR"] - oh["DR"]
        print(f"    ΔDR transferred=+{d_trans:.3f}  ΔDR full(peeks holdout GT)=+{d_full:.3f}  "
              f"-> selection-bias share={max(0,d_full-d_trans):.3f}", flush=True)

    # ── TRAIN classes: in-sample TDM gain (selection sees own GT) ──
    train = [c for c in DIOR_CLASSES if c not in HOLDOUT]
    in_dr = []
    for c in train:
        oh = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, c,
                              u.PREFIXES["overhead"](c), cache_only=True)
        full_r = u.evaluate_class(MODEL_KEY, proc, model, DEVICE, gt, c,
                                 suite["tdm"]["per_class"][c]["prompt"], cache_only=True)
        g = full_r["DR"] - oh["DR"]
        in_dr.append(g)
        out["train_gain"][c] = float(g)
    print(f"\n  TRAIN in-sample mean ΔDR (TDM vs overhead) = {np.mean(in_dr):+.4f} "
          f"(n_up={sum(1 for x in in_dr if x>0.01)})", flush=True)

    # summary
    h_trans = [out["per_holdout"][h]["transferred"]["DR"] - out["per_holdout"][h]["overhead"]["DR"] for h in HOLDOUT]
    h_full = [out["per_holdout"][h]["full_tdm"]["DR"] - out["per_holdout"][h]["overhead"]["DR"] for h in HOLDOUT]
    out["summary"] = {
        "holdout_mean_transferred_DR_gain": float(np.mean(h_trans)),
        "holdout_mean_full_DR_gain": float(np.mean(h_full)),
        "train_mean_in_sample_DR_gain": float(np.mean(in_dr)),
        "selection_bias_DR_share": float(max(0, np.mean(h_full) - np.mean(h_trans))),
    }
    s = out["summary"]
    print(f"\n{'='*60}", flush=True)
    print(f"意见十 leave-one-class-out:", flush=True)
    print(f"  holdout transferred-TDM mean ΔDR = {s['holdout_mean_transferred_DR_gain']:+.4f}", flush=True)
    print(f"  holdout full-TDM      mean ΔDR = {s['holdout_mean_full_DR_gain']:+.4f} (peeks holdout GT)", flush=True)
    print(f"  train in-sample TDM   mean ΔDR = {s['train_mean_in_sample_DR_gain']:+.4f}", flush=True)
    print(f"  selection-bias DR share (full-transferred) = {s['selection_bias_DR_share']:+.4f}", flush=True)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"tdm_leave_one_out_gdino-tiny_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}", flush=True)


if __name__ == "__main__":
    main()
