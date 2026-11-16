# -*- coding: utf-8 -*-
"""Summarize OFAT per-axis DR/AP deltas from the newest dior_suite json (opinion 7)."""
import os, json, glob
import numpy as np

RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
js = sorted(glob.glob(os.path.join(RESULTS_DIR, "dior_suite_gdino-tiny_*.json")))
suite = json.load(open(js[-1], encoding="utf-8"))
ofat = suite["ofat"]

axes = sorted({a for cls in ofat for a in ofat[cls]})
print(f"using suite: {os.path.basename(js[-1])}")
print(f"{'axis':<12s} {'#cls_pos_DR':>11s} {'#cls_pos_AP':>11s} {'mean_DRΔ':>10s} {'mean_APΔ':>10s}")
for ax in axes:
    dr_d = [ofat[c][ax][lv]["dr_delta"] for c in ofat for lv in ofat[c][ax]]
    ap_d = [ofat[c][ax][lv]["ap_delta"] for c in ofat for lv in ofat[c][ax]]
    n_pos_dr = sum(1 for x in dr_d if x > 0)
    n_pos_ap = sum(1 for x in ap_d if x > 0)
    print(f"  {ax:<10s} {n_pos_dr:>8d}/{len(dr_d):<3d} {n_pos_ap:>8d}/{len(ap_d):<3d} {np.mean(dr_d):>+9.4f} {np.mean(ap_d):>+9.4f}")

# overall: count classes where ANY positive axis exists (TDM construction basis)
n_any_pos = sum(1 for c in ofat if any(ofat[c][ax][lv]["dr_delta"] > 0 for ax in ofat[c] for lv in ofat[c][ax]))
print(f"\nclasses with >=1 positive (axis,level) DR delta: {n_any_pos}/20")
