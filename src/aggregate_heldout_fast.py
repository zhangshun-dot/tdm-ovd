# -*- coding: utf-8 -*-
"""
Fast, protocol-identical re-implementation of aggregate_heldout.py.

WHY
  The original calls hp.metrics_on() for *every* candidate prompt of every class
  of every split, and metrics_on computes full tp/fp/fn + 11-point AP through a
  pure-Python O(n_pred * n_gt) IoU loop (unified_eval.box_iou). Selecting the
  top-2 prompts only needs DR ("does this image yield any box?"), so >90% of that
  work is waste -- on 20 classes x 50 splits the original takes hours.

WHAT IS IDENTICAL
  - same candidate prompt list (overhead + all single-axis variants)
  - same split construction (seed = hp.BASE_SEED + i, rng.permutation, K=100)
  - same selection rule (DR gain over overhead on calib, top-2, merged combo)
  - same skip rule (combo prompt missing from dump -> class reported missing)
  - same statistics (pooled micro-F1, DR, mAP, paired-t + sign test)
  - same output schema, same filename pattern

WHAT DIFFERS (pure speed, no semantics)
  - DR for the selection phase is computed directly (O(#images))
  - the IoU matrix is built with numpy instead of a Python double loop
  - per-class images / dumps / GT boxes are built once instead of per call

Usage:
  python aggregate_heldout_fast.py gdino-tiny
  python aggregate_heldout_fast.py owlv2
"""
import os, sys, json, time, math
import numpy as np

import heldout_probe_v2 as hp
import unified_eval as u

MK = sys.argv[1] if len(sys.argv) > 1 else "gdino-tiny"
# AGG_SPLITS is for equivalence testing only; leave unset for the real 50-split run.
N_SPLITS = int(os.environ.get("AGG_SPLITS", hp.N_SPLITS))
# ── experiment knobs (both default to the paper's setting, so the plain call
#    `aggregate_heldout_fast.py <model>` behaves exactly as before) ─────────
# AGG_K: calibration-set size K (paper: 100). Drives _heldout_k_sweep.py.
# AGG_CRITERION: which signal selects the top-2 prompts on the calibration set.
#   DR  (default) = label-FREE: only asks "does this image yield any box".
#   F1  / mAP     = uses calibration GT boxes, for the "why DR?" ablation.
K_OVERRIDE = int(os.environ.get("AGG_K", "0")) or hp.K
CRIT = os.environ.get("AGG_CRITERION", "DR").upper()
assert CRIT in ("DR", "F1", "MAP"), CRIT
# AGG_TOPK: how many of the top-ranked prompts go into the merged TDM prompt.
#   2 (paper setting) = top-2 merged into a two-axis combo prompt.
#   1 = take the single best prompt only.
# WHY 1 EXISTS: the two-axis combo is a *new* prompt string, so it is only usable
# if it happens to be in the dump. The dumps were filled with the combos that the
# DR ranking selects (see _fill_combo_prompts.py), NOT the ones F1/mAP ranking
# selects -- so the F1/mAP-criterion rows silently lose ~3/4 of the classes
# (line 208 skip) and end up computed on a non-comparable 6/20 subset. TOPK=1
# always lands on a dumped single-axis prompt, so all three criteria are scored
# on the *identical* candidate pool and the *identical* 20/20 classes.
TOPK = int(os.environ.get("AGG_TOPK", "0")) or 2
assert TOPK in (1, 2), TOPK
# AGG_CONF: re-threshold the dumps at a HIGHER score threshold than the one the
# detector already applied. IMPORTANT: the dumps only store boxes with
# score >= u.CONF_THRESH (0.25), so this knob can only move the threshold UP
# (tau >= 0.25); lower values are NOT recoverable without re-running the GPU.
# WHY IT EXISTS: a strong detector can saturate the image-level discovery rate
# (DR = "does this image yield any box >= tau") at tau=0.25 on the whole pool.
# Then every candidate prompt ties, the DR-ranking has no headroom, the selector
# falls back to the overhead prompt, and TDM becomes the identity -> the
# reported delta is exactly 0.0000 (not a small effect, a degenerate one).
# Raising tau restores headroom. Default 0.0 = paper setting (no re-threshold).
CONF_FILTER = float(os.environ.get("AGG_CONF", "0") or 0.0)
assert CONF_FILTER >= 0.0, CONF_FILTER
gt = u.load_gt()
mdir = os.path.join(hp.DUMP_DIR, MK)
t0 = time.time()

prompts_needed = {}
for cls in u.DIOR_CLASSES:
    plist = {"__overhead__": u.PREFIXES["overhead"](cls)}
    for axis in hp.MODIFIER_AXES:
        for level in u.AXIS_CONFIG[axis]["perturbations"]:
            plist[f"{axis}::{level}"] = hp.variant_prompt(cls, axis, level)
    prompts_needed[cls] = plist


# ── vectorised metrics, semantics matched to unified_eval ──────────────
def _iou_matrix(P, G):
    if len(P) == 0 or len(G) == 0:
        return np.zeros((len(P), len(G)), dtype=np.float64)
    x1 = np.maximum(P[:, None, 0], G[None, :, 0])
    y1 = np.maximum(P[:, None, 1], G[None, :, 1])
    x2 = np.minimum(P[:, None, 2], G[None, :, 2])
    y2 = np.minimum(P[:, None, 3], G[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
    ab = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-10)


def _greedy(P, G, thr):
    """Same greedy rule and same processing order as unified_eval.tp_fp_fn."""
    n_g = len(G)
    if n_g == 0:
        return 0, len(P), 0
    M = _iou_matrix(P, G)
    matched = np.zeros(n_g, dtype=bool)
    tp = fp = 0
    for i in range(len(P)):
        row = M[i].copy()
        row[matched] = -1.0          # skip already-matched GT, same as the `if matched[j]: continue`
        bj = int(row.argmax())
        if float(row[bj]) >= thr:    # original needs iou > 0 AND >= thr; thr=0.25 so equivalent
            matched[bj] = True; tp += 1
        else:
            fp += 1
    return tp, fp, int((~matched).sum())


def _ap(P, S, G, thr):
    """11-point Pascal VOC AP, same as unified_eval.compute_ap."""
    if len(G) == 0:
        return 0.0 if len(P) > 0 else 1.0
    # EXACTLY unified_eval.compute_ap: np.argsort(list)[::-1] (quicksort + reverse,
    # NOT stable) -- keep it identical so score ties break the same way.
    order = np.argsort(np.asarray(S, dtype=np.float64))[::-1]
    P = P[order]
    M = _iou_matrix(P, G)
    matched = np.zeros(len(G), dtype=bool)
    tp = np.zeros(len(P)); fp = np.zeros(len(P))
    for i in range(len(P)):
        row = M[i].copy()
        row[matched] = -1.0
        bj = int(row.argmax())
        if float(row[bj]) >= thr:
            matched[bj] = True; tp[i] = 1
        else:
            fp[i] = 1
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec = tpc / len(G); prec = tpc / (tpc + fpc + 1e-10)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        ap += float(np.max(prec[m])) if m.any() else 0.0
    return ap / 11.0


def _dr(dmap, fnames):
    """Image-level discovery rate: fraction of images with >= 1 box.

    With CONF_FILTER > 0 the SAME definition is applied after re-thresholding the
    stored scores, so the number stays comparable across tau (only the threshold
    changes). CONF_FILTER == 0 reproduces the original "has any box" semantics,
    because the dumps were already filtered at u.CONF_THRESH.
    """
    if not fnames:
        return 0.0
    if CONF_FILTER <= 0:
        return float(np.mean([len(dmap[f]["boxes"]) > 0 for f in fnames]))
    return float(np.mean([any(s >= CONF_FILTER for s in dmap[f]["scores"]) for f in fnames]))


def _boxes(dmap, fnames):
    B, S = [], []
    for f in fnames:
        d = dmap[f]
        if CONF_FILTER <= 0:
            B.extend(d["boxes"]); S.extend(d["scores"])
        else:
            for b, s in zip(d["boxes"], d["scores"]):
                if s >= CONF_FILTER:
                    B.append(b); S.append(s)
    return np.asarray(B, dtype=np.float64).reshape(-1, 4), np.asarray(S, dtype=np.float64)


# ── precompute per-class images + dumps once ───────────────────────────
CLS_IMGS = {c: hp.class_imgs(c, gt) for c in u.DIOR_CLASSES}
CLS_DUMP = {}
for c in u.DIOR_CLASSES:
    p = os.path.join(mdir, f"{c}.json")
    CLS_DUMP[c] = hp.load_dump(p) if os.path.exists(p) else {}
print(f"[fast-agg] {MK}: loaded {sum(1 for c in CLS_DUMP if CLS_DUMP[c])} class dumps "
      f"in {time.time()-t0:.1f}s", flush=True)

seeds = [hp.BASE_SEED + i for i in range(N_SPLITS)]
per_seed = []
missing = []
for si, sd in enumerate(seeds):
    rng = np.random.default_rng(sd)
    T = F = N = 0; To = Fo = No = 0
    dr_t, dr_o, ap_t, ap_o, used = [], [], [], [], 0
    for cls in u.DIOR_CLASSES:
        imgs = CLS_IMGS[cls]
        if len(imgs) < 200:
            missing.append(cls); continue
        dump = CLS_DUMP[cls]
        if not dump:
            missing.append(cls); continue
        oh_p = u.PREFIXES["overhead"](cls)
        if oh_p not in dump:
            missing.append(cls); continue
        idx = rng.permutation(len(imgs))
        calib = [imgs[i] for i in idx[:K_OVERRIDE]]
        test = [imgs[i] for i in idx[K_OVERRIDE:]]
        cal = [f for f, _ in calib]; tst = [f for f, _ in test]

        # ---- selection phase ----
        if CRIT == "DR":
            oh_dr_cal = _dr(dump[oh_p], cal)

            def score(pr, _b=oh_dr_cal):
                return _dr(dump[pr], cal) - _b
        else:
            gtb_cal = np.asarray(hp.gt_boxes_for(cls, gt, calib),
                                 dtype=np.float64).reshape(-1, 4)
            Po_c, So_c = _boxes(dump[oh_p], cal)
            if CRIT == "F1":
                _t, _f, _n = _greedy(Po_c, gtb_cal, u.IOU_F1)
                base_c = u.micro_f1(_t, _f, _n)

                def score(pr, _b=base_c, _g=gtb_cal):
                    Pc, Sc = _boxes(dump[pr], cal)
                    t, f, n = _greedy(Pc, _g, u.IOU_F1)
                    return u.micro_f1(t, f, n) - _b
            else:                                   # MAP (GT on calib)
                base_c = _ap(Po_c, So_c, gtb_cal, u.IOU_MAP)

                def score(pr, _b=base_c, _g=gtb_cal):
                    Pc, Sc = _boxes(dump[pr], cal)
                    return _ap(Pc, Sc, _g, u.IOU_MAP) - _b
        pos = []
        for tag, pr in prompts_needed[cls].items():
            if tag == "__overhead__" or pr not in dump:
                continue
            if any(f not in dump[pr] for f in cal):
                continue
            dd = score(pr)
            if dd > 0:
                pos.append((tag, pr, dd))
        pos.sort(key=lambda x: -x[2])
        top2 = pos[:TOPK]
        tdm_p = oh_p
        if len(top2) == 1:
            tdm_p = top2[0][1]
        elif len(top2) == 2:
            a1, l1 = top2[0][0].split("::"); a2, l2 = top2[1][0].split("::")
            tdm_p = hp.variant_prompt(cls, mods={a1: l1, a2: l2})

        if tdm_p not in dump or any(f not in dump[tdm_p] for f in tst):
            missing.append(f"{cls}:tdm_test"); continue

        gtb = np.asarray(hp.gt_boxes_for(cls, gt, test), dtype=np.float64).reshape(-1, 4)
        Pt, St = _boxes(dump[tdm_p], tst)
        Po, So = _boxes(dump[oh_p], tst)
        tp, fp, fn = _greedy(Pt, gtb, u.IOU_F1)
        to, fo, no = _greedy(Po, gtb, u.IOU_F1)
        T += tp; F += fp; N += fn
        To += to; Fo += fo; No += no
        dr_t.append(_dr(dump[tdm_p], tst)); dr_o.append(_dr(dump[oh_p], tst))
        ap_t.append(_ap(Pt, St, gtb, u.IOU_MAP))
        ap_o.append(_ap(Po, So, gtb, u.IOU_MAP))
        used += 1
    per_seed.append({
        "seed": sd, "n_classes": used,
        "tdm_microF1": u.micro_f1(T, F, N), "oh_microF1": u.micro_f1(To, Fo, No),
        "tdm_DR": float(np.mean(dr_t)) if dr_t else 0.0,
        "oh_DR": float(np.mean(dr_o)) if dr_o else 0.0,
        "tdm_mAP": float(np.mean(ap_t)) if ap_t else 0.0,
        "oh_mAP": float(np.mean(ap_o)) if ap_o else 0.0,
    })
    if (si + 1) % 10 == 0:
        print(f"  [split {si+1}/{len(seeds)}] {time.time()-t0:.0f}s", flush=True)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _paired_p(diffs):
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    d = np.array(diffs, dtype=float)
    mean_d = d.mean(); std_d = d.std(ddof=1)
    if std_d == 0:
        p_t = 1.0 if mean_d == 0 else 0.0
    else:
        z = mean_d / (std_d / math.sqrt(n))
        p_t = 2.0 * (1.0 - _phi(abs(z)))
    k = int((d > 0).sum())
    p_lo = sum(math.comb(n, x) * (0.5 ** n) for x in range(0, k + 1))
    p_hi = sum(math.comb(n, x) * (0.5 ** n) for x in range(k, n + 1))
    return float(p_t), float(2.0 * min(p_lo, p_hi))


tdm_arr = np.array([s["tdm_microF1"] for s in per_seed])
oh_arr = np.array([s["oh_microF1"] for s in per_seed])
p_t, p_sign = _paired_p(list(tdm_arr - oh_arr))

out = {
    "model": MK, "K": K_OVERRIDE, "n_splits": N_SPLITS, "base_seed": hp.BASE_SEED,
    "selection_criterion": CRIT, "topk_merged": TOPK,
    "criterion_uses_calib_gt": CRIT != "DR",
    "iou_f1": u.IOU_F1, "iou_map": u.IOU_MAP, "conf": u.CONF_THRESH,
    "conf_filter": CONF_FILTER,
    "aggregated": True, "fast_aggregate": True,
    "note": "computed from chunked dumps with vectorised metrics",
    "heldout_splits": per_seed,
    "heldout_summary": {
        "tdm_microF1_mean": float(tdm_arr.mean()), "tdm_microF1_std": float(tdm_arr.std()),
        "tdm_microF1_min": float(tdm_arr.min()), "tdm_microF1_max": float(tdm_arr.max()),
        "oh_microF1_mean": float(oh_arr.mean()), "oh_microF1_std": float(oh_arr.std()),
        "tdm_minus_oh_mean": float((tdm_arr - oh_arr).mean()),
        "frac_splits_tdm_gt_oh": float((tdm_arr > oh_arr).mean()),
        "paired_t_pvalue": p_t, "sign_test_pvalue": p_sign,
        "mean_classes_per_split": float(np.mean([s["n_classes"] for s in per_seed])),
        "tdm_DR_mean": float(np.mean([s["tdm_DR"] for s in per_seed])),
        "oh_DR_mean": float(np.mean([s["oh_DR"] for s in per_seed])),
        "tdm_mAP_mean": float(np.mean([s["tdm_mAP"] for s in per_seed])),
        "oh_mAP_mean": float(np.mean([s["oh_mAP"] for s in per_seed])),
    },
    "missing_classes": sorted(set(missing)),
}
ts = time.strftime("%Y%m%d_%H%M%S")
_tag = "" if N_SPLITS == hp.N_SPLITS else f"_TEST{N_SPLITS}"
if K_OVERRIDE != hp.K or CRIT != "DR" or TOPK != 2 or CONF_FILTER > 0:
    _tag += f"_K{K_OVERRIDE}_{CRIT}"                   # paper's filename pattern is untouched
    if TOPK != 2:
        _tag += f"_TOP{TOPK}"
    if CONF_FILTER > 0:
        _tag += f"_TAU{CONF_FILTER:g}"
path = os.path.join(hp.RESULTS_DIR, f"heldout_v2_{MK}_aggregated_fast{_tag}_{ts}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

hs = out["heldout_summary"]
print(f"\n=== {MK} (FAST aggregate) ==="
      f"  tau={CONF_FILTER if CONF_FILTER > 0 else u.CONF_THRESH:g}", flush=True)
print(f"  classes/split  = {hs['mean_classes_per_split']:.1f} / {len(u.DIOR_CLASSES)}", flush=True)
print(f"  TDM F1 = {hs['tdm_microF1_mean']:.4f} +/- {hs['tdm_microF1_std']:.4f} "
      f"[{hs['tdm_microF1_min']:.4f},{hs['tdm_microF1_max']:.4f}]", flush=True)
print(f"  OH  F1 = {hs['oh_microF1_mean']:.4f} +/- {hs['oh_microF1_std']:.4f}", flush=True)
print(f"  TDM-OH = {hs['tdm_minus_oh_mean']:+.4f}   splits TDM>OH: {hs['frac_splits_tdm_gt_oh']:.0%}", flush=True)
print(f"  paired-t p = {hs['paired_t_pvalue']:.4g}   sign-test p = {hs['sign_test_pvalue']:.4g}", flush=True)
print(f"  DR: TDM {hs['tdm_DR_mean']:.4f} vs OH {hs['oh_DR_mean']:.4f}"
      f"   (dDR {hs['tdm_DR_mean']-hs['oh_DR_mean']:+.4f})", flush=True)
print(f"  mAP: TDM {hs['tdm_mAP_mean']:.4f} vs OH {hs['oh_mAP_mean']:.4f}"
      f"   (dmAP {hs['tdm_mAP_mean']-hs['oh_mAP_mean']:+.4f})", flush=True)
if out["missing_classes"]:
    print(f"  WARNING missing ({len(out['missing_classes'])}): {out['missing_classes'][:12]}", flush=True)
print(f"[saved] {path}  ({time.time()-t0:.0f}s)", flush=True)
