# -*- coding: utf-8 -*-
"""
Native TDM selection on a NON-tiny detector (gdino-base / owlv2).

Replicates run_dior_suite's selection logic but for an arbitrary MODEL_KEY, so
we can answer: does the TDM *method* (local prompt selection) raise DR on a
detector other than gdino-tiny? This is the "method transfer" test that
complements the earlier "prompt transfer" test (cross_detector_tdm.py).

For owlv2 the text tower caps at 16 tokens, so EVERY evaluated prompt is
truncated to <=16 tokens (incl. special tokens) before detection — keeping
selection and evaluation on the same truncated-prompt DR deltas (internally
consistent). gdino-base is not truncated.

Outputs dior_suite_{MODEL_KEY}_{ts}.json with TDM summary + paired bootstrap
of ΔDR/ΔF1/ΔmAP (TDM vs overhead).
"""
import os, sys, json, time, datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES, AXIS_CONFIG, build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
MODELS = ["gdino-base", "owlv2"]
N_BOOT = 10000
SEED = 0

MODIFIER_AXES = [a for a in AXIS_CONFIG if a not in ("grammar", "negation")]
MODIFIER_STACK = [
    ("spatial", "context"), ("geometric", "geometry"), ("scale", "comparative"),
    ("spectral", "spectral"), ("texture", "pattern"), ("granularity", "fine"),
]


def truncate_prompt(text, proc, max_len=16):
    """Word-boundary truncate so processor-encoded length (incl. special) <= max_len.
    OWLV2 text tower caps at 16 position embeddings; processor adds 2 special tokens.
    """
    try:
        nspec = len(proc.tokenizer.encode("x", add_special_tokens=True)) \
               - len(proc.tokenizer.encode("x", add_special_tokens=False))
    except Exception:
        nspec = 2
    words = text.split()
    kept = ""
    for w in words:
        cand = (kept + " " + w).strip() if kept else w
        if len(proc.tokenizer.encode(cand, add_special_tokens=True)) <= max_len:
            kept = cand
        else:
            break
    return kept if kept else text


def overhead_with_mods(cls, mods):
    kw = {"granularity": "standard", "spatial": "none", "geometric": "none",
          "scale": "none", "spectral": "none", "texture": "none"}
    kw.update(mods)
    core = build_prompt(cls, grammar="bare", **kw)
    return f"an overhead image of {core}"


def run_model(MK, proc, model, gt, truncate):
    max_len = 16 if truncate else 10**9
    def ev(prompt, cls):
        p = truncate_prompt(prompt, proc, max_len) if truncate else prompt
        return u.evaluate_class(MK, proc, model, DEVICE, gt, cls, p)

    out = {"model": MK, "truncate": truncate,
           "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
           "conf_thresh": u.CONF_THRESH, "iou_map": u.IOU_MAP, "iou_f1": u.IOU_F1}

    # overhead baseline per class
    overhead_pc = {cls: ev(u.PREFIXES["overhead"](cls), cls) for cls in DIOR_CLASSES}

    # OFAT
    print(f"  [OFAT] {MK}", flush=True)
    ofat = {}
    for ci, cls in enumerate(DIOR_CLASSES):
        oh = overhead_pc[cls]
        ofat[cls] = {}
        for axis in MODIFIER_AXES:
            ofat[cls][axis] = {}
            for lvl in AXIS_CONFIG[axis]["perturbations"]:
                p = overhead_with_mods(cls, {axis: lvl})
                r = ev(p, cls)
                ofat[cls][axis][lvl] = {
                    "dr": r["DR"], "ap": r["AP"],
                    "dr_delta": round(r["DR"] - oh["DR"], 4),
                    "ap_delta": round(r["AP"] - oh["AP"], 4),
                }
        if (ci + 1) % 5 == 0:
            print(f"    OFAT [{ci+1}/{len(DIOR_CLASSES)}] {cls}", flush=True)

    # TDM: top-2 positive modifiers by DR delta
    print(f"  [TDM] {MK}", flush=True)
    tdm = {"per_class": {}, "summary": None}
    tp = fp = fn = 0
    for cls in DIOR_CLASSES:
        oh = overhead_pc[cls]
        pos = []
        for axis in MODIFIER_AXES:
            for lvl in AXIS_CONFIG[axis]["perturbations"]:
                d = ofat[cls][axis][lvl]["dr_delta"]
                if d > 0:
                    pos.append((axis, lvl, d))
        pos.sort(key=lambda x: -x[2])
        top2 = pos[:2]
        mods = {a: l for a, l, _ in top2}
        p = overhead_with_mods(cls, mods)
        r = ev(p, cls)
        tdm["per_class"][cls] = {
            "DR": r["DR"], "AP": r["AP"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "n_gt": r["n_gt"], "prompt": p,
            "selected": [(a, l, round(d, 3)) for a, l, d in top2],
            "dr_delta": round(r["DR"] - oh["DR"], 4),
            "ap_delta": round(r["AP"] - oh["AP"], 4),
        }
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
    tdm["summary"] = {
        "DR": float(np.mean([r["DR"] for r in tdm["per_class"].values()])),
        "mAP": float(np.mean([r["AP"] for r in tdm["per_class"].values()])),
        "F1": u.micro_f1(tp, fp, fn),
        "n_tp": tp, "n_fp": fp, "n_fn": fn,
        "n_up": sum(1 for r in tdm["per_class"].values() if r["dr_delta"] > 0.01),
        "n_down": sum(1 for r in tdm["per_class"].values() if r["dr_delta"] < -0.01),
    }
    oh_dr = [overhead_pc[c]["DR"] for c in DIOR_CLASSES]
    oh_ap = [overhead_pc[c]["AP"] for c in DIOR_CLASSES]
    td_dr = [tdm["per_class"][c]["DR"] for c in DIOR_CLASSES]
    td_ap = [tdm["per_class"][c]["AP"] for c in DIOR_CLASSES]
    print(f"    TDM summary: DR={tdm['summary']['DR']:.4f} mAP={tdm['summary']['mAP']:.4f} "
          f"F1={tdm['summary']['F1']:.4f} (up={tdm['summary']['n_up']} down={tdm['summary']['n_down']})", flush=True)

    # paired bootstrap of Δ (TDM - overhead)
    rng = np.random.default_rng(SEED)
    n = len(DIOR_CLASSES)
    dF1, dDR, dAP = [], [], []
    To = Fo = No = Tt = Ft = Nt = 0
    for c in DIOR_CLASSES:
        ot, of_, on = (overhead_pc[c]["tp"], overhead_pc[c]["fp"], overhead_pc[c]["fn"])
        tt, tf_, tn = (tdm["per_class"][c]["tp"], tdm["per_class"][c]["fp"], tdm["per_class"][c]["fn"])
        To += ot; Fo += of_; No += on; Tt += tt; Ft += tf_; Nt += tn
    pt_f1 = u.micro_f1(Tt, Ft, Nt) - u.micro_f1(To, Fo, No)
    pt_dr = float(np.mean(td_dr) - np.mean(oh_dr))
    pt_ap = float(np.mean(td_ap) - np.mean(oh_ap))
    for b in range(N_BOOT):
        sel = rng.integers(0, n, n)
        To2 = Fo2 = No2 = Tt2 = Ft2 = Nt2 = 0
        dro, drt, apo, apt = [], [], [], []
        for i in sel:
            c = DIOR_CLASSES[i]
            ot, of_, on = (overhead_pc[c]["tp"], overhead_pc[c]["fp"], overhead_pc[c]["fn"])
            tt, tf_, tn = (tdm["per_class"][c]["tp"], tdm["per_class"][c]["fp"], tdm["per_class"][c]["fn"])
            To2 += ot; Fo2 += of_; No2 += on; Tt2 += tt; Ft2 += tf_; Nt2 += tn
            dro.append(overhead_pc[c]["DR"]); drt.append(tdm["per_class"][c]["DR"])
            apo.append(overhead_pc[c]["AP"]); apt.append(tdm["per_class"][c]["AP"])
        dF1.append(u.micro_f1(Tt2, Ft2, Nt2) - u.micro_f1(To2, Fo2, No2))
        dDR.append(np.mean(drt) - np.mean(dro))
        dAP.append(np.mean(apt) - np.mean(apo))
    dF1 = np.array(dF1); dDR = np.array(dDR); dAP = np.array(dAP)
    boot = {
        "point": {"dF1": pt_f1, "dDR": pt_dr, "dmAP": pt_ap},
        "dF1": {"mean": float(dF1.mean()), "lo": float(np.percentile(dF1,2.5)), "hi": float(np.percentile(dF1,97.5)), "p_gt0": float((dF1>0).mean())},
        "dDR": {"mean": float(dDR.mean()), "lo": float(np.percentile(dDR,2.5)), "hi": float(np.percentile(dDR,97.5)), "p_gt0": float((dDR>0).mean())},
        "dmAP": {"mean": float(dAP.mean()), "lo": float(np.percentile(dAP,2.5)), "hi": float(np.percentile(dAP,97.5)), "p_gt0": float((dAP>0).mean())},
    }
    print(f"    paired bootstrap N={N_BOOT}: ΔDR mean={dDR.mean():.4f} "
          f"95% CI=[{np.percentile(dDR,2.5):.4f},{np.percentile(dDR,97.5):.4f}] P(>0)={boot['dDR']['p_gt0']:.3f}", flush=True)

    out["ofat"] = ofat
    out["tdm"] = tdm
    out["bootstrap"] = boot
    out["overhead"] = {c: {"DR": overhead_pc[c]["DR"], "AP": overhead_pc[c]["AP"],
                           "tp": overhead_pc[c]["tp"], "fp": overhead_pc[c]["fp"], "fn": overhead_pc[c]["fn"]}
                       for c in DIOR_CLASSES}
    return out


def main():
    t0 = time.time()
    gt = u.load_gt()
    all_out = {}
    for MK in MODELS:
        truncate = (MK == "owlv2")
        print(f"\n{'='*64}\n=== native TDM selection: {MK} (truncate={truncate}) ===\n{'='*64}", flush=True)
        try:
            proc, model = u.load_model(MK, DEVICE)
        except Exception as e:
            print(f"  FAILED load {MK}: {e}", flush=True)
            all_out[MK] = {"error": f"load failed: {e}"}
            continue
        out = run_model(MK, proc, model, gt, truncate)
        all_out[MK] = out
        del model, proc
        torch.cuda.empty_cache()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"dior_suite_{MK}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_out, f, indent=2, ensure_ascii=False)
        print(f"  [saved] {path}", flush=True)
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
