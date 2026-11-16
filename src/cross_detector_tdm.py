# -*- coding: utf-8 -*-
"""
Cross-detector generalization of TDM (意见 contributing-weight lever).

Take the TDM per-class prompts SELECTED on gdino-tiny (from the suite) and
EVALUATE them on gdino-base and owlv2 WITHOUT re-selecting. If TDM's gain
transfers, it is a general method, not a gdino-tiny quirk.

For each detector: overhead prefix vs transferred TDM prompt -> DR/AP/F1 +
paired bootstrap (N=10000) of ΔDR/ΔF1/ΔmAP. Reuses detection cache per model_key.
"""
import os, sys, json, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from axes_definition import DIOR_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = r"<REPO_ROOT>\results\comparison"
MODELS = ["gdino-base", "owlv2"]
N_BOOT = 10000
SEED = 0


def truncate_prompt(text, proc, max_len=16):
    """Truncate a prompt so the processor's encoded length (incl. special tokens)
    stays <= max_len. OWLv2's text tower is capped at 16 position embeddings and
    the processor adds 2 special tokens, so word-boundary truncation is the safe
    transfer test for a tiny-selected verbose TDM prompt.
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


def eval_pair(MK, proc, model, gt, cls, tdm_prompt, truncate=False, max_len=16):
    oh_p = u.PREFIXES["overhead"](cls)
    td_p = tdm_prompt
    if truncate:
        oh_p = truncate_prompt(oh_p, proc, max_len)
        td_p = truncate_prompt(td_p, proc, max_len)
    oh = u.evaluate_class(MK, proc, model, DEVICE, gt, cls, oh_p, cache_only=False)
    td = u.evaluate_class(MK, proc, model, DEVICE, gt, cls, td_p, cache_only=False)
    return oh, td, oh_p, td_p


def main():
    gt = u.load_gt()
    js = sorted(glob.glob(os.path.join(RESULTS_DIR, "dior_suite_gdino-tiny_*.json")))
    suite = json.load(open(js[-1], encoding="utf-8"))
    tdm_prompts = {c: suite["tdm"]["per_class"][c]["prompt"] for c in DIOR_CLASSES}
    print(f"device={DEVICE}  TDM prompts from: {os.path.basename(js[-1])}", flush=True)

    all_out = {}
    for MK in MODELS:
        trunc = (MK == "owlv2")
        print(f"\n{'='*64}\n=== detector: {MK} (truncate={trunc}) ===\n{'='*64}", flush=True)
        try:
            proc, model = u.load_model(MK, DEVICE)
        except Exception as e:
            print(f"  FAILED to load {MK}: {e}", flush=True)
            all_out[MK] = {"error": f"load failed: {e}"}
            continue
        per = {}
        for cls in DIOR_CLASSES:
            oh, td, oh_p, td_p = eval_pair(MK, proc, model, gt, cls, tdm_prompts[cls], trunc)
            per[cls] = {
                "oh": {"DR": oh["DR"], "AP": oh["AP"], "f1": (oh["tp"], oh["fp"], oh["fn"]), "prompt": oh_p},
                "td": {"DR": td["DR"], "AP": td["AP"], "f1": (td["tp"], td["fp"], td["fn"]), "prompt": td_p},
            }
            print(f"  {cls:<16s} oh DR={oh['DR']:.3f} TDM DR={td['DR']:.3f} "
                  f"(Δ={td['DR']-oh['DR']:+.3f})", flush=True)

        # point estimates
        To = Fo = No = Tt = Ft = Nt = 0
        oh_dr = [per[c]["oh"]["DR"] for c in DIOR_CLASSES]
        td_dr = [per[c]["td"]["DR"] for c in DIOR_CLASSES]
        oh_ap = [per[c]["oh"]["AP"] for c in DIOR_CLASSES]
        td_ap = [per[c]["td"]["AP"] for c in DIOR_CLASSES]
        for c in DIOR_CLASSES:
            ot, of, on = per[c]["oh"]["f1"]; tt, tf, tn = per[c]["td"]["f1"]
            To += ot; Fo += of; No += on; Tt += tt; Ft += tf; Nt += tn
        pt_f1 = u.micro_f1(Tt, Ft, Nt) - u.micro_f1(To, Fo, No)
        pt_dr = float(np.mean(td_dr) - np.mean(oh_dr))
        pt_ap = float(np.mean(td_ap) - np.mean(oh_ap))
        print(f"  point ΔF1={pt_f1:+.4f}  ΔDR={pt_dr:+.4f}  ΔmAP={pt_ap:+.4f}", flush=True)

        # bootstrap
        rng = np.random.default_rng(SEED)
        n = len(DIOR_CLASSES)
        dF1, dDR, dAP = [], [], []
        for b in range(N_BOOT):
            sel = rng.integers(0, n, n)
            To2 = Fo2 = No2 = Tt2 = Ft2 = Nt2 = 0
            dro = []; drt = []; apo = []; apt = []
            for i in sel:
                c = DIOR_CLASSES[i]
                ot, of, on = per[c]["oh"]["f1"]; tt, tf, tn = per[c]["td"]["f1"]
                To2 += ot; Fo2 += of; No2 += on; Tt2 += tt; Ft2 += tf; Nt2 += tn
                dro.append(per[c]["oh"]["DR"]); drt.append(per[c]["td"]["DR"])
                apo.append(per[c]["oh"]["AP"]); apt.append(per[c]["td"]["AP"])
            dF1.append(u.micro_f1(Tt2, Ft2, Nt2) - u.micro_f1(To2, Fo2, No2))
            dDR.append(np.mean(drt) - np.mean(dro))
            dAP.append(np.mean(apt) - np.mean(apo))
        dF1 = np.array(dF1); dDR = np.array(dDR); dAP = np.array(dAP)
        print(f"  paired bootstrap N={N_BOOT}:", flush=True)
        print(f"    ΔF1 : mean={dF1.mean():.4f} 95% CI=[{np.percentile(dF1,2.5):.4f},{np.percentile(dF1,97.5):.4f}] P(>0)={(dF1>0).mean():.3f}", flush=True)
        print(f"    ΔDR : mean={dDR.mean():.4f} 95% CI=[{np.percentile(dDR,2.5):.4f},{np.percentile(dDR,97.5):.4f}] P(>0)={(dDR>0).mean():.3f}", flush=True)
        print(f"    ΔmAP: mean={dAP.mean():.4f} 95% CI=[{np.percentile(dAP,2.5):.4f},{np.percentile(dAP,97.5):.4f}] P(>0)={(dAP>0).mean():.3f}", flush=True)

        all_out[MK] = {
            "point": {"dF1": pt_f1, "dDR": pt_dr, "dmAP": pt_ap},
            "bootstrap": {
                "dF1": {"mean": float(dF1.mean()), "lo": float(np.percentile(dF1,2.5)), "hi": float(np.percentile(dF1,97.5)), "p_gt0": float((dF1>0).mean())},
                "dDR": {"mean": float(dDR.mean()), "lo": float(np.percentile(dDR,2.5)), "hi": float(np.percentile(dDR,97.5)), "p_gt0": float((dDR>0).mean())},
                "dmAP": {"mean": float(dAP.mean()), "lo": float(np.percentile(dAP,2.5)), "hi": float(np.percentile(dAP,97.5)), "p_gt0": float((dAP>0).mean())},
            },
            "per_class": {c: {"oh_DR": per[c]["oh"]["DR"], "td_DR": per[c]["td"]["DR"],
                              "oh_AP": per[c]["oh"]["AP"], "td_AP": per[c]["td"]["AP"],
                              "oh_prompt": per[c]["oh"]["prompt"], "td_prompt": per[c]["td"]["prompt"]} for c in DIOR_CLASSES},
        }
        del model, proc
        torch.cuda.empty_cache()
        # incremental save so a later-model failure doesn't lose this
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"cross_detector_tdm_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_out, f, indent=2, ensure_ascii=False)
        print(f"  [saved incrementally] {path}", flush=True)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"cross_detector_tdm_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}", flush=True)


if __name__ == "__main__":
    main()
