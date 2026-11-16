# -*- coding: utf-8 -*-
"""
DOTA TDM probe: overhead vs the DIOR-calibrated TDM prompt, evaluated in ONE pass
over the same tiles, so the two results are exactly paired (no cross-run
comparability assumption). Reports micro-F1@IoU0.25, mAP@IoU0.5 and per-image
discovery, plus McNemar on image-level discovery.

Tiling is done once per image and shared by both prompts; only the forward passes
are duplicated.

AUTO-RESUME: every `CKPT_EVERY` images the FULL state (per-image hits + all raw
boxes in global coords + GT boxes + the output path) is checkpointed to
`dota_tdm_ckpt_<model>_<tag>_<limit>.json`. On startup, a checkpoint whose
signature (model/root/label/limit/tile/overlap/conf/tdm file) matches is loaded
and the already-processed images are skipped, appending to the SAME metrics json.
So an interruption (or a kill) costs at most CKPT_EVERY images, not the run.
Disable with --no-resume.

Usage:
  python third_dataset_dota_tdm.py --model gdino-tiny ^
      --root "<REPO_ROOT>/dior_data/DOTA/images" ^
      --label "<REPO_ROOT>/dior_data/DOTA/labelTxt" ^
      --tdm "<REPO_ROOT>/results/comparison/tdm_prompts_gdino-tiny_full20.json" ^
      --limit 100 --tag pilot100
"""
import os, sys, json, time, math, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import unified_eval as u
from third_dataset_eval_dota import (load_dota_gt, tile_image, nms,
                                     DIOR_CLASSES_USED, DEVICE)

OUT_DIR = r"<REPO_ROOT>\results\third_dataset"
CKPT_EVERY = 2


def mcnemar_exact(b, c):
    # int/int division instead of comb(n,k) * 0.5**n: the latter raises
    # OverflowError once comb(n, k) exceeds float range (n >~ 1030 at the central
    # term), which the NWPU pooled counts really did reach.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    num = sum(math.comb(n, x) for x in range(0, k + 1))
    return float(min(1.0, 2.0 * (num / (2 ** n))))


def _sig(args):
    """Identity of the run: any change here invalidates an old checkpoint."""
    try:
        mt = os.path.getmtime(args.tdm)
    except OSError:
        mt = 0.0
    return {"model": args.model, "root": args.root, "label": args.label,
            "limit": args.limit, "tile": args.tile, "overlap": args.overlap,
            "conf": args.conf, "tdm": args.tdm, "tdm_mtime": mt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="gdino-tiny")
    ap.add_argument("--tdm", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--conf", type=float, default=u.CONF_THRESH)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start over")
    args = ap.parse_args()

    tdm_map = json.load(open(args.tdm, encoding="utf-8"))
    gt = load_dota_gt(args.label)
    proc, model = u.load_model(args.model, DEVICE)
    print(f"[{args.model}] device={DEVICE} conf={args.conf} GT imgs={len(gt)}", flush=True)

    imgs = [f for f in sorted(os.listdir(args.root))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))][:args.limit]
    print(f"images={len(imgs)} classes={len(DIOR_CLASSES_USED)}", flush=True)

    VARIANTS = ("overhead", "tdm")
    pred = {v: {d: {"boxes": [], "scores": []} for d in DIOR_CLASSES_USED} for v in VARIANTS}
    gtbox = {d: [] for d in DIOR_CLASSES_USED}
    hit = {v: {d: [] for d in DIOR_CLASSES_USED} for v in VARIANTS}

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{('_'+args.tag) if args.tag else ''}_{args.limit}"
    op = os.path.join(OUT_DIR, f"dota_tdm_{args.model}{('_'+args.tag) if args.tag else ''}_{ts}.json")
    ckpt_path = os.path.join(OUT_DIR, f"dota_tdm_ckpt_{args.model}{suffix}.json")

    # ── auto-resume ───────────────────────────────────────────────────
    done_stems = set()
    resumed = False
    if not args.no_resume and os.path.exists(ckpt_path):
        try:
            s = json.load(open(ckpt_path, encoding="utf-8"))
        except Exception as e:
            s = None
            print(f"[resume] checkpoint unreadable ({e}); starting fresh", flush=True)
        if s and s.get("sig") == _sig(args):
            pred = {v: {d: {"boxes": list(s["pred"][v][d]["boxes"]),
                            "scores": list(s["pred"][v][d]["scores"])}
                        for d in DIOR_CLASSES_USED} for v in VARIANTS}
            gtbox = {d: list(s["gtbox"].get(d, [])) for d in DIOR_CLASSES_USED}
            hit = {v: {d: list(s["hit"][v][d]) for d in DIOR_CLASSES_USED} for v in VARIANTS}
            done_stems = set(s.get("done_stems", []))
            if s.get("out_path") and os.path.exists(s["out_path"]):
                op = s["out_path"]
            resumed = True
            print(f"[resume] checkpoint {os.path.basename(ckpt_path)}: "
                  f"{len(done_stems)} image(s) already done -> skipping them; "
                  f"appending to {os.path.basename(op)}", flush=True)
        elif s:
            print("[resume] checkpoint signature differs (args changed); starting fresh", flush=True)

    def save_ckpt():
        json.dump({"sig": _sig(args), "out_path": op, "ts": ts,
                   "done_stems": sorted(done_stems),
                   "pred": pred, "gtbox": gtbox, "hit": hit},
                  open(ckpt_path, "w", encoding="utf-8"), ensure_ascii=False)

    def build(complete=False):
        per_class = {}
        Tp = Fp = Nn = 0
        for d in DIOR_CLASSES_USED:
            e = {}
            for v in VARIANTS:
                pb, ps, gb = pred[v][d]["boxes"], pred[v][d]["scores"], gtbox[d]
                tp, fp, fn = u.tp_fp_fn(pb, gb, u.IOU_F1)
                e[v] = {"tp": tp, "fp": fp, "fn": fn, "microF1": u.micro_f1(tp, fp, fn),
                        "mAP": u.compute_ap(pb, ps, gb, u.IOU_MAP),
                        "n_pred": len(pb), "n_gt": len(gb),
                        "DR": float(np.mean(hit[v][d])) if hit[v][d] else 0.0}
            b = sum(1 for a_, c_ in zip(hit["overhead"][d], hit["tdm"][d]) if (not a_) and c_)
            c = sum(1 for a_, c_ in zip(hit["overhead"][d], hit["tdm"][d]) if a_ and (not c_))
            per_class[d] = {"overhead": e["overhead"], "tdm": e["tdm"],
                            "dF1": e["tdm"]["microF1"] - e["overhead"]["microF1"],
                            "dmAP": e["tdm"]["mAP"] - e["overhead"]["mAP"],
                            "dDR": e["tdm"]["DR"] - e["overhead"]["DR"],
                            "n_oh_miss_tdm_hit": b, "n_oh_hit_tdm_miss": c}
            Tp += e["tdm"]["tp"]; Fp += e["tdm"]["fp"]; Nn += e["tdm"]["fn"]
        oh_f1 = float(np.mean([v["overhead"]["microF1"] for v in per_class.values()])) if per_class else 0.0
        td_f1 = float(np.mean([v["tdm"]["microF1"] for v in per_class.values()])) if per_class else 0.0
        oh_map = float(np.mean([v["overhead"]["mAP"] for v in per_class.values()])) if per_class else 0.0
        td_map = float(np.mean([v["tdm"]["mAP"] for v in per_class.values()])) if per_class else 0.0
        oh_dr = float(np.mean([v["overhead"]["DR"] for v in per_class.values()])) if per_class else 0.0
        td_dr = float(np.mean([v["tdm"]["DR"] for v in per_class.values()])) if per_class else 0.0
        n_imgs_done = len(hit["overhead"][DIOR_CLASSES_USED[0]])
        B = sum(sum(1 for a_, c_ in zip(hit["overhead"][d], hit["tdm"][d]) if (not a_) and c_)
                for d in DIOR_CLASSES_USED)
        C = sum(sum(1 for a_, c_ in zip(hit["overhead"][d], hit["tdm"][d]) if a_ and (not c_))
                for d in DIOR_CLASSES_USED)
        o = {"dataset": "DOTA", "model": args.model, "device": DEVICE,
             "tile": args.tile, "overlap": args.overlap, "conf": args.conf,
             "n_images": n_imgs_done, "n_classes": len(per_class), "complete": complete,
             "overhead": {"macroF1": oh_f1, "macro_mAP": oh_map, "meanDR": oh_dr},
             "tdm": {"macroF1": td_f1, "macro_mAP": td_map, "meanDR": td_dr},
             "delta": {"macroF1": td_f1 - oh_f1, "macro_mAP": td_map - oh_map, "meanDR": td_dr - oh_dr},
             "pooled_microF1_tdm": u.micro_f1(Tp, Fp, Nn),
             "pooled_n_oh_miss_tdm_hit": B, "pooled_n_oh_hit_tdm_miss": C,
             "pooled_mcnemar_p": mcnemar_exact(B, C),
             "per_class": per_class}
        json.dump(o, open(op, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return o

    n_skip = 0
    for i, f in enumerate(imgs):
        stem = os.path.splitext(f)[0]
        if stem in done_stems:
            n_skip += 1
            continue
        img = Image.open(os.path.join(args.root, f)).convert("RGB")
        tiles = tile_image(img, args.tile, args.overlap)
        gtb = gt.get(stem, {})
        local = {}
        for v in VARIANTS:
            prompt_map = (tdm_map if v == "tdm" else
                          {d: u.PREFIXES["overhead"](d) for d in DIOR_CLASSES_USED})
            for d in DIOR_CLASSES_USED:
                pr = prompt_map.get(d) or u.PREFIXES["overhead"](d)
                tile_list = [(f"{stem}_{x0}_{y0}", c) for (x0, y0), c in tiles]
                dets = u.detect_all(proc, model, pr, tile_list, DEVICE, conf=args.conf)
                pb, ps = [], []
                for (x0, y0), _ in tiles:
                    dvec = dets.get(f"{stem}_{x0}_{y0}", {"boxes": [], "scores": []})
                    for box, sc in zip(dvec["boxes"], dvec["scores"]):
                        pb.append([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0])
                        ps.append(sc)
                pb, ps = nms(pb, ps, 0.5)
                pred[v][d]["boxes"] += pb; pred[v][d]["scores"] += ps
                local.setdefault(d, {})[v] = len(pb) > 0
        for d in DIOR_CLASSES_USED:
            gtbox[d] += gtb.get(d, [])
            for v in VARIANTS:
                hit[v][d].append(bool(local.get(d, {}).get(v, False)))
        done_stems.add(stem)
        if len(done_stems) % CKPT_EVERY == 0 or (i + 1) == len(imgs):
            save_ckpt()
            o = build()
            print(f"  [{len(done_stems)}/{len(imgs)}] {f} tiles={len(tiles)}  "
                  f"macroF1 {o['overhead']['macroF1']:.4f}->{o['tdm']['macroF1']:.4f}  "
                  f"mAP {o['overhead']['macro_mAP']:.4f}->{o['tdm']['macro_mAP']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    o = build(complete=True)
    save_ckpt()
    print(f"\n=== DOTA / {args.model}: overhead -> TDM ===")
    print(f"  macroF1 {o['overhead']['macroF1']:.4f} -> {o['tdm']['macroF1']:.4f}  "
          f"({o['delta']['macroF1']:+.4f})")
    print(f"  macro mAP {o['overhead']['macro_mAP']:.4f} -> {o['tdm']['macro_mAP']:.4f}  "
          f"({o['delta']['macro_mAP']:+.4f})")
    print(f"  mean DR {o['overhead']['meanDR']:.4f} -> {o['tdm']['meanDR']:.4f}  "
          f"({o['delta']['meanDR']:+.4f})")
    print(f"  McNemar pooled: miss->hit {o['pooled_n_oh_miss_tdm_hit']}, "
          f"hit->miss {o['pooled_n_oh_hit_tdm_miss']}, p={o['pooled_mcnemar_p']:.3g}")
    print(f"[saved] {op}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
