# -*- coding: utf-8 -*-
"""
Unified evaluation harness for the TDM paper rewrite (2026-08-15).

Locks one reproducible口径 for the whole paper:
  batch   = modeling/dior_data/full20_images (400 img, 20 classes x 20)
  weights = local HF snapshots (offline, deterministic version)
  conf    = 0.25 (post_process threshold)
  DR      = mean over images of [num_detections > 0]          (discovery metric)
  mAP     = mean over classes of AP (IoU 0.5, 11-pt Pascal)   (standard metric)
  F1      = micro-averaged over all images/classes, IoU 0.25  (continuity metric)

Fixes vs. old compute_map.py:
  - load_gt keyed by image_id (the old `break` bug dropped GT for multi-class
    images under every class name except one). Here one image_id -> full GT,
    looked up by parsing the filename, so every class gets its GT.
"""
import os, sys, json, re, hashlib, glob
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from axes_definition import DIOR_CLASSES, GRANULARITY, AXIS_CONFIG, build_prompt

# ── paths ────────────────────────────────────────────────────────────
IMG_DIR = r"<REPO_ROOT>\dior_data\eval_v2_images"
PARQUET_DIR = r"<REPO_ROOT>\dior_data\data"
CACHE_DIR = r"<REPO_ROOT>\results\comparison\_det_cache"

CONF_THRESH = 0.25
IMAGES_PER_CLASS = 200
IOU_MAP = 0.5   # mAP IoU
IOU_F1 = 0.25   # F1 continuity IoU

# ── model specs (offline snapshots lock the weight version) ──────────
HF_HUB = r"<HF_CACHE>\hub"
MODEL_SPECS = {
    "gdino-tiny": {
        "type": "hf-gdino",
        "snapshot": os.path.join(HF_HUB, r"models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71"),
        "cache_key": "gdinotiny",
    },
    "gdino-base": {
        "type": "hf-gdino",
        "snapshot": os.path.join(HF_HUB, r"models--IDEA-Research--grounding-dino-base\snapshots\12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
        "cache_key": "gdinobase",
    },
    "owlv2": {
        "type": "hf-gdino",
        "snapshot": os.path.join(HF_HUB, r"models--google--owlv2-base-patch16-ensemble\snapshots\cfd3195ba4ea9592eec887ded089f4c08eff231d"),
        "cache_key": "owlv2",
    },
}

# ── 5 domain prefixes (exact strings from paper TABLE III/IV) ────────
PREFIXES = {
    "bare":           lambda cls: GRANULARITY[cls]["standard"],
    "satellite":      lambda cls: f"a satellite image of {GRANULARITY[cls]['standard']}",
    "aerial":         lambda cls: f"an aerial view of {GRANULARITY[cls]['standard']}",
    "overhead":       lambda cls: f"an overhead image of {GRANULARITY[cls]['standard']}",
    "remote_sensing": lambda cls: f"a remote sensing image of {GRANULARITY[cls]['standard']}",
}

MODIFIER_AXES = [a for a in AXIS_CONFIG if a not in ("grammar", "negation")]


# ════════════════════════════════════════════════════════════════════
# GT loading (fixed)
# ════════════════════════════════════════════════════════════════════
def load_gt():
    """Return {image_id(int): {w, h, bboxes_xywh(list), cats(list)}}."""
    gt = {}
    for shard in glob.glob(os.path.join(PARQUET_DIR, "train-*.parquet")):
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            img_id = int(row["image_id"])
            objs = row["objects"]
            n = len(objs["bbox"])
            gt[img_id] = {
                "w": int(row["width"]), "h": int(row["height"]),
                "bboxes": [objs["bbox"][i].tolist() for i in range(n)],
                "cats": [int(objs["category"][i]) for i in range(n)],
            }
    return gt


def parse_filename(fname):
    """'airplane_2003.jpg' -> ('airplane', 2003)."""
    m = re.match(r"(.+?)_(\d+)\.jpg$", fname)
    return m.group(1), int(m.group(2))


def class_images(cls, gt):
    """Return list of (fname, image_id) for a class, in deterministic order."""
    out = []
    for f in sorted(os.listdir(IMG_DIR)):
        if not f.endswith(".jpg"):
            continue
        c, iid = parse_filename(f)
        if c == cls and iid in gt:
            out.append((f, iid))
            if len(out) >= IMAGES_PER_CLASS:
                break
    return out


# ════════════════════════════════════════════════════════════════════
# Detection with disk cache
# ════════════════════════════════════════════════════════════════════
def _cache_path(model_key, prompt):
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, model_key, f"{h}.json")


def load_detections(model_key, prompt):
    """Return cached {fname: {boxes, scores}} or None."""
    p = _cache_path(model_key, prompt)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_detections(model_key, prompt, dets):
    p = _cache_path(model_key, prompt)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dets, f, ensure_ascii=False)


def detect_all(processor, model, prompt, images, device, conf=CONF_THRESH):
    """Run prompt over list of (fname, PIL.Image). Return {fname: {boxes, scores}}."""
    out = {}
    for fname, img in images:
        inputs = processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = processor.post_process_grounded_object_detection(
            outputs, threshold=conf, target_sizes=target)[0]
        out[fname] = {
            "boxes": r["boxes"].cpu().numpy().tolist(),
            "scores": [float(s) for s in r["scores"].cpu().numpy()],
        }
    return out


def load_model(model_key, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    spec = MODEL_SPECS[model_key]
    proc = AutoProcessor.from_pretrained(spec["snapshot"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        spec["snapshot"], dtype=torch.float32).to(device).eval()
    return proc, model


# ════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════
def det_rate(dets):
    """DR = mean over images of [>=1 detection]."""
    n = len(dets)
    if n == 0:
        return 0.0
    return float(np.mean([len(d["boxes"]) > 0 for d in dets.values()]))


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-10)


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=IOU_MAP):
    """11-point Pascal VOC AP for one class."""
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    idx = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in idx]
    tp = np.zeros(len(pred_boxes)); fp = np.zeros(len(pred_boxes))
    matched = [False] * len(gt_boxes)
    for i, pb in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp[i] = 1; matched[best_j] = True
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-10)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += np.max(precisions[mask]) if np.any(mask) else 0.0
    return ap / 11.0


def tp_fp_fn(pred_boxes, gt_boxes, iou_thresh):
    """Return (tp, fp, fn) for one class given its GT boxes."""
    matched = [False] * len(gt_boxes)
    tp = 0; fp = 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1; matched[best_j] = True
        else:
            fp += 1
    fn = sum(1 for m in matched if not m)
    return tp, fp, fn


def micro_f1(tp, fp, fn):
    if tp + fp == 0:
        p = 0.0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        r = 0.0
    else:
        r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ════════════════════════════════════════════════════════════════════
# High-level: evaluate a prompt-builder over all classes
# ════════════════════════════════════════════════════════════════════
def evaluate_class(model_key, processor, model, device, gt, cls, prompt, cache_only=False):
    """Evaluate one (class, prompt): return {DR, AP, tp, fp, fn, n_gt, prompt}."""
    images = [(f, Image.open(os.path.join(IMG_DIR, f)).convert("RGB"))
              for f, _ in class_images(cls, gt)]
    dets = load_detections(model_key, prompt)
    if dets is None:
        if cache_only:
            return None
        dets = detect_all(processor, model, prompt, images, device)
        save_detections(model_key, prompt, dets)

    gt_boxes_cls = []
    for f, iid in class_images(cls, gt):
        g = gt[iid]
        for b, c in zip(g["bboxes"], g["cats"]):
            if _cat_name(c) == cls:
                gt_boxes_cls.append(xywh_to_xyxy(b))

    pred_boxes = [b for d in dets.values() for b in d["boxes"]]
    pred_scores = [s for d in dets.values() for s in d["scores"]]

    tp, fp, fn = tp_fp_fn(pred_boxes, gt_boxes_cls, IOU_F1)
    return {
        "DR": det_rate(dets),
        "AP": compute_ap(pred_boxes, pred_scores, gt_boxes_cls, IOU_MAP),
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": len(gt_boxes_cls),
        "prompt": prompt,
        "pred_boxes": pred_boxes, "pred_scores": pred_scores, "gt_boxes": gt_boxes_cls,
    }


def evaluate_builder(model_key, processor, model, device, gt, builder, cache_only=False):
    """
    builder: cls -> prompt string (per-class).
    Returns summary dict with per-class DR/AP and pooled micro-F1.
    """
    per_class = {}
    tp_sum = fp_sum = fn_sum = 0
    for cls in DIOR_CLASSES:
        r = evaluate_class(model_key, processor, model, device, gt, cls,
                           builder(cls), cache_only=cache_only)
        if r is None:
            return None
        per_class[cls] = r
        tp_sum += r["tp"]; fp_sum += r["fp"]; fn_sum += r["fn"]

    summary = {
        "DR": float(np.mean([r["DR"] for r in per_class.values()])),
        "mAP": float(np.mean([r["AP"] for r in per_class.values()])),
        "F1": micro_f1(tp_sum, fp_sum, fn_sum),
        "n_tp": tp_sum, "n_fp": fp_sum, "n_fn": fn_sum,
    }
    return {"summary": summary, "per_class": per_class}


def _cat_name(cat_id):
    # HF DIOR category ids follow DIOR_CATS order (compute_map.py ID2CAT)
    DIOR_CATS = ["airplane", "airport", "baseball_field", "basketball_court", "bridge",
                 "chimney", "dam", "expressway_service_area", "expressway_toll_station",
                 "golf_course", "ground_track_field", "harbor", "overpass", "ship",
                 "stadium", "storage_tank", "tennis_court", "train_station", "vehicle", "windmill"]
    return DIOR_CATS[cat_id] if 0 <= cat_id < len(DIOR_CATS) else None
