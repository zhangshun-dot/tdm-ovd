"""
Phase 1: One-Factor-At-a-Time (OFAT) Experiment Pipeline

Tests each prompt axis independently by varying one axis at a time while
holding all others at baseline. For each DIOR class × axis × level, runs
GroundingDINO inference on 20 images.

Usage:
    python ofat_experiment.py                    # Run all axes, all classes
    python ofat_experiment.py --model gdino      # GroundingDINO only
    python ofat_experiment.py --model yoloworld   # YOLO-World (to be added)
    python ofat_experiment.py --axis granularity  # Single axis only
    python ofat_experiment.py --cls airplane,ship # Specific classes
"""
import os, sys, json, time, glob, argparse
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from scipy.stats import wilcoxon

# Import axes definition
sys.path.insert(0, os.path.dirname(__file__))
from axes_definition import (
    DIOR_CLASSES, AXIS_CONFIG,
    build_prompt, build_baseline, build_ofat_variants,
    GRANULARITY, SPATIAL, GEOMETRIC, SCALE, GRAMMAR
)

# ============================================================
# Configuration
# ============================================================
IMAGE_DIR = r"<REPO_ROOT>\dior_data\full20_images"
OUTPUT_DIR = r"<REPO_ROOT>\results\ofat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOX_THRESHOLD = 0.25
IMAGES_PER_CLASS = 20


# ============================================================
# Model Wrapper
# ============================================================
class GroundingDINOInference:
    """Reusable GroundingDINO inference wrapper."""

    def __init__(self, model_id='IDEA-Research/grounding-dino-tiny', device='cpu'):
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, dtype=torch.float32
        ).to(device).eval()
        self.device = device
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  Model loaded: {n_params:.0f}M params, device={device}")

    def detect(self, image, text_prompt):
        """Run detection on one image with one prompt. Returns dict with stats."""
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=BOX_THRESHOLD, target_sizes=target_sizes
        )[0]

        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy() if len(boxes) > 0 else np.array([])

        return {
            'num_detections': len(boxes),
            'boxes': boxes.tolist() if len(boxes) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'max_score': float(np.max(scores)) if len(scores) > 0 else 0.0,
            'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0.0,
        }


# ============================================================
# Image Loader
# ============================================================
def load_class_images(cls, n=IMAGES_PER_CLASS):
    """Load image paths for a given DIOR class."""
    pattern = os.path.join(IMAGE_DIR, f"{cls}_*.jpg")
    paths = sorted(glob.glob(pattern))[:n]
    return paths


# ============================================================
# Result Persistence (resumption-aware)
# ============================================================
def load_existing_results(output_path):
    """Load existing results if available (for resumption)."""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(output_path, data):
    """Save results atomically."""
    tmp = output_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


# ============================================================
# OFAT Runner
# ============================================================
class OFATRunner:
    def __init__(self, model, classes=None, axes=None):
        self.model = model
        self.classes = classes or DIOR_CLASSES
        self.axes = axes or [a for a in AXIS_CONFIG if not AXIS_CONFIG[a].get("phase2_only")]

    def run(self):
        total_variants = sum(
            len(AXIS_CONFIG[a]["perturbations"]) for a in self.axes
        )
        total_inferences = len(self.classes) * (1 + total_variants) * IMAGES_PER_CLASS
        print(f"\n{'='*60}")
        print(f"OFAT EXPERIMENT")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Axes: {len(self.axes)} ({', '.join(self.axes)})")
        print(f"  Variants: 1 baseline + {total_variants} perturbations")
        print(f"  Total inferences: {total_inferences}")
        print(f"  Est. time: {total_inferences * 1.5 / 3600:.1f}h (CPU)")
        print(f"{'='*60}\n")

        run_idx = 0
        t_start = time.time()

        for cls in self.classes:
            # Load images
            img_paths = load_class_images(cls)
            if not img_paths:
                print(f"  [{cls}] No images found — skipping")
                continue
            images = [(os.path.basename(p), Image.open(p).convert("RGB"))
                      for p in img_paths]

            # Output file per class
            out_path = os.path.join(OUTPUT_DIR, f"{cls}.json")
            cls_results = load_existing_results(out_path)

            # --- BASELINE ---
            baseline_prompt = build_baseline(cls)
            if "baseline" not in cls_results:
                cls_results["baseline"] = {"prompt": baseline_prompt, "results": []}
                for img_name, img in images:
                    run_idx += 1
                    r = self.model.detect(img, baseline_prompt)
                    cls_results["baseline"]["results"].append({
                        "image": img_name, **r
                    })
                save_results(out_path, cls_results)
                print(f"  [{cls}] baseline: \"{baseline_prompt}\" — done ({len(images)} imgs)")

            # --- AXIS PERTURBATIONS ---
            ofat = build_ofat_variants(cls)
            for axis_name in self.axes:
                if axis_name not in ofat:
                    continue
                if axis_name not in cls_results:
                    cls_results[axis_name] = {}

                for level, prompt in ofat[axis_name].items():
                    if level in cls_results[axis_name]:
                        continue  # Already done (resumption)

                    cls_results[axis_name][level] = {"prompt": prompt, "results": []}
                    for img_name, img in images:
                        run_idx += 1
                        r = self.model.detect(img, prompt)
                        cls_results[axis_name][level]["results"].append({
                            "image": img_name, **r
                        })

                    save_results(out_path, cls_results)

                    # Live summary for this level
                    dets = [x['num_detections'] for x in cls_results[axis_name][level]["results"]]
                    det_rate = sum(1 for d in dets if d > 0) / len(dets) * 100
                    avg_det = np.mean(dets)
                    elapsed = time.time() - t_start
                    remaining = (total_inferences - run_idx) * 1.5
                    print(f"  [{cls}] {axis_name:12s} | {level:12s} | "
                          f"avg_det={avg_det:.1f} det_rate={det_rate:.0f}% | "
                          f"[{run_idx}/{total_inferences}] "
                          f"elapsed={elapsed/60:.0f}m ETA={remaining/60:.0f}m")

            save_results(out_path, cls_results)

        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"OFAT COMPLETE: {run_idx} inferences in {elapsed/60:.1f} min")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"{'='*60}")


# ============================================================
# Quick Analysis (run after all OFAT data collected)
# ============================================================
def analyze_ofat_results():
    """Analyze completed OFAT results across all classes."""
    print(f"\n{'='*60}")
    print("OFAT ANALYSIS")
    print(f"{'='*60}")

    summary = {}
    for cls in DIOR_CLASSES:
        path = os.path.join(OUTPUT_DIR, f"{cls}.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        cls_summary = {}
        # Baseline
        if "baseline" in data:
            dets = [r['num_detections'] for r in data['baseline']['results']]
            cls_summary['baseline'] = {
                'avg_det': float(np.mean(dets)),
                'det_rate': float(sum(1 for d in dets if d > 0) / len(dets)),
                'avg_score': float(np.mean([r['max_score'] for r in data['baseline']['results']])),
            }

        # Each axis
        for axis_name in AXIS_CONFIG:
            if axis_name not in data:
                continue
            cls_summary[axis_name] = {}
            for level, ldata in data[axis_name].items():
                dets = [r['num_detections'] for r in ldata['results']]
                cls_summary[axis_name][level] = {
                    'prompt': ldata['prompt'],
                    'avg_det': float(np.mean(dets)),
                    'det_rate': float(sum(1 for d in dets if d > 0) / len(dets)),
                    'avg_score': float(np.mean([r['max_score'] for r in ldata['results']])),
                }

            # Simple Wilcoxon test: baseline vs each perturbation
            if 'baseline' in data:
                base_dets = [r['num_detections'] for r in data['baseline']['results']]
                for level in AXIS_CONFIG[axis_name]['perturbations']:
                    if level in data[axis_name]:
                        pert_dets = [r['num_detections'] for r in data[axis_name][level]['results']]
                        try:
                            w, p = wilcoxon(base_dets, pert_dets, zero_method='zsplit')
                            cls_summary[axis_name][level]['wilcoxon_p'] = float(p)
                            cls_summary[axis_name][level]['wilcoxon_sig'] = (
                                '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                            )
                        except:
                            cls_summary[axis_name][level]['wilcoxon_p'] = None

        summary[cls] = cls_summary

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")

    # Print top-line results
    print(f"\n{'─'*80}")
    print(f"{'Class':<25} {'Axis':<14} {'Level':<12} {'AvgDet':>7} {'Rate':>7} {'vsBase':>10}")
    print(f"{'─'*80}")
    for cls, cdata in summary.items():
        base_det = cdata.get('baseline', {}).get('avg_det', 0)
        for axis, adata in cdata.items():
            if axis == 'baseline':
                continue
            for level, ldata in adata.items():
                delta = ldata['avg_det'] - base_det
                sig = ldata.get('wilcoxon_sig', '')
                print(f"{cls:<25} {axis:<14} {level:<12} {ldata['avg_det']:>7.2f} "
                      f"{ldata['det_rate']:>6.0%} {delta:>+9.2f} {sig}")

    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gdino", choices=["gdino", "yoloworld"])
    parser.add_argument("--axis", type=str, help="Run single axis only")
    parser.add_argument("--cls", type=str, help="Comma-separated class names")
    parser.add_argument("--analyze", action="store_true", help="Analyze results only")
    args = parser.parse_args()

    if args.analyze:
        analyze_ofat_results()
        sys.exit(0)

    # Filter classes and axes
    classes = [c.strip() for c in args.cls.split(",")] if args.cls else None
    axes = [args.axis] if args.axis else None

    if args.model == "gdino":
        model = GroundingDINOInference()
    else:
        print("YOLO-World not yet implemented — use --model gdino")
        sys.exit(1)

    runner = OFATRunner(model, classes=classes, axes=axes)
    runner.run()

    # Auto-analyze after completion
    analyze_ofat_results()
