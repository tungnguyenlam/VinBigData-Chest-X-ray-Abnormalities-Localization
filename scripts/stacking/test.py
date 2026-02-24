"""
Fuses predictions from multiple models using Weighted Box Fusion (WBF).

Reads JSONL prediction files from each model's output directory and generates
a single combined predictions JSONL dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import EnsembleConfig, get_output_root
from scripts.predict_logger import stack_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse predictions using WBF")
    parser.add_argument(
        "--preds",
        nargs="+",
        required=True,
        help="Paths to JSONL prediction files to fuse (e.g. outputs/yolo/predictions_val.jsonl)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each prediction file (e.g. 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--ensemble_iou",
        type=float,
        default=0.5,
        help="WBF IoU threshold for ensemble fusion.",
    )
    parser.add_argument(
        "--ensemble_skip",
        type=float,
        default=0.05,
        help="WBF skip_box_thr for ensemble fusion.",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="wbf",
        choices=["wbf", "nms", "soft_nms"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensemble_cfg = EnsembleConfig(
        iou_thr=args.ensemble_iou,
        skip_box_thr=args.ensemble_skip,
        method=args.ensemble_method,
    )
    
    fused_path = Path(get_output_root()) / "ensemble" / f"fused_predictions_{args.ensemble_method}.jsonl"
    
    stack_predictions(
        pred_files=args.preds,
        output_path=fused_path,
        cfg=ensemble_cfg,
        weights=args.weights,
    )
    
    print(f"\nAll done. Fused predictions saved to → {fused_path}")


if __name__ == "__main__":
    main()
