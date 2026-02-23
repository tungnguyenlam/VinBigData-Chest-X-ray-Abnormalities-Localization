"""
Entry point for training models and building the stacking ensemble.

Paths are read from paths.yaml at the repo root — edit that file instead
of passing --data every time.

Examples
--------
# Train YOLO with settings from paths.yaml
python -m src.train --arch yolo --preset medium

# Train Faster R-CNN on Kaggle (override path for this run only)
python -m src.train --arch faster_rcnn --preset medium \
    --data /kaggle/input/vinbigdata-chest-xray-abnormalities-detection

# Train all three models sequentially, predict val set image-by-image,
# then fuse predictions with WBF
python -m src.train --arch all --preset medium
"""
from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
from typing import Literal

import torch

from src.config import (
    DataConfig,
    EnsembleConfig,
    HardwarePreset,
    ModelConfig,
    get_data_root,
    get_output_root,
)
from src.data.dataset import build_dataloader
from src.models.base import BaseDetector
from src.predict_logger import evaluate_predictions, predict_and_log, stack_predictions


Arch = Literal["yolo", "faster_rcnn", "detr", "all"]
Preset = Literal["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(arch: str, data_cfg: DataConfig, model_cfg: ModelConfig) -> BaseDetector:
    if arch == "yolo":
        from src.models.yolo import YOLODetector
        return YOLODetector(data_cfg, model_cfg)
    elif arch == "faster_rcnn":
        from src.models.faster_rcnn import FasterRCNNDetector
        return FasterRCNNDetector(data_cfg, model_cfg)
    elif arch == "detr":
        from src.models.detr import DETRDetector
        return DETRDetector(data_cfg, model_cfg)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Single-model workflow: train → predict val → save → free memory
# ---------------------------------------------------------------------------

def train_and_predict(
    arch: str,
    preset: HardwarePreset,
    epochs: int | None,
    device: str | None,
    score_threshold: float,
    pred_batch_size: int,
) -> Path:
    """
    1. Train the model.
    2. Predict the val split one image at a time, write to JSONL.
    3. Save weights.
    4. Delete model from memory.

    Returns the path to the prediction JSONL file.
    """
    data_cfg = preset.data
    model_cfg = copy.deepcopy(preset.model)
    model_cfg.arch = arch

    if epochs is not None:
        model_cfg.epochs = epochs
    if device is not None:
        model_cfg.device = device

    print(f"\n{'='*60}")
    print(f"Training {arch.upper()}")
    print(f"  data root   : {data_cfg.root}")
    print(f"  image size  : {data_cfg.image_size}")
    print(f"  batch size  : {data_cfg.batch_size}")
    print(f"  num workers : {data_cfg.num_workers}")
    print(f"  epochs      : {model_cfg.epochs}")
    print(f"  device      : {model_cfg.device}")
    print(f"{'='*60}\n")

    # ---- Train ----
    model = build_model(arch, data_cfg, model_cfg)

    if arch == "yolo":
        model.train_model(train_loader=None, val_loader=None)
    else:
        output_format = "torchvision"
        train_loader = build_dataloader(
            root=data_cfg.root, split="train", cfg=data_cfg, output_format=output_format
        )
        val_loader = build_dataloader(
            root=data_cfg.root, split="val", cfg=data_cfg, output_format=output_format
        )
        model.train_model(train_loader, val_loader)

    # ---- Save weights ----
    weights_path = Path(model_cfg.output_dir) / arch / "best.pt"
    model.save(weights_path)
    print(f"  Saved weights → {weights_path}")

    # ---- Predict val (image by image) ----
    pred_path = Path(model_cfg.output_dir) / arch / "predictions_val.jsonl"
    predict_and_log(
        model=model,
        data_cfg=data_cfg,
        split="val",
        output_path=pred_path,
        score_threshold=score_threshold,
        batch_size=pred_batch_size,
    )

    # ---- Free memory before the next model ----
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  {arch.upper()} done — predictions → {pred_path}\n")
    return pred_path


# ---------------------------------------------------------------------------
# "all" workflow: train each model sequentially, then fuse
# ---------------------------------------------------------------------------

def train_all(
    preset: HardwarePreset,
    epochs: int | None,
    device: str | None,
    ensemble_cfg: EnsembleConfig,
    score_threshold: float,
    pred_batch_size: int,
) -> Path:
    archs = ["yolo", "faster_rcnn", "detr"]
    pred_files: list[Path] = []

    for arch in archs:
        arch_preset = copy.deepcopy(preset)
        pred_file = train_and_predict(
            arch=arch,
            preset=arch_preset,
            epochs=epochs,
            device=device,
            score_threshold=score_threshold,
            pred_batch_size=pred_batch_size,
        )
        pred_files.append(pred_file)

    # Optional: compute per-model mAP to set ensemble weights
    weights: list[float] | None = None
    data_cfg = preset.data
    try:
        maps: list[float] = []
        for arch, pf in zip(archs, pred_files):
            print(f"  Evaluating {arch} predictions...")
            result = evaluate_predictions(pf, data_cfg, split="val")
            maps.append(result.get("map", 0.0))
            print(f"    mAP = {maps[-1]:.4f}")
        total = sum(maps) or 1.0
        weights = [m / total for m in maps]
        print(f"  Ensemble weights: {weights}")
    except ImportError:
        print("  torchmetrics not installed — using equal weights.")

    # Fuse all predictions on disk (no models in memory)
    fused_path = Path(get_output_root()) / "ensemble" / "predictions_val.jsonl"
    stack_predictions(
        pred_files=pred_files,
        output_path=fused_path,
        cfg=ensemble_cfg,
        weights=weights,
    )

    print(f"\nEnsemble fused predictions → {fused_path}")
    return fused_path


# ---------------------------------------------------------------------------
# Predict-only (load saved weights and re-predict)
# ---------------------------------------------------------------------------

def predict_only(
    arch: str,
    weights_path: str | Path,
    preset: HardwarePreset,
    split: str,
    score_threshold: float,
    pred_batch_size: int,
    device: str | None,
) -> Path:
    """Load saved weights and predict without retraining."""
    data_cfg = preset.data
    model_cfg = copy.deepcopy(preset.model)
    model_cfg.arch = arch
    if device is not None:
        model_cfg.device = device

    model = build_model(arch, data_cfg, model_cfg)
    model.load(weights_path)
    print(f"  Loaded {arch} weights from {weights_path}")

    pred_path = Path(model_cfg.output_dir) / arch / f"predictions_{split}.jsonl"
    predict_and_log(
        model=model,
        data_cfg=data_cfg,
        split=split,
        output_path=pred_path,
        score_threshold=score_threshold,
        batch_size=pred_batch_size,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VinBigData localization training")

    parser.add_argument(
        "--arch",
        type=str,
        default="yolo",
        choices=["yolo", "faster_rcnn", "detr", "all"],
        help="Model to train. 'all' trains all three sequentially then fuses.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Hardware preset (controls batch size, image size, workers).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override data root path (default: read from paths.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output root path (default: read from paths.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from preset.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.0,
        help="Score threshold when logging predictions (0 = keep all).",
    )
    parser.add_argument(
        "--pred_batch",
        type=int,
        default=1,
        help="Batch size during prediction (1 = one image at a time, safest for RAM).",
    )
    parser.add_argument(
        "--predict_only",
        type=str,
        default=None,
        metavar="WEIGHTS_PATH",
        help="Skip training; load weights from this path and predict.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to predict when using --predict_only.",
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

    # Resolve paths: CLI arg > paths.yaml > default
    data_root = args.data or get_data_root()
    output_root = args.output or get_output_root()

    preset_factory = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[args.preset]
    preset = preset_factory(root=data_root, arch=args.arch if args.arch != "all" else "yolo")
    preset.model.output_dir = output_root

    ensemble_cfg = EnsembleConfig(
        iou_thr=args.ensemble_iou,
        skip_box_thr=args.ensemble_skip,
        method=args.ensemble_method,
    )

    if args.predict_only:
        pred_path = predict_only(
            arch=args.arch,
            weights_path=args.predict_only,
            preset=preset,
            split=args.split,
            score_threshold=args.score_thr,
            pred_batch_size=args.pred_batch,
            device=args.device,
        )
        print(f"\nPredictions saved → {pred_path}")
        return

    if args.arch == "all":
        fused_path = train_all(
            preset=preset,
            epochs=args.epochs,
            device=args.device,
            ensemble_cfg=ensemble_cfg,
            score_threshold=args.score_thr,
            pred_batch_size=args.pred_batch,
        )
        print(f"\nAll done. Fused predictions → {fused_path}")
    else:
        pred_path = train_and_predict(
            arch=args.arch,
            preset=preset,
            epochs=args.epochs,
            device=args.device,
            score_threshold=args.score_thr,
            pred_batch_size=args.pred_batch,
        )
        print(f"\n{args.arch.upper()} done. Predictions → {pred_path}")


if __name__ == "__main__":
    main()
