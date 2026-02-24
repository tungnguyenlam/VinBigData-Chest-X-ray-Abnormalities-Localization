"""
YOLO proof-of-concept training script.

Usage
-----
# Default: reads paths from paths.yaml, medium preset
python scripts/models/train_yolo.py

# Dataset already prepared — skip DICOM conversion, start training immediately
python scripts/models/train_yolo.py --prepared_dataset data/processed

# Small hardware (CPU, tiny model, 5 epochs)
python scripts/models/train_yolo.py --preset small --epochs 5

# Custom data path (e.g. Kaggle)
python scripts/models/train_yolo.py --data /kaggle/input/vinbigdata-chest-xray-abnormalities-detection

# Larger model, more epochs, specific GPU
python scripts/models/train_yolo.py --preset large --epochs 50 --device cuda:0

# Skip training, just run predictions from existing weights
# python scripts/models/train_yolo.py --predict_only outputs/yolo/train/weights/best.pt --prepared_dataset data/processed
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

# Ensure the repo root (two levels up from scripts/models/) is on sys.path so
# `src.*` imports work when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import (
    DataConfig,
    ModelConfig,
    HardwarePreset,
    get_data_root,
    get_output_root,
    get_processed_data_root,
)
from scripts.models.yolo import YOLODetector
from scripts.predict_logger import predict_and_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_preset(
    preset_name: str,
    data_root: str,
    output_root: str,
    model_size: str,
) -> tuple[DataConfig, ModelConfig]:
    preset = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[preset_name](root=data_root, arch="yolo")

    preset.model.output_dir = output_root
    preset.model.backbone_size = model_size
    return preset.data, preset.model


def print_config(data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
    print("\n" + "=" * 60)
    print("YOLO Training Configuration")
    print("=" * 60)
    print(f"  data root     : {data_cfg.root}")
    print(f"  image size    : {data_cfg.image_size}px")
    print(f"  batch size    : {data_cfg.batch_size}")
    print(f"  num workers   : {data_cfg.num_workers}")
    print(f"  val split     : {data_cfg.val_split:.0%}")
    print(
        f"  no-finding    : {'included (hard negatives)' if data_cfg.include_no_finding else 'excluded'}"
    )
    print(f"  WBF iou thr   : {data_cfg.wbf_iou_thr}")
    print(f"  model size    : yolov8{model_cfg.backbone_size}")
    print(f"  epochs        : {model_cfg.epochs}")
    print(f"  learning rate : {model_cfg.lr}")
    print(f"  device        : {model_cfg.device}")
    print(f"  AMP           : {model_cfg.amp}")
    print(f"  output dir    : {model_cfg.output_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on VinBigData chest X-ray localization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        default="medium",
        help=(
            "Hardware preset. "
            "small=CPU/<=8 GB (yolov8n, bs=4, img=640), "
            "medium=GPU 8-16 GB (yolov8s, bs=8, img=640), "
            "large=GPU >=24 GB (yolov8l, bs=16, img=1024)."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset root path. Overrides paths.yaml if given.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output root path. Overrides paths.yaml if given.",
    )
    parser.add_argument(
        "--model_size",
        choices=["n", "s", "m", "l", "x"],
        default=None,
        help="YOLOv8 model size. Overrides the preset default.",
    )
    parser.add_argument(
        "--no_finding",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="include_no_finding",
        help=(
            "Include 'No finding' images as hard negatives (default: yes). "
            "Use --no-no_finding to exclude them and train only on images with boxes."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size for training and prediction. Overrides preset default.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides the preset default.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        dest="batch_size",
        help="Training batch size. Overrides the preset default.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        dest="num_workers",
        help="Number of DataLoader workers. Overrides the preset default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu | cuda | cuda:0 | mps. Overrides the preset default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data or get_data_root()
    output_root = args.output or get_output_root()

    # Resolve default model size from preset, then allow override
    preset_default_size = {"small": "n", "medium": "s", "large": "l"}[args.preset]
    model_size = args.model_size or preset_default_size

    data_cfg, model_cfg = build_preset(args.preset, data_root, output_root, model_size)

    if args.image_size is not None:
        data_cfg.image_size = args.image_size
    if args.epochs is not None:
        model_cfg.epochs = args.epochs
    if args.batch_size is not None:
        data_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        data_cfg.num_workers = args.num_workers

    if args.device is not None:
        model_cfg.device = args.device
    else:
        import torch

        if torch.cuda.is_available():
            model_cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            model_cfg.device = "mps"
        else:
            model_cfg.device = "cpu"
    data_cfg.include_no_finding = args.include_no_finding

    # Optimization for MPS (Apple Silicon)
    if model_cfg.device == "mps":
        # Increase batch size for small models to saturate GPU (only if not explicitly set)
        if model_size == "n" and data_cfg.batch_size < 16 and args.batch_size is None:
            print(
                f"  MPS optimization: increasing batch size {data_cfg.batch_size} -> 16"
            )
            data_cfg.batch_size = 16
        # Reduce workers on macOS to avoid memory pressure/overhead (only if not explicitly set)
        if data_cfg.num_workers > 2 and args.num_workers is None:
            print(f"  MPS optimization: reducing workers {data_cfg.num_workers} -> 2")
            data_cfg.num_workers = 2

    prepared_dataset_root = Path(get_processed_data_root())

    print_config(data_cfg, model_cfg)
    if prepared_dataset_root:
        print(f"  prepared dataset : {prepared_dataset_root}\n")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    detector = YOLODetector(data_cfg, model_cfg)
    detector.train_model()

    # Save final weights explicitly
    final_weights = Path(output_root) / "yolo" / "best.pt"
    detector.save(final_weights)
    print(f"\nStep 2/3 — Weights saved → {final_weights}")

    # ------------------------------------------------------------------
    # Predict val split image-by-image
    # ------------------------------------------------------------------
    print("\nStep 3/3 — Running val predictions (one image at a time)...")
    pred_path = Path(output_root) / "yolo" / "predictions_val.jsonl"
    predict_and_log(
        model=detector,
        data_cfg=data_cfg,
        split="val",
        output_path=pred_path,
        score_threshold=0.0,
        batch_size=1,
        prepared_dataset_root=prepared_dataset_root,
    )

    del detector
    gc.collect()

    print("\nDone.")
    print(f"  Weights      → {final_weights}")
    print(f"  Predictions  → {pred_path}")
    print(
        "\nTo inspect predictions:\n"
        f'  python -c "'
        f"from scripts.predict_logger import load_predictions; "
        f"p = load_predictions('{pred_path}'); "
        f'img_id = next(iter(p)); print(img_id, p[img_id])"'
    )


if __name__ == "__main__":
    main()
