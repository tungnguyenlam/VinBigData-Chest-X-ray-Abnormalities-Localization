import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on VinBigData.")
    parser.add_argument(
        "--preset", type=str, default="medium", choices=["small", "medium", "large"]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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
        "--no_finding",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="include_no_finding",
        help="Include 'No finding' images as hard negatives.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size for training and prediction. Overrides preset default.",
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
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet50v2", "resnet101"],
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional tag appended to the run directory name, e.g. 'exp1'.",
    )
    args = parser.parse_args()

    # Enable TF32 for faster training
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster training")

    # ------------------------------------------------------------------ #
    # Resolve roots
    # ------------------------------------------------------------------ #
    from scripts.config import get_data_root, get_output_root

    data_root = args.data or get_data_root()
    output_root = args.output or get_output_root()

    # ------------------------------------------------------------------ #
    # Build config from preset + CLI overrides
    # ------------------------------------------------------------------ #
    preset = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[args.preset](root=data_root, arch="faster_rcnn")
    data_cfg = preset.data
    model_cfg = preset.model

    if args.device:
        model_cfg.device = args.device
    if args.epochs:
        model_cfg.epochs = args.epochs
    if args.image_size is not None:
        data_cfg.image_size = args.image_size
    if args.batch_size is not None:
        data_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        data_cfg.num_workers = args.num_workers
    if args.backbone is not None:
        model_cfg.backbone_size = args.backbone
    data_cfg.include_no_finding = args.include_no_finding

    # ------------------------------------------------------------------ #
    # Create a unique, timestamped run directory
    #   outputs/faster_rcnn/runs/<timestamp>[_<run_name>]/
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    run_name = f"{timestamp}{run_suffix}"
    run_dir = Path(output_root) / "faster_rcnn" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory : {run_dir}")

    # ------------------------------------------------------------------ #
    # Save config snapshot
    # ------------------------------------------------------------------ #
    config_snapshot = {
        "run_name": run_name,
        "preset": args.preset,
        "arch": model_cfg.arch,
        "backbone": model_cfg.backbone_size,
        "epochs": model_cfg.epochs,
        "lr": model_cfg.lr,
        "weight_decay": model_cfg.weight_decay,
        "device": model_cfg.device,
        "amp": model_cfg.amp,
        "grad_clip": model_cfg.grad_clip,
        "warmup_epochs": model_cfg.warmup_epochs,
        "image_size": data_cfg.image_size,
        "batch_size": data_cfg.batch_size,
        "num_workers": data_cfg.num_workers,
        "include_no_finding": data_cfg.include_no_finding,
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # ------------------------------------------------------------------ #
    # Build model & data loaders
    # ------------------------------------------------------------------ #
    from scripts.data.dataset import build_dataloader
    from scripts.models.faster_rcnn import FasterRCNNDetector

    model = FasterRCNNDetector(data_cfg, model_cfg, run_dir=run_dir)

    train_loader = build_dataloader(
        root=data_cfg.root, split="train", cfg=data_cfg, output_format="torchvision"
    )
    val_loader = build_dataloader(
        root=data_cfg.root, split="val", cfg=data_cfg, output_format="torchvision"
    )

    print(
        f"Training Faster R-CNN  (epochs={model_cfg.epochs}, batch={data_cfg.batch_size}, "
        f"backbone={model_cfg.backbone_size}, device={model_cfg.device})"
    )

    # ------------------------------------------------------------------ #
    # Train — returns list of per-epoch dicts
    # ------------------------------------------------------------------ #
    history = model.train_model(train_loader, val_loader)

    # ------------------------------------------------------------------ #
    # Save loss history as CSV via pandas
    # ------------------------------------------------------------------ #
    history_df = pd.DataFrame(history)          # columns: epoch, train_loss, val_loss, lr
    csv_path = run_dir / "loss_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Loss history  : {csv_path}")

    # ------------------------------------------------------------------ #
    # Update config snapshot with finish time
    # ------------------------------------------------------------------ #
    config_snapshot["finished_at"] = datetime.now().isoformat()
    config_snapshot["best_val_loss"] = float(history_df["val_loss"].min())
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"Training complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
