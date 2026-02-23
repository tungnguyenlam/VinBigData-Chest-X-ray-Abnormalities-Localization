import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import HardwarePreset
from src.data.dataset import build_dataloader
from src.models.detr import DETRDetector

def main():
    parser = argparse.ArgumentParser(description="Train DETR on VinBigData.")
    parser.add_argument("--preset", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config without resetting values
    preset = {"small": HardwarePreset.small, "medium": HardwarePreset.medium, "large": HardwarePreset.large}[args.preset](arch="detr")
    data_cfg = preset.data
    model_cfg = preset.model
    if args.device:
        model_cfg.device = args.device
    if args.epochs:
        model_cfg.epochs = args.epochs

    print(f"Training DETR (Epochs: {model_cfg.epochs}, Batch: {data_cfg.batch_size})...")
    model = DETRDetector(data_cfg, model_cfg)

    train_loader = build_dataloader(
        root=data_cfg.root, split="train", cfg=data_cfg, output_format="torchvision"
    )
    val_loader = build_dataloader(
        root=data_cfg.root, split="val", cfg=data_cfg, output_format="torchvision"
    )
    
    model.train_model(train_loader, val_loader)

if __name__ == "__main__":
    main()
