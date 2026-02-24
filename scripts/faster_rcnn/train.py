import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset
from scripts.data.dataset import build_dataloader
from scripts.models.faster_rcnn import FasterRCNNDetector


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
    args = parser.parse_args()

    # Resolve roots
    from scripts.config import get_data_root, get_output_root

    data_root = args.data or get_data_root()
    output_root = args.output or get_output_root()

    # Load config without resetting values
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
    model_cfg.output_dir = output_root

    print(
        f"Training Faster R-CNN (Epochs: {model_cfg.epochs}, Batch: {data_cfg.batch_size})..."
    )
    model = FasterRCNNDetector(data_cfg, model_cfg)

    train_loader = build_dataloader(
        root=data_cfg.root, split="train", cfg=data_cfg, output_format="torchvision"
    )
    val_loader = build_dataloader(
        root=data_cfg.root, split="val", cfg=data_cfg, output_format="torchvision"
    )

    model.train_model(train_loader, val_loader)


if __name__ == "__main__":
    main()
