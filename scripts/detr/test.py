import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset
from scripts.models.detr import DETRDetector
from scripts.predict_logger import predict_and_log

def main():
    parser = argparse.ArgumentParser(description="Test DETR on VinBigData.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt or hf_directory")
    parser.add_argument("--preset", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--score_thr", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config without resetting values
    preset = {"small": HardwarePreset.small, "medium": HardwarePreset.medium, "large": HardwarePreset.large}[args.preset](arch="detr")
    data_cfg = preset.data
    model_cfg = preset.model
    if args.device:
        model_cfg.device = args.device

    print(f"Testing DETR on {args.split} split...")
    model = DETRDetector(data_cfg, model_cfg)
    model.load(args.weights)

    out_path = Path(model_cfg.output_path) / "detr" / f"predictions_{args.split}.jsonl"
    
    predict_and_log(
        model=model,
        data_cfg=data_cfg,
        split=args.split,
        output_path=out_path,
        score_threshold=args.score_thr,
        batch_size=1,
    )
    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
