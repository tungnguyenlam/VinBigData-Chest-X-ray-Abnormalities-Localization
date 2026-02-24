import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset, get_processed_data_root
from scripts.models.yolo import YOLODetector
from scripts.predict_logger import predict_and_log, evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Test YOLO on VinBigData.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--preset", type=str, default="medium", choices=["small", "medium", "large"]
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--score_thr", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config without resetting values
    preset = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[args.preset](arch="yolo")
    data_cfg = preset.data
    model_cfg = preset.model
    if args.device:
        model_cfg.device = args.device

    print(f"Testing YOLO on {args.split} split...")
    model = YOLODetector(data_cfg, model_cfg)
    model.load(args.weights)

    out_path = Path(model_cfg.output_path) / "yolo" / f"predictions_{args.split}.jsonl"
    prepared_dataset_root = Path(get_processed_data_root())
    if not (prepared_dataset_root / "dataset.yaml").exists():
        prepared_dataset_root = None

    predict_and_log(
        model=model,
        data_cfg=data_cfg,
        split=args.split,
        output_path=out_path,
        score_threshold=args.score_thr,
        batch_size=1,
        prepared_dataset_root=prepared_dataset_root,
    )
    print(f"Predictions saved to {out_path}")

    if args.split != "test":
        print(f"Evaluating predictions on {args.split} split...")
        try:
            metrics = evaluate_predictions(
                out_path,
                data_cfg,
                split=args.split,
                prepared_dataset_root=prepared_dataset_root,
            )
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
