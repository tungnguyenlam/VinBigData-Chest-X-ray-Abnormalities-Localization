import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset, get_processed_data_root
from scripts.models.faster_rcnn import FasterRCNNDetector
from scripts.predict_logger import predict_and_log, evaluate_predictions, evaluate_froc


def main():
    parser = argparse.ArgumentParser(description="Test Faster R-CNN on VinBigData.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--preset", type=str, default="medium", choices=["small", "medium", "large"]
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--score_thr", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet50v2", "resnet101"],
        help="Backbone architecture.",
    )
    args = parser.parse_args()

    # Load config without resetting values
    preset = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[args.preset](arch="faster_rcnn")
    data_cfg = preset.data
    model_cfg = preset.model
    # Try to auto-detect backbone from metadata if not explicitly provided
    weights_path = Path(args.weights)
    meta_path = weights_path.with_suffix(".meta.json")
    
    if args.backbone:
        model_cfg.backbone_size = args.backbone
    elif meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "backbone" in meta:
                model_cfg.backbone_size = meta["backbone"]
                print(f"Auto-detected backbone from {meta_path.name}: {model_cfg.backbone_size}")
        except Exception as e:
            print(f"Failed to load metadata: {e}")

    if args.device:
        model_cfg.device = args.device

    prepared_dataset_root = Path(get_processed_data_root())
    if not (prepared_dataset_root / "dataset.yaml").exists():
        prepared_dataset_root = None

    print(f"Testing Faster R-CNN on {args.split} split...")
    model = FasterRCNNDetector(data_cfg, model_cfg)
    model.load(args.weights)

    out_path = (
        Path(model_cfg.output_path) / "faster_rcnn" / f"predictions_{args.split}.jsonl"
    )

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

    print(f"Evaluating predictions on {args.split} split...")
    try:
        metrics = evaluate_predictions(
            out_path,
            data_cfg,
            split=args.split,
            prepared_dataset_root=prepared_dataset_root,
        )

        display_names = {
            "map": "mAP@50-95",
            "map_50": "mAP@50",
            "map_75": "mAP@75",
        }
        print("\n  === Results ===")
        for k, v in metrics.items():
            label = display_names.get(k, k)
            if isinstance(v, float):
                print(f"  {label}: {v:.4f}")
            else:
                print(f"  {label}: {v}")

        # --- FROC evaluation ---
        print(f"\n  Computing FROC score on {args.split} split...")
        froc_result = evaluate_froc(
            out_path,
            data_cfg,
            split=args.split,
            prepared_dataset_root=prepared_dataset_root,
        )

        print(f"\n  === FROC Results (IoU={froc_result['iou_threshold']}) ===")
        print(f"  FROC Score: {froc_result['froc_score']:.4f}")
        print(
            f"  GT Lesions: {froc_result['total_gt_lesions']}  |  Images: {froc_result['total_images']}"
        )
        for rate, sens in zip(froc_result["fp_rates"], froc_result["sensitivities"]):
            print(f"    Sensitivity @ {rate} FP/img: {sens:.4f}")

        # Merge FROC into metrics
        metrics["froc_score"] = froc_result["froc_score"]
        metrics["froc_sensitivities"] = froc_result["sensitivities"]
        metrics["froc_fp_rates"] = froc_result["fp_rates"]
        metrics["froc_iou_threshold"] = froc_result["iou_threshold"]

        # Save metrics to JSON
        metrics_path = out_path.with_suffix(".metrics.json")
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                serializable[k] = round(v, 6)
            elif isinstance(v, (int, str, bool, list)):
                serializable[k] = v
            else:
                serializable[k] = str(v)
        serializable["split"] = args.split
        serializable["weights"] = args.weights
        serializable["score_threshold"] = args.score_thr

        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Metrics saved to {metrics_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
