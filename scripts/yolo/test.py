import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import HardwarePreset, get_processed_data_root
from scripts.models.yolo import YOLODetector
from scripts.predict_logger import predict_and_log, evaluate_predictions, evaluate_froc


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
    parser.add_argument(
        "--native-eval",
        action="store_true",
        help="Use Ultralytics native val() for evaluation (matches training metrics)",
    )
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

    prepared_dataset_root = Path(get_processed_data_root())
    if not (prepared_dataset_root / "dataset.yaml").exists():
        prepared_dataset_root = None

    # --- Native Ultralytics evaluation ---
    if args.native_eval:
        yaml_path = (
            prepared_dataset_root / "dataset.yaml" if prepared_dataset_root else None
        )
        if yaml_path is None or not yaml_path.exists():
            print("Error: --native-eval requires data/processed/dataset.yaml")
            return

        print(f"Running Ultralytics native val() on {args.split} split...")
        results = model.model.val(
            data=str(yaml_path),
            split=args.split,
            imgsz=data_cfg.image_size,
            batch=data_cfg.batch_size,
            device=model_cfg.device,
            verbose=True,
        )

        metrics = {
            "map": float(results.box.map),
            "map_50": float(results.box.map50),
            "map_75": float(results.box.map75),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        # Per-class maps if available
        if hasattr(results.box, "maps") and len(results.box.maps) > 0:
            for i, v in enumerate(results.box.maps):
                metrics[f"map_class_{i}"] = float(v)

        display_names = {
            "map": "mAP@50-95",
            "map_50": "mAP@50",
            "map_75": "mAP@75",
            "precision": "Precision",
            "recall": "Recall",
        }
        
        keys_to_keep = ["map", "map_50", "map_75", "precision", "recall"]
        metrics = {k: v for k, v in metrics.items() if k in keys_to_keep}
        print("\n  === Results (Ultralytics native) ===")
        for k, v in metrics.items():
            label = display_names.get(k, k)
            print(f"  {label}: {v:.4f}")

        # Save
        metrics_path = (
            Path(model_cfg.output_path) / "yolo" / f"native_metrics_{args.split}.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics["split"] = args.split
        metrics["weights"] = args.weights
        metrics["eval_mode"] = "ultralytics_native"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to {metrics_path}")
        return

    # --- Standard predict + torchmetrics evaluation ---
    out_path = Path(model_cfg.output_path) / "yolo" / f"predictions_{args.split}.jsonl"

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

        # --- FROC evaluation (moved up to extract Precision/Recall) ---
        print(f"\n  Computing FROC score on {args.split} split...")
        froc_result = evaluate_froc(
            out_path,
            data_cfg,
            split=args.split,
            prepared_dataset_root=prepared_dataset_root,
        )
        
        metrics["precision"] = froc_result["precision"]
        metrics["recall"] = froc_result["recall"]

        # Print with explicit mAP50-95 label
        display_names = {
            "map": "mAP@50-95",
            "map_50": "mAP@50",
            "map_75": "mAP@75",
            "precision": "Precision",
            "recall": "Recall",
        }
        
        keys_to_keep = ["map", "map_50", "map_75", "precision", "recall"]
        metrics_print = {k: v for k, v in metrics.items() if k in keys_to_keep}
        
        print("\n  === Results ===")
        for k, v in metrics_print.items():
            label = display_names.get(k, k)
            if isinstance(v, float):
                print(f"  {label}: {v:.4f}")
            else:
                print(f"  {label}: {v}")

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
