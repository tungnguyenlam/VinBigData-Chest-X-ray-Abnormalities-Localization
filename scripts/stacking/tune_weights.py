"""
Auto-tunes ensemble weights using val predictions.

Two strategies available:
1. mAP-proportional (default): evaluates each model's val JSONL individually, 
   and uses its mAP score as the proportional weight.
2. Grid search (--grid-search): sweeps combination of weights from 0.0 to 2.0 
   in steps of 0.1, evaluating fused mAP for each, returning the best.

Outputs `tuned_weights.json` to the val output folder.
"""
from __future__ import annotations

import argparse
import json
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.config import EnsembleConfig, HardwarePreset, get_output_root, get_processed_data_root
from scripts.predict_logger import stack_predictions, evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ensemble weights from val predictions")
    
    # Model prediction flags
    parser.add_argument(
        "--yolo-pred",
        type=str,
        help="Path to YOLO predictions JSONL (must be val split)",
    )
    parser.add_argument(
        "--frcnn-pred",
        type=str,
        help="Path to Faster R-CNN predictions JSONL (must be val split)",
    )
    parser.add_argument(
        "--detr-pred",
        type=str,
        help="Path to DETR predictions JSONL (must be val split)",
    )
    
    # Strategy
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use grid search instead of mAP-proportional (slower, but can find better combinations).",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Step size for grid search sweep (default: 0.1).",
    )
    parser.add_argument(
        "--grid-max",
        type=float,
        default=2.0,
        help="Max weight value to test in grid search (default: 2.0).",
    )
    
    # Ensemble parameters (used by grid search during fusion)
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="wbf",
        choices=["wbf", "nms", "soft_nms"],
    )
    parser.add_argument(
        "--ensemble-iou",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ensemble-skip",
        type=float,
        default=0.05,
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
    )

    return parser.parse_args()


def tune_map_proportional(
    model_preds: list[str],
    model_names: list[str],
    data_cfg,
    prepared_dataset_root: Path | None,
) -> tuple[list[float], float]:
    """
    Evaluates each model independently. Weights are proportional to mAP / total_map.
    """
    print("\n[Strategy: mAP-proportional]")
    maps = []
    
    for name, pred_path in zip(model_names, model_preds):
        print(f"  Evaluating {name} ({pred_path})...")
        try:
            metrics = evaluate_predictions(
                pred_file=pred_path,
                data_cfg=data_cfg,
                split="val",
                prepared_dataset_root=prepared_dataset_root,
            )
            model_map = metrics["map"]
            print(f"    mAP@50-95: {model_map:.4f}")
            maps.append(model_map)
        except Exception as e:
            print(f"    Failed to evaluate {name}: {e}")
            sys.exit(1)
            
    total_map = sum(maps)
    if total_map == 0:
        print("Warning: Sum of all mAP is 0. Returning equal weights.")
        return [1.0] * len(model_names), 0.0
        
    weights = [round(m / total_map, 4) for m in maps]
    
    # Calculate expected theoretical val mAP (just an approximation)
    print(f"\n  Resulting proportional weights: {weights}")
    
    return weights, max(maps) # We don't have fused mAP yet without running stack_predictions


def tune_grid_search(
    model_preds: list[str],
    model_names: list[str],
    data_cfg,
    ensemble_cfg,
    prepared_dataset_root: Path | None,
    grid_step: float = 0.5,
    grid_max: float = 2.0,
) -> tuple[list[float], float]:
    """
    Sweeps combinations of weights to find the best functioning ensemble val mAP.
    NOTE: this dynamically writes over a temp fused file. 
    """
    print("\n[Strategy: Grid Search]")
    
    # Generate weight combinations
    # Make step a bit larger by default if many models to save time
    possible_values = [i * grid_step for i in range(1, int(grid_max/grid_step) + 1)]
    combinations = list(itertools.product(possible_values, repeat=len(model_names)))
    
    print(f"  Evaluating {len(combinations)} combinations... (This might take a while)")
    
    best_weights = []
    best_map = -1.0
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fused_path = Path(temp_dir) / f"temp_fused_{ensemble_cfg.method}.jsonl"
        
        # Optionally silence stdout for stack_predictions and evaluate
        import os
        
        for idx, weights in enumerate(combinations):
            print(f"  [{idx+1}/{len(combinations)}] Testing weights: {weights}", end="\r")
            try:
                # Redirect stdout to devnull to avoid spamming terminal
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                # Stack
                stack_predictions(
                    pred_files=model_preds,
                    output_path=temp_fused_path,
                    cfg=ensemble_cfg,
                    weights=list(weights),
                )
                
                # Evaluate
                metrics = evaluate_predictions(
                    pred_file=temp_fused_path,
                    data_cfg=data_cfg,
                    split="val",
                    prepared_dataset_root=prepared_dataset_root,
                )
                
                sys.stdout = old_stdout # restore terminal output
                
                fused_map = metrics["map"]
                if fused_map > best_map:
                    best_map = fused_map
                    best_weights = list(weights)
                    
            except Exception as e:
                sys.stdout = old_stdout
                print(f"\n  Error during run with weights {weights}: {e}")
                
    print(f"\n\n  Best Fused mAP: {best_map:.4f}")
    print(f"  Best Weights: {best_weights}")
    
    return best_weights, best_map


def main() -> None:
    args = parse_args()

    # Collect inputs
    model_preds = []
    model_names = []
    
    if args.yolo_pred:
        model_preds.append(args.yolo_pred)
        model_names.append("yolo")
    if args.frcnn_pred:
        model_preds.append(args.frcnn_pred)
        model_names.append("frcnn")
    if args.detr_pred:
        model_preds.append(args.detr_pred)
        model_names.append("detr")
        
    if len(model_preds) < 2:
        print("Error: At least two prediction files are required for tuning.")
        sys.exit(1)

    # Configs
    preset_cfg = {
        "small": HardwarePreset.small,
        "medium": HardwarePreset.medium,
        "large": HardwarePreset.large,
    }[args.preset](arch="yolo")
    data_cfg = preset_cfg.data
    
    ensemble_cfg = EnsembleConfig(
        iou_thr=args.ensemble_iou,
        skip_box_thr=args.ensemble_skip,
        method=args.ensemble_method,
    )
    
    prepared_dataset_root = Path(get_processed_data_root())
    if not (prepared_dataset_root / "dataset.yaml").exists():
        prepared_dataset_root = None

    print(f"\n--- Tuning Ensemble Weights ---")
    print(f"Models: {', '.join(model_names)}")
    
    if args.grid_search:
        weights_list, best_map = tune_grid_search(
            model_preds, model_names, 
            data_cfg, ensemble_cfg, prepared_dataset_root,
            grid_step=args.grid_step, grid_max=args.grid_max
        )
        strategy = "grid_search"
    else:
        weights_list, best_map = tune_map_proportional(
            model_preds, model_names, 
            data_cfg, prepared_dataset_root
        )
        strategy = "map_proportional"
        
    # Output structure
    output_dir = Path(get_output_root()) / "ensemble" / "val"
    output_dir.mkdir(parents=True, exist_ok=True)
    tuned_path = output_dir / "tuned_weights.json"
    
    weights_dict = {name: weight for name, weight in zip(model_names, weights_list)}
    
    result = {
        "strategy": strategy,
        "split": "val",
        "models": model_names,
        "weights": weights_dict,
        "ordered_weights": weights_list,
        "best_val_map": best_map,
        "ensemble_method": args.ensemble_method if args.grid_search else "N/A"
    }

    with open(tuned_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"\nDone! Tuned weights saved to → {tuned_path}")
    print(f"You can now use this via: --tuned-weights {tuned_path}")


if __name__ == "__main__":
    main()
