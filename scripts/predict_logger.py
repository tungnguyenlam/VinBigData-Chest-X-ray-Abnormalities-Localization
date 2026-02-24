"""
Disk-based prediction logger.

Each model writes its predictions image-by-image to a JSONL file:
    outputs/<arch>/predictions_<split>.jsonl

One line per image:
    {"image_id": "abc123", "boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}

The stacking step reads all per-model JSONL files and fuses them with WBF —
without ever loading more than one model into memory at a time.
"""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None  # type: ignore

from scripts.config import DataConfig, EnsembleConfig
from scripts.data.dataset import build_dataloader
from scripts.models.base import BaseDetector, Detection


# ---------------------------------------------------------------------------
# JSONL writer / reader
# ---------------------------------------------------------------------------


class PredictionWriter:
    """Appends one Detection per line to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, image_id: str, det: Detection) -> None:
        record = {
            "image_id": image_id,
            "boxes": det.boxes.tolist(),
            "scores": det.scores.tolist(),
            "labels": det.labels.tolist(),
        }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def iter_predictions(path: str | Path) -> Iterator[tuple[str, Detection]]:
    """Yield (image_id, Detection) from a JSONL prediction file."""
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            det = Detection(
                boxes=np.array(rec["boxes"], dtype=np.float32),
                scores=np.array(rec["scores"], dtype=np.float32),
                labels=np.array(rec["labels"], dtype=np.int32),
            )
            yield rec["image_id"], det


def load_predictions(path: str | Path) -> dict[str, Detection]:
    """Load an entire JSONL prediction file into a dict."""
    return {img_id: det for img_id, det in iter_predictions(path)}


# ---------------------------------------------------------------------------
# Per-model prediction runner (one image at a time)
# ---------------------------------------------------------------------------


def predict_and_log(
    model: BaseDetector,
    data_cfg: DataConfig,
    split: Literal["train", "val", "test"],
    output_path: str | Path,
    score_threshold: float = 0.0,
    batch_size: int = 1,
    prepared_dataset_root: str | Path | None = None,
) -> Path:
    """
    Run model inference over the entire split, writing each prediction
    immediately to disk.  Uses batch_size=1 by default to minimise peak RAM.

    When the model supports `predict_from_paths` (e.g. YOLO), images are
    passed as file paths directly — avoiding the lossy normalize/denormalize
    round-trip and matching training-time preprocessing exactly.

    prepared_dataset_root : path to the output of prepare_dataset.py
        (e.g. "data/processed").  When provided, images are loaded
        from pre-converted 16-bit PNGs instead of raw DICOMs.

    Returns the path to the written JSONL file.
    """
    output_path = Path(output_path)

    # --- Direct path mode (YOLO) ---
    if hasattr(model, "predict_from_paths"):
        from scripts.data.dataset import VinBigDataset

        dataset = VinBigDataset(
            root=data_cfg.root,
            split=split,
            cfg=data_cfg,
            output_format="torchvision",
            prepared_dataset_root=prepared_dataset_root,
        )
        total = len(dataset)
        print(f"  Predicting {total} images → {output_path} (direct path mode)")

        with PredictionWriter(output_path) as writer:
            for idx in tqdm(range(total), desc="predict"):
                img_id = dataset.image_ids[idx]
                img_path = dataset._resolve_image_path(img_id)
                dets = model.predict_from_paths(
                    [img_path], image_size=data_cfg.image_size
                )
                filtered = dets[0].filter_by_score(score_threshold)
                writer.write(img_id, filtered)

        return output_path

    # --- Fallback: dataloader mode (Faster R-CNN, DETR, etc.) ---
    import copy

    cfg_1 = copy.copy(data_cfg)
    cfg_1.batch_size = batch_size
    cfg_1.num_workers = min(data_cfg.num_workers, 2)

    output_format = "torchvision"
    loader = build_dataloader(
        root=data_cfg.root,
        split=split,
        cfg=cfg_1,
        output_format=output_format,
        shuffle=False,
        prepared_dataset_root=prepared_dataset_root,
    )

    total = len(loader.dataset)  # type: ignore[arg-type]
    print(f"  Predicting {total} images → {output_path}")

    with PredictionWriter(output_path) as writer:
        for images, targets in tqdm(loader, desc="predict", total=len(loader)):
            if isinstance(images, torch.Tensor):
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                image_list = list(images)

            detections = model.predict(image_list, image_size=data_cfg.image_size)

            for det, target in zip(detections, targets):
                img_id = target["image_id"]
                filtered = det.filter_by_score(score_threshold)
                writer.write(img_id, filtered)

    return output_path


# ---------------------------------------------------------------------------
# Stacking from saved prediction files
# ---------------------------------------------------------------------------


def stack_predictions(
    pred_files: list[str | Path],
    output_path: str | Path,
    cfg: EnsembleConfig | None = None,
    weights: list[float] | None = None,
) -> Path:
    """
    Fuse per-model JSONL prediction files using WBF — no models in memory.

    pred_files : one JSONL file per model, in the same order as `weights`
    output_path: where to write the fused JSONL file
    """
    from scripts.stacking.ensemble import StackingEnsemble
    from scripts.config import EnsembleConfig

    if cfg is None:
        cfg = EnsembleConfig()
    if weights is not None:
        cfg.weights = weights

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all image IDs (union across all files)
    all_ids: list[str] = []
    seen: set[str] = set()
    for pf in pred_files:
        for img_id, _ in iter_predictions(pf):
            if img_id not in seen:
                all_ids.append(img_id)
                seen.add(img_id)

    # Load all predictions into memory — only boxes/scores/labels, not images
    per_model: list[dict[str, Detection]] = [load_predictions(pf) for pf in pred_files]

    # Dummy ensemble with no models — we call _fuse_single directly
    dummy_ensemble = StackingEnsemble.__new__(StackingEnsemble)
    dummy_ensemble.models = []
    dummy_ensemble.cfg = cfg

    print(
        f"  Fusing {len(all_ids)} images from {len(pred_files)} models → {output_path}"
    )

    with PredictionWriter(output_path) as writer:
        for img_id in tqdm(all_ids, desc="fuse"):
            img_preds = [
                model_preds.get(img_id, Detection()) for model_preds in per_model
            ]
            fused = dummy_ensemble._fuse_single(img_preds)
            writer.write(img_id, fused)

    return output_path


# ---------------------------------------------------------------------------
# mAP evaluation from a prediction JSONL file
# ---------------------------------------------------------------------------


def evaluate_predictions(
    pred_file: str | Path,
    data_cfg: DataConfig,
    split: Literal["train", "val", "test"] = "val",
    prepared_dataset_root: str | Path | None = None,
) -> dict:
    """
    Compute mAP from a saved JSONL prediction file against ground truth.
    Requires torchmetrics.

    prepared_dataset_root : path to the prepared dataset (data/processed).
        Must be provided when raw DICOMs are not available locally, so that
        GT boxes are read from the pre-built YOLO .txt label files instead.
    """
    if MeanAveragePrecision is None:
        raise ImportError("pip install torchmetrics")

    preds_dict = load_predictions(pred_file)

    loader = build_dataloader(
        root=data_cfg.root,
        split=split,
        cfg=data_cfg,
        output_format="torchvision",
        shuffle=False,
        prepared_dataset_root=prepared_dataset_root,
    )

    metric: Any = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    # --- Diagnostics ---
    n_total = 0
    n_pred_empty = 0
    n_gt_empty = 0
    n_both_nonempty = 0
    diag_printed = 0

    for _, targets in loader:
        for target in targets:
            img_id = target["image_id"]
            det = preds_dict.get(img_id, Detection())

            # Fix empty box shape: (0,) → (0, 4)
            pred_boxes = det.boxes
            if pred_boxes.ndim == 1 and len(pred_boxes) == 0:
                pred_boxes = pred_boxes.reshape(0, 4)

            pred_boxes_abs = pred_boxes * data_cfg.image_size

            pred_fmt = {
                "boxes": torch.as_tensor(pred_boxes_abs, dtype=torch.float32),
                "scores": torch.as_tensor(det.scores, dtype=torch.float32),
                "labels": torch.as_tensor(det.labels, dtype=torch.long),
            }
            gt_fmt = {
                "boxes": target["boxes"],
                "labels": target["labels"],
            }

            n_total += 1
            if len(det.scores) == 0:
                n_pred_empty += 1
            if len(target["boxes"]) == 0:
                n_gt_empty += 1
            if len(det.scores) > 0 and len(target["boxes"]) > 0:
                n_both_nonempty += 1
                if diag_printed < 3:
                    diag_printed += 1
                    print(f"  [DIAG] image={img_id}")
                    print(f"    pred boxes (abs): {pred_boxes_abs[:2].tolist()}")
                    print(f"    pred scores:      {det.scores[:2].tolist()}")
                    print(f"    pred labels:      {det.labels[:2].tolist()}")
                    print(f"    gt   boxes:       {target['boxes'][:2].tolist()}")
                    print(f"    gt   labels:      {target['labels'][:2].tolist()}")

            metric.update(preds=[pred_fmt], target=[gt_fmt])

    print(
        f"  [DIAG] total={n_total} pred_empty={n_pred_empty} "
        f"gt_empty={n_gt_empty} both_nonempty={n_both_nonempty}"
    )

    result: dict = metric.compute()
    return {k: float(v.item()) if hasattr(v, "item") else v for k, v in result.items()}


# ---------------------------------------------------------------------------
# FROC evaluation from a prediction JSONL file
# ---------------------------------------------------------------------------


def _compute_iou_matrix(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Compute IoU between every pred box and every GT box. Both in xyxy format."""
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]
    )
    union = area_pred[:, None] + area_gt[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def evaluate_froc(
    pred_file: str | Path,
    data_cfg: DataConfig,
    split: Literal["train", "val", "test"] = "val",
    iou_threshold: float = 0.5,
    fp_rates: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    prepared_dataset_root: str | Path | None = None,
) -> dict:
    """
    Compute FROC score from a saved JSONL prediction file against ground truth.

    The FROC curve plots Lesion Localization Fraction (sensitivity) vs
    average false positives per image.  The FROC score is the mean
    sensitivity at the standard FP/image operating points.

    Returns
    -------
    dict with keys:
        froc_score        – mean sensitivity across operating points
        sensitivities     – list of sensitivity values at each fp_rate
        fp_rates          – the operating points used
        total_gt_lesions  – number of ground truth lesions
        total_images      – number of images evaluated
        iou_threshold     – IoU threshold used for TP matching
    """
    preds_dict = load_predictions(pred_file)

    loader = build_dataloader(
        root=data_cfg.root,
        split=split,
        cfg=data_cfg,
        output_format="torchvision",
        shuffle=False,
        prepared_dataset_root=prepared_dataset_root,
    )

    # Collect all (score, is_tp) pairs and count GT lesions / images
    all_scores: list[float] = []
    all_is_tp: list[bool] = []
    total_gt_lesions = 0
    total_images = 0

    for _, targets in loader:
        for target in targets:
            img_id = target["image_id"]
            det = preds_dict.get(img_id, Detection())
            total_images += 1

            gt_boxes = target["boxes"].numpy()  # (M, 4) absolute xyxy
            n_gt = len(gt_boxes)
            total_gt_lesions += n_gt

            pred_boxes = det.boxes  # (K, 4) normalized [0,1]
            pred_scores = det.scores

            if len(pred_boxes) == 0:
                continue

            # Convert predicted boxes to absolute coords for IoU
            pred_boxes_abs = pred_boxes.copy()
            if pred_boxes_abs.ndim == 1:
                continue
            pred_boxes_abs[:, [0, 2]] *= data_cfg.image_size
            pred_boxes_abs[:, [1, 3]] *= data_cfg.image_size

            # Sort predictions by descending score
            order = np.argsort(-pred_scores)
            pred_boxes_abs = pred_boxes_abs[order]
            pred_scores = pred_scores[order]

            if n_gt == 0:
                # All predictions are FP
                for s in pred_scores:
                    all_scores.append(float(s))
                    all_is_tp.append(False)
                continue

            # Greedy matching: each GT box can be matched at most once
            iou_matrix = _compute_iou_matrix(pred_boxes_abs, gt_boxes)
            matched_gt = set()

            for i in range(len(pred_scores)):
                best_gt = -1
                best_iou = iou_threshold  # minimum to qualify

                for j in range(n_gt):
                    if j in matched_gt:
                        continue
                    if iou_matrix[i, j] >= best_iou:
                        best_iou = iou_matrix[i, j]
                        best_gt = j

                if best_gt >= 0:
                    all_is_tp.append(True)
                    matched_gt.add(best_gt)
                else:
                    all_is_tp.append(False)

                all_scores.append(float(pred_scores[i]))

    # Build FROC curve by sweeping thresholds
    if len(all_scores) == 0:
        sensitivities_at_rates = [0.0] * len(fp_rates)
        return {
            "froc_score": 0.0,
            "sensitivities": sensitivities_at_rates,
            "fp_rates": list(fp_rates),
            "total_gt_lesions": total_gt_lesions,
            "total_images": total_images,
            "iou_threshold": iou_threshold,
        }

    scores_arr = np.array(all_scores)
    is_tp_arr = np.array(all_is_tp)

    # Sort by descending score
    order = np.argsort(-scores_arr)
    is_tp_sorted = is_tp_arr[order]

    cum_tp = np.cumsum(is_tp_sorted).astype(np.float64)
    cum_fp = np.cumsum(~is_tp_sorted).astype(np.float64)

    sensitivity = cum_tp / max(total_gt_lesions, 1)
    fp_per_image = cum_fp / max(total_images, 1)

    # Interpolate sensitivity at each standard FP/image rate
    sensitivities_at_rates: list[float] = []
    for rate in fp_rates:
        # Find the last index where fp_per_image <= rate
        indices = np.where(fp_per_image <= rate)[0]
        if len(indices) == 0:
            sensitivities_at_rates.append(0.0)
        else:
            sensitivities_at_rates.append(float(sensitivity[indices[-1]]))

    froc_score = float(np.mean(sensitivities_at_rates))

    return {
        "froc_score": froc_score,
        "sensitivities": sensitivities_at_rates,
        "fp_rates": list(fp_rates),
        "total_gt_lesions": total_gt_lesions,
        "total_images": total_images,
        "iou_threshold": iou_threshold,
    }
