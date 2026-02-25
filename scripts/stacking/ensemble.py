from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.config import EnsembleConfig, NUM_CLASSES, DataConfig
from scripts.models.base import BaseDetector, Detection


FusionMethod = Literal["wbf", "nms", "soft_nms"]


class StackingEnsemble:
    """
    Model-agnostic stacking ensemble for object detection.

    Combines predictions from any number of BaseDetector models using
    Weighted Box Fusion (WBF), NMS, or Soft-NMS.

    Usage:
        ensemble = StackingEnsemble(
            models=[yolo_detector, frcnn_detector, detr_detector],
            cfg=EnsembleConfig(iou_thr=0.5, skip_box_thr=0.05),
        )
        detections = ensemble.predict(images)
    """

    def __init__(
        self,
        models: list[BaseDetector],
        cfg: Optional[EnsembleConfig] = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")
        self.models = models
        self.cfg = cfg or EnsembleConfig()

        if self.cfg.weights is not None and len(self.cfg.weights) != len(models):
            raise ValueError(
                f"weights length ({len(self.cfg.weights)}) must match "
                f"number of models ({len(models)})."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        image_size: Optional[int] = None,
        score_threshold: float = 0.05,
    ) -> list[Detection]:
        """
        Run all models on the same batch and fuse their predictions.

        Returns one Detection per image.
        """
        image_size = image_size or DataConfig().image_size

        # Collect per-model predictions  shape: [num_models][num_images]
        all_preds: list[list[Detection]] = []
        for model in self.models:
            preds = model.predict(images, image_size=image_size)
            all_preds.append(preds)

        num_images = len(all_preds[0])
        fused: list[Detection] = []

        for img_idx in range(num_images):
            img_preds = [model_preds[img_idx] for model_preds in all_preds]
            det = self._fuse_single(img_preds)
            det = det.filter_by_score(score_threshold)
            fused.append(det)

        return fused

    def predict_loader(
        self,
        loader: DataLoader,
        image_size: Optional[int] = None,
        score_threshold: float = 0.05,
    ) -> dict[str, Detection]:
        """
        Run ensemble inference over an entire DataLoader.
        Returns a dict mapping image_id → Detection.
        """
        image_size = image_size or DataConfig().image_size

        results: dict[str, Detection] = {}

        for images, targets in loader:
            if isinstance(images, torch.Tensor):
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                image_list = images

            detections = self.predict(
                image_list, image_size=image_size, score_threshold=score_threshold
            )

            for det, target in zip(detections, targets):
                results[target["image_id"]] = det

        return results

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------

    def _fuse_single(self, preds: list[Detection]) -> Detection:
        """Fuse Detection objects from multiple models for one image."""
        boxes_list = [p.boxes.tolist() for p in preds]
        scores_list = [p.scores.tolist() for p in preds]
        labels_list = [p.labels.tolist() for p in preds]

        # Match weights array if we have one
        weights = (
            self.cfg.weights
            if self.cfg.weights is not None
            else [None] * len(boxes_list)
        )

        # Remove empty predictions
        non_empty = [
            (b, s, lbl, w)
            for b, s, lbl, w in zip(boxes_list, scores_list, labels_list, weights)
            if len(b) > 0
        ]
        if not non_empty:
            return Detection()

        boxes_list, scores_list, labels_list, active_weights = zip(*non_empty)

        # Normalize labels to [0, 1] range for WBF (it expects class scores, not IDs)
        # We encode per-class separately then reassemble
        return self._wbf_multiclass(
            list(boxes_list),
            list(scores_list),
            list(labels_list),
            list(active_weights) if active_weights[0] is not None else None,
        )

    def _wbf_multiclass(
        self,
        boxes_list: list[list],
        scores_list: list[list],
        labels_list: list[list],
        active_weights: list[float] | None,
    ) -> Detection:
        """
        Run WBF per class then combine, which gives cleaner results than
        running WBF on multi-class boxes with label-encoded scores.
        """
        from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for cls_id in range(NUM_CLASSES):
            cls_boxes_list, cls_scores_list = [], []

            for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
                boxes_arr = np.array(boxes, dtype=np.float32)
                scores_arr = np.array(scores, dtype=np.float32)
                labels_arr = np.array(labels, dtype=np.int32)

                mask = labels_arr == cls_id
                if mask.sum() == 0:
                    cls_boxes_list.append([])
                    cls_scores_list.append([])
                else:
                    cls_boxes_list.append(boxes_arr[mask].tolist())
                    cls_scores_list.append(scores_arr[mask].tolist())

            if all(len(b) == 0 for b in cls_boxes_list):
                continue

            # Replace empty with dummy to satisfy ensemble_boxes API
            cls_boxes_list_safe = [b if b else [[0, 0, 0, 0]] for b in cls_boxes_list]
            cls_scores_list_safe = [s if s else [0.0] for s in cls_scores_list]
            cls_labels_list = [[cls_id] * len(b) for b in cls_boxes_list_safe]

            method = self.cfg.method
            if method == "wbf":
                f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                    cls_boxes_list_safe,
                    cls_scores_list_safe,
                    cls_labels_list,
                    weights=active_weights,
                    iou_thr=self.cfg.iou_thr,
                    skip_box_thr=self.cfg.skip_box_thr,
                )
            elif method == "soft_nms":
                f_boxes, f_scores, f_labels = soft_nms(
                    cls_boxes_list_safe,
                    cls_scores_list_safe,
                    cls_labels_list,
                    weights=active_weights,
                    iou_thr=self.cfg.iou_thr,
                    thresh=self.cfg.skip_box_thr,
                )
            else:  # nms
                f_boxes, f_scores, f_labels = nms(
                    cls_boxes_list_safe,
                    cls_scores_list_safe,
                    cls_labels_list,
                    weights=active_weights,
                    iou_thr=self.cfg.iou_thr,
                )

            if len(f_boxes) > 0:
                all_boxes.append(f_boxes.astype(np.float32))
                all_scores.append(f_scores.astype(np.float32))
                all_labels.append(f_labels.astype(np.int32))

        if not all_boxes:
            return Detection()

        return Detection(
            boxes=np.concatenate(all_boxes, axis=0),
            scores=np.concatenate(all_scores, axis=0),
            labels=np.concatenate(all_labels, axis=0),
        )

    # ------------------------------------------------------------------
    # Weight tuning
    # ------------------------------------------------------------------

    def tune_weights_by_map(
        self,
        val_loader: DataLoader,
        image_size: Optional[int] = None,
    ) -> list[float]:
        """
        Compute per-model validation mAP and use as ensemble weights.
        Updates self.cfg.weights in place and returns the weights.

        Requires torchmetrics: pip install torchmetrics
        """
        image_size = image_size or DataConfig().image_size
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError:
            raise ImportError("pip install torchmetrics to use tune_weights_by_map")

        maps: list[float] = []
        for model in self.models:
            from typing import Any

            metric: Any = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            for images, targets in val_loader:
                if isinstance(images, torch.Tensor):
                    image_list = [images[i] for i in range(images.shape[0])]
                else:
                    image_list = images

                preds = model.predict(image_list, image_size=image_size)

                preds_fmt = [
                    {
                        "boxes": torch.tensor(d.boxes * image_size),
                        "scores": torch.tensor(d.scores),
                        "labels": torch.tensor(d.labels, dtype=torch.long),
                    }
                    for d in preds
                ]
                targets_fmt = [
                    {
                        "boxes": t["boxes"],
                        "labels": t["labels"],
                    }
                    for t in targets
                ]
                metric.update(preds=preds_fmt, target=targets_fmt)  # type: ignore

            result = metric.compute()  # type: ignore
            maps.append(float(result["map"].item()))
            print(f"  Model {type(model).__name__}: mAP={maps[-1]:.4f}")

        total = sum(maps) or 1.0
        weights = [m / total for m in maps]
        self.cfg.weights = weights
        print(f"Ensemble weights tuned: {weights}")
        return weights
