from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class Detection:
    """
    Unified detection output for a single image.

    boxes  : (N, 4) float32 — normalized xyxy in [0, 1]
    scores : (N,)   float32
    labels : (N,)   int32
    """
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.scores)

    def filter_by_score(self, threshold: float) -> "Detection":
        mask = self.scores >= threshold
        return Detection(self.boxes[mask], self.scores[mask], self.labels[mask])


class BaseDetector(ABC):
    """Abstract interface that all detection models must implement."""

    @abstractmethod
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Run the full training loop."""

    @abstractmethod
    def predict(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        image_size: int,
    ) -> list[Detection]:
        """
        Run inference on a batch of images.

        images     : list of CHW float tensors (already normalized) OR a NCHW batch tensor
        image_size : the square size the images were resized to (for denormalization if needed)

        Returns a list of Detection objects, one per image.
        All bounding boxes must be normalized xyxy in [0, 1].
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model weights to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model weights from disk."""

    # ------------------------------------------------------------------
    # Optional helpers with default implementations
    # ------------------------------------------------------------------

    def predict_loader(
        self,
        loader: DataLoader,
        image_size: int,
        score_threshold: float = 0.05,
    ) -> dict[str, Detection]:
        """
        Run predict over an entire DataLoader.
        Accumulates all results in memory — prefer predict_logger.predict_and_log
        when memory is constrained.

        Returns a dict mapping image_id → Detection.
        """
        results: dict[str, Detection] = {}

        for images, targets in loader:
            if isinstance(images, torch.Tensor):
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                image_list = list(images)

            detections = self.predict(image_list, image_size=image_size)

            for det, target in zip(detections, targets):
                img_id = target["image_id"]
                results[img_id] = det.filter_by_score(score_threshold)

        return results
