from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import src.config as app_config
from src.config import DataConfig, NO_FINDING_CLASS_ID
from src.data.transforms import get_test_transforms, get_train_transforms, get_val_transforms
from src.data.utils import aggregate_annotations, get_image_dims, load_annotations, load_image


OutputFormat = Literal["torchvision", "yolo"]


class VinBigDataset(Dataset):
    """
    Dataset for VinBigData Chest X-ray Abnormalities Localization.

    If a prepared dataset (built by src.data.prepare_dataset) exists at
    <output_root>/prepared_dataset, images are loaded from its pre-converted
    16-bit PNGs (fast, shared by all models). Otherwise falls back to reading
    raw DICOMs on the fly.

    output_format="torchvision":
        Returns (image_tensor, target) where target is a dict:
            {"boxes": FloatTensor[N,4] xyxy absolute,
             "labels": Int64Tensor[N],
             "image_id": str}

    output_format="yolo":
        Returns (image_tensor, target) where target is a dict:
            {"boxes": FloatTensor[N,4] normalized cx cy w h,
             "labels": Int64Tensor[N],
             "image_id": str}
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"],
        cfg: DataConfig,
        output_format: OutputFormat = "torchvision",
        transforms: Optional[Callable] = None,
        image_ids: Optional[list[str]] = None,
        prepared_dataset_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.output_format = output_format

        # Resolve prepared PNG directory if available
        self._prepared_img_dir: Path | None = None
        if prepared_dataset_root is not None:
            candidate = Path(prepared_dataset_root) / split / "images"
            if candidate.is_dir():
                self._prepared_img_dir = candidate

        if split == "test" and prepared_dataset_root is None:
            self.img_dir = self.root / "test"
            self.ann_df = pd.DataFrame()
            all_ids = [p.stem for p in sorted(self.img_dir.glob("*.dicom"))]
            self.image_ids = image_ids if image_ids is not None else all_ids
        else:
            self.img_dir = self.root / "train"
            self.ann_df = load_annotations(self.root / "train.csv")

            # If a manifest exists in the prepared dataset, reuse its splits
            # so image assignment is identical to what prepare_dataset built.
            if prepared_dataset_root is not None:
                manifest_path = Path(prepared_dataset_root) / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    all_ids = manifest.get(split, [])
                else:
                    all_ids = self._split_image_ids()
            else:
                all_ids = self._split_image_ids()

            self.image_ids = image_ids if image_ids is not None else all_ids

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(cfg.image_size)
        elif split == "val":
            self.transforms = get_val_transforms(cfg.image_size)
        else:
            self.transforms = get_test_transforms(cfg.image_size)

        # Optional: cache decoded images to speed up subsequent epochs
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_image_ids(self) -> list[str]:
        """
        Return train or val image IDs based on val_split fraction.

        When cfg.include_no_finding=True (default), "No finding" images are
        included as hard negatives — they have no boxes, so the model learns
        to suppress false positives on normal X-rays.
        When False, only images with at least one abnormality box are used.
        """
        if self.cfg.include_no_finding:
            # Use all unique image IDs from the raw CSV
            full_csv = pd.read_csv(self.root / "train.csv")
            all_ids = full_csv["image_id"].unique().tolist()
        else:
            # ann_df already has No-finding rows removed
            all_ids = self.ann_df["image_id"].unique().tolist()

        abnormality_ids = set(self.ann_df["image_id"].unique().tolist())

        rng = np.random.default_rng(self.cfg.seed)
        pos = [img_id for img_id in all_ids if img_id in abnormality_ids]
        neg = [img_id for img_id in all_ids if img_id not in abnormality_ids]
        rng.shuffle(pos)
        rng.shuffle(neg)

        def split_group(group_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
            n = len(group_ids)
            if n == 0:
                return [], [], []

            n_val = int(round(n * self.cfg.val_split))
            n_test = int(round(n * self.cfg.test_split))
            n_val = min(max(n_val, 0), n)
            n_test = min(max(n_test, 0), n - n_val)

            val_ids = group_ids[:n_val]
            test_ids = group_ids[n_val : n_val + n_test]
            train_ids = group_ids[n_val + n_test :]
            return train_ids, val_ids, test_ids

        train_pos, val_pos, test_pos = split_group(pos)
        train_neg, val_neg, test_neg = split_group(neg)

        train_ids = train_pos + train_neg
        val_ids = val_pos + val_neg
        test_ids = test_pos + test_neg

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        if self.split == "val":
            return val_ids
        elif self.split == "test":
            return test_ids
        else:
            return train_ids

    def _resolve_image_path(self, image_id: str) -> Path:
        """Return the best available image file: prepared PNG first, then DICOM."""
        if self._prepared_img_dir is not None:
            png = self._prepared_img_dir / f"{image_id}.png"
            if png.exists():
                return png
        return self.img_dir / f"{image_id}.dicom"

    def _load_image(self, image_id: str) -> np.ndarray:
        if self.cfg.cache_images:
            if image_id not in self._cache:
                self._cache[image_id] = load_image(
                    self._resolve_image_path(image_id), self.cfg.image_size
                )
            return self._cache[image_id].copy()
        return load_image(self._resolve_image_path(image_id), self.cfg.image_size)

    def _get_annotations(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns boxes (N,4) normalized xyxy and labels (N,) for one image.
        Loads original DICOM dims for normalization.
        """
        if self.ann_df.empty:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

        dicom_path = self.img_dir / f"{image_id}.dicom"
        if not dicom_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

        orig_h, orig_w = get_image_dims(dicom_path)
        boxes, labels = aggregate_annotations(
            image_id, self.ann_df, orig_h, orig_w, iou_thr=self.cfg.wbf_iou_thr
        )
        return boxes, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image_id = self.image_ids[idx]

        image = self._load_image(image_id)  # uint8 BGR (H, W, 3)
        boxes_norm, labels = self._get_annotations(image_id)  # normalized xyxy

        # Convert normalized xyxy → absolute xyxy for albumentations (pascal_voc)
        h, w = image.shape[:2]
        if len(boxes_norm) > 0:
            abs_boxes = boxes_norm.copy()
            abs_boxes[:, [0, 2]] *= w
            abs_boxes[:, [1, 3]] *= h
            abs_boxes = abs_boxes.tolist()
            labels_list = labels.tolist()
        else:
            abs_boxes = []
            labels_list = []

        # Apply albumentations (train has bbox_params; test does not)
        if self.split != "test" and len(abs_boxes) > 0:
            transformed = self.transforms(
                image=image, bboxes=abs_boxes, labels=labels_list
            )
            image = transformed["image"]
            abs_boxes = list(transformed["bboxes"])
            labels_list = list(transformed["labels"])
        elif self.split == "test":
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            transformed = self.transforms(
                image=image, bboxes=abs_boxes, labels=labels_list
            )
            image = transformed["image"]

        # image is now a CHW float tensor from ToTensorV2
        out_h, out_w = self.cfg.image_size, self.cfg.image_size

        if len(abs_boxes) > 0:
            # Map labels to contiguous indices (handles 1-class collapsing)
            mapped_labels = [app_config.CLASS_ID_TO_IDX.get(int(l), 0) for l in labels_list]
            boxes_t = torch.as_tensor(abs_boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(mapped_labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)

        if self.output_format == "yolo":
            # Convert absolute xyxy → normalized cx cy w h
            if len(boxes_t) > 0:
                cx = (boxes_t[:, 0] + boxes_t[:, 2]) / 2 / out_w
                cy = (boxes_t[:, 1] + boxes_t[:, 3]) / 2 / out_h
                bw = (boxes_t[:, 2] - boxes_t[:, 0]) / out_w
                bh = (boxes_t[:, 3] - boxes_t[:, 1]) / out_h
                boxes_t = torch.stack([cx, cy, bw, bh], dim=1)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
        }
        return image, target


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_torchvision(batch: list) -> tuple[list[torch.Tensor], list[dict]]:
    """Collate for Faster R-CNN / DETR — returns list of images and list of targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def collate_yolo(batch: list) -> tuple[torch.Tensor, list[dict]]:
    """Collate for YOLO — stacks images into a single batch tensor."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    root: str | Path,
    split: Literal["train", "val", "test"],
    cfg: DataConfig,
    output_format: OutputFormat = "torchvision",
    image_ids: Optional[list[str]] = None,
    shuffle: Optional[bool] = None,
    prepared_dataset_root: str | Path | None = None,
) -> DataLoader:
    dataset = VinBigDataset(
        root=root,
        split=split,
        cfg=cfg,
        output_format=output_format,
        image_ids=image_ids,
        prepared_dataset_root=prepared_dataset_root,
    )

    if shuffle is None:
        shuffle = split == "train"

    collate_fn = collate_yolo if output_format == "yolo" else collate_torchvision

    loader_kwargs: dict = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": cfg.num_workers > 0 and torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)
