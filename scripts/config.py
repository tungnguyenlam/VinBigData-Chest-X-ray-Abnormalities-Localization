from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


# ---------------------------------------------------------------------------
# Path resolution — reads paths.yaml at the repo root
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """Walk up from this file until we find paths.yaml."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "paths.yaml").exists():
            return candidate
    return Path.cwd()


def load_paths(paths_file: str | Path | None = None) -> dict:
    """
    Load paths.yaml.  Resolution order:
      1. explicit paths_file argument
      2. paths.yaml next to the repo root (auto-detected)
      3. safe defaults
    """
    if paths_file is None:
        paths_file = _find_repo_root() / "paths.yaml"

    paths_file = Path(paths_file)
    if paths_file.exists():
        with open(paths_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_data_root(paths_file: str | Path | None = None) -> str:
    return load_paths(paths_file).get("data_root", "data")


def get_output_root(paths_file: str | Path | None = None) -> str:
    return load_paths(paths_file).get("output_root", "outputs")


def get_processed_data_root(paths_file: str | Path | None = None) -> str:
    return load_paths(paths_file).get("processed_data_root", "data/processed")


# ---------------------------------------------------------------------------
# Class Configuration
# ---------------------------------------------------------------------------

# SET TO True to collapse all 14 abnormality classes into a single "Abnormality" class.
LOCALIZE_ONLY = True

NUM_CLASSES = 1 if LOCALIZE_ONLY else 14

CLASS_NAMES = (
    ["Abnormality"]
    if LOCALIZE_ONLY
    else [
        "Aortic enlargement",  # 0
        "Atelectasis",  # 1
        "Calcification",  # 2
        "Cardiomegaly",  # 3
        "Consolidation",  # 4
        "ILD",  # 5
        "Infiltration",  # 6
        "Lung Opacity",  # 7
        "Nodule/Mass",  # 8
        "Other lesion",  # 9
        "Pleural effusion",  # 10
        "Pleural thickening",  # 11
        "Pneumothorax",  # 12
        "Pulmonary fibrosis",  # 13
    ]
)

# Maps original class_id (0-13) to contiguous index used in models (0 or 0-13)
CLASS_ID_TO_IDX: dict[int, int] = {i: (0 if LOCALIZE_ONLY else i) for i in range(14)}
NO_FINDING_CLASS_ID = 14


@dataclass
class DataConfig:
    root: str = "data"
    processed_root: str = "data/processed"
    image_size: int = 1024
    batch_size: int = 8
    num_workers: int = 4
    cache_images: bool = False
    cache_dir: str = "data/cache"
    wbf_iou_thr: float = 0.5
    wbf_skip_box_thr: float = 0.0001
    val_split: float = 0.05
    test_split: float = 0.1
    seed: int = 42
    include_no_finding: bool = True  # include "No finding" images as hard negatives

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def train_img_dir(self) -> Path:
        return self.root_path / "train"

    @property
    def test_img_dir(self) -> Path:
        return self.root_path / "test"

    @property
    def train_csv(self) -> Path:
        return self.root_path / "train.csv"


@dataclass
class ModelConfig:
    arch: Literal["yolo", "faster_rcnn", "detr"] = "yolo"
    # yolo: n/s/m/l/x  |  faster_rcnn: resnet50/resnet101  |  detr: resnet-50/resnet-101
    backbone_size: str = "s"
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    output_dir: str = "outputs"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.arch


@dataclass
class EnsembleConfig:
    iou_thr: float = 0.5
    skip_box_thr: float = 0.05
    weights: list[float] | None = None
    method: Literal["wbf", "nms", "soft_nms"] = "wbf"


@dataclass
class HardwarePreset:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    @classmethod
    def small(
        cls,
        root: str | None = None,
        arch: Literal["yolo", "faster_rcnn", "detr"] = "yolo",
    ) -> "HardwarePreset":
        """CPU or single low-VRAM GPU (<=8 GB)."""
        if root is None:
            root = get_data_root()
        return cls(
            data=DataConfig(
                root=root,
                processed_root=get_processed_data_root(),
                image_size=640,
                batch_size=4,
                num_workers=0,
                cache_images=False,
            ),
            model=ModelConfig(
                arch=arch,
                backbone_size="n" if arch == "yolo" else "resnet50",
                epochs=10,
                device="cpu",
                amp=False,
                output_dir=get_output_root(),
            ),
        )

    @classmethod
    def medium(
        cls,
        root: str | None = None,
        arch: Literal["yolo", "faster_rcnn", "detr"] = "yolo",
    ) -> "HardwarePreset":
        """Single mid-range GPU (8–16 GB)."""
        if root is None:
            root = get_data_root()
        return cls(
            data=DataConfig(
                root=root,
                processed_root=get_processed_data_root(),
                image_size=640,
                batch_size=8,
                num_workers=4,
                cache_images=False,
            ),
            model=ModelConfig(
                arch=arch,
                backbone_size="s" if arch == "yolo" else "resnet50",
                epochs=20,
                device="cuda",
                amp=True,
                output_dir=get_output_root(),
            ),
        )

    @classmethod
    def large(
        cls,
        root: str | None = None,
        arch: Literal["yolo", "faster_rcnn", "detr"] = "yolo",
    ) -> "HardwarePreset":
        """Multi-GPU or high-VRAM (>= 24 GB)."""
        if root is None:
            root = get_data_root()
        return cls(
            data=DataConfig(
                root=root,
                processed_root=get_processed_data_root(),
                image_size=1024,
                batch_size=16,
                num_workers=8,
                cache_images=False,
            ),
            model=ModelConfig(
                arch=arch,
                backbone_size="l" if arch == "yolo" else "resnet101",
                epochs=30,
                device="cuda",
                amp=True,
                output_dir=get_output_root(),
            ),
        )
