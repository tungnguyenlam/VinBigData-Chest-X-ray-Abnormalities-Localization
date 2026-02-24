"""
Shared dataset preparation: DICOM → 8-bit PNG + YOLO-format labels.

All detection models (YOLO, Faster R-CNN, DETR) can load images from the
prepared directory instead of reading raw DICOMs at training time.

On-disk layout after preparation
---------------------------------
<output_root>/prepared_dataset/
    train/
        images/   <image_id>.png   (8-bit grayscale, resized to image_size)
        labels/   <image_id>.txt   (YOLO format: cls cx cy w h, normalized)
    val/
        images/   <image_id>.png
        labels/   <image_id>.txt
    dataset.yaml                   (for Ultralytics YOLO)
    manifest.json                  (split lists + metadata, for other models)

Usage (CLI)
-----------
# Default paths from paths.yaml
python src/data/prepare_dataset.py

# Override paths
python src/data/prepare_dataset.py --data /path/to/data --output outputs

# Force full rebuild even if already prepared
python src/data/prepare_dataset.py --force

# Exclude "No finding" images (only abnormality images)
python src/data/prepare_dataset.py --no_finding false
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    DataConfig,
    get_data_root,
    get_output_root,
    get_processed_data_root,
)
from src.data.utils import (
    build_yolo_dataset,
    load_annotations,
    write_yolo_yaml,
)


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _convert_worker(args: tuple) -> None:
    """Convert one DICOM to a 3-channel 8-bit PNG in a subprocess."""
    import cv2
    from src.data.utils import dicom_to_3channel_8bit

    img_id, src_dir, dst_dir, image_size = args
    src_dicom = Path(src_dir) / f"{img_id}.dicom"
    dst_png = Path(dst_dir) / f"{img_id}.png"

    if dst_png.exists():
        return
    if not src_dicom.exists():
        return

    # Create 3-channel representation for dual-phase training
    arr3 = dicom_to_3channel_8bit(src_dicom, image_size=image_size)
    cv2.imwrite(str(dst_png), arr3)


def _current_class_names() -> list[str]:
    import src.config as app_config

    return list(app_config.CLASS_NAMES)


def _current_localize_only() -> bool:
    import src.config as app_config

    return bool(app_config.LOCALIZE_ONLY)


def _stratified_three_way_split(
    all_ids: list[str],
    positive_ids: set[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split image IDs into train/val/test while preserving finding ratio."""
    rng = np.random.default_rng(seed)

    pos = [img_id for img_id in all_ids if img_id in positive_ids]
    neg = [img_id for img_id in all_ids if img_id not in positive_ids]

    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_group(group_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(group_ids)
        if n == 0:
            return [], [], []

        n_val = int(round(n * val_split))
        n_test = int(round(n * test_split))
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

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Core preparation function (importable by model modules)
# ---------------------------------------------------------------------------


def prepare_dataset(
    data_cfg: DataConfig,
    output_root: str | Path,
    force: bool = False,
) -> Path:
    """
    Convert DICOMs to 16-bit PNGs and write YOLO label files for all splits.

    Parameters
    ----------
    data_cfg    : DataConfig  — controls image_size, val_split, seed, include_no_finding
    output_root : where to write the prepared dataset (e.g. "outputs")
    force       : if True, rebuild even if dataset.yaml already exists

    Returns
    -------
    Path to the prepared dataset root (contains train/, val/, dataset.yaml,
    manifest.json).
    """
    import pandas as pd
    from tqdm import tqdm

    dataset_root = Path(data_cfg.processed_root)
    yaml_path = dataset_root / "dataset.yaml"
    manifest_path = dataset_root / "manifest.json"

    if yaml_path.exists() and manifest_path.exists() and not force:
        try:
            with open(manifest_path) as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = {}

        expected = {
            "image_size": data_cfg.image_size,
            "include_no_finding": data_cfg.include_no_finding,
            "seed": data_cfg.seed,
            "val_split": data_cfg.val_split,
            "test_split": data_cfg.test_split,
            "localize_only": _current_localize_only(),
            "class_names": _current_class_names(),
        }

        mismatch = [
            key
            for key, value in expected.items()
            if existing_manifest.get(key) != value
        ]
        if not mismatch:
            print(
                f"[prepare_dataset] Already prepared at {dataset_root}. "
                "Pass force=True to rebuild."
            )
            return dataset_root
        print(
            "[prepare_dataset] Existing prepared dataset is incompatible with "
            f"current config ({', '.join(mismatch)}). Rebuilding..."
        )

    if force and dataset_root.exists():
        print(
            f"[prepare_dataset] Force=True: Clearing existing dataset at {dataset_root}..."
        )
        import shutil

        # Delete subdirectories to ensure no stale files from old splits remain
        for sub in ["train", "val", "test"]:
            path = dataset_root / sub
            if path.exists():
                shutil.rmtree(path)

    ann_df = load_annotations(data_cfg.train_csv)

    if data_cfg.include_no_finding:
        full_csv = pd.read_csv(data_cfg.train_csv)
        all_ids = full_csv["image_id"].unique().tolist()
    else:
        all_ids = ann_df["image_id"].unique().tolist()

    abnormality_ids = set(ann_df["image_id"].unique().tolist())

    train_ids, val_ids, test_ids = _stratified_three_way_split(
        all_ids=all_ids,
        positive_ids=abnormality_ids,
        val_split=data_cfg.val_split,
        test_split=data_cfg.test_split,
        seed=data_cfg.seed,
    )

    if data_cfg.num_workers is None:
        n_workers = os.cpu_count() or 4
    else:
        n_workers = max(0, int(data_cfg.num_workers))

    if n_workers <= 0:
        n_workers = 1

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        img_out_dir = dataset_root / split_name / "images"
        lbl_out_dir = dataset_root / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        no_finding_count = sum(1 for i in ids if i not in abnormality_ids)
        print(
            f"[prepare_dataset] {split_name}: {len(ids)} images "
            f"({len(ids) - no_finding_count} with boxes, "
            f"{no_finding_count} no-finding)"
        )

        # --- DICOM → PNG (parallel, CPU-bound) ---
        src_img_dir = data_cfg.train_img_dir
        worker_args = [
            (img_id, str(src_img_dir), str(img_out_dir), data_cfg.image_size)
            for img_id in ids
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(
                tqdm(
                    executor.map(_convert_worker, worker_args, chunksize=8),
                    total=len(ids),
                    desc=f"DICOM→PNG [{split_name}]",
                )
            )

        # --- Write YOLO label .txt files ---
        ids_with_boxes = [i for i in ids if i in abnormality_ids]
        build_yolo_dataset(
            ann_df=ann_df,
            image_ids=ids_with_boxes,
            img_dir=data_cfg.train_img_dir,
            label_dir=lbl_out_dir,
            iou_thr=data_cfg.wbf_iou_thr,
        )

        # Empty label files for no-finding images (hard negatives for YOLO)
        for img_id in ids:
            if img_id not in abnormality_ids:
                lbl_path = lbl_out_dir / f"{img_id}.txt"
                if not lbl_path.exists():
                    lbl_path.write_text("")

        # --- Write Phase 2 Classification CSV ---
        split_ann = ann_df[ann_df["image_id"].isin(ids_with_boxes)].copy()
        csv_out_path = dataset_root / split_name / f"{split_name}_boxes.csv"
        # We need to compute the scaled coordinates but WBF gives them normalized:
        # Instead of recalculating, we just use the raw coordinates from split_ann,
        # but what if Phase 2 works on the resized images?
        # Actually, Phase 2 dataset class can read the normalized [0, 1] boxes from labels/
        # or we just write the raw boxes here. Let's write raw boxes and let phase 2 read metadata
        # or bounding boxes as needed, keeping it simple.
        split_ann.to_csv(csv_out_path, index=False)
        print(
            f"[prepare_dataset] Saved {len(split_ann)} raw annotations to {csv_out_path.name}"
        )

    # --- dataset.yaml (Ultralytics) ---
    write_yolo_yaml(
        yaml_path=yaml_path,
        dataset_path=".",
        train_img_dir="train/images",
        val_img_dir="val/images",
        test_img_dir="test/images",
        class_names=_current_class_names(),
    )

    # --- manifest.json (used by VinBigDataset and other models) ---
    manifest = {
        "image_size": data_cfg.image_size,
        "include_no_finding": data_cfg.include_no_finding,
        "seed": data_cfg.seed,
        "val_split": data_cfg.val_split,
        "test_split": data_cfg.test_split,
        "localize_only": _current_localize_only(),
        "class_names": _current_class_names(),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[prepare_dataset] Done. Dataset at: {dataset_root}")
    return dataset_root


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    default_cfg = DataConfig()
    p = argparse.ArgumentParser(
        description="Prepare VinBigData dataset (DICOM → PNG + YOLO labels)"
    )
    p.add_argument(
        "--data", default=None, help="Path to dataset root (overrides paths.yaml)"
    )
    p.add_argument("--output", default=None, help="Output root (overrides paths.yaml)")
    p.add_argument("--image_size", type=int, default=default_cfg.image_size)
    p.add_argument("--val_split", type=float, default=default_cfg.val_split)
    p.add_argument("--test_split", type=float, default=default_cfg.test_split)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--no_finding",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=default_cfg.include_no_finding,
        metavar="true|false",
        help=f"Include 'No finding' images as hard negatives (default: {str(default_cfg.include_no_finding).lower()})",
    )
    p.add_argument(
        "--force", action="store_true", help="Rebuild even if already prepared"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    data_root = args.data or get_data_root()
    output_root = args.output or get_output_root()

    cfg = DataConfig(
        root=data_root,
        processed_root=get_processed_data_root()
        if args.output is None
        else args.output,
        image_size=args.image_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        num_workers=args.workers,
        include_no_finding=args.no_finding,
    )

    prepare_dataset(cfg, output_root=output_root, force=args.force)
