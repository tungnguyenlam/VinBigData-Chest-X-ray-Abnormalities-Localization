import os
import argparse
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def create_zip_parts(
    source_dir: Path, output_dir: Path, prefix: str, chunk_size: int = 5000
):
    """
    Zips images and labels in chunks.
    Each zip will contain 'images/' and 'labels/' subfolders.
    """
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"

    if not img_dir.exists():
        return []

    all_images = sorted(list(img_dir.glob("*.png")))
    num_parts = (len(all_images) + chunk_size - 1) // chunk_size

    zip_paths = []

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(all_images))
        chunk_images = all_images[start_idx:end_idx]

        part_name = f"{prefix}_part{i + 1}"
        part_work_dir = output_dir / part_name
        (part_work_dir / "images").mkdir(parents=True, exist_ok=True)
        (part_work_dir / "labels").mkdir(parents=True, exist_ok=True)

        print(f"  Preparing {part_name} ({len(chunk_images)} images)...")
        for img_path in chunk_images:
            # Copy image
            shutil.copy2(img_path, part_work_dir / "images" / img_path.name)
            # Copy label if exists
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, part_work_dir / "labels" / lbl_path.name)

        # Create zip
        zip_file = shutil.make_archive(
            str(output_dir / part_name), "zip", str(part_work_dir)
        )
        zip_paths.append(Path(zip_file))

        # Clean up work dir
        shutil.rmtree(part_work_dir)

    return zip_paths


def _clean_remote_files(api: HfApi, repo_id: str, keep_files: set[str]) -> int:
    """
    Delete stale remote files so repeated uploads don't accumulate old parts
    and exceed Hugging Face file-count limits.

    We keep:
      - .gitattributes
      - files present in keep_files (new zip shards + metadata)
    We remove:
      - old zip/json/yaml artifacts not in keep_files
      - legacy raw trees under train/, val/, test/
    """
    remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    deleted = 0
    for path in remote_files:
        if path == ".gitattributes" or path in keep_files:
            continue

        remove_for_refresh = (
            path.endswith(".zip")
            or path.endswith(".yaml")
            or path.endswith(".json")
            or path.endswith(".csv")
        )
        remove_legacy_raw = (
            path.startswith("train/")
            or path.startswith("val/")
            or path.startswith("test/")
        )
        if not (remove_for_refresh or remove_legacy_raw):
            continue

        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Delete stale file: {path}",
        )
        deleted += 1

    return deleted


def upload_to_hf(repo_id: str, source_folder: str, chunk_size: int, clean_remote: bool):
    api = HfApi()
    source_path = Path(source_folder)

    print(f"Creating repository {repo_id} (if it doesn't exist)...")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Note: Repository creation might have failed: {e}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        upload_files = []

        # 1. Zip the splits
        for split in ["train", "val", "test"]:
            split_dir = source_path / split
            if split_dir.exists():
                print(f"Processing {split} split...")
                zips = create_zip_parts(
                    split_dir, temp_path, split, chunk_size=chunk_size
                )
                upload_files.extend(zips)
                # Copy the split_boxes.csv if it exists
                csv_path = split_dir / f"{split}_boxes.csv"
                if csv_path.exists():
                    shutil.copy2(csv_path, temp_path / f"{split}_boxes.csv")
                    upload_files.append(temp_path / f"{split}_boxes.csv")

        # 2. Add metadata files
        for meta in ["dataset.yaml", "manifest.json"]:
            meta_path = source_path / meta
            if meta_path.exists():
                shutil.copy2(meta_path, temp_path / meta)
                upload_files.append(temp_path / meta)

        # 3. Upload everything in the temp folder
        keep_files = {p.name for p in upload_files}

        if clean_remote:
            print(f"Cleaning stale remote files in {repo_id}...")
            removed = _clean_remote_files(api, repo_id, keep_files)
            print(f"  Removed {removed} stale file(s).")

        print(f"Uploading {len(upload_files)} files to {repo_id}...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload zipped dataset (chunks of {chunk_size})",
        )

    print(
        f"Upload complete! View your dataset at https://huggingface.co/datasets/{repo_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face as multiple zip files"
    )
    parser.add_argument(
        "--repo_id",
        default="TheBlindMaster/VinBigData-Chest-X-ray-Prepared",
        help="Hugging Face repo ID",
    )
    parser.add_argument("--folder", default="data/processed", help="Folder to upload")
    parser.add_argument(
        "--chunk_size", type=int, default=5000, help="Number of images per zip file"
    )
    parser.add_argument(
        "--clean_remote",
        action="store_true",
        help="Delete stale remote zip/json/yaml files and legacy train/val/test trees before upload",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Folder {args.folder} does not exist.")
    elif args.chunk_size <= 0:
        print("Error: --chunk_size must be > 0")
    else:
        upload_to_hf(args.repo_id, args.folder, args.chunk_size, args.clean_remote)
