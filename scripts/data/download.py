import argparse
import shutil
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download


def download_and_extract(repo_id: str, output_folder: str):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {repo_id}...")
    # Download everything to a cache/temp location
    download_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.zip", "*.yaml", "*.json", "*.csv"],
        )
    )

    print(f"Extracting files to {output_folder}...")

    # 1. Copy metadata files
    for meta in ["dataset.yaml", "manifest.json"]:
        src = download_dir / meta
        if src.exists():
            print(f"  Copying {meta}...")
            shutil.copy2(src, output_path / meta)

    for split in ["train", "val", "test"]:
        csv_file = f"{split}_boxes.csv"
        src = download_dir / csv_file
        if src.exists():
            split_dir = output_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Copying {csv_file} to {split}/...")
            shutil.copy2(src, split_dir / csv_file)

    # 2. Extract zip parts
    # Zip files are named like 'train_part1.zip', 'val_part2.zip', etc.
    all_zips = sorted(list(download_dir.glob("*.zip")))

    for zip_path in all_zips:
        # Determine split from filename (the part before the first underscore)
        split = zip_path.stem.split("_")[0]
        target_split_dir = output_path / split
        target_split_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting {zip_path.name} to {split}/...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_split_dir)

    print(f"\nSuccess! Dataset restored at: {output_path.resolve()}")
    print("Structure:")
    for sub in ["train", "val", "test"]:
        img_dir = output_path / sub / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.png")))
            print(f"  - {sub}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and unzip dataset from Hugging Face"
    )
    parser.add_argument(
        "--repo_id",
        default="TheBlindMaster/VinBigData-Chest-X-ray-Prepared",
        help="Hugging Face repo ID",
    )
    parser.add_argument("--output", default="data/processed", help="Target folder")

    args = parser.parse_args()

    download_and_extract(args.repo_id, args.output)
