# VinBigData-Chest-X-ray-Abnormalities-Localization

## Dataset Management

### 1. Preparation
Convert raw DICOM images to 8-bit grayscale PNGs and generate aggregated YOLO labels (WBF). The original training data is split into **train (85%)**, **val (5%)**, and **local test (10%)** sets.

```bash
python src/data/prepare_dataset.py --force
```
*Note: Images are processed at 16-bit for quality but saved as 8-bit for YOLOv8/v11 compatibility.*

### 2. Hugging Face Upload
Upload the processed dataset to Hugging Face as multiple zip files (chunks) to handle large file counts reliably.

Hugging Face dataset repos can hit limits when a single folder contains too many files (commonly around 10k entries). Keep using zip shards and clean stale remote files on re-upload.

```bash
python src/data/upload_to_hf.py \
    --repo_id "TheBlindMaster/VinBigData-Chest-X-ray-Prepared" \
    --folder "outputs/prepared_dataset" \
    --chunk_size 5000 \
    --clean_remote
```

### 3. Hugging Face Download
Restore the exact dataset structure from Hugging Face on a new machine.

```bash
python src/data/download_from_hf.py \
    --repo_id "TheBlindMaster/VinBigData-Chest-X-ray-Prepared" \
    --output "outputs/prepared_dataset"
```

## GPU utilization

GPU usage can look uneven (spikes and dips) for a few reasons:

- **Data loading**: The GPU often waits for the next batch. If CPU workers or disk I/O can’t keep up, utilization drops between batches.
- **Validation**: Running validation (e.g. every epoch) uses a different workload or smaller batches, so utilization changes during those phases.
- **Per-batch cost**: Augmentation and number of boxes per image vary, so some batches take longer than others.
- **Logging and checkpoints**: Saving metrics or weights forces sync points and short stalls.

**What to try:**

- Increase **`num_workers`** in `src/config.py` (or the preset you use) so more batches are prepared in parallel.
- Increase **batch size** if VRAM allows (more work per batch, fewer syncs).
- For YOLO training with a prepared dataset, **`cache='disk'`** is enabled so images are cached and I/O is less of a bottleneck.
