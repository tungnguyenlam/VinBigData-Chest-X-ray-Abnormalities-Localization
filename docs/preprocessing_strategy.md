# Chest X-Ray Preprocessing Strategy
## Why Convert 16-Bit 1-Channel DICOMs to 8-Bit 3-Channel PNGs?

In computer vision pipelines applied to medical imaging, there is a recurring architectural challenge: **Modern object detectors (YOLOv8, Faster R-CNN, DETR) are aggressively optimized for 8-bit, 3-channel RGB images** like those found in the COCO dataset. 

Raw chest radiographs (DICOM format) operate fundamentally differently: they are single-channel grayscale, and they contain up to 16 bits of dynamic range (65,536 shades of gray) to capture subtle differences in tissue density.

To bridge this gap without losing critical medical data or sacrificing the speed of pre-trained models, our pipeline explicitly converts the raw 16-bit DICOMs into custom 3-channel 8-bit PNGs before training. 

### 1. Framework & Pre-trained Architecture Compatibility
Pre-trained CNNs and Vision Transformers expect an input shape of `(Batch, 3, Height, Width)`. If we feed the network a single-channel 1-channel image, we either:
* Have to duplicate that single channel across all 3 RGB layers (wasting channel space).
* Have to heavily modify the model's first convolution layer, instantly destroying the value of any ImageNet or COCO pre-trained weights.

By giving the models exactly what they expect—a 3-channel 8-bit matrix—we preserve structural compatibility and leverage transfer learning seamlessly.

### 2. Maximizing Visual Information Density (The "Multi-Window" Technique)
Naively compressing a 16-bit DICOM down into a single 8-bit grayscale image results in catastrophic data loss. Subtle abnormalities—like a faint pulmonary nodule hiding behind a rib—will be crushed into the same gray pixel value as surrounding tissue.

Since we are generating 3 channels (Red, Green, Blue) for compatibility anyway, we use these 3 channels to store **three different mathematical interpretations** of the original 16-bit radiograph:

* **Channel 1 (Standard Window):** We apply a standard percentile-based linear scaling. This acts as the "baseline" view, preserving global patient anatomy and overall contrast just as a radiologist would see in a standard viewer.
* **Channel 2 (CLAHE - Contrast Limited Adaptive Histogram Equalization):** We compute the adaptive local contrast algorithm on the raw 16-bit data, map the result down to 8-bits, and store it here. CLAHE aggressively pulls out local textures that are otherwise invisible in extremely bright regions (like the mediastinum) or extremely dark regions (like over-penetrated lung fields).
* **Channel 3 (Laplacian Filter):** We apply a second-order derivative edge-enhancement filter. This computes the rate of change between adjacent pixels, completely isolating and highlighting high-frequency spatial gradients (edges). This is medically critical for preserving sharp, subtle features like micro-nodules or hairline fractures that would otherwise be destroyed by interpolation blur during resizing.

When the neural network reads this "RGB" image, it is actually reading a stacked feature map of Standard Anatomy, Local Contrast, and Edge Maps all at once.

### 3. I/O Speed and CPU Bottlenecks
Reading DICOM files dynamically requires `pydicom`, a CPU-intensive library, and calculating CLAHE and Laplacian filters on a per-batch basis inside a PyTorch DataLoader creates a severe bottleneck. Your GPU will constantly stall at 0% utilization while waiting for the CPU to compute the medical filters.

By performing this heavy transformation purely offline during the `scripts/data/prepare.py` step, we bake the complex logic into a static, highly optimized `.png` format. During training, we can read these PNGs practically instantly using standard libraries (`cv2.imread` or `PIL`) with blazing fast zero-copy optimizations, ensuring maximum GPU throughput.
