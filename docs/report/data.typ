= Dataset and Preprocessing

== Data Overview
This study utilizes the original VinBigData Chest X-ray dataset, which consists of high-resolution DICOM files. The dataset features a significant number of images categorized broadly into "No finding" and positive images containing varied bounding box (BBox) annotations representing 14 distinct chest abnormalities. 

The original dataset comprises 18,000 chest X-ray images, divided into 15,000 annotated training images and 3,000 test images without annotations, which were reserved for evaluation in the competition. The annotations for the training images are provided as 67,914 bounding-box coordinates (stored as `x_max`, `y_max`, `x_min`, `y_min`), stored in a tabular format where each record is associated with an image identifier and the corresponding ground-truth anomaly class label.

Overall, 10,606 images contain no detected anomalies and are labeled as the "No Finding" class, accounting for approximately 67% of the dataset.

#figure(
  [
  #image("images/class_name.png", width: 70%)
  ],
  caption: [Input Image Annotation Classes]
)

== File Structure and Stratified Splitting
From the original dataset, only the `train` folder was utilized for our comprehensive modeling pipeline. To effectively evaluate and train the detectors, we extracted and transformed these images into a clean, custom `processed` directory. 

We split the available data into `train`, `val`, and `test` subsets using a rigorous stratified split. This stratification ensured that the proportion of "No finding" images to images with positive findings remained consistent across all three splits, preventing class imbalance skew during evaluation.

== Preprocessing Pipeline
Prior to model development, we established preprocessing pipelines to convert the original medical .dicom arrays into formats directly suitable for deep learning object detectors. 

#figure(
  [
  #image("images/pipeline.pdf", width: 100%)
  ],
  caption: [Data Preparation Pipeline]
)

=== Baseline Image Processing (1-channel 8-bit)
For the foundational image processing branch, the pixel array representing the medical image was extracted from the .dicom files and converted into NumPy arrays using the pydicom library. We further applied the Volume of Interest Look-Up Table (VOI LUT) transformation to improve the interpretability of the 16-bit DICOM images by enhancing clinically relevant intensity ranges. Finally, pixel intensities were normalized and inverted so that anatomical structures appear bright against a dark background. 

Because the original images were stored at varying resolutions, all images were resized to a uniform resolution of 1024×1024 pixels to ensure consistency during model training. In addition, contrast enhancement was applied using Contrast Limited Adaptive Histogram Equalization (CLAHE), which improves local contrast and helps emphasize clinically relevant structures within the images. This single-channel extraction served as the evaluation point for our foundational comparison tests.

#figure(
  [
  #image("images/image_processing.pdf", width: 100%)
  ],
  caption: [Comparison between Histogram Equalization and CLAHE Extracted Processing]
)

=== Advanced Image Processing (3-channel 8-bit, Localization Only)
To fully leverage standard computer vision architectures (which typically expect rich 3-channel 8-bit RGB inputs natively), we implemented a deterministic feature extraction conversion during advanced preprocessing for the YOLO and Faster R-CNN pipelines. Crucially, this advanced 3-channel pipeline was designed and utilized *strictly for localization tasks* (i.e., binary classification of "Finding" vs. "No Finding" bounding boxes), bypassing the 14-class differentiation during this specific phase. 

The arrays are stacked into a ($H times W times 3$) representation utilizing multi-band transformations:

- *Channel 1 (Standard Window):* The raw pixel array is normalized utilizing percentile clipping (0.5% to 99.5%) to remove outliers, scaled to $[0, 1]$, and converted directly to standard 8-bit (`uint8`) space. This serves as the baseline visual representation.
- *Channel 2 (Local Contrast):* Contrast Limited Adaptive Histogram Equalization (CLAHE) is tightly computed over the 16-bit array with a clip limit of $2.0$ to enhance local boundary contrast.
- *Channel 3 (Spatial Detail):* A Laplacian filter (kernel size $3$) is applied to extract spatial details and edge enhancements.

This 3-channel extraction ensures that baseline windowing, local contrast mapping, and high-frequency edge details are distinctly preserved for the backbone CNNs, bypassing standard gray-to-RGB duplications.

=== Baseline Label Processing
For label processing, bounding-box coordinates were first rescaled to correspond to the standardized 1024×1024 image resolution. The rescaled coordinates were then converted into required structural models like normalized center mapping constraints (`cx`, `cy`, `w`, `h`) for initial training.

In addition, redundant bounding boxes were optimized by merging boxes belonging to the same class when their Intersection over Union (IoU) exceeded a threshold of 0.3. This procedure generated consolidated bounding boxes that more effectively represented each anomaly class while reducing annotation redundancy directly.

#figure(
  [
  #image("images/merging.png", width: 95%)
  ],
  caption: [Example of IoU bounding box coordinate merging]
)
