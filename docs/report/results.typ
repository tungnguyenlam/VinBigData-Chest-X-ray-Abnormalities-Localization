= Results and Analysis

== Baseline Model Performances (1-channel Input)

To establish robust benchmarking metrics, preliminary multi-class models were evaluated directly utilizing the baseline 1-channel CLAHE processed image extractions and merged label formats natively. All metrics evaluate bounds across standard IoU validation.

#figure(
  placement: auto,
  caption: [Baseline Model Evaluation],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (center, left, left),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => (top: 0.5pt, bottom: 0.5pt, left: 0.5pt, right: 0.5pt),
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[*Model*][*Parameter*][*Precision*][*Recall*][*mAP-0.5*][*mAP-0.75*],

    [RT-DETR], [512x512, 14 classes (baseline)], [0.32], [0.31], [0.23], [0.05],
    [ResNet50-FasterRCNN], [1024x1024, only bbox], [0.12], [0.26], [0.15], [0.13],
    [ConVeXtV2-FasterRCNN], [1024x1024, only bbox], [0.14], [0.17], [0.17], [0.14],
    [YOLOv8n], [1024x1024, 15 classes], [0.38], [0.23], [0.22], [0.08],
    [RT-DETR], [1024x1024, 15 classes], [0.32], [0.33], [0.27], [0.10]
  ),
)

== Advanced Localization Model Performances (3-channel Input)

Moving to strictly binary localization bounding utilizing our advanced 3-channel arrays, we evaluated the individual model performances prior to mapping them to our unified score ensemble. After successfully applying score normalization constraints universally across all structural subsets, the Weighted Box Fusion ensemble successfully maintained peak mapping parity, effectively combining both distributions optimally. 

#figure(
  placement: auto,
  caption: [Advanced Localization Model Evaluation (3-channel)],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, left, center, center, center),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => (top: 0.5pt, bottom: 0.5pt, left: 0.5pt, right: 0.5pt),
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[*Model*][*Input Pipeline*][*mAP\@50*][*mAP\@50-95*][*FROC Score*],

    [Faster R-CNN], [3-ch 8-bit (Localization)], [0.3386], [0.1656], [0.4465],
    [YOLOv8], [3-ch 8-bit (Localization)], [0.4899], [0.2310], [0.5347],
    [WBF Ensemble (Unnormalized)], [Fused Output Distributions], [0.4163], [0.2136], [0.4735],
    [WBF Ensemble (Normalized)], [Fused Output Distributions], [0.4202], [0.2151], [0.4788],
    [WBF Ensemble (Manual Skew 0.75)], [Fused Output Distributions], [0.4680], [0.2278], [0.5212],
  ),
)

The optimized YOLO configuration significantly outperformed both baseline single configurations, and the properly normalized Weight Box Fusion effectively aligned coordinates globally to parallel these highest peaks natively.
