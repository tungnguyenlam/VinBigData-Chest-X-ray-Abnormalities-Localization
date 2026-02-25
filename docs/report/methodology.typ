= Methodology

== Detection Models
Two distinct architectures were utilized independently prior to combining their outputs via an ensemble methodology. All advanced processing models were constrained to simple structure localization tasks.

1. *YOLO (You Only Look Once):* Acted as our primary rapid, single-stage anchor-free detector framework. It maps spatial coordinates effectively with highly calibrated and inherently conservative bounding box confidences.
2. *Faster R-CNN:* Acted as our robust two-stage detector region-proposal framework. While generally powerful, it intrinsically generated predictions with highly inflated confidence boundaries natively scaling aggressively toward $0.999$.

== Ensemble Formulation (Weighted Box Fusion)
To combine predictions from YOLO and Faster R-CNN, we utilized Weighted Box Fusion (WBF). Unlike basic Non-Maximum Suppression (NMS) which simply discards overlapping bounding boxes, WBF merges them by recalculating the final bounding box coordinates. The recalculation averages the coordinates of closely overlapping predictions _weighted strictly by their respective confidence scores_. 

=== Confidence Score Discrepancy
Unlike single-model configurations, the models output significantly different raw confidence bounds natively across validation objects:
- *Faster R-CNN Output Bounds:* $bracket.l 0.050, 0.999 bracket.r$
- *YOLO Output Bounds:* $bracket.l 0.001, 0.847 bracket.r$

Because WBF mathematically refactors fused coordinates directly using the corresponding detection confidence metric of the underlying bounding boxes, merging these two isolated confidence paradigms independently skewed spatial representation. Faster R-CNN structurally acts as a heavier mathematical vector point during convergence natively due to its artificially inflated scalar intervals.

=== Min-Max Normalization Integration
To completely align mapping constraints prior to applying WBF structure combinations, a deterministic `Min-Max Scaling` logic was encoded universally across the inference pipeline. Prior to any fusion parameter extraction, our system sequentially processes the independent subsets and maps all scores directly into normalized scalar parameters scaling specifically from $[0.0, 1.0]$: 
$ S_"norm" = (S - S_"min") / (S_"max" - S_"min") $

Once both models were forced onto a uniform predictive structure matrix independently per image subset (utilizing a $0.0$-score fallback to maintain weight ratios on negative zero-finding subset arrays), WBF successfully converged bounding box locations globally without inherent bias weighting logic errors toward the regional proposal structure.

== Evaluation Metrics
To robustly capture localization precision and sensitivity, our pipeline measured the following:
- *mAP\@50:* Mean Average Precision computed at an Intersection-over-Union (IoU) threshold of $0.5$.
- *mAP\@50-95:* Mean Average Precision averaged across IoU thresholds ranging from $0.5$ to $0.95$ (step $0.05$). This is the primary strict metric for exact bounding box alignment.
- *FROC (Free-Response Receiver Operating Characteristic):* Used natively in medical lesion detection, it measures the average sensitivity (lesion localization fraction) explicitly plotted against discrete average False Positive (FP) rates per image (e.g., $0.125, 0.25, ..., 8.0$ FPs/image).
