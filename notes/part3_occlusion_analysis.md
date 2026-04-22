# Part3 Occlusion Analysis

## Run Scope

- Scenario: `Part3` occlusion and emergence
- Sequence: `part3_5m_01`
- Bag-aligned timestamp range:
  `1760934688.877809286` to `1760934694.223556995`
- Evaluation run timestamp: `20260422_130121`
- Sensor rate: `10 Hz`

## Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_130121_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_130121_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_130121_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_130121.txt`

## Core Result

For this `5 m` emergence sequence, the model already produced visible pedestrian boxes in RViz shortly after the person started to appear from the crop. However, the strict `TP@IoU>=0.5` result remained very low. The key reason is not that the model failed to detect the person at all, but that the predicted box overlap with the ground-truth box was often not sufficient to pass the strict IoU criterion during the partially occluded stage.

This run contained `54` ground-truth frames in the `mid_5_15m` distance bin. Under the strict evaluation criterion `IoU >= 0.5`, only `3` frames were counted as true positives, giving a strict recall of `5.56%`. After adding the loose criterion `IoU >= 0.3`, `48` frames were considered successfully detected, corresponding to a loose recall of `88.89%`. Using center error `<= 0.3 m` as a coarse localization criterion, `52` frames were successful, corresponding to a coarse localization recall of `96.30%`.

These numbers show that the model can usually perceive the person early and place the box center near the true person location, but the predicted box size, shape, or orientation is still inaccurate under partial occlusion. In other words, the main weakness in this scene is strict box regression quality rather than target existence perception.

## Metrics Summary

- `Total GT`: `54`
- `Total TP`: `3`
- `Total FP`: `49`
- `Total FN`: `51`
- `Strict Recall @ IoU>=0.5`: `5.56%`
- `Loose Recall @ IoU>=0.3`: `88.89%`
- `Coarse Localization Recall @ center_error<=0.3m`: `96.30%`
- `Mean IoU` over strict TP: `0.5242`
- `Mean Confidence` over strict TP: `0.9216`
- `Mean Latency`: `51` frames = `5.1 s`
- `Mean Center Error` over strict TP: `0.0923 m`
- `Mean GT Points in Box`: `58.41`
- `Min Trigger Points`: `78`

## Distance-bin Detail

Only one distance bin was involved in this run.

- Distance bin: `mid_5_15m`
- `GT = 54`
- `TP = 3`
- `FN = 51`
- `Recall = 5.56%`
- `Loose TP = 48`
- `Loose Recall = 88.89%`
- `Coarse TP = 52`
- `Coarse Localization Recall = 96.30%`

## Interpretation for Part3

For the occlusion and emergence scenario, a single strict IoU-based TP metric is not sufficient to describe model behavior. In this sequence, RViz visualization looked qualitatively reasonable because the detector often generated a box around the emerging person. However, during partial visibility, even modest box size or boundary errors can reduce IoU sharply, which causes many visually plausible detections to be counted as false negatives under a strict `IoU >= 0.5` rule.

Therefore, Part3 should be reported with at least two complementary views:

- A strict metric such as `TP@IoU>=0.5`, which measures whether the predicted box is accurate enough for final high-precision localization.
- A loose or emergence-oriented metric such as `detected@IoU>=0.3` or `center_error<=0.3 m`, which measures whether the system has already perceived the person and roughly localized them.

For safety-related agricultural machinery, the second view is operationally important. Early awareness that a person is emerging from vegetation can still be useful for slowing down, issuing warnings, or increasing caution, even before the detector produces a tightly aligned bounding box.

## Second 5m Sequence

### Run Scope

- Sequence: `part3_5m_02`
- Bag-aligned timestamp range:
  `1760934721.053413391` to `1760934724.079257965`
- Evaluation run timestamp: `20260422_131300`

### Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_131300_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_131300_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_131300_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_131300.txt`

### Metrics Summary

- `Total GT`: `31`
- `Total TP`: `0`
- `Total FP`: `29`
- `Total FN`: `31`
- `Strict Recall @ IoU>=0.5`: `0.00%`
- `Loose Recall @ IoU>=0.3`: `87.10%`
- `Coarse Localization Recall @ center_error<=0.3m`: `93.55%`
- `Mean IoU` over strict TP: `0.0`
- `Mean Confidence` over strict TP: `0.0`
- `Mean Latency`: `null`
- `Mean GT Points in Box`: `53.94`

### Interpretation

The second `5 m` sequence shows the same overall behavior as the first one, but under a stricter outcome. The detector frequently produced person boxes and still maintained high loose detection recall and high coarse localization recall, yet none of the frames reached the strict `IoU >= 0.5` requirement. As a result, strict recall dropped to `0%`, while the broader emergence-oriented view still indicates that the person was perceived in most frames.

This strengthens the conclusion that the main bottleneck in Part3 is not target presence detection but box overlap quality under partial occlusion. The model usually reacts to the emerging person, but the predicted box does not align tightly enough with the annotated full-body box to be counted as a strict true positive.

## Combined 5m Conclusion

Combining the two `5 m` sequences gives a more stable picture of model behavior in the emergence scenario.

- Combined `GT`: `85`
- Combined strict `TP`: `3`
- Combined strict recall: `3.53%`
- Combined loose `TP`: `75`
- Combined loose recall: `88.24%`
- Combined coarse localization `TP`: `81`
- Combined coarse localization recall: `95.29%`

Across both `5 m` runs, the detector consistently shows high early awareness and high coarse localization capability, but very weak strict IoU-based success. This means the model is often operationally aware that a person is emerging from the crop, yet its box regression remains too unstable during partial visibility to satisfy a final high-precision evaluation criterion.

## 6m Sequence

### Run Scope

- Sequence: `part3_6m_01`
- Bag-aligned timestamp range:
  `1760934771.384391785` to `1760934775.721526623`
- Evaluation run timestamp: `20260422_132146`

### Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132146_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132146_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132146_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_132146.txt`

### Metrics Summary

- `Total GT`: `44`
- `Total TP`: `2`
- `Total FP`: `96`
- `Total FN`: `42`
- `Strict Recall @ IoU>=0.5`: `4.55%`
- `Loose Recall @ IoU>=0.3`: `95.45%`
- `Coarse Localization Recall @ center_error<=0.3m`: `97.73%`
- `Mean IoU` over strict TP: `0.5239`
- `Mean Confidence` over strict TP: `0.8725`
- `Mean Latency`: `null`
- `Mean Center Error` over strict TP: `0.1159 m`
- `Mean GT Points in Box`: `53.50`
- `Min Trigger Points`: `51`

### Interpretation

The `6 m` result is consistent with the two `5 m` sequences. Strict IoU-based detection remains very low, but loose detection recall and coarse localization recall are both extremely high. This means the model almost always reacts to the emerging pedestrian and places a box near the correct region, yet the overlap with the annotated full-body box is still too weak in most frames to pass the strict evaluation rule.

Compared with `5 m`, the `6 m` sequence shows similarly weak strict recall, but even stronger loose and coarse localization recall. The dominant limitation therefore still appears to be box regression quality under partial occlusion, not whether the model notices the target at all.

### Annotation Note

During manual inspection of the `6 m` run, the detector sometimes produced boxes for a second real person in the scene, but that person is not currently annotated in the ground truth. Because of this, frame-level `FP` counts and any metric that depends on unmatched predictions, especially `MOTA`, are not fully reliable for this sequence. Until the second person is annotated, the most trustworthy Part3 indicators are the ground-truth-centered metrics for the target occluded pedestrian:

- strict recall
- loose recall
- coarse localization recall
- center error
- latency

The `FP` and `MOTA` values for the current `6 m` sequence should therefore be treated as provisional only.

## 8m Sequence

### Run Scope

- Sequence: `part3_8m_01`
- Bag-aligned timestamp range:
  `1760934792.969187260` to `1760934796.196823120`
- Evaluation run timestamp: `20260422_132638`

### Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132638_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132638_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132638_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_132638.txt`

### Metrics Summary

- `Total GT`: `33`
- `Total TP`: `13`
- `Total FP`: `70`
- `Total FN`: `20`
- `Strict Recall @ IoU>=0.5`: `39.39%`
- `Loose Recall @ IoU>=0.3`: `96.97%`
- `Coarse Localization Recall @ center_error<=0.3m`: `96.97%`
- `Mean IoU` over strict TP: `0.5749`
- `Mean Confidence` over strict TP: `0.7616`
- `Mean Latency`: `25` frames = `2.5 s`
- `Mean Center Error` over strict TP: `0.0854 m`
- `Mean GT Points in Box`: `47.39`
- `Min Trigger Points`: `39`

### Interpretation

The `8 m` sequence is the strongest Part3 result so far. Strict recall rises substantially to `39.39%`, while loose detection recall and coarse localization recall remain extremely high at `96.97%`. This suggests that, as the target appearance pattern in this sequence becomes more favorable, the model is no longer limited only to coarse awareness: a meaningful portion of frames now also satisfy the strict IoU criterion.

Compared with `5 m` and `6 m`, this `8 m` run indicates better strict box quality while preserving similarly strong emergence awareness. The model still benefits from a loose interpretation for early detection analysis, but the gap between “roughly sees the person” and “strictly localizes the person” becomes noticeably smaller here.

### Annotation Note

Manual inspection also showed a second upright person in the `8 m` scene who is not currently annotated in the ground truth. Therefore, as in the `6 m` case, `FP` and `MOTA` are not fully reliable for this sequence and should be treated as provisional until the second person is annotated. The most trustworthy metrics remain the ground-truth-centered target metrics:

- strict recall
- loose recall
- coarse localization recall
- center error
- latency

## 10m First Sequence

### Run Scope

- Sequence: `part3_10m_01`
- Bag-aligned timestamp range:
  `1760934810.620367527` to `1760934815.058284760`
- Evaluation run timestamp: `20260422_132938`

### Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132938_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132938_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_132938_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_132938.txt`

### Metrics Summary

- `Total GT`: `45`
- `Total TP`: `40`
- `Total FP`: `63`
- `Total FN`: `5`
- `Strict Recall @ IoU>=0.5`: `88.89%`
- `Loose Recall @ IoU>=0.3`: `97.78%`
- `Coarse Localization Recall @ center_error<=0.3m`: `97.78%`
- `Mean IoU` over strict TP: `0.5980`
- `Mean Confidence` over strict TP: `0.7590`
- `Mean Latency`: `2` frames = `0.2 s`
- `Mean Center Error` over strict TP: `0.0647 m`
- `Mean GT Points in Box`: `36.73`
- `Min Trigger Points`: `33`

### Interpretation

The first `10 m` sequence is by far the strongest Part3 result so far. Strict recall rises to `88.89%`, which means the detector not only notices the emerging person early but also produces boxes that satisfy the strict IoU criterion in most frames. Loose detection recall and coarse localization recall remain very high as well, so this sequence shows both strong emergence awareness and strong final localization quality.

This result is notable because it is counter to the usual intuition that greater distance should always degrade detection. In the current Part3 data, the `10 m` sequence appears easier for the model than the `5 m` and `6 m` sequences, likely because the geometry, visibility progression, or background structure of this specific sequence is more favorable. This should be interpreted as a sequence-level effect, not automatically as a general rule that `10 m` is easier than `5 m`.

### Annotation Note

This `10 m` sequence also contains a second real person who is not yet annotated in the ground truth. Therefore, as with the `6 m` and `8 m` runs, `FP` and `MOTA` remain provisional. The most trustworthy indicators are still the target-centered metrics for the annotated occluded pedestrian:

- strict recall
- loose recall
- coarse localization recall
- center error
- latency

## 10m Second Sequence

### Run Scope

- Sequence: `part3_10m_02`
- Bag-aligned timestamp range:
  `1760934839.049735308` to `1760934842.981302738`
- Evaluation run timestamp: `20260422_133149`

### Saved Result Files

- Summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_133149_summary.json`
- Frame metrics:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_133149_frame_metrics.csv`
- Performance summary:
  `result/eval/archive/part2_iou_sweep/part3_occlusion_20260422_133149_performance_summary.json`
- Detection text log:
  `result/Rice field/Part3_Occlusion-emerge_20260422_133149.txt`

### Metrics Summary

- `Total GT`: `40`
- `Total TP`: `31`
- `Total FP`: `7`
- `Total FN`: `9`
- `Strict Recall @ IoU>=0.5`: `77.50%`
- `Loose Recall @ IoU>=0.3`: `95.00%`
- `Coarse Localization Recall @ center_error<=0.3m`: `95.00%`
- `Mean IoU` over strict TP: `0.6073`
- `Mean Confidence` over strict TP: `0.8105`
- `Mean Latency`: `5` frames = `0.5 s`
- `Mean Center Error` over strict TP: `0.0649 m`
- `Mean GT Points in Box`: `31.58`
- `Min Trigger Points`: `25`
- `MOTA`: `0.60`

### Interpretation

The second `10 m` sequence is also strong, although slightly weaker than the first `10 m` run in strict recall. It still achieves high strict recall, high loose detection recall, and high coarse localization recall, while maintaining low localization error and short latency. Importantly, this sequence contains only the annotated target pedestrian, so the `FP` and `MOTA` values are directly meaningful here.

Because `FP = 7` and `MOTA = 0.60` are not contaminated by an unannotated second person, this sequence provides the cleanest overall Part3 evaluation among the current runs. It shows that under a cleaner single-target setup at `10 m`, the detector can both recognize the emerging pedestrian early and localize them with reasonably strong final accuracy.

## Part3 Summary Table

| Sequence | GT | Strict Recall | Loose Recall | Coarse Localization Recall | Mean Latency | Mean Center Error (m) | Annotation Reliability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `part3_5m_01` | 54 | 5.56% | 88.89% | 96.30% | 5.1 s | 0.0923 | reliable for target-centered metrics |
| `part3_5m_02` | 31 | 0.00% | 87.10% | 93.55% | N/A | N/A | reliable for target-centered metrics |
| `part3_6m_01` | 44 | 4.55% | 95.45% | 97.73% | N/A | 0.1159 | second person unannotated, FP/MOTA provisional |
| `part3_8m_01` | 33 | 39.39% | 96.97% | 96.97% | 2.5 s | 0.0854 | second person unannotated, FP/MOTA provisional |
| `part3_10m_01` | 45 | 88.89% | 97.78% | 97.78% | 0.2 s | 0.0647 | second person unannotated, FP/MOTA provisional |
| `part3_10m_02` | 40 | 77.50% | 95.00% | 95.00% | 0.5 s | 0.0649 | clean single-target result |

## Part3 Overall Conclusion

Across all currently evaluated Part3 sequences, the most stable pattern is that loose recall and coarse localization recall remain very high, while strict recall varies strongly by sequence. This means the detector usually notices the emerging pedestrian and places a box near the correct location even under occlusion, but whether that box is accurate enough to satisfy a strict `IoU >= 0.5` criterion depends heavily on the specific sequence.

The `5 m` and `6 m` sequences show the weakest strict performance, indicating that early emergence under heavier partial occlusion still causes unstable box overlap even when target awareness is already present. The `8 m` sequence improves substantially, and both `10 m` sequences perform best overall. In particular, `part3_10m_02` is currently the cleanest evaluation because it contains only the annotated target pedestrian, so its `FP` and `MOTA` can be interpreted directly.

At the current stage, the most defensible Part3 conclusion is:

- the model has strong early awareness of emerging pedestrians
- coarse localization is consistently strong across distances
- strict final localization quality is sequence-dependent
- for multi-person sequences with incomplete annotation, target-centered metrics are more trustworthy than `FP` or `MOTA`
