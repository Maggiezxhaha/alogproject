# Adversarial Safety Metric

Safety is defined as robustness against adversarial perturbations. The default pipeline evaluates all tasks with a white-box PGD (L∞) attack and reports both clean and adversarial metrics.

## Threat Model
- Attack: PGD (L∞)
- eps: 8/255 (inputs normalized to [0, 1])
- steps: 10
- alpha: 2/255
- random_start: true
- norm: linf
- targeted: false

These parameters are configurable under `configs/evaluation/*.yaml` and are printed in `report.html` for reproducibility.

## Task-specific measurement
- **Classification**: clean accuracy vs. adversarial accuracy on the same split; `safety_score` = adv accuracy (clamped [0,1]).
- **Detection**: PGD steps backprop through the detector loss (`forward_loss` if available); clean/adv metrics are reported as simple mAP-style overlap using IoU ≥ 0.5. When detectors do not support loss in eval mode, the provided heuristic loss on detection scores is used.
- **Segmentation**: PGD uses pixel-wise cross-entropy on masks; clean vs. adversarial mIoU are compared.

## Reported fields (written to `metrics_raw.csv`)
- `safety_clean_metric`: clean accuracy / mAP / mIoU
- `safety_adv_metric`: adversarial counterpart
- `safety_drop`: difference between clean and adversarial metric
- `safety_retention`: adversarial / clean ratio
- `safety_score`: primary safety score (adv metric clamped to [0,1])
