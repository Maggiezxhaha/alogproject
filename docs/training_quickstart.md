# Training Quickstart

This platform ships with twelve trainable demo algorithms spanning classification, detection, and segmentation. Training is YAML-driven and uses the same adapter/registry system as evaluation.

## Available demo algorithms and train configs

### Classification (CIFAR-10)
- ResNet-18: `configs/algorithms/demo_cls_resnet18.yaml` + `configs/train/cls_resnet18_cifar10.yaml`
- ResNet-50: `configs/algorithms/demo_cls_resnet50.yaml` + `configs/train/cls_resnet50_cifar10.yaml`
- MobileNetV3-Small: `configs/algorithms/demo_cls_mobilenetv3s.yaml` + `configs/train/cls_mobilenetv3s_cifar10.yaml`
- EfficientNet-B0: `configs/algorithms/demo_cls_efficientnetb0.yaml` + `configs/train/cls_efficientnetb0_cifar10.yaml`

### Detection (COCO-style)
- Faster R-CNN ResNet50 FPN: `configs/algorithms/demo_det_frcnn_r50.yaml` + `configs/train/det_frcnn_r50_coco.yaml`
- RetinaNet ResNet50 FPN: `configs/algorithms/demo_det_retinanet_r50.yaml` + `configs/train/det_retinanet_r50_coco.yaml`
- FCOS ResNet50 FPN: `configs/algorithms/demo_det_fcos_r50.yaml` + `configs/train/det_fcos_r50_coco.yaml`
- SSD300 VGG16: `configs/algorithms/demo_det_ssd300_vgg16.yaml` + `configs/train/det_ssd300_vgg16_coco.yaml`

### Segmentation (COCO-style masks)
- DeepLabV3 ResNet50: `configs/algorithms/demo_seg_deeplabv3_r50.yaml` + `configs/train/seg_deeplabv3_r50_coco.yaml`
- DeepLabV3 ResNet101: `configs/algorithms/demo_seg_deeplabv3_r101.yaml` + `configs/train/seg_deeplabv3_r101_coco.yaml`
- FCN ResNet50: `configs/algorithms/demo_seg_fcn_r50.yaml` + `configs/train/seg_fcn_r50_coco.yaml`
- FCN ResNet101: `configs/algorithms/demo_seg_fcn_r101.yaml` + `configs/train/seg_fcn_r101_coco.yaml`

## Running training

Use the unified CLI:

```bash
python -m runners.train --task classification \
  --algo configs/algorithms/demo_cls_resnet18.yaml \
  --data configs/datasets/demo_cifar.yaml \
  --train configs/train/cls_resnet18_cifar10.yaml \
  --device cuda
```

Example detection launch:

```bash
python -m runners.train --task detection \
  --algo configs/algorithms/demo_det_frcnn_r50.yaml \
  --data configs/datasets/demo_coco_det.yaml \
  --train configs/train/det_frcnn_r50_coco.yaml \
  --device cuda
```

Segmentation follows the same pattern with `--task segmentation`.

## Outputs and checkpoints

Run artifacts are stored under `checkpoints/<task>/<algo_name>/<dataset_name>/<timestamp>/` and contain:

- `last.pth` and `best.pth` (model + optimizer state)
- `train_log.csv` (epoch-level metrics)
- `configs/` (copies of algo/data/train YAMLs)
- Optional `eval/` folder if `eval_after_train: true` and an evaluation YAML is provided; this folder mirrors the evaluation CLI outputs (CSV, plots, HTML report).

## Notes

- COCO demos expect dataset paths in `configs/datasets/demo_coco_det.yaml` and `configs/datasets/demo_coco_seg.yaml`; adjust `train_root`/`val_root` and annotation files to match your local layout.
- Enable AMP by setting `amp: true` in the train YAML when training on CUDA.
- The evaluation runner can consume `best.pth` by setting `weights` in the algo YAML or by using `eval_after_train: true` in the train config.
