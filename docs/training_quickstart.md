# Training Quickstart

This platform ships with twelve trainable demo algorithms spanning classification, detection, and segmentation. Training is YAML-driven and uses the same adapter/registry system as evaluation.

## Available demo algorithms and train configs

### Classification (CIFAR-10)
- ResNet-18: `configs/algorithms/demo_cls_resnet18.yaml`
  - Scratch: `configs/train/cls_resnet18_cifar10_scratch.yaml` (SGD 0.1, 100 epochs)
  - Finetune: `configs/train/cls_resnet18_cifar10.yaml` (SGD 0.01, 40 epochs)
- ResNet-50: `configs/algorithms/demo_cls_resnet50.yaml`
  - Scratch: `configs/train/cls_resnet50_cifar10_scratch.yaml`
  - Finetune: `configs/train/cls_resnet50_cifar10.yaml`
- MobileNetV3-Small: `configs/algorithms/demo_cls_mobilenetv3s.yaml`
  - Scratch: `configs/train/cls_mobilenetv3s_cifar10_scratch.yaml`
  - Finetune: `configs/train/cls_mobilenetv3s_cifar10.yaml`
- EfficientNet-B0: `configs/algorithms/demo_cls_efficientnetb0.yaml`
  - Scratch: `configs/train/cls_efficientnetb0_cifar10_scratch.yaml`
  - Finetune: `configs/train/cls_efficientnetb0_cifar10.yaml`
  
Typical convergence: with the finetune presets, CIFAR-10 models surpass ~80% accuracy after ~30–50 epochs on a single GPU; scratch presets reach similar accuracy by ~60–80 epochs.

### Detection (COCO-style)
- Faster R-CNN ResNet50 FPN: `configs/algorithms/demo_det_frcnn_r50.yaml`
  - Scratch: `configs/train/det_frcnn_r50_coco_scratch.yaml` (24 epochs, LR 0.005)
  - Finetune: `configs/train/det_frcnn_r50_coco.yaml` (12-epoch 1x schedule, LR 0.0025)
- RetinaNet ResNet50 FPN: `configs/algorithms/demo_det_retinanet_r50.yaml`
  - Scratch: `configs/train/det_retinanet_r50_coco_scratch.yaml`
  - Finetune: `configs/train/det_retinanet_r50_coco.yaml`
- FCOS ResNet50 FPN: `configs/algorithms/demo_det_fcos_r50.yaml`
  - Scratch: `configs/train/det_fcos_r50_coco_scratch.yaml`
  - Finetune: `configs/train/det_fcos_r50_coco.yaml`
- SSD300 VGG16: `configs/algorithms/demo_det_ssd300_vgg16.yaml`
  - Scratch: `configs/train/det_ssd300_vgg16_coco_scratch.yaml`
  - Finetune: `configs/train/det_ssd300_vgg16_coco.yaml`

Defaults follow a COCO 1x schedule (12 epochs, multistep at 8/11). Expect validation loss to stabilize within the first 6–8 epochs on COCO-style datasets when using pretrained weights.

### Segmentation (COCO-style masks)
- DeepLabV3 ResNet50: `configs/algorithms/demo_seg_deeplabv3_r50.yaml`
  - Scratch: `configs/train/seg_deeplabv3_r50_coco_scratch.yaml`
  - Finetune: `configs/train/seg_deeplabv3_r50_coco.yaml`
- DeepLabV3 ResNet101: `configs/algorithms/demo_seg_deeplabv3_r101.yaml`
  - Scratch: `configs/train/seg_deeplabv3_r101_coco_scratch.yaml`
  - Finetune: `configs/train/seg_deeplabv3_r101_coco.yaml`
- FCN ResNet50: `configs/algorithms/demo_seg_fcn_r50.yaml`
  - Scratch: `configs/train/seg_fcn_r50_coco_scratch.yaml`
  - Finetune: `configs/train/seg_fcn_r50_coco.yaml`
- FCN ResNet101: `configs/algorithms/demo_seg_fcn_r101.yaml`
  - Scratch: `configs/train/seg_fcn_r101_coco_scratch.yaml`
  - Finetune: `configs/train/seg_fcn_r101_coco.yaml`

Segmentation presets use AdamW + cosine decay for 60–80 epochs and monitor `val_miou`. Finetune runs typically climb past 0.5 mIoU on standard COCO-like subsets within the first half of training when labels are balanced.

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

Segmentation follows the same pattern with `--task segmentation`. Swap in the `_scratch` presets to train from random initialization (set the algo YAML `pretrained: false` when going fully from scratch).

## Outputs and checkpoints

Run artifacts are stored under `checkpoints/<task>/<algo_name>/<dataset_name>/<timestamp>/` and contain:

- `last.pth` and `best.pth` (model + optimizer state)
- `train_log.csv` (epoch-level metrics)
- `configs/` (copies of algo/data/train YAMLs)
- Optional `eval/` folder if `eval_after_train: true` and an evaluation YAML is provided; this folder mirrors the evaluation CLI outputs (CSV, plots, HTML report).

## Notes

- COCO demos expect dataset paths in `configs/datasets/demo_coco_det.yaml` and `configs/datasets/demo_coco_seg.yaml`; adjust `train_root`/`val_root` and annotation files to match your local layout.
- AMP is enabled by default in the provided CUDA presets. Disable only if you encounter numerical issues.
- The evaluation runner can consume `best.pth` by setting `weights` in the algo YAML or by using `eval_after_train: true` in the train config.

## 中文快速上手

使用统一入口启动训练（以 CIFAR-10 分类 ResNet-18 为例）：

```bash
python -m runners.train --task classification \
  --algo configs/algorithms/demo_cls_resnet18.yaml \
  --data configs/datasets/demo_cifar.yaml \
  --train configs/train/cls_resnet18_cifar10.yaml \
  --device cuda
```

- `_scratch` 预设适合从随机初始化开始（请在算法 YAML 中将 `pretrained` 设为 `false`）。
- `_finetune`/默认预设使用 ImageNet/COCO 权重，小学习率收敛更快。

训练结果保存在 `checkpoints/<task>/<algo_name>/<dataset_name>/<timestamp>/` 下，包含 `last.pth`、`best.pth`、`train_log.csv` 以及所用配置文件副本；如果训练配置里设置了 `eval_after_train: true` 并提供评测 YAML，会额外生成 `eval/` 子目录存放评测产物（CSV、图表、HTML 报告）。
