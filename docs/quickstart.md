# Quickstart

This platform evaluates pretrained models for **classification / detection / segmentation** with unified adapters, eight-dimension metrics, adversarial safety, and multi-layer scoring.

## Prepare environment
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   CPU-only PyTorch wheels are pinned; if you have CUDA you can swap to GPU wheels from the PyTorch index.

2. Optional: download COCO data for detection/segmentation demos under `data/coco` (follow standard COCO layout).

## Data repository layout
- Classification demo: CIFAR-10 auto-downloads to the path in `configs/datasets/demo_cifar.yaml`.
- Detection demo expects COCO val2017 images and `instances_val2017.json`.
- Segmentation demo reuses the same COCO annotations with masks.

## Model weights
- Place user weights and set the `weights` field in the algorithm config. If omitted, pretrained backbones are used with randomly initialized heads.

## Run the end-to-end demo
Classification + safety + report (falls back to CPU if CUDA is unavailable):
```bash
python -m runners.run --algo configs/algorithms/demo_cls.yaml --data configs/datasets/demo_cifar.yaml --eval configs/evaluation/8dims.yaml --device cuda
```

Detection (requires COCO + pycocotools):
```bash
python -m runners.run --algo configs/algorithms/demo_det.yaml --data configs/datasets/demo_coco_det.yaml --eval configs/evaluation/8dims.yaml --device cuda
```

Segmentation (COCO masks):
```bash
python -m runners.run --algo configs/algorithms/demo_seg.yaml --data configs/datasets/demo_coco_seg.yaml --eval configs/evaluation/8dims.yaml --device cuda
```

## Outputs
Results are written to `results/<timestamp>/<algo>/<dataset>/`:
- `metrics_raw.csv`: per-dimension raw fields (including safety_clean_metric / safety_adv_metric / safety_retention / drop)
- `metrics_norm.csv`: normalized eight-dimension `*_score`
- `score_final.csv`: weighted AHP + TOPSIS summary and rank
- Visuals: `radar.png`, `bars.png`, `acc_latency.png`, `safety.png`
- `report.html`: embeds charts, threat model (PGD params), and CSV paths for reproducibility
