import argparse
import csv
import os
import time
from datetime import datetime

import torch

from algorithms import build_algo
from core import simple_yaml
from datasets import build_dataloader
from evaluation.aggregation import load_weights, weighted_sum, topsis
from evaluation.protocols import build_protocol_loaders
from metrics.eight_dims import EightDimensionSuite, DIMENSIONS
from visualization import plots
from visualization.report import render_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="Algorithm config path")
    parser.add_argument("--data", required=True, help="Dataset config path")
    parser.add_argument("--eval", required=True, help="Evaluation config path")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        content = f.read()
    return simple_yaml.safe_load(content)


def main():
    args = parse_args()
    algo_cfg = load_yaml(args.algo)
    data_cfg = load_yaml(args.data)
    eval_cfg = load_yaml(args.eval)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device == "cpu" and args.device != "cpu":
        print("CUDA not available, falling back to CPU. This may be slow.")

    algo = build_algo(algo_cfg, device=device)

    batch_size = eval_cfg.get("batch_size", 8)
    num_workers = eval_cfg.get("num_workers", 0)
    protocol_loaders, protocol_cfg = build_protocol_loaders(data_cfg, eval_cfg, batch_size=batch_size, num_workers=num_workers)

    metrics = EightDimensionSuite(eval_cfg, task_type=getattr(algo, "task_type", "classification"))
    norm_scores, raw_metrics = metrics.run(algo, protocol_loaders, device)

    weight_vector = load_weights(eval_cfg.get("ahp", {}), [f"{d}_score" for d in DIMENSIONS])
    final_score = weighted_sum(norm_scores, [f"{d}_score" for d in DIMENSIONS], weight_vector)
    topsis_results = topsis([{"name": algo_cfg.get("name", "algo"), **norm_scores}], [f"{d}_score" for d in DIMENSIONS], weight_vector)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", timestamp, algo_cfg.get("name", "algo"), data_cfg.get("name", "dataset"))
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "metrics_raw.csv")
    norm_path = os.path.join(out_dir, "metrics_norm.csv")
    final_path = os.path.join(out_dir, "score_final.csv")

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_metrics.keys()))
        writer.writeheader()
        writer.writerow(raw_metrics)

    with open(norm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f"{d}_score" for d in DIMENSIONS])
        writer.writeheader()
        writer.writerow(norm_scores)

    with open(final_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "score", "rank"])
        writer.writeheader()
        writer.writerow({"name": algo_cfg.get("name"), "score": final_score, "rank": 1})

    radar_path = os.path.join(out_dir, "radar.png")
    bar_path = os.path.join(out_dir, "bars.png")
    scatter_path = os.path.join(out_dir, "acc_latency.png")
    safety_path = os.path.join(out_dir, "safety.png")

    plots.radar_chart({algo_cfg.get("name", "algo"): norm_scores}, radar_path)
    plots.bar_chart({algo_cfg.get("name", "algo"): norm_scores}, bar_path)
    plots.scatter_latency_accuracy([
        {
            "name": algo_cfg.get("name", "algo"),
            "latency": raw_metrics.get("efficiency_latency", 0),
            "accuracy": norm_scores.get("accuracy_score", 0),
        }
    ], scatter_path)
    plots.safety_plot([
        {
            "name": algo_cfg.get("name", "algo"),
            "clean": raw_metrics.get("safety_safety_clean_metric", 0),
            "adv": raw_metrics.get("safety_safety_adv_metric", 0),
        }
    ], safety_path)

    render_report(
        out_dir,
        {
            "radar": os.path.basename(radar_path),
            "bar": os.path.basename(bar_path),
            "scatter": os.path.basename(scatter_path),
            "safety": os.path.basename(safety_path),
            "metrics_table": topsis_results,
            "config": {"algo_name": algo_cfg.get("name"), "dataset_name": data_cfg.get("name")},
            "threat_model": eval_cfg.get("safety", {}),
            "raw_csv": os.path.basename(raw_path),
            "norm_csv": os.path.basename(norm_path),
            "score_csv": os.path.basename(final_path),
        },
    )

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
