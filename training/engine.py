import argparse
import csv
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from algorithms import build_algo
from runners.run import load_yaml as load_yaml_file
from evaluation.aggregation import load_weights, weighted_sum, topsis
from evaluation.protocols import build_protocol_loaders
from metrics.eight_dims import EightDimensionSuite, DIMENSIONS
from datasets import build_dataloader
from visualization import plots
from visualization.report import render_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_split_cfg(data_cfg: Dict, split_key: str, train_cfg: Dict) -> Dict:
    base_cfg = deepcopy(data_cfg.get(split_key) or data_cfg)
    if split_key == "train":
        split_value = train_cfg.get("train_split") or base_cfg.get("train_split") or base_cfg.get("split", "train")
        base_cfg["split"] = split_value
        if base_cfg.get("train_root"):
            base_cfg["root"] = base_cfg["train_root"]
        if base_cfg.get("train_ann_file"):
            base_cfg["ann_file"] = base_cfg["train_ann_file"]
    else:
        split_value = train_cfg.get("val_split") or base_cfg.get("val_split") or base_cfg.get("split", "val")
        base_cfg["split"] = split_value
        if base_cfg.get("val_root"):
            base_cfg["root"] = base_cfg["val_root"]
        if base_cfg.get("val_ann_file"):
            base_cfg["ann_file"] = base_cfg["val_ann_file"]
    return base_cfg


def _build_optimizer(params, cfg: Dict):
    name = (cfg.get("name") or "sgd").lower()
    lr = cfg.get("lr", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0)
    if name == "sgd":
        momentum = cfg.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer, cfg: Dict):
    if not cfg:
        return None
    name = (cfg.get("name") or "").lower()
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.1)
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.get("milestones", [30, 60]), gamma=cfg.get("gamma", 0.1)
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("t_max", 50))
    return None


def _save_configs(run_dir: str, algo_path: str, data_path: str, train_path: str, eval_path: Optional[str] = None):
    cfg_dir = os.path.join(run_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    shutil.copyfile(algo_path, os.path.join(cfg_dir, os.path.basename(algo_path)))
    shutil.copyfile(data_path, os.path.join(cfg_dir, os.path.basename(data_path)))
    shutil.copyfile(train_path, os.path.join(cfg_dir, os.path.basename(train_path)))
    if eval_path:
        shutil.copyfile(eval_path, os.path.join(cfg_dir, os.path.basename(eval_path)))


def _evaluate_best(algo_cfg: Dict, data_cfg: Dict, eval_cfg_path: str, device: str, weight_path: str, run_dir: str):
    eval_cfg = load_yaml_file(eval_cfg_path)
    algo_eval_cfg = deepcopy(algo_cfg)
    algo_eval_cfg["weights"] = weight_path

    algo = build_algo(algo_eval_cfg, device=device)
    batch_size = eval_cfg.get("batch_size", 8)
    num_workers = eval_cfg.get("num_workers", 0)
    protocol_loaders, protocol_cfg = build_protocol_loaders(data_cfg, eval_cfg, batch_size=batch_size, num_workers=num_workers)

    metrics = EightDimensionSuite(eval_cfg, task_type=getattr(algo, "task_type", "classification"))
    norm_scores, raw_metrics = metrics.run(algo, protocol_loaders, device)

    weight_vector = load_weights(eval_cfg.get("ahp", {}), [f"{d}_score" for d in DIMENSIONS])
    final_score = weighted_sum(norm_scores, [f"{d}_score" for d in DIMENSIONS], weight_vector)
    topsis_results = topsis([{"name": algo_eval_cfg.get("name", "algo"), **norm_scores}], [f"{d}_score" for d in DIMENSIONS], weight_vector)

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    raw_path = os.path.join(eval_dir, "metrics_raw.csv")
    norm_path = os.path.join(eval_dir, "metrics_norm.csv")
    final_path = os.path.join(eval_dir, "score_final.csv")

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
        writer.writerow({"name": algo_eval_cfg.get("name"), "score": final_score, "rank": 1})

    radar_path = os.path.join(eval_dir, "radar.png")
    bar_path = os.path.join(eval_dir, "bars.png")
    scatter_path = os.path.join(eval_dir, "acc_latency.png")
    safety_path = os.path.join(eval_dir, "safety.png")

    plots.radar_chart({algo_eval_cfg.get("name", "algo"): norm_scores}, radar_path)
    plots.bar_chart({algo_eval_cfg.get("name", "algo"): norm_scores}, bar_path)
    plots.scatter_latency_accuracy(
        [
            {
                "name": algo_eval_cfg.get("name", "algo"),
                "latency": raw_metrics.get("efficiency_latency", 0),
                "accuracy": norm_scores.get("accuracy_score", 0),
            }
        ],
        scatter_path,
    )
    plots.safety_plot(
        [
            {
                "name": algo_eval_cfg.get("name", "algo"),
                "clean": raw_metrics.get("safety_clean_metric", 0),
                "adv": raw_metrics.get("safety_adv_metric", 0),
            }
        ],
        safety_path,
    )

    render_report(
        eval_dir,
        {
            "radar": os.path.basename(radar_path),
            "bar": os.path.basename(bar_path),
            "scatter": os.path.basename(scatter_path),
            "safety": os.path.basename(safety_path),
            "metrics_table": topsis_results,
            "config": {"algo_name": algo_eval_cfg.get("name"), "dataset_name": data_cfg.get("name")},
            "threat_model": eval_cfg.get("safety", {}),
            "raw_csv": os.path.basename(raw_path),
            "norm_csv": os.path.basename(norm_path),
            "score_csv": os.path.basename(final_path),
        },
    )


def run_training(task: str, algo_cfg: Dict, data_cfg: Dict, train_cfg: Dict, device: str, paths: Dict):
    set_seed(train_cfg.get("seed", 42))
    algo = build_algo(algo_cfg, device=device)

    train_cfg_data = _resolve_split_cfg(data_cfg, "train", train_cfg)
    val_cfg_data = _resolve_split_cfg(data_cfg, "val", train_cfg)

    batch_size = train_cfg.get("batch_size", 32)
    val_batch_size = train_cfg.get("val_batch_size", batch_size)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader: DataLoader = build_dataloader(train_cfg_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader: DataLoader = build_dataloader(val_cfg_data, batch_size=val_batch_size, num_workers=num_workers, shuffle=False)

    optimizer = _build_optimizer(algo.parameters(), train_cfg.get("optimizer", {}))
    scheduler = _build_scheduler(optimizer, train_cfg.get("scheduler", {}))

    epochs = train_cfg.get("epochs", 1)
    amp_enabled = train_cfg.get("amp", False) and device.startswith("cuda")
    scaler = GradScaler(enabled=amp_enabled)

    monitor = train_cfg.get("monitor", "val_loss")
    monitor_mode = train_cfg.get("monitor_mode", "min")
    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    best_path = os.path.join(paths["run_dir"], "best.pth")
    last_path = os.path.join(paths["run_dir"], "last.pth")

    log_path = os.path.join(paths["run_dir"], "train_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_miou"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    for epoch in range(1, epochs + 1):
        algo.model.train() if hasattr(algo, "model") else None
        train_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast(enabled=amp_enabled):
                loss_dict = algo.training_step(batch, device=device)
                loss = loss_dict.get("loss")
            if loss is None:
                raise ValueError("training_step must return a dict with key 'loss'")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                scheduler.step()
            train_loss += loss.item()
            steps += 1
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            scheduler.step()

        # validation
        val_loss = None
        val_accuracy = None
        val_miou = None
        if val_loader:
            val_metrics = []
            for batch in val_loader:
                metrics = algo.validation_step(batch, device=device)
                val_metrics.append(metrics)
            if val_metrics:
                if any("val_loss" in m for m in val_metrics):
                    val_loss = float(np.mean([m.get("val_loss") for m in val_metrics if m.get("val_loss") is not None]))
                if any("val_accuracy" in m for m in val_metrics):
                    val_accuracy = float(np.mean([m.get("val_accuracy") for m in val_metrics if m.get("val_accuracy") is not None]))
                if any("val_miou" in m for m in val_metrics):
                    val_miou = float(np.mean([m.get("val_miou") for m in val_metrics if m.get("val_miou") is not None]))

        monitor_value = None
        if monitor == "val_accuracy":
            monitor_value = val_accuracy
        elif monitor == "val_miou":
            monitor_value = val_miou
        else:
            monitor_value = val_loss

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, steps),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_miou": val_miou,
        }
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(row)

        state = {
            "epoch": epoch,
            "model_state": algo.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "monitor": monitor_value,
        }
        torch.save(state, last_path)

        improved = False
        if monitor_value is not None:
            if monitor_mode == "max" and monitor_value > best_metric:
                improved = True
            if monitor_mode == "min" and monitor_value < best_metric:
                improved = True
        if improved:
            best_metric = monitor_value
            torch.save(state, best_path)

        print(
            f"Epoch {epoch}/{epochs} - train_loss: {row['train_loss']:.4f} "
            f"val_loss: {val_loss if val_loss is not None else 'n/a'} "
            f"val_acc: {val_accuracy if val_accuracy is not None else 'n/a'} val_miou: {val_miou if val_miou is not None else 'n/a'}"
        )

    if not os.path.exists(best_path):
        shutil.copyfile(last_path, best_path)

    return {"last": last_path, "best": best_path, "log": log_path}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["classification", "detection", "segmentation"])
    parser.add_argument("--algo", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval", default=None, help="Optional eval config to run after training")
    return parser.parse_args()


def main():
    args = parse_args()
    algo_cfg = load_yaml_file(args.algo)
    data_cfg = load_yaml_file(args.data)
    train_cfg = load_yaml_file(args.train)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device == "cpu" and args.device != "cpu":
        print("CUDA not available, falling back to CPU. This may be slow.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", args.task, algo_cfg.get("name", "algo"), data_cfg.get("name", "dataset"), timestamp)
    os.makedirs(run_dir, exist_ok=True)

    _save_configs(run_dir, args.algo, args.data, args.train, args.eval)

    artifacts = run_training(args.task, algo_cfg, data_cfg, train_cfg, device, {"run_dir": run_dir})

    if train_cfg.get("eval_after_train") and args.eval:
        best_path = artifacts.get("best") or artifacts.get("last")
        if best_path and os.path.exists(best_path):
            _evaluate_best(algo_cfg, data_cfg, args.eval, device, best_path, run_dir)

    print(f"Training finished. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
