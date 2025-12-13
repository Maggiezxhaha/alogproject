# runners/run_classification.py
import argparse
import time
from pathlib import Path

import yaml
import pandas as pd

from backends import classification_backend


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo-config",
        type=str,
        required=True,
        help="算法配置文件路径，例如 configs/algorithms/cls_resnet18_cifar10.yaml",
    )
    parser.add_argument(
        "--datasets-config",
        type=str,
        default="configs/datasets.yaml",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="results/merged/base_metrics.csv",
    )
    args = parser.parse_args()

    algo_cfg = load_yaml(args.algo_config)
    ds_cfg = load_yaml(args.datasets_config)

    dataset_name = algo_cfg["dataset"]
    dataset_info = ds_cfg["datasets"][dataset_name]

    output_dir = Path(algo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 调用后端训练 + 评估 ===
    start_time = time.time()
    result = classification_backend.train_and_eval(
        data_path=dataset_info["root"],
        model_name=algo_cfg["model"],
        epochs=algo_cfg["epochs"],
        batch_size=algo_cfg["batch_size"],
        lr=algo_cfg["lr"],
        output_dir=str(output_dir),
    )
    end_time = time.time()

    # 写入统一 base_metrics.csv
    metrics_row = {
        "algo_name": algo_cfg["name"],
        "task": "classification",
        "dataset": dataset_name,

        "accuracy": result.get("top1", 0.0),
        "cross_domain": 0.0,   # 先留空位，后续做跨域测试时再填
        "efficiency": 1.0 / max(result.get("test_time_s_per_image", 1.0), 1e-6),
        "learnability": algo_cfg["epochs"],
        "autonomy": 0.0,
        "capability": result.get("params_m", 0.0),
        "stability": 0.0,
        "safety": 0.0,
        "train_time_s": end_time - start_time,
    }

    metrics_csv = Path(args.metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        df = pd.concat([df, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics_row])
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] {algo_cfg['name']} 运行完成，指标已写入 {metrics_csv}")


if __name__ == "__main__":
    main()
