# eval_8dims/evaluate_8dims.py
import argparse
from pathlib import Path

import yaml
import pandas as pd

from .normalize_scores import normalize_all
from .ahp_topsis import compute_composite_scores
from .hard_constraints import apply_hard_constraints


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="results/merged/base_metrics.csv",
        help="输入的基础指标文件（前面分类/检测/分割写入的 CSV）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_8dims.yaml",
        help="8 维度配置文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/merged/scores_8dims.csv",
        help="输出的 8 维度打分结果 CSV",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dim_cfg = cfg.get("dimensions", {})
    hard_cfg = cfg.get("hard_thresholds", {})

    df = pd.read_csv(args.metrics_csv)

    # 1. 归一化（添加 *_norm 列）
    df_norm = normalize_all(df)

    # 2. AHP/TOPSIS（这里是加权总分的简化版）
    df_scores = compute_composite_scores(df_norm, dim_cfg)

    # 3. 硬门槛
    pass_list, reasons = apply_hard_constraints(df_scores, hard_cfg)
    df_scores["pass"] = pass_list
    df_scores["fail_reason"] = reasons

    # 4. 保存结果并打印排行榜
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== 8 维度综合评价结果 ===")
    print(df_scores[["algo_name", "task", "dataset", "final_score", "pass", "fail_reason"]]
          .sort_values(by="final_score", ascending=False))


if __name__ == "__main__":
    main()
