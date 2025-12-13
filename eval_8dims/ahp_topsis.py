# eval_8dims/ahp_topsis.py
from typing import Dict

import pandas as pd


DIM_COLS = [
    "accuracy_norm",
    "efficiency_norm",
    "safety_norm",
    "stability_norm",
    "capability_norm",
    "adaptability_norm",
    "autonomy_norm",
    "learnability_norm",
]


def compute_composite_scores(df_norm: pd.DataFrame, dim_cfg: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    df_norm: 已经包含 *_norm 列的 DataFrame
    dim_cfg: 来自 yaml 的 dimensions 配置，如：
        {
          "accuracy": {"weight": 0.25},
          "efficiency": {"weight": 0.15},
          ...
        }
    返回：在 df_norm 的基础上增加各维度权重列和最终得分 final_score
    """
    df = df_norm.copy()

    # 1. 提取权重
    weights = {}
    for dim_name, dim_info in dim_cfg.items():
        w = dim_info.get("weight", 0.0)
        weights[dim_name] = float(w)

    # 2. 归一化权重（防止总和不是 1）
    total_w = sum(weights.values())
    if total_w < 1e-12:
        # 权重全 0，就平均分配
        avg_w = 1.0 / len(weights)
        for k in weights:
            weights[k] = avg_w
    else:
        for k in weights:
            weights[k] /= total_w

    # 3. 计算每个算法的加权总分（你可以以后把这部分换成真正的 TOPSIS）
    final_scores = []
    for _, row in df.iterrows():
        score = 0.0
        # 按维度累加： dim_norm * dim_weight
        for dim_name, w in weights.items():
            col_norm = dim_name + "_norm"
            if col_norm in df.columns:
                score += float(row[col_norm]) * w
        final_scores.append(score)

    df["final_score"] = final_scores
    return df
