# eval_8dims/normalize_scores.py
import pandas as pd

# 哪些维度是“越大越好”（效益型指标）
BENEFIT_COLS = [
    "accuracy",
    "cross_domain",
    "efficiency",
    "autonomy",
    "capability",
    "stability",
    "safety",
]

# 哪些维度是“越小越好”（成本型指标），这里只把 learnability 当成成本型：epoch 越少越好
COST_COLS = [
    "learnability",
]


def _min_max_normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    min-max 归一化到 [0, 1]。
    reverse=True 时表示“越小越好”，会先做反向再归一化。
    """
    s = series.astype(float)
    if reverse:
        s = s.max() - s  # 越小越好 -> 数值反向

    min_v = s.min()
    max_v = s.max()
    if max_v - min_v < 1e-12:
        # 所有算法都一样，直接给 1.0
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)


def normalize_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：base_metrics.csv 读出的原始 df
    输出：加入 *_norm 列的 df（不会删除原始列）
    """
    df = df.copy()

    # 1. 先确保所有需要的列存在（不存在就补 0）
    for col in BENEFIT_COLS + COST_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # 2. 对每个维度做归一化
    for col in BENEFIT_COLS:
        df[col + "_norm"] = _min_max_normalize(df[col], reverse=False)

    for col in COST_COLS:
        df[col + "_norm"] = _min_max_normalize(df[col], reverse=True)

    return df
