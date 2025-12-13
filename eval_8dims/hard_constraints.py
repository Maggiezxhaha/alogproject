# eval_8dims/hard_constraints.py
import pandas as pd
from typing import Dict, List, Tuple


def apply_hard_constraints(df: pd.DataFrame, thresholds: Dict[str, float]) -> Tuple[List[bool], List[str]]:
    """
    对每个算法应用硬门槛：
      - thresholds 是从 yaml 读出来的，如 {"capability":0.60, "stability":0.75, "safety":0.70}
    返回：
      - pass_list: [True/False,...]
      - reasons:   ["", "能力边界 < 0.60; 安全性 < 0.70", ...]
    注意：这里用的是原始值，不是 *_norm。
    """
    pass_list = []
    reasons = []

    for _, row in df.iterrows():
        algo_ok = True
        reason_list = []

        # 能力边界
        if "capability" in thresholds:
            if row.get("capability", 0.0) < thresholds["capability"]:
                algo_ok = False
                reason_list.append(f"能力边界 < {thresholds['capability']:.2f}")

        # 稳定性
        if "stability" in thresholds:
            if row.get("stability", 0.0) < thresholds["stability"]:
                algo_ok = False
                reason_list.append(f"稳定性 < {thresholds['stability']:.2f}")

        # 安全性
        if "safety" in thresholds:
            if row.get("safety", 0.0) < thresholds["safety"]:
                algo_ok = False
                reason_list.append(f"安全性 < {thresholds['safety']:.2f}")

        pass_list.append(algo_ok)
        reasons.append("; ".join(reason_list))

    return pass_list, reasons
