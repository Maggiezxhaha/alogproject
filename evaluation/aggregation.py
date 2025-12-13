from typing import Dict, List
import numpy as np


def load_weights(cfg: Dict, dims: List[str]):
    if "pairwise" in cfg:
        matrix = np.array(cfg["pairwise"], dtype=float)
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_idx = eigvals.real.argmax()
        weights = eigvecs[:, max_idx].real
        weights = weights / weights.sum()
    elif "weights" in cfg:
        if isinstance(cfg["weights"], dict):
            weights = np.array([cfg["weights"].get(d, 1.0) for d in dims], dtype=float)
        else:
            weights = np.array(cfg["weights"], dtype=float)
    else:
        weights = np.ones(len(dims))
    weights = weights / weights.sum()
    return weights


def weighted_sum(scores: Dict[str, float], dims: List[str], weights: List[float]):
    return float(sum(weights[i] * scores.get(dims[i], 0.0) for i in range(len(dims))))


def topsis(rank_matrix: List[Dict[str, float]], dims: List[str], weights: List[float]):
    if len(rank_matrix) == 1:
        return [{"name": rank_matrix[0].get("name", "algo"), "score": 1.0, "rank": 1}]
    matrix = np.array([[row.get(dim, 0.0) for dim in dims] for row in rank_matrix])
    weights_arr = np.array(weights)
    norm = np.linalg.norm(matrix, axis=0)
    norm_matrix = matrix / np.maximum(norm, 1e-8)
    weighted = norm_matrix * weights_arr
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    scores = []
    for idx, row in enumerate(weighted):
        d_pos = np.linalg.norm(row - ideal)
        d_neg = np.linalg.norm(row - nadir)
        closeness = d_neg / (d_pos + d_neg + 1e-8)
        scores.append({"name": rank_matrix[idx].get("name", f"algo_{idx}"), "score": float(closeness)})
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(scores, 1):
        s["rank"] = i
    return scores
