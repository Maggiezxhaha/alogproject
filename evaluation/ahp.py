import numpy as np
from typing import Dict


def normalize_weights_from_pairwise(matrix):
    matrix = np.array(matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals)
    principal = np.real(eigvecs[:, max_idx])
    weights = principal / principal.sum()
    return weights.real


def load_weights(config: Dict[str, any], dimensions: list):
    if "pairwise" in config:
        pairwise = config["pairwise"]
        weights = normalize_weights_from_pairwise(pairwise)
        return {dim: float(w) for dim, w in zip(dimensions, weights)}
    if "weights" in config:
        w = config["weights"]
        total = sum(w.get(dim, 1.0) for dim in dimensions)
        return {dim: w.get(dim, 1.0) / total for dim in dimensions}
    # default equal
    equal = 1.0 / len(dimensions)
    return {dim: equal for dim in dimensions}


def aggregate(scores: Dict[str, float], weights: Dict[str, float]):
    final = 0.0
    for dim, weight in weights.items():
        final += scores.get(dim, 0.0) * weight
    return final
