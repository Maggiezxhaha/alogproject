from copy import deepcopy
from typing import Dict

from datasets import build_dataloader


def default_protocol():
    return {
        "accuracy": {},
        "efficiency": {"max_batches": 10},
        "stability": {"noise_std": 0.05, "max_batches": 5},
        "safety": {},
        "capability_boundary": {"max_batches": 10},
        "adaptability": {"dataset_overrides": {"transform": None}, "max_batches": 10},
        "autonomy": {"max_batches": 5},
        "learnability": {"subset": 256, "max_batches": 5},
    }


def build_protocol_loaders(data_cfg: Dict, eval_cfg: Dict, batch_size: int, num_workers: int):
    protocol_cfg = deepcopy(default_protocol())
    protocol_cfg.update(eval_cfg.get("protocol", {}))
    loaders = {}
    for dim, dim_cfg in protocol_cfg.items():
        dataset_cfg = deepcopy(data_cfg)
        overrides = dim_cfg.get("dataset_overrides", {})
        if overrides:
            dataset_cfg.update(overrides)
        if dim_cfg.get("subset"):
            dataset_cfg["subset"] = dim_cfg["subset"]
        collate_map = dataset_cfg.get("task", dataset_cfg.get("name"))
        if dataset_cfg.get("task") == "detection":
            dataset_cfg["collate"] = "detection"
        if dataset_cfg.get("task") == "segmentation":
            dataset_cfg["collate"] = "segmentation"
        loaders[dim] = build_dataloader(dataset_cfg, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loaders, protocol_cfg
