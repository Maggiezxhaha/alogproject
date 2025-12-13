from copy import deepcopy
from typing import Any, Dict
from torch.utils.data import DataLoader

from core.registry import dataset_registry

# ensure registrations
from .classification import cifar10_adapter  # noqa: F401
from .detection import coco_adapter  # noqa: F401
from .segmentation import coco_mask_adapter  # noqa: F401


def build_dataset(cfg: Dict[str, Any]):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Dataset config must include a 'name'")
    adapter_cls = dataset_registry.get(name)
    dataset = adapter_cls(**cfg)
    return dataset


def build_dataloader(cfg: Dict[str, Any], batch_size: int, num_workers: int = 0, shuffle: bool = False):
    dataset_cfg = deepcopy(cfg)
    dataset_cfg.pop("train", None)
    dataset_cfg.pop("val", None)
    for drop_key in [
        "train_split",
        "val_split",
        "train_root",
        "val_root",
        "train_ann_file",
        "val_ann_file",
    ]:
        dataset_cfg.pop(drop_key, None)
    dataset = build_dataset(dataset_cfg)
    collate = cfg.get("collate")
    if collate == "detection":
        from .detection.coco_adapter import collate_fn as coco_collate
        collate_fn = coco_collate
    elif collate == "segmentation":
        from .segmentation.coco_mask_adapter import collate_fn as mask_collate
        collate_fn = mask_collate
    else:
        collate_fn = None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
