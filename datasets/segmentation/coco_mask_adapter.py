from typing import Any, Dict
import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools import mask as mask_utils

from core.interfaces import DatasetAdapter
from core.registry import dataset_registry


class COCOSegmentationAdapter(DatasetAdapter):
    def __init__(self, root: str, ann_file: str, transforms_cfg: Dict[str, Any] = None, subset: int = None, **kwargs):
        t_list = [transforms.ToTensor()]
        if transforms_cfg and transforms_cfg.get("resize"):
            size = transforms_cfg["resize"]
            t_list.insert(0, transforms.Resize(size))
        self.transform = transforms.Compose(t_list)
        self.dataset = CocoDetection(root=root, annFile=ann_file, transform=self.transform)
        if subset is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, list(range(min(len(self.dataset), subset))))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, target = self.dataset[idx]
        h, w = image.shape[-2:]
        masks = []
        labels = []
        for ann in target:
            seg = ann.get("segmentation", None)
            if seg:
                rles = mask_utils.frPyObjects(seg, h, w)
                mask = mask_utils.decode(rles)
                mask_tensor = torch.as_tensor(mask, dtype=torch.uint8)
                if mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.any(dim=2)
                masks.append(mask_tensor)
                labels.append(ann.get("category_id", 0))
        target_dict = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, h, w), dtype=torch.uint8),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return {"image": image, "target": target_dict, "meta": {"index": idx}}

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    images = [b["image"] for b in batch]
    targets = [b["target"] for b in batch]
    metas = [b.get("meta", {}) for b in batch]
    return {"image": images, "target": targets, "meta": metas}


dataset_registry.register("coco_mask", COCOSegmentationAdapter)
