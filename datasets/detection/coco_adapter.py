from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

from core.interfaces import DatasetAdapter
from core.registry import dataset_registry


class COCODetectionAdapter(DatasetAdapter):
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
        boxes = []
        labels = []
        for ann in target:
            bbox = ann.get("bbox", None)
            if bbox:
                # coco bbox format [x,y,w,h] -> [x1,y1,x2,y2]
                x1, y1, w, h = bbox
                boxes.append([x1, y1, x1 + w, y1 + h])
                labels.append(ann.get("category_id", 0))
        target_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return {"image": image, "target": target_dict, "meta": {"index": idx}}

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch: List[Dict[str, Any]]):
    images = [b["image"] for b in batch]
    targets = [b["target"] for b in batch]
    metas = [b.get("meta", {}) for b in batch]
    return {"image": images, "target": targets, "meta": metas}


dataset_registry.register("coco_detection", COCODetectionAdapter)
