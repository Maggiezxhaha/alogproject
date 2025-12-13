import torch
from torchvision import datasets, transforms
from typing import Any, Dict
from core.interfaces import DatasetAdapter
from core.registry import dataset_registry


class CIFAR10Adapter(DatasetAdapter):
    def __init__(self, root: str, split: str = "test", download: bool = True, transform=None, subset: int = None, **kwargs):
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        train = split == "train"
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=self.transform, download=download)
        if subset is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, list(range(min(len(self.dataset), subset))))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, target = self.dataset[idx]
        return {"image": image, "target": target, "meta": {"index": idx}}

    def __len__(self):
        return len(self.dataset)


dataset_registry.register("cifar10", CIFAR10Adapter)


