from typing import Any
import os
import torch
import torch.nn.functional as F
from torchvision import models

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


def _miou(preds: torch.Tensor, target: torch.Tensor, num_classes: int):
    preds = preds.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / max(1, len(ious))


class _BaseSegmentation(AlgoAdapter):
    task_type = "segmentation"

    def __init__(self, num_classes: int = 21, device: str = "cpu", pretrained: bool = True, **kwargs):
        super().__init__(device=device)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        raise NotImplementedError

    def load(self, weights_path: str = None, **kwargs):
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        return self

    def training_step(self, batch: Any, device: str = "cpu"):
        self.model.train()
        images = batch["image"]
        targets = batch["target"]
        if isinstance(images, list):
            images = torch.stack([img.to(device) for img in images])
        else:
            images = images.to(device)
        if isinstance(targets, list):
            targets = torch.stack([t.to(device) for t in targets]).long()
        else:
            targets = targets.long().to(device)
        outputs = self.model(images)["out"]
        loss = F.cross_entropy(outputs, targets)
        return {"loss": loss}

    def validation_step(self, batch: Any, device: str = "cpu"):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"]
            targets = batch["target"]
            if isinstance(images, list):
                images = torch.stack([img.to(device) for img in images])
            else:
                images = images.to(device)
            if isinstance(targets, list):
                targets = torch.stack([t.to(device) for t in targets]).long()
            else:
                targets = targets.long().to(device)
            outputs = self.model(images)["out"]
            loss = F.cross_entropy(outputs, targets)
            miou = _miou(outputs, targets, self.num_classes)
        return {"val_loss": loss.item(), "val_miou": miou}

    @torch.no_grad()
    def predict(self, batch: Any):
        if isinstance(batch, torch.Tensor):
            images = batch.to(self.device)
        else:
            images_in = batch["image"]
            if isinstance(images_in, list):
                images = torch.stack([img.to(self.device) for img in images_in])
            else:
                images = images_in.to(self.device)
        outputs = self.model(images)["out"]
        return outputs


class DeepLabV3ResNet50Segmentation(_BaseSegmentation):
    def _build_model(self):
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if self.pretrained else None
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights, num_classes=self.num_classes)


class DeepLabV3ResNet101Segmentation(_BaseSegmentation):
    def _build_model(self):
        weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if self.pretrained else None
        self.model = models.segmentation.deeplabv3_resnet101(weights=weights, num_classes=self.num_classes)


class FCNResNet50Segmentation(_BaseSegmentation):
    def _build_model(self):
        weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT if self.pretrained else None
        self.model = models.segmentation.fcn_resnet50(weights=weights, num_classes=self.num_classes)


class FCNResNet101Segmentation(_BaseSegmentation):
    def _build_model(self):
        weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT if self.pretrained else None
        self.model = models.segmentation.fcn_resnet101(weights=weights, num_classes=self.num_classes)


algo_registry.register("deeplabv3_resnet50", DeepLabV3ResNet50Segmentation)
algo_registry.register("deeplabv3_resnet101", DeepLabV3ResNet101Segmentation)
algo_registry.register("fcn_resnet50", FCNResNet50Segmentation)
algo_registry.register("fcn_resnet101", FCNResNet101Segmentation)
