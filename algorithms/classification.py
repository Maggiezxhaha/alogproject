from typing import Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


class _BaseClassifier(AlgoAdapter):
    task_type = "classification"

    def __init__(self, num_classes: int = 10, device: str = "cpu", pretrained: bool = True, **kwargs):
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
        images = batch["image"].to(device)
        target = batch["target"].to(device)
        logits = self.model(images)
        loss = F.cross_entropy(logits, target)
        return {"loss": loss}

    def validation_step(self, batch: Any, device: str = "cpu"):
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(device)
            target = batch["target"].to(device)
            logits = self.model(images)
            loss = F.cross_entropy(logits, target)
            acc = (logits.argmax(dim=1) == target).float().mean().item()
        return {"val_loss": loss.item(), "val_accuracy": acc}

    @torch.no_grad()
    def predict(self, batch: Any):
        images = batch.to(self.device) if isinstance(batch, torch.Tensor) else batch["image"].to(self.device)
        logits = self.model(images)
        return logits


class ResNet18Classifier(_BaseClassifier):
    def _build_model(self):
        weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


class ResNet50Classifier(_BaseClassifier):
    def _build_model(self):
        weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


class MobileNetV3SmallClassifier(_BaseClassifier):
    def _build_model(self):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)


class EfficientNetB0Classifier(_BaseClassifier):
    def _build_model(self):
        weights = models.EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)


algo_registry.register("resnet18", ResNet18Classifier)
algo_registry.register("resnet50", ResNet50Classifier)
algo_registry.register("mobilenet_v3_small", MobileNetV3SmallClassifier)
algo_registry.register("efficientnet_b0", EfficientNetB0Classifier)
