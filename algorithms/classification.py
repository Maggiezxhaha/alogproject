from typing import Any
import os
import torch
import torch.nn as nn
from torchvision import models

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


class ResNet18Classifier(AlgoAdapter):
    task_type = "classification"

    def __init__(self, num_classes: int = 10, device: str = "cpu", pretrained: bool = True, **kwargs):
        super().__init__(device=device)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)
        self.model.eval()

    def load(self, weights_path: str = None, **kwargs):
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        return self

    @torch.no_grad()
    def predict(self, batch: Any):
        images = batch.to(self.device)
        logits = self.model(images)
        return logits


algo_registry.register("resnet18", ResNet18Classifier)
