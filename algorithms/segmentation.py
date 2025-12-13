from typing import Any
import os
import torch
from torchvision import models

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


class DeepLabV3Segmentation(AlgoAdapter):
    task_type = "segmentation"

    def __init__(self, num_classes: int = 21, device: str = "cpu", pretrained: bool = True, **kwargs):
        super().__init__(device=device)
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights, num_classes=num_classes)
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
        outputs = self.model(images)["out"]
        return outputs


algo_registry.register("deeplabv3_resnet50", DeepLabV3Segmentation)
