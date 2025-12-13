from typing import Any, Dict, List
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


class FasterRCNNDetector(AlgoAdapter):
    task_type = "detection"

    def __init__(self, num_classes: int = 91, device: str = "cpu", pretrained: bool = True, **kwargs):
        super().__init__(device=device)
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
        if num_classes != 91:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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
    def predict(self, batch: Any) -> List[Dict[str, torch.Tensor]]:
        images = [img.to(self.device) for img in batch]
        outputs = self.model(images)
        return outputs

    def forward_loss(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        self.model.train()
        losses = self.model(images, targets)
        loss_sum = sum(v for v in losses.values())
        self.model.eval()
        return loss_sum


algo_registry.register("fasterrcnn_resnet50", FasterRCNNDetector)
