from typing import Any, Dict, List
import os
import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    fcos_resnet50_fpn,
    ssd300_vgg16,
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

from core.interfaces import AlgoAdapter
from core.registry import algo_registry


class _BaseDetector(AlgoAdapter):
    task_type = "detection"

    def __init__(self, num_classes: int = 91, device: str = "cpu", pretrained: bool = True, **kwargs):
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
        images = [img.to(device) for img in batch["image"]]
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tgt.items()}
            for tgt in batch["target"]
        ]
        losses = self.model(images, targets)
        loss_sum = sum(v for v in losses.values())
        return {"loss": loss_sum, **{f"loss_{k}": v.item() for k, v in losses.items()}}

    def validation_step(self, batch: Any, device: str = "cpu"):
        self.model.train()  # detection losses expect train mode
        with torch.no_grad():
            images = [img.to(device) for img in batch["image"]]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tgt.items()}
                for tgt in batch["target"]
            ]
            losses = self.model(images, targets)
            loss_sum = sum(v for v in losses.values())
        return {"val_loss": loss_sum.item()}

    @torch.no_grad()
    def predict(self, batch: Any) -> List[Dict[str, torch.Tensor]]:
        images = [img.to(self.device) for img in batch]
        self.model.eval()
        outputs = self.model(images)
        return outputs

    def forward_loss(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        self.model.train()
        losses = self.model(images, targets)
        loss_sum = sum(v for v in losses.values())
        self.model.eval()
        return loss_sum


class FasterRCNNDetector(_BaseDetector):
    def _build_model(self):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        if self.num_classes != 91:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)


class RetinaNetDetector(_BaseDetector):
    def _build_model(self):
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = retinanet_resnet50_fpn(weights=weights)
        if self.num_classes != 91:
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head.classification_head = RetinaNetClassificationHead(
                self.model.backbone.out_channels, num_anchors, self.num_classes
            )


class FCOSDetector(_BaseDetector):
    def _build_model(self):
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = fcos_resnet50_fpn(weights=weights, num_classes=self.num_classes)


class SSD300VGG16Detector(_BaseDetector):
    def _build_model(self):
        weights = SSD300_VGG16_Weights.DEFAULT if self.pretrained else None
        self.model = ssd300_vgg16(weights=weights)
        if self.num_classes != 91:
            num_anchors = self.model.head.classification_head.num_anchors
            out_channels = self.model.head.classification_head.out_channels
            self.model.head.classification_head = SSDClassificationHead(out_channels, num_anchors, self.num_classes)


algo_registry.register("fasterrcnn_resnet50", FasterRCNNDetector)
algo_registry.register("retinanet_resnet50", RetinaNetDetector)
algo_registry.register("fcos_resnet50", FCOSDetector)
algo_registry.register("ssd300_vgg16", SSD300VGG16Detector)
