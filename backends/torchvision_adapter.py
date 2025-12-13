# backends/torchvision_adapter.py
from typing import List
from PIL import Image

import os

import torch
from torch import nn
from torchvision import transforms, models

from .classification_base import ClassificationAlgoBase


def build_torchvision_model(name: str, num_classes: int) -> nn.Module:
    """
    根据名字构建一个 torchvision 模型，并把最后一层改成 num_classes。
    不依赖你自己的 model.py。
    """
    name = name.lower()

    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

    elif name in ["shufflenet_v2_x1_0", "shufflenet"]:
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name in ["efficientnet_b0", "effnet_b0"]:
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

    else:
        raise ValueError(f"未知的 model_name: {name}")

    return m


class TorchVisionClassificationAlgo(ClassificationAlgoBase):
    """
    用 torchvision 自带的分类模型作为后端。
    - 不依赖你工程里的 model.py
    - 只需要在 algo-config 里写 model.name 就可以：resnet18 / resnet34 / mobilenet_v2 等
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        self.device = device
        model_cfg = config["model"]

        model_name = model_cfg.get("name", "resnet18")
        num_classes = model_cfg.get("num_classes", 10)
        ckpt_path = model_cfg.get("checkpoint", None)

        # 1) 构建模型
        self.model = build_torchvision_model(model_name, num_classes)

        # 2) 尝试加载 checkpoint（如果有）
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("state_dict", state)
            self.model.load_state_dict(state_dict)
            print(f"[INFO] 从 {ckpt_path} 加载权重")
        else:
            print(f"[WARN] 未找到 checkpoint: {ckpt_path}，使用随机初始化权重。")

        self.model.to(self.device)
        self.model.eval()

        # 3) CIFAR-10 标准预处理（和你之前训练的一致）
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])

    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors, dim=0)  # [B, C, H, W]
        return batch.to(self.device)

    def predict(self, images_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(images_tensor)
        return logits

    def supports_grad(self) -> bool:
        # torchvision 模型都是可导的，将来可以用于对抗攻击
        return True
