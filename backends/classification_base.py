# backends/classification_base.py
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import torch


class ClassificationAlgoBase(ABC):
    """
    所有分类算法的统一接口。
    - 外部算法只要继承这个类，并实现 preprocess() 和 predict()，就能被你的平台调用。
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

    @abstractmethod
    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """
        输入：一批 PIL.Image 对象（来自 Dataset）
        输出：一个 PyTorch Tensor，形状 [B, C, H, W]，放在 CPU 或 GPU 上都可以。
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, images_tensor: torch.Tensor) -> torch.Tensor:
        """
        输入：preprocess 返回的 Tensor [B, C, H, W]
        输出：logits 或概率 [B, num_classes]
        """
        raise NotImplementedError

    def supports_grad(self) -> bool:
        """
        是否支持对输入做可微分的前向传播（用于对抗攻击）。
        默认 False，外部算法可以按需重写。
        """
        return False
