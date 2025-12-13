from typing import Any, Dict, Tuple
import torch


class AlgoAdapter:
    task_type: str = "base"

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if device != str(self.device):
            print(f"[AlgoAdapter] Falling back to {self.device} because requested device {device} is unavailable.")

    def load(self, weights_path: str = None, **kwargs):
        raise NotImplementedError

    def predict(self, batch: Any):
        raise NotImplementedError


class DatasetAdapter(torch.utils.data.Dataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Metric:
    name: str = "metric"

    def update(self, pred: Any, target: Any, meta: Dict[str, Any]):
        raise NotImplementedError

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
