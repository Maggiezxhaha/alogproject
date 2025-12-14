from typing import Any, Dict

from core.registry import algo_registry

# ensure registrations
from . import classification  # noqa: F401
from . import detection  # noqa: F401
from . import segmentation  # noqa: F401


def build_algo(cfg: Dict[str, Any], device: str = "cpu"):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Algorithm config must include a 'name'")
    adapter_cls = algo_registry.get(name)
    adapter = adapter_cls(device=device, **cfg)
    weights_path = cfg.get("weights")
    if weights_path:
        print(f"Using weights from local path: {weights_path}")
    elif cfg.get("pretrained"):
        print("Using torchvision pretrained weights (no local weights provided)")
    else:
        print("Initializing model without pretrained weights")
    adapter.load(weights_path=weights_path, device=device)
    return adapter
