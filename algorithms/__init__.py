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
    adapter.load(weights_path=cfg.get("weights"), device=device)
    return adapter
