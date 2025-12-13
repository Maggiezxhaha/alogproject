from typing import Any, Callable, Dict


class Registry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, cls: Callable):
        self._registry[name] = cls

    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"{name} not found in registry. Available: {list(self._registry.keys())}")
        return self._registry[name]


algo_registry = Registry()
dataset_registry = Registry()
metric_registry = Registry()
