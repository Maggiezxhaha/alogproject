from typing import Dict, Any
import numpy as np

from evaluation import task_metrics
from evaluation.safety import evaluate_safety

DIMENSIONS = [
    "accuracy",
    "efficiency",
    "stability",
    "safety",
    "capability_boundary",
    "adaptability",
    "autonomy",
    "learnability",
]


class EightDimensionSuite:
    def __init__(self, eval_cfg: Dict[str, Any], task_type: str):
        self.eval_cfg = eval_cfg
        self.task_type = task_type

    def _score_from_latency(self, latency: float, target: float = 0.05):
        return max(0.0, min(1.0, target / (latency + target)))

    def _score_from_variation(self, values):
        if not values:
            return 0.0
        return float(1.0 / (1.0 + np.std(values)))

    def evaluate_dimension(self, dim: str, algo, dataloaders: Dict[str, Any], device: str):
        cfg = self.eval_cfg.get("protocol", {}).get(dim, {})
        max_batches = cfg.get("max_batches")
        if self.task_type == "classification":
            if dim == "accuracy":
                res = task_metrics.classification_accuracy(algo, dataloaders[dim], device, max_batches=max_batches)
                score = res["acc"]
                return score, {"clean_accuracy": res["acc"]}
            if dim == "efficiency":
                res = task_metrics.classification_accuracy(algo, dataloaders.get(dim), device, max_batches=max_batches)
                score = self._score_from_latency(res["latency"])
                return score, {"latency": res["latency"]}
            if dim == "stability":
                res = task_metrics.classification_accuracy(algo, dataloaders.get(dim), device, max_batches=max_batches, noise_std=cfg.get("noise_std", 0.05))
                score = self._score_from_variation(res["confidences"])
                return score, {"confidence_std": float(np.std(res["confidences"]) if res["confidences"] else 0.0)}
            if dim == "capability_boundary":
                top5 = task_metrics.classification_topk(algo, dataloaders.get(dim), device, k=5, max_batches=max_batches)
                return float(top5), {"top5": top5}
            if dim == "adaptability":
                res = task_metrics.classification_accuracy(algo, dataloaders.get(dim), device, max_batches=max_batches)
                return res["acc"], {"aug_acc": res["acc"]}
            if dim == "autonomy":
                res = task_metrics.classification_accuracy(algo, dataloaders.get(dim), device, max_batches=max_batches)
                conf = res["confidences"]
                confident_ratio = float(np.mean(np.array(conf) > cfg.get("threshold", 0.9))) if conf else 0.0
                return confident_ratio, {"confident_ratio": confident_ratio}
            if dim == "learnability":
                res = task_metrics.classification_accuracy(algo, dataloaders.get(dim), device, max_batches=max_batches)
                return res["acc"], {"fewshot_acc": res["acc"]}
            if dim == "safety":
                base_cfg = self.eval_cfg.get("safety", {})
                clean_res = task_metrics.classification_accuracy(algo, dataloaders.get("accuracy"), device, max_batches=max_batches)
                adv_metric = evaluate_safety(algo, dataloaders.get(dim), device, self.task_type, base_cfg, max_batches=max_batches, base_metric_fn=None)
                drop = max(0.0, clean_res["acc"] - adv_metric)
                retention = adv_metric / clean_res["acc"] if clean_res["acc"] > 0 else 0.0
                return max(0.0, min(1.0, adv_metric)), {
                    "safety_clean_metric": clean_res["acc"],
                    "safety_adv_metric": adv_metric,
                    "safety_drop": drop,
                    "safety_retention": retention,
                }
        elif self.task_type == "detection":
            from evaluation.task_metrics import detection_eval
            if dim == "accuracy":
                res = detection_eval(algo, dataloaders[dim], device, max_batches=max_batches)
                return res["map"], {"map": res["map"]}
            if dim == "efficiency":
                res = detection_eval(algo, dataloaders.get(dim), device, max_batches=max_batches)
                return self._score_from_latency(res["latency"]), {"latency": res["latency"]}
            if dim == "safety":
                base_cfg = self.eval_cfg.get("safety", {})
                res = detection_eval(algo, dataloaders.get("accuracy"), device, max_batches=max_batches)
                adv_metric = evaluate_safety(algo, dataloaders.get(dim), device, self.task_type, base_cfg, max_batches=max_batches, base_metric_fn=lambda outputs, targets: task_metrics.detection_map(outputs, targets))
                drop = max(0.0, res["map"] - adv_metric)
                retention = adv_metric / res["map"] if res["map"] > 0 else 0.0
                return max(0.0, min(1.0, adv_metric)), {
                    "safety_clean_metric": res["map"],
                    "safety_adv_metric": adv_metric,
                    "safety_drop": drop,
                    "safety_retention": retention,
                }
            # other dims reuse accuracy or latency
            res = detection_eval(algo, dataloaders.get(dim, dataloaders["accuracy"]), device, max_batches=max_batches)
            return res["map"], {"map": res["map"], "latency": res.get("latency", 0.0)}
        elif self.task_type == "segmentation":
            from evaluation.task_metrics import segmentation_eval
            if dim == "accuracy":
                res = segmentation_eval(algo, dataloaders[dim], device, max_batches=max_batches)
                return res["miou"], {"miou": res["miou"]}
            if dim == "efficiency":
                res = segmentation_eval(algo, dataloaders.get(dim), device, max_batches=max_batches)
                return self._score_from_latency(res["latency"]), {"latency": res["latency"]}
            if dim == "safety":
                base_cfg = self.eval_cfg.get("safety", {})
                res = segmentation_eval(algo, dataloaders.get("accuracy"), device, max_batches=max_batches)
                adv_metric = evaluate_safety(algo, dataloaders.get(dim), device, self.task_type, base_cfg, max_batches=max_batches, base_metric_fn=None)
                drop = max(0.0, res["miou"] - adv_metric)
                retention = adv_metric / res["miou"] if res["miou"] > 0 else 0.0
                return max(0.0, min(1.0, adv_metric)), {
                    "safety_clean_metric": res["miou"],
                    "safety_adv_metric": adv_metric,
                    "safety_drop": drop,
                    "safety_retention": retention,
                }
            res = segmentation_eval(algo, dataloaders.get(dim, dataloaders["accuracy"]), device, max_batches=max_batches)
            return res["miou"], {"miou": res["miou"], "latency": res.get("latency", 0.0)}

        return 0.0, {}

    def run(self, algo, dataloaders: Dict[str, Any], device: str):
        norm_scores: Dict[str, float] = {}
        raw_metrics: Dict[str, Any] = {}
        for dim in DIMENSIONS:
            score, details = self.evaluate_dimension(dim, algo, dataloaders, device)
            norm_scores[f"{dim}_score"] = float(score)
            raw_metrics.update({f"{dim}_{k}": v for k, v in details.items()})
        return norm_scores, raw_metrics
