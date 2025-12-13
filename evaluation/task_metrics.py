import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def classification_accuracy(algo, dataloader, device, max_batches=None, noise_std: float = 0.0):
    correct = 0
    total = 0
    latencies = []
    confidences: List[float] = []
    for bidx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        if noise_std > 0:
            images = torch.clamp(images + torch.randn_like(images) * noise_std, 0, 1)
        targets = batch["target"].to(device)
        start = time.time()
        logits = algo.predict(images)
        latency = time.time() - start
        latencies.append(latency / max(1, images.shape[0]))
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)
        conf, _ = probs.max(dim=1)
        confidences.extend(conf.detach().cpu().tolist())
        correct += (preds == targets).sum().item()
        total += targets.numel()
        if max_batches and (bidx + 1) >= max_batches:
            break
    acc = correct / max(1, total)
    return {
        "acc": acc,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
        "confidences": confidences,
    }


def classification_topk(algo, dataloader, device, k=5, max_batches=None):
    correct = 0
    total = 0
    for bidx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        logits = algo.predict(images)
        _, pred = logits.topk(k, dim=1)
        correct += (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.numel()
        if max_batches and (bidx + 1) >= max_batches:
            break
    return correct / max(1, total)


def detection_map(outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]], iou_thr: float = 0.5):
    tp = 0
    fp = 0
    fn = 0
    for pred, tgt in zip(outputs, targets):
        if pred.get("boxes") is None or len(pred["boxes"]) == 0:
            fn += len(tgt["boxes"])
            continue
        ious = box_iou(pred["boxes"], tgt["boxes"]) if len(tgt["boxes"]) > 0 else torch.zeros((len(pred["boxes"]), 0))
        matched = set()
        for i, scores in enumerate(ious):
            best = scores.argmax().item() if scores.numel() > 0 else -1
            if best >= 0 and scores[best] >= iou_thr and best not in matched:
                tp += 1
                matched.add(best)
            else:
                fp += 1
        fn += max(0, len(tgt["boxes"]) - len(matched))
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def detection_eval(algo, dataloader, device, max_batches=None):
    aps = []
    latencies = []
    for bidx, batch in enumerate(dataloader):
        images = [img.to(device) for img in batch["image"]]
        targets = batch["target"]
        start = time.time()
        outputs = algo.predict(images)
        latency = time.time() - start
        latencies.append(latency / max(1, len(images)))
        aps.append(detection_map(outputs, targets))
        if max_batches and (bidx + 1) >= max_batches:
            break
    return {
        "map": float(np.mean(aps)) if aps else 0.0,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
    }


def segmentation_eval(algo, dataloader, device, max_batches=None):
    ious = []
    latencies = []
    for bidx, batch in enumerate(dataloader):
        images = torch.stack(batch["image"]).to(device) if isinstance(batch["image"], list) else batch["image"].to(device)
        targets = batch["target"]
        target_masks = targets if torch.is_tensor(targets) else targets.get("masks")
        target_masks = target_masks.to(device)
        start = time.time()
        logits = algo.predict(images)
        latency = time.time() - start
        latencies.append(latency / max(1, images.shape[0]))
        preds = logits.argmax(dim=1)
        intersection = (preds & target_masks).float().sum(dim=(1, 2))
        union = (preds | target_masks).float().sum(dim=(1, 2))
        iou = (intersection / torch.clamp(union, min=1.0)).mean().item()
        ious.append(iou)
        if max_batches and (bidx + 1) >= max_batches:
            break
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
    }
