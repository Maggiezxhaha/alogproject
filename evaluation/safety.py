from typing import Dict
import torch
import torch.nn.functional as F


def pgd_attack(algo, images, target, task_type: str, eps: float, alpha: float, steps: int, random_start: bool, norm: str):
    device = images.device
    adv = images.clone().detach()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1)
    adv.requires_grad = True

    for _ in range(steps):
        adv.requires_grad = True
        if task_type == "classification":
            logits = algo.model(adv) if hasattr(algo, "model") else algo.predict(adv)
            loss = F.cross_entropy(logits, target)
        elif task_type == "segmentation":
            outputs = algo.model(adv)["out"] if hasattr(algo, "model") else algo.predict(adv)
            loss = F.cross_entropy(outputs, target.long())
        elif task_type == "detection":
            if hasattr(algo, "forward_loss"):
                loss = algo.forward_loss([adv], target)
            else:
                outputs = algo.model(adv) if hasattr(algo, "model") else algo.predict([adv])
                scores = torch.stack([o.get("scores", torch.tensor(0, device=device)).sum() for o in outputs])
                loss = -scores.mean()
        else:
            raise ValueError(f"Unsupported task_type {task_type}")

        loss.backward()
        with torch.no_grad():
            grad_sign = adv.grad.sign()
            adv = adv + alpha * grad_sign
            perturb = torch.clamp(adv - images, min=-eps, max=eps)
            adv = torch.clamp(images + perturb, 0, 1)
        adv = adv.detach()
    return adv


def evaluate_safety(algo, dataloader, device, task_type: str, config: Dict, max_batches=None, base_metric_fn=None):
    eps = config.get("eps", 8 / 255)
    alpha = config.get("alpha", 2 / 255)
    steps = config.get("steps", 10)
    random_start = config.get("random_start", True)
    norm = config.get("norm", "linf")

    total = 0
    matched = 0
    for bidx, batch in enumerate(dataloader):
        if task_type == "classification":
            images = batch["image"].to(device)
            target = batch["target"].to(device)
        elif task_type == "segmentation":
            images = batch["image"].to(device)
            target = batch["target"].to(device)
        else:
            images = torch.stack(batch["image"]).to(device) if isinstance(batch["image"], list) else batch["image"].to(device)
            target = batch["target"]

        adv_images = pgd_attack(algo, images, target, task_type, eps, alpha, steps, random_start, norm)
        if base_metric_fn is None:
            raise ValueError("base_metric_fn is required for safety evaluation")
        # reuse metric on adv images
        if task_type == "classification":
            logits = algo.predict(adv_images)
            preds = logits.argmax(dim=1)
            matched += (preds == target).sum().item()
            total += target.numel()
        elif task_type == "segmentation":
            logits = algo.predict(adv_images)
            preds = logits.argmax(dim=1)
            matched += (preds == target).sum().item()
            total += target.numel()
        elif task_type == "detection":
            outputs = algo.predict([img for img in adv_images])
            adv_metric = base_metric_fn(outputs, target)
            matched += adv_metric
            total += 1
        if max_batches and (bidx + 1) >= max_batches:
            break

    return matched / max(1, total)
