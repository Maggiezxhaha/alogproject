# backends/classification_backend.py
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


def train_and_eval(
    data_path: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: str,
) -> Dict[str, float]:
    """
    在 CIFAR-10 上训练一个分类模型，并返回指标。
    注意：强制使用 CUDA，如果没有 GPU 会直接报错。
    """
    # 1. 检查 CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA GPU，但工程要求必须使用 GPU。")

    device = torch.device("cuda")
    print("Using device:", device, torch.cuda.get_device_name(0))

    # 2. 数据集与 DataLoader（自动下载 CIFAR-10 到 data_path）
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. 模型
    if model_name.lower() == "resnet18":
        model = torchvision.models.resnet18(num_classes=10)
    else:
        raise ValueError(f"暂不支持的模型: {model_name}")

    model.to(device)

    # 统计参数量
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # 4. 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 5. 训练
    model.train()
    total_steps = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_steps += 1

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"Step {i}/{len(train_loader)}, "
                    f"loss = {running_loss / (i+1):.4f}"
                )

    # 6. 在测试集上评估 Top-1 准确率，并测推理时间
    model.eval()
    correct = 0
    total = 0
    total_infer_time = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            start = time.time()
            outputs = model(images)
            end = time.time()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_infer_time += (end - start)

    top1 = correct / total
    test_time_s_per_image = total_infer_time / max(total, 1)

    print(f"Test Top-1 Accuracy: {top1:.4f}")
    print(f"Inference time per image: {test_time_s_per_image * 1000:.3f} ms")

    return {
        "top1": float(top1),
        "top5": 0.0,  # CIFAR-10 这里先不算 Top-5
        "params_m": float(params_m),
        "test_time_s_per_image": float(test_time_s_per_image),
    }
