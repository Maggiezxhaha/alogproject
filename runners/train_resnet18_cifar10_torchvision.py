# runners/train_resnet18_cifar10_torchvision.py
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_resnet18(num_classes: int = 10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_cifar10_datasets(raw_root: str):
    """
    raw_root 例如: E:/QY/coding/algo_datasets/classification/cifar10/raw
    torchvision 会在这个目录下建 cifar-10-batches-py 之类的文件夹。
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_ds = datasets.CIFAR10(root=raw_root, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=raw_root, train=False, download=True, transform=test_transform)
    return train_ds, test_ds


def train_resnet18_cifar10(
    raw_root: str,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 使用设备:", device)

    print("[INFO] 加载 CIFAR-10 数据集...")
    train_ds, test_ds = build_cifar10_datasets(raw_root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    model = build_resnet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"[Train] Epoch {epoch+1}/{epochs}, loss={train_loss:.4f}, acc={train_acc:.4f}")

        # 每个 epoch 简单在 test 上评估一下
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct_test += preds.eq(labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        print(f"[Eval] test_acc={test_acc:.4f}")

        # 保存最好的一版
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "test_acc": best_acc,
                },
                save_path,
            )
            print(f"[INFO] 保存当前最佳模型到 {save_path} (acc={best_acc:.4f})")

    t1 = time.time()
    print(f"[DONE] 训练完成，总耗时 {t1 - t0:.1f} 秒, best_acc={best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=str,
        default="E:/QY/coding/algo_datasets/classification/cifar10/raw",
        help="CIFAR-10 原始数据目录 (raw)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/cifar10/resnet18_cifar10_baseline.pth",
        help="模型保存路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数",
    )
    args = parser.parse_args()

    train_resnet18_cifar10(
        raw_root=args.raw_root,
        save_path=args.save_path,
        epochs=args.epochs,
    )
