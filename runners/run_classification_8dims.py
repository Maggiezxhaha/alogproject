# runners/run_classification_8dims.py
"""
功能：
- 支持多种模型（ResNet18 / ResNet34 / MobileNetV2）在 CIFAR-10 上的 8 维评估。
- 对每个模型：
  1）用多种训练集占比（10%, 25%, 50%, 100%）训练，计算学习性（learnability）。
  2）用 100% 训练集的模型计算：
       - accuracy       : baseline test 准确率
       - efficiency     : 每秒推理图片数（images/sec）
       - safety         : 高风险子集上的 accuracy
       - capability     : hard 子集 acc / acc_base
       - stability      : (acc_noise + acc_blur)/2 / acc_base
       - cross_domain   : acc_noise / acc_base
       - learnability   : 归一化 AUC
       - autonomy       : 当前置 0.0
  3）将指标追加写入 results/merged/base_metrics.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# ========================
# 通用小工具函数
# ========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy_on_loader(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def measure_efficiency(model, device, loader, max_batches=50):
    """
    在给定 loader 上测平均推理时间（毫秒/张），只取前 max_batches 个 batch 估计。
    返回：
      ms_per_image, images_per_second
    """
    model.eval()
    total_images = 0
    total_time = 0.0

    batches = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.time()

            batch_time = (t1 - t0) * 1000.0  # ms
            total_time += batch_time
            total_images += images.size(0)
            batches += 1

            if batches >= max_batches:
                break

    if total_images == 0:
        return 0.0, 0.0

    ms_per_image = total_time / total_images
    images_per_second = 1000.0 / ms_per_image if ms_per_image > 0 else 0.0
    return ms_per_image, images_per_second


# ========================
# 模型工厂：根据名称创建不同网络结构
# ========================

def create_model(model_name: str, num_classes: int = 10):
    name = model_name.lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)

    elif name == "resnet34":
        model = torchvision.models.resnet34(num_classes=num_classes)

    elif name == "mobilenet_v2":
        m = torchvision.models.mobilenet_v2()
        # 替换最后一层分类头
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        model = m

    elif name == "shufflenet_v2":
        # 使用 shufflenet_v2_x1_0 作为默认结构
        m = torchvision.models.shufflenet_v2_x1_0()
        # 最后一层是 fc
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        model = m

    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0()
        # classifier 通常是 [Dropout, Linear]
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        model = m

    else:
        raise ValueError(
            f"不支持的模型名称: {model_name}. "
            f"请使用 resnet18 / resnet34 / mobilenet_v2 / shufflenet_v2 / efficientnet_b0"
        )
    return model



def get_algo_name(model_name: str):
    """
    根据模型名称生成 algo_name，写入 CSV 用。
    """
    name = model_name.lower()
    if name == "resnet18":
        return "ResNet18_CIFAR10_8dims"
    if name == "resnet34":
        return "ResNet34_CIFAR10_8dims"
    if name == "mobilenet_v2":
        return "MobileNetV2_CIFAR10_8dims"
    if name == "shufflenet_v2":
        return "ShuffleNetV2_CIFAR10_8dims"
    if name == "efficientnet_b0":
        return "EfficientNetB0_CIFAR10_8dims"
    return f"{model_name}_CIFAR10_8dims"



# ========================
# 基于 processed 的测试集 Dataset
# ========================

class ProcessedCIFAR10Test(torch.utils.data.Dataset):
    """
    从 algo_datasets/classification/cifar10/processed 中读取 test/test_noise/test_blur。
    使用 index.csv 中的 filename 和 label。
    """
    def __init__(self, processed_dir: Path, variant: str = "test", transform=None):
        """
        variant: "test" / "test_noise" / "test_blur"
        """
        self.processed_dir = processed_dir
        self.images_dir = processed_dir / variant
        self.transform = transform

        index_path = processed_dir / "index.csv"
        df = pd.read_csv(index_path)
        self.df = df[df["split"] == "test"].copy().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = int(row["label"])
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ========================
# 安全性 / 能力边界 / 稳定性
# ========================

def compute_safety_score_adv(
        model,
        device,
        cifar_test,
        acc_base: float,
        epsilon: float = 8.0 / 255.0,
):
    """
    基于对抗攻击的安全性指标（FGSM）：

    - 在整个 CIFAR-10 test 集上做 FGSM 攻击：
        x_adv = x + epsilon * sign(grad_x L(model(x), y))
    - 计算对抗样本上的准确率 acc_adv
    - 安全性得分 = acc_adv / acc_base

    说明：
    - 这里 epsilon 是对未归一化图片的典型尺度（8/255），
      但由于我们在归一化后的空间做攻击，所以这是一个近似；
      重点是统一标准，便于算法横向比较。
    """
    if acc_base <= 1e-8:
        print("[WARN] acc_base 太小，安全性指标置为 0.0")
        return 0.0

    test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    robust_correct = 0
    total = 0

    # 在归一化后的空间中使用一个近似的 epsilon
    # 这里直接使用 0.03 左右的量级（约等于 8/255）
    eps = float(epsilon)  # 默认 8/255 ≈ 0.031

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        # FGSM：沿梯度符号方向走一步
        grad_sign = images.grad.sign()
        adv_images = images + eps * grad_sign

        # 做一个粗略的 clamp，避免数值过大
        adv_images = torch.clamp(adv_images, -3.0, 3.0)

        outputs_adv = model(adv_images)
        preds_adv = outputs_adv.argmax(dim=1)

        robust_correct += (preds_adv == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        print("[WARN] 测试集中没有样本，安全性指标置为 0.0")
        return 0.0

    acc_adv = robust_correct / total
    safety_score = acc_adv / acc_base

    print(f"[RESULT] Adversarial accuracy (FGSM) = {acc_adv:.4f}")
    print(f"[RESULT] Safety score (acc_adv/acc_base) = {safety_score:.4f}")

    return float(safety_score)



def compute_capability_score(model, device, processed_dir: Path, splits_dir: Path, cifar_test, acc_base: float):
    """
    能力边界：从 capability_hard_ids.txt 中读高难样本的 global_id，
    通过 index.csv 找到对应 orig_index，在该子集上的 accuracy / acc_base 作为能力边界得分。
    """
    index_path = processed_dir / "index.csv"
    df = pd.read_csv(index_path)

    cap_ids_path = splits_dir / "capability_hard_ids.txt"
    if not cap_ids_path.exists():
        print(f"[WARN] 找不到 {cap_ids_path}，能力边界指标置为 0.0")
        return 0.0

    with cap_ids_path.open("r", encoding="utf-8") as f:
        global_ids = [int(line.strip()) for line in f if line.strip()]

    test_df = df[df["split"] == "test"]
    sub_df = test_df[test_df["global_id"].isin(global_ids)]
    orig_indices = sub_df["orig_index"].astype(int).tolist()

    if len(orig_indices) == 0 or acc_base <= 0:
        print("[WARN] capability 子集中没有样本或 acc_base<=0，能力边界指标置为 0.0")
        return 0.0

    subset = Subset(cifar_test, orig_indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=4)
    acc_hard = accuracy_on_loader(model, device, loader)
    capability_score = acc_hard / acc_base
    return capability_score


def compute_stability_and_cross_domain(model, device, processed_dir: Path, acc_base: float):
    """
    稳定性 + cross_domain：
      - 在 test_noise 和 test_blur 上计算 acc_noise / acc_blur
      - stability = ((acc_noise + acc_blur)/2) / acc_base
      - cross_domain = acc_noise / acc_base
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    noise_ds = ProcessedCIFAR10Test(processed_dir, variant="test_noise", transform=transform)
    blur_ds = ProcessedCIFAR10Test(processed_dir, variant="test_blur", transform=transform)

    noise_loader = DataLoader(noise_ds, batch_size=256, shuffle=False, num_workers=4)
    blur_loader = DataLoader(blur_ds, batch_size=256, shuffle=False, num_workers=4)

    acc_noise = accuracy_on_loader(model, device, noise_loader)
    acc_blur = accuracy_on_loader(model, device, blur_loader)

    if acc_base <= 1e-8:
        stability = 0.0
        cross_domain = 0.0
    else:
        stability = ((acc_noise + acc_blur) / 2.0) / acc_base
        cross_domain = acc_noise / acc_base

    return stability, cross_domain, acc_noise, acc_blur


# ========================
# 训练 + 学习性
# ========================

def train_model_cifar10(
        model_name: str,
        raw_root: Path,
        device,
        epochs: int = 2,
        batch_size: int = 128,
        lr: float = 0.1,
        subset_fraction: float = 1.0,
        seed: int = 42,
):
    """
    在 CIFAR-10 上训练一个指定模型（resnet18/resnet34/mobilenet_v2）。
    subset_fraction: 使用多少比例的训练数据（0<subset_fraction<=1.0）
    返回：
      - model: 训练完的模型
      - acc_base: 在原始 test 上的 accuracy
      - ms_per_image, images_per_second: 效率指标（仅 full-data 时有意义）
      - train_time_s: 训练总时长（秒）
      - cifar_test: test 数据集对象
      - subset_size: 实际使用的训练样本数
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    print(f"[INFO] 使用 CIFAR-10 原始数据，root={raw_root}, subset_fraction={subset_fraction}, model={model_name}")

    cifar_train = torchvision.datasets.CIFAR10(
        root=str(raw_root),
        train=True,
        download=True,
        transform=train_transform,
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root=str(raw_root),
        train=False,
        download=True,
        transform=test_transform,
    )

    n_total = len(cifar_train)
    if subset_fraction >= 1.0:
        subset_indices = list(range(n_total))
    else:
        n_sub = max(1, int(n_total * subset_fraction))
        rng = np.random.RandomState(seed)
        subset_indices = rng.choice(n_total, size=n_sub, replace=False).tolist()
    subset_size = len(subset_indices)

    train_subset = Subset(cifar_train, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model = create_model(model_name, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    print(f"[INFO] 训练 {model_name}：epochs={epochs}, batch_size={batch_size}, lr={lr}, subset_size={subset_size}")
    t_train_start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if i % 100 == 0:
                train_acc = correct / total if total > 0 else 0.0
                avg_loss = running_loss / total if total > 0 else 0.0
                print(f"[{model_name}][Epoch {epoch+1}/{epochs}] Step {i}, loss={avg_loss:.4f}, train_acc={train_acc:.4f}")

        scheduler.step()

    t_train_end = time.time()
    train_time_s = t_train_end - t_train_start
    print(f"[INFO] 训练完成，总耗时 {train_time_s:.1f} 秒 (subset_fraction={subset_fraction}, model={model_name})")

    acc_base = accuracy_on_loader(model, device, test_loader)
    print(f"[RESULT] Baseline Test Accuracy (subset_fraction={subset_fraction}, model={model_name}): {acc_base:.4f}")

    ms_per_image, images_per_second = measure_efficiency(model, device, test_loader, max_batches=50)

    return model, acc_base, ms_per_image, images_per_second, train_time_s, cifar_test, subset_size


def compute_learnability(fractions, accs, acc_full):
    """
    基于 (fraction, acc) 计算学习性：
      learnability = AUC / (acc_full * 1.0)
    其中 AUC 用梯形法在 [min_fraction, 1.0] 上近似。
    """
    if acc_full <= 1e-8 or len(fractions) < 2:
        return 0.0

    pairs = sorted(zip(fractions, accs), key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(pairs)):
        f0, a0 = pairs[i - 1]
        f1, a1 = pairs[i]
        dx = f1 - f0
        area += dx * (a0 + a1) / 2.0

    ideal_area = acc_full * 1.0
    learnability = area / ideal_area if ideal_area > 0 else 0.0
    return float(learnability)


# ========================
# 写 CSV
# ========================

def append_metrics_to_csv(csv_path: Path, metrics_row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = [
        "algo_name",
        "task",
        "dataset",
        "accuracy",
        "cross_domain",
        "efficiency",
        "learnability",
        "autonomy",
        "capability",
        "stability",
        "safety",
        "train_time_s",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)

    print(f"[OK] 指标已写入 {csv_path}")


# ========================
# 主入口
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=str,
        default="E:/QY/coding/algo_datasets",
        help="数据仓库根目录（含 classification/cifar10/...）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="每次训练的 epoch 数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="训练/测试 batch 大小",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="初始学习率",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="模型名称：resnet18 / resnet34 / mobilenet_v2",
    )
    args = parser.parse_args()

    model_name = args.model
    repo_root = Path(args.repo_root)
    cifar_root = repo_root / "classification" / "cifar10"
    raw_root = cifar_root / "raw"
    processed_dir = cifar_root / "processed"
    splits_dir = cifar_root / "splits"

    device = get_device()
    print(f"[INFO] 使用设备: {device}, 模型: {model_name}")

    # -------- 学习性：多个训练集占比 --------
    learn_fractions = [0.10, 0.25, 0.50, 1.00]
    learn_accs = []

    model_full = None
    acc_full = None
    ms_full = None
    imgps_full = None
    train_time_full = None
    cifar_test_full = None

    total_train_time_s = 0.0

    for frac in learn_fractions:
        print(f"\n[LEARN] 开始学习性训练：subset_fraction={frac}, model={model_name}")
        model, acc_base, ms_per_image, images_per_second, train_time_s, cifar_test, subset_size = \
            train_model_cifar10(
                model_name=model_name,
                raw_root=raw_root,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                subset_fraction=frac,
                seed=42,
            )
        learn_accs.append(acc_base)
        total_train_time_s += train_time_s

        if abs(frac - 1.0) < 1e-6:
            # 记录 full-data 版本，用于后续安全/稳定等指标
            model_full = model
            acc_full = acc_base
            ms_full = ms_per_image
            imgps_full = images_per_second
            train_time_full = train_time_s
            cifar_test_full = cifar_test

    if model_full is None or acc_full is None:
        print("[ERROR] 未能完成 full-data 训练，无法计算后续指标。")
        return

    learnability = compute_learnability(learn_fractions, learn_accs, acc_full)
    print(f"[RESULT] Learnability (normalized AUC) = {learnability:.4f}")

    # -------- 使用 full-data 模型计算其他维度 --------
    acc_base = acc_full
    images_per_second = imgps_full


    # 安全性（基于对抗攻击能力）
    safety_score = compute_safety_score_adv(
        model=model_full,
        device=device,
        cifar_test=cifar_test_full,
        acc_base=acc_base,
        epsilon=8.0 / 255.0,  # 可以日后在配置文件里调
    )
    print(f"[RESULT] Safety (adversarial robustness score): {safety_score:.4f}")

    # 能力边界
    capability_score = compute_capability_score(
        model=model_full,
        device=device,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        cifar_test=cifar_test_full,
        acc_base=acc_base,
    )
    print(f"[RESULT] Capability (hard subset acc/acc_base): {capability_score:.4f}")

    # 稳定性 + 跨域
    stability, cross_domain, acc_noise, acc_blur = compute_stability_and_cross_domain(
        model=model_full,
        device=device,
        processed_dir=processed_dir,
        acc_base=acc_base,
    )
    print(f"[RESULT] acc_noise={acc_noise:.4f}, acc_blur={acc_blur:.4f}")
    print(f"[RESULT] Stability: {stability:.4f}, Cross-domain: {cross_domain:.4f}")

    # 自主性：目前占位，后续你可以设计真正的指标
    autonomy = 0.0

    # 训练时间：记录四次训练的总耗时
    total_time_to_record = total_train_time_s

    results_dir = Path("results") / "merged"
    csv_path = results_dir / "base_metrics.csv"

    algo_name = get_algo_name(model_name)

    metrics_row = {
        "algo_name": algo_name,
        "task": "classification",
        "dataset": "cifar10",
        "accuracy": float(acc_base),
        "cross_domain": float(cross_domain),
        "efficiency": float(images_per_second),
        "learnability": float(learnability),
        "autonomy": float(autonomy),
        "capability": float(capability_score),
        "stability": float(stability),
        "safety": float(safety_score),
        "train_time_s": float(total_time_to_record),
    }

    append_metrics_to_csv(csv_path, metrics_row)

    print(f"[DONE] 模型 {model_name} 的八维相关指标（含学习性）计算完成。")


if __name__ == "__main__":
    main()
