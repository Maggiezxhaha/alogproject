# runners/run_classification_8dims_external.py
import os
import time
import argparse
import importlib
from typing import List, Tuple

import yaml
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# ========= 工具：根据 "backends.xxx.ClassName" 字符串加载类 =========
def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


# ========= 数据集封装：用于 processed/test_noise, test_blur =========
class ProcessedCIFARSubset(Dataset):
    """
    直接读 processed 目录下的 PNG：
      - 每个文件名形如 00000_label3.png
      - 从文件名里解析标签
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.files = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(".png"):
                self.files.append(fname)
        self.files.sort()

        self.labels = []
        for fname in self.files:
            # 假设文件名格式: XXXXX_labelY.png
            # e.g. 00001_label3.png
            try:
                base = os.path.splitext(fname)[0]
                # 拆成 ["00001", "label3"]
                parts = base.split("_label")
                label = int(parts[1])
            except Exception:
                raise ValueError(f"无法从文件名解析标签: {fname}")
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            # 注意：这里 transform 一般不需要，因为预处理在 adapter 里做
            img = self.transform(img)
        return img, label


# ========= 从 capability_hard_ids.txt 构建 hard 子集 =========
def make_hard_subset_from_ids(
        cifar_test_raw: datasets.CIFAR10,
        splits_dir: str,
        hard_ids_file: str = "capability_hard_ids.txt"
) -> Dataset:
    """
    capability_hard_ids.txt 里存的是 "难样本" 的索引。

    注意：
    - 你现在的 test 文件命名是 50000~59999（延续全局编号）
    - 而 torchvision 的 CIFAR10(test) 只有 0~9999

    所以这里做一个映射规则：
      如果 idx 在 [50000, 60000) 之间，先做 idx = idx - 50000
      其它 idx 保持不变，然后再检查 0 <= idx < len(test)
    """
    path = os.path.join(splits_dir, hard_ids_file)
    if not os.path.exists(path):
        print(f"[WARN] 找不到 {path}，能力边界子集为空。")
        return None

    raw_indices = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_indices.append(int(line))
            except ValueError:
                continue

    # 先做 50000~59999 -> 0~9999 的映射
    mapped_indices = []
    for idx in raw_indices:
        if 50000 <= idx < 60000:
            idx = idx - 50000
        mapped_indices.append(idx)

    n_test = len(cifar_test_raw)  # CIFAR-10 test = 10000
    valid_indices = [i for i in mapped_indices if 0 <= i < n_test]

    dropped = len(mapped_indices) - len(valid_indices)
    if dropped > 0:
        print(
            f"[WARN] hard 子集中有 {dropped} 个索引越界(不在 [0, {n_test})), 已自动丢弃，"
            f"有效样本数 {len(valid_indices)}。"
        )
    else:
        print(f"[INFO] hard 子集索引全部有效，总数 {len(valid_indices)}。")

    if len(valid_indices) == 0:
        print("[WARN] hard 子集过滤后为空，将能力边界指标置为 0.0。")
        return None

    from torch.utils.data import Subset
    return Subset(cifar_test_raw, valid_indices)


# ========= 专用 collate：保持图片是 PIL，标签变成 Tensor =========
def pil_collate(batch):
    """
    batch: List[(img, label)]
      - img: PIL.Image.Image
      - label: int
    返回:
      - imgs: List[PIL.Image.Image]
      - labels: torch.LongTensor  [B]
    """
    imgs, labels = zip(*batch)  # imgs 是 tuple[PIL]，labels 是 tuple[int]
    import torch
    labels = torch.tensor(labels, dtype=torch.long)
    return list(imgs), labels

# ========= 通用准确率计算（通过 adapter） =========
def accuracy_with_adapter(
        adapter,
        dataset: Dataset,
        device,
        batch_size: int = 256
) -> float:
    """
    使用 adapter 计算某个数据集上的分类准确率。
    Dataset 返回 (PIL.Image, label)，DataLoader 用 pil_collate。
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # 为避免 Windows 下多进程问题，先用 0
        collate_fn=pil_collate  # 保持图片是 PIL 列表
    )

    correct = 0
    total = 0
    for pil_list, labels in loader:
        # pil_list: List[PIL.Image.Image]
        # labels:  torch.LongTensor [B]
        batch = adapter.preprocess(pil_list)   # [B,C,H,W] on device
        logits = adapter.predict(batch)        # [B,num_classes]

        preds = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total


# ========= 效率：images/sec =========
def measure_efficiency_with_adapter(
        adapter,
        dataset: Dataset,
        device,
        batch_size: int = 256,
        max_batches: int = 50
) -> Tuple[float, float]:
    """
    使用 adapter 估计推理时间（ms/image）和吞吐量（images/sec）。
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pil_collate
    )

    total_images = 0
    total_time = 0.0
    n_batches = 0

    for pil_list, _ in loader:
        batch = adapter.preprocess(pil_list)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        _ = adapter.predict(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()

        dt = t1 - t0
        total_time += dt
        total_images += batch.size(0)

        n_batches += 1
        if n_batches >= max_batches:
            break

    if total_images == 0 or total_time == 0:
        return 0.0, 0.0

    ms_per_img = total_time * 1000.0 / total_images
    images_per_sec = total_images / total_time
    return ms_per_img, images_per_sec



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo-config",
        type=str,
        required=True,
        help="算法配置文件，例如 configs/algorithms/cls_resnet18_cifar10.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="E:/QY/coding/algo_datasets/classification/cifar10",
        help="CIFAR-10 数据根目录（下面包含 raw/ processed/ splits/）",
    )
    args = parser.parse_args()

    # ---- 读算法配置 ----
    with open(args.algo_config, "r", encoding="utf-8") as f:
        algo_cfg = yaml.safe_load(f)

    algo_name = algo_cfg["algo_name"]
    backend_path = algo_cfg["backend"]
    model_cfg = algo_cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 使用设备:", device)

    # ---- 实例化 adapter ----
    BackendCls = import_class(backend_path)
    adapter = BackendCls(config={"model": model_cfg}, device=device)
    print(f"[INFO] 已加载算法: {algo_name} ({backend_path})")

    # ---- 构建数据集 ----
    raw_root = os.path.join(args.data_root, "raw")
    processed_root = os.path.join(args.data_root, "processed")
    splits_root = os.path.join(args.data_root, "splits")

    # CIFAR-10 原始 test（transform=None => PIL）
    cifar_test_raw = datasets.CIFAR10(
        root=raw_root, train=False, download=False, transform=None
    )

    # processed test_noise / test_blur
    noise_dir = os.path.join(processed_root, "test_noise")
    blur_dir = os.path.join(processed_root, "test_blur")

    if not os.path.exists(noise_dir):
        print(f"[WARN] {noise_dir} 不存在，稳定性/跨域中的噪声集无法计算，将视为 0.")
    if not os.path.exists(blur_dir):
        print(f"[WARN] {blur_dir} 不存在，稳定性中的模糊集无法计算，将视为 0.")

    noise_ds = ProcessedCIFARSubset(noise_dir) if os.path.exists(noise_dir) else None
    blur_ds = ProcessedCIFARSubset(blur_dir) if os.path.exists(blur_dir) else None

    # hard 子集
    hard_ds = make_hard_subset_from_ids(cifar_test_raw, splits_root)

    # ---- 1) 准确率（Accuracy） ----
    acc_base = accuracy_with_adapter(adapter, cifar_test_raw, device)
    print(f"[RESULT] Base Accuracy: {acc_base:.4f}")

    # ---- 2) 效率（Efficiency） ----
    ms_per_img, images_per_sec = measure_efficiency_with_adapter(
        adapter, cifar_test_raw, device
    )
    print(f"[RESULT] Inference: {ms_per_img:.4f} ms/image, {images_per_sec:.2f} images/sec")

    # ---- 3) 稳定性 & 跨域（噪声/模糊） ----
    if noise_ds is not None:
        acc_noise = accuracy_with_adapter(adapter, noise_ds, device)
    else:
        acc_noise = 0.0

    if blur_ds is not None:
        acc_blur = accuracy_with_adapter(adapter, blur_ds, device)
    else:
        acc_blur = 0.0

    print(f"[RESULT] acc_noise={acc_noise:.4f}, acc_blur={acc_blur:.4f}")

    if acc_base > 0:
        stability = ((acc_noise + acc_blur) / 2.0) / acc_base
        cross_domain = acc_noise / acc_base
    else:
        stability = 0.0
        cross_domain = 0.0

    print(f"[RESULT] Stability={stability:.4f}, Cross-domain={cross_domain:.4f}")

    # ---- 4) 能力边界（Hard subset） ----
    if hard_ds is not None:
        acc_hard = accuracy_with_adapter(adapter, hard_ds, device)
        capability = acc_hard / acc_base if acc_base > 0 else 0.0
        print(f"[RESULT] Capability (hard subset acc/acc_base)={capability:.4f}")
    else:
        acc_hard = 0.0
        capability = 0.0

    # ---- 5) 安全性 / 学习性 / 自主性：先占位为 0.0（之后可以再升级） ----
    safety = 0.0
    learnability = 0.0
    autonomy = 0.0

    # ---- 写入 results/merged/base_metrics.csv ----
    metrics_row = {
        "algo_name": algo_name,
        "task": "classification",
        "dataset": "cifar10",
        "accuracy": acc_base,
        "cross_domain": cross_domain,
        "efficiency": images_per_sec,
        "learnability": learnability,
        "autonomy": autonomy,
        "capability": capability,
        "stability": stability,
        "safety": safety,
        "train_time_s": 0.0,  # 外部算法我们评测时不训练
    }

    os.makedirs("results/merged", exist_ok=True)
    csv_path = "results/merged/base_metrics.csv"

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([metrics_row])

    df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 指标已写入 {csv_path}")


if __name__ == "__main__":
    main()
