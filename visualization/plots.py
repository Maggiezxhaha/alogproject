import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

DIM_ORDER = [
    "accuracy_score",
    "efficiency_score",
    "stability_score",
    "safety_score",
    "capability_boundary_score",
    "adaptability_score",
    "autonomy_score",
    "learnability_score",
]

LABEL_MAP = {
    "accuracy_score": "Accuracy",
    "efficiency_score": "Efficiency",
    "stability_score": "Stability",
    "safety_score": "Safety",
    "capability_boundary_score": "Capability",
    "adaptability_score": "Adaptability",
    "autonomy_score": "Autonomy",
    "learnability_score": "Learnability",
}


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def radar_chart(results: Dict[str, Dict[str, float]], save_path: str):
    ensure_dir(save_path)
    labels = [LABEL_MAP[d] for d in DIM_ORDER]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for name, scores in results.items():
        values = [scores.get(dim, 0.0) for dim in DIM_ORDER]
        values += values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def bar_chart(results: Dict[str, Dict[str, float]], save_path: str):
    ensure_dir(save_path)
    labels = [LABEL_MAP[d] for d in DIM_ORDER]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(results))
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, (name, scores) in enumerate(results.items()):
        values = [scores.get(dim, 0.0) for dim in DIM_ORDER]
        ax.bar(x + idx * width, values, width, label=name)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def scatter_latency_accuracy(points: List[Dict], save_path: str):
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(5, 4))
    for p in points:
        ax.scatter(p.get("latency", 0), p.get("accuracy", 0), label=p.get("name"))
        ax.annotate(p.get("name"), (p.get("latency", 0), p.get("accuracy", 0)))
    ax.set_xlabel("Latency (s/batch)")
    ax.set_ylabel("Accuracy score")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def safety_plot(points: List[Dict], save_path: str):
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(5, 4))
    clean = [p.get("clean", 0) for p in points]
    adv = [p.get("adv", 0) for p in points]
    names = [p.get("name") for p in points]
    x = np.arange(len(names))
    ax.bar(x - 0.15, clean, width=0.3, label="Clean")
    ax.bar(x + 0.15, adv, width=0.3, label="Adversarial")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Primary metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
