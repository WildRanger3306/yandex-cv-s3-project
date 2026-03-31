"""
Визуализация метрик эксперимента 9 из MLflow.
Выгружает loss, lr, grad_norm, identity_similarity и сохраняет в metrics.png
Сглаживание через EMA (как в MLflow UI) — начинается с шага 0.
"""

import requests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

MLFLOW_URL = "http://188.243.201.66:5000"
RUN_ID = "5d6a1b4fe0bb4a60b1563ddf4a9997ba"

METRICS = ["loss", "lr", "grad_norm", "identity_similarity"]


def fetch_metric(metric_key):
    """Выгружает историю метрики из MLflow API"""
    url = f"{MLFLOW_URL}/api/2.0/mlflow/metrics/get-history"
    resp = requests.get(url, params={"run_id": RUN_ID, "metric_key": metric_key})
    resp.raise_for_status()
    metrics = resp.json().get("metrics", [])
    # Сортируем по шагу на всякий случай
    metrics.sort(key=lambda m: m["step"])
    steps = [m["step"] for m in metrics]
    values = [m["value"] for m in metrics]
    return steps, values


def ema_smooth(values, weight=0.9):
    """
    Exponential Moving Average — как в MLflow UI.
    Начинается с первой точки, не съедает начало.
    weight: 0 = без сглаживания, 1 = константа (первое значение)
    """
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


print("Выгрузка метрик из MLflow...")
data = {}
for key in METRICS:
    print(f"  -> {key}...")
    steps, values = fetch_metric(key)
    data[key] = (steps, values)
    print(f"     Получено {len(steps)} точек")

# ================================
# Визуализация
# ================================
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

CONFIGS = [
    {
        "key": "loss",
        "title": "Training Loss",
        "color": "#ff6b6b",
        "ylabel": "Loss",
        "smooth": True,
        "ema_weight": 0.98,
    },
    {
        "key": "lr",
        "title": "Learning Rate (UNet)",
        "color": "#4ecdc4",
        "ylabel": "LR",
        "smooth": False,
    },
    {
        "key": "grad_norm",
        "title": "Gradient Norm",
        "color": "#ffe66d",
        "ylabel": "Grad Norm",
        "smooth": True,
        "ema_weight": 0.98,
    },
    {
        "key": "identity_similarity",
        "title": "Identity Similarity (CLIP vs reference)",
        "color": "#a29bfe",
        "ylabel": "Cosine Similarity",
        "smooth": False,
    },
]

for i, cfg in enumerate(CONFIGS):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#1a1d27")

    steps, values = data[cfg["key"]]

    if cfg.get("smooth") and len(values) > 1:
        # Сырые данные — полупрозрачно (как в MLflow)
        ax.plot(steps, values, color=cfg["color"], alpha=0.2, linewidth=0.8, zorder=1)
        # EMA поверх — яркая линия с шага 0
        sm = ema_smooth(values, weight=cfg.get("ema_weight", 0.9))
        ax.plot(steps, sm, color=cfg["color"], linewidth=2.0, zorder=2, label="EMA smooth")
        ax.legend(facecolor="#2d3142", edgecolor="none", labelcolor="white", fontsize=8)
    else:
        ax.plot(steps, values, color=cfg["color"], linewidth=2.0, zorder=2)

    # Для identity_similarity — аннотации пика
    if cfg["key"] == "identity_similarity" and values:
        peak_idx = int(np.argmax(values))
        ax.axvline(steps[peak_idx], color="#fd79a8", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"Peak: {values[peak_idx]:.4f}\n@ step {steps[peak_idx]}",
            xy=(steps[peak_idx], values[peak_idx]),
            xytext=(steps[peak_idx] + max(steps) * 0.07, values[peak_idx] - 0.008),
            color="#fd79a8",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#fd79a8", lw=1.2),
        )
        ax.scatter(steps, values, color=cfg["color"], s=60, zorder=3)
        for s, v in zip(steps, values):
            ax.annotate(
                f"{v:.4f}",
                xy=(s, v),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color="white",
                fontsize=8,
            )

    ax.set_title(cfg["title"], color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Step", color="#aaaaaa", fontsize=9)
    ax.set_ylabel(cfg["ylabel"], color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(color="#2a2d3e", linewidth=0.5, linestyle="--")

fig.suptitle(
    f"Experiment 9 — Training Metrics\nRun: {RUN_ID[:16]}...",
    color="white",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

out_path = "metrics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ График сохранён: {out_path}")
plt.close()
