import os
import csv
import base64
import io
from typing import Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_FILE = os.path.join("logs", "predictions_log.csv")

COLUMNS = [
    "timestamp", "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course", "reading_score", "writing_score",
    "predicted_score"
]


def log_prediction(raw_features: dict, prediction: float) -> None:
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        row = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            **raw_features,
            "predicted_score": round(prediction, 2)
        }
        writer.writerow(row)


def get_logs() -> Optional[pd.DataFrame]:
    if not os.path.isfile(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return None
    return df


def get_stats(df: pd.DataFrame) -> dict:
    scores = df["predicted_score"]
    return {
        "total": len(df),
        "mean": round(scores.mean(), 1),
        "min": round(scores.min(), 1),
        "max": round(scores.max(), 1),
        "std": round(scores.std(), 1) if len(df) > 1 else 0.0,
    }


def build_trend_chart(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(range(len(df)), df["predicted_score"].values,
            color="#3b82f6", linewidth=2, marker="o", markersize=4)
    ax.fill_between(range(len(df)), df["predicted_score"].values,
                    alpha=0.15, color="#3b82f6")

    ax.set_xlabel("Prediction #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Math Score", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(axis="y", color="#334155", linewidth=0.5, linestyle="--")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
