"""
Analysis: Random In vs Random Out

Computes two diversity metrics across noise scales and problem categories:
  1. Self-BLEU : measures text-level similarity across trials. Lower = more lexically diverse outputs.
  2. Semantic Distance : measures meaning-level distance across trials. Higher = more conceptually distinct outputs.

Also flags coherence collapse, outputs that are empty or clearly off-topic,
which indicates the noise scale was too high and broke the model.

Outputs:
  - results/latent_experiment_summary.csv   (per problem × noise scale metrics)
  - assets/figures/self_bleu.png
  - assets/figures/semantic_distance.png
  - assets/figures/coherence_collapse.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

# Config
RESULTS_CSV  = "results/latent_perturbation_results.csv"
SUMMARY_CSV  = "results/latent_experiment_summary.csv"
FIGURES_DIR  = "assets/figures"

NOISE_ORDER  = [0.0, 0.02, 0.05, 0.1, 0.2]
CATEGORY_MAP = {
    "logic_reasoning"  : "Logic",
    "code_optimization": "Code",
    "system_design"    : "Architecture",
    "math_derivation"  : "Math",
    "strategy_creative": "Strategy",
}

os.makedirs(FIGURES_DIR, exist_ok=True)


df = pd.read_csv(RESULTS_CSV)
df["output"] = df["output"].fillna("").astype(str).str.strip()

print(f"[Load] {len(df)} rows loaded from {RESULTS_CSV}\n")


# Embedding model
print("[Embed] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embed] Ready.\n")


# Metric helpers
def compute_self_bleu(texts: list[str]) -> float:
    valid = [t for t in texts if t]
    if len(valid) < 2:
        return 0.0

    smooth = SmoothingFunction().method1
    tokenized = [t.split() for t in valid]
    scores = []

    for i, candidate in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        if candidate and refs:
            scores.append(sentence_bleu(refs, candidate, smoothing_function=smooth))

    return float(np.mean(scores)) if scores else 0.0


def compute_semantic_distance(texts: list[str]) -> float:
    valid = [t for t in texts if t]
    if len(valid) < 2:
        return 0.0

    embeddings = embed_model.encode(valid, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    n = len(valid)
    distances = [
        1 - sim_matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(distances)) if distances else 0.0


def coherence_collapse_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0

    def is_collapsed(t: str) -> bool:
        if not t:
            return True
        if len(t) < 30:
            return True
        non_ascii = sum(1 for c in t if ord(c) > 127)
        if non_ascii / max(len(t), 1) > 0.3:   # >30% non-ASCII → language drift
            return True
        return False

    collapsed = sum(1 for t in texts if is_collapsed(t))
    return collapsed / len(texts)


print("[Analysis] Computing metrics per (problem, noise_scale) ...")

latent_df = df[df["method"].isin(["Random_In_Latent", "Latent_Perturbation"])].copy()

rows = []
for (problem_id, noise_scale), group in latent_df.groupby(
    ["problem_id", "noise_scale"], sort=False, observed=True
):
    outputs = group["output"].tolist()
    rows.append({
        "Problem"          : CATEGORY_MAP.get(problem_id, problem_id),
        "Problem_ID"       : problem_id,
        "Noise_Scale"      : noise_scale,
        "Self_BLEU"        : round(compute_self_bleu(outputs),            4),
        "Semantic_Distance": round(compute_semantic_distance(outputs),     4),
        "Collapse_Rate"    : round(coherence_collapse_rate(outputs),       4),
        "N_Trials"         : len(outputs),
    })

summary = pd.DataFrame(rows).sort_values(["Problem_ID", "Noise_Scale"])
summary.to_csv(SUMMARY_CSV, index=False)

print(f"\n{'='*62}")
print("EXPERIMENTAL ANALYSIS METRICS")
print(f"{'='*62}")
print(summary[["Problem", "Noise_Scale", "Self_BLEU",
               "Semantic_Distance", "Collapse_Rate"]].to_string(index=False))
print(f"{'='*62}\n")


COLORS = {
    "Logic"       : "#4C72B0",
    "Code"        : "#DD8452",
    "Architecture": "#55A868",
    "Math"        : "#C44E52",
    "Strategy"    : "#8172B2",
}

def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved → {path}")



fig, ax = plt.subplots(figsize=(8, 4.5))

for problem_id, label in CATEGORY_MAP.items():
    sub = summary[summary["Problem_ID"] == problem_id].sort_values("Noise_Scale")
    ax.plot(
        sub["Noise_Scale"], sub["Self_BLEU"],
        marker="o", label=label, color=COLORS[label], linewidth=2
    )

ax.set_title("Self-BLEU vs Noise Scale\n(lower = more lexical diversity)", fontsize=13)
ax.set_xlabel("Noise Scale (σ)", fontsize=11)
ax.set_ylabel("Self-BLEU", fontsize=11)
ax.set_xticks(NOISE_ORDER)
ax.set_ylim(-0.05, 1.10)
ax.axvline(x=0.02, color="grey", linestyle="--", linewidth=1, alpha=0.6,
           label="Sweet spot (0.02)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "self_bleu.png")


fig, ax = plt.subplots(figsize=(8, 4.5))

for problem_id, label in CATEGORY_MAP.items():
    sub = summary[summary["Problem_ID"] == problem_id].sort_values("Noise_Scale")
    ax.plot(
        sub["Noise_Scale"], sub["Semantic_Distance"],
        marker="s", label=label, color=COLORS[label], linewidth=2
    )

ax.set_title("Semantic Distance vs Noise Scale\n(higher = more conceptual diversity)",
             fontsize=13)
ax.set_xlabel("Noise Scale (σ)", fontsize=11)
ax.set_ylabel("Mean Pairwise Cosine Distance", fontsize=11)
ax.set_xticks(NOISE_ORDER)
ax.axvline(x=0.02, color="grey", linestyle="--", linewidth=1, alpha=0.6,
           label="Sweet spot (0.02)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "semantic_distance.png")



fig, ax = plt.subplots(figsize=(8, 4.5))

for problem_id, label in CATEGORY_MAP.items():
    sub = summary[summary["Problem_ID"] == problem_id].sort_values("Noise_Scale")
    ax.plot(
        sub["Noise_Scale"], sub["Collapse_Rate"],
        marker="^", label=label, color=COLORS[label], linewidth=2
    )

ax.set_title("Coherence Collapse Rate vs Noise Scale\n(fraction of broken/empty outputs)",
             fontsize=13)
ax.set_xlabel("Noise Scale (σ)", fontsize=11)
ax.set_ylabel("Collapse Rate", fontsize=11)
ax.set_xticks(NOISE_ORDER)
ax.set_ylim(-0.05, 1.10)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax.axvline(x=0.02, color="grey", linestyle="--", linewidth=1, alpha=0.6,
           label="Sweet spot (0.02)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, "coherence_collapse.png")

print("\n[Done] All metrics computed and figures saved.")
