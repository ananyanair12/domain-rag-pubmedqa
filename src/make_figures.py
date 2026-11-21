import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

def load(*candidates):
    for c in candidates:
        p = ROOT / c
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError(candidates)

small = load("results_corpus11k/retrieval_baseline_whole.json")
big = load("results_corpus62k/retrieval_baseline_whole_62k.json",
           "results/retrieval_baseline_whole_62k.json")
NAVY, TEAL = "#0B2B4F", "#1F8AA8"

# Chart 1: 11k vs 62k library
metrics = [("hit@5", "Hit@5"), ("mrr@10", "MRR@10")]
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
b1 = ax.bar([i - w / 2 for i in range(2)], [small[k] for k, _ in metrics], w,
            label="11,000 abstracts", color=TEAL)
b2 = ax.bar([i + w / 2 for i in range(2)], [big[k] for k, _ in metrics], w,
            label="62,249 abstracts", color=NAVY)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=10)
ax.set_xticks(range(2))
ax.set_xticklabels([l for _, l in metrics])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Baseline (all-MiniLM-L6-v2): effect of library size")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "baseline_11k_vs_62k.png")
plt.close(fig)

# Chart 2: Hit@k on the 62k library
ks = [1, 3, 5, 10]
vals = [big[f"hit@{k}"] for k in ks]
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
bars = ax.bar([f"Hit@{k}" for k in ks], vals, color=NAVY)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Fraction of questions")
ax.set_title("Baseline on 62,249 abstracts: right abstract found in top-k")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "baseline_hit_at_k_62k.png")
plt.close(fig)
print("saved to", OUT)