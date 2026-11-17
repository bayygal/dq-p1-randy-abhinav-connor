"""
Builds slide-ready images in presentation_images/ from saved results.

Needs: factorial_results.csv, quality_results.csv (factorial_clops.py),
parameter_updates_results.csv (parameter_updates_clops.py),
combined_results.csv (combined_experiment.py).
"""
import pandas as pd
import matplotlib.pyplot as plt
from paths import DATA, PRESENTATION as OUT

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                     "legend.fontsize": 12})

factorial = pd.read_csv(DATA / "factorial_results.csv")
quality = pd.read_csv(DATA / "quality_results.csv")
updates = pd.read_csv(DATA / "parameter_updates_results.csv")
combined = pd.read_csv(DATA / "combined_results.csv")

speed_by_layers = factorial[factorial["S"] == 1000].groupby("D")["clops"].mean() / 1e6


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)


# 1. Speed vs QV layers
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(speed_by_layers.index, speed_by_layers.values, marker="o", color="tab:blue", lw=2.5, ms=9)
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("CLOPS (millions)")
ax.set_title("More QV Layers → Higher CLOPS (Speed)")
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
save(fig, "1_qv_layers_speed.png")

# 2. Accuracy vs QV layers
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(quality["D"], quality["hop"], marker="o", color="tab:red", lw=2.5, ms=9)
ax.axhline(2 / 3, ls="--", color="gray", label="Passing score (2/3)")
ax.axhline(0.5, ls=":", color="black", label="Random guessing (0.5)")
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("Accuracy (heavy output probability)")
ax.set_title("More QV Layers → Lower Accuracy")
ax.set_ylim(0.45, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "2_qv_layers_accuracy.png")

# 3. Speed and accuracy on one chart
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(speed_by_layers.index, speed_by_layers.values, marker="o", color="tab:blue",
        lw=2.5, ms=9, label="Speed (CLOPS)")
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("CLOPS (millions)", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(bottom=0)
ax2 = ax.twinx()
ax2.plot(quality["D"], quality["hop"], marker="s", color="tab:red", lw=2.5, ms=9,
         label="Accuracy")
ax2.axhline(2 / 3, ls="--", color="gray", label="Passing accuracy (2/3)")
ax2.set_ylabel("Accuracy", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0.45, 1.0)
ax.set_title("QV Layers: Speed Goes Up, Accuracy Goes Down")
fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15))
ax.grid(True, alpha=0.3)
save(fig, "3_qv_layers_speed_vs_accuracy.png")

# 4. All three together: speed
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for shots, group in factorial.groupby("S"):
    avg = group.groupby("D")["clops"].mean() / 1e6
    ax.plot(avg.index, avg.values, marker="o", lw=2.5, ms=8, label=f"{shots} shots")
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("CLOPS (millions)")
ax.set_title("QV Layers + Shots")
ax.legend()
ax = axes[1]
for shots, group in updates.groupby("shots"):
    ax.plot(group["updates"], group["clops"] / 1e3, marker="o", lw=2.5, ms=8,
            label=f"{shots} shots")
ax.set_xlabel("Number of Parameter Updates")
ax.set_ylabel("CLOPS (thousands)")
ax.set_title("Parameter Updates + Shots")
ax.set_ylim(bottom=0)
ax.legend()
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Speed: Shots and QV Layers Raise CLOPS, Parameter Updates Don't", fontsize=17)
save(fig, "4_together_speed.png")

# 5. All three together: accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[1]
for d, group in combined[combined["S"] == 1000].groupby("D"):
    avg = group.groupby("K")["accuracy"].mean()
    ax.plot(avg.index, avg.values, marker="o", lw=2.5, ms=8,
            label=f"{d} layer{'s' if d > 1 else ''}")
ax.set_xlabel("Number of Parameter Updates")
ax.set_ylabel("Accuracy")
ax.set_title("More Updates Help, but More Layers\nLower the Best Possible Accuracy")
ax.set_ylim(0.3, 1.0)
ax.set_xticks(sorted(combined["K"].unique()))
ax.legend(loc="lower right")
ax = axes[0]
for s, group in combined[combined["K"] == 20].groupby("S"):
    avg = group.groupby("D")["accuracy"].mean()
    ax.plot(avg.index, avg.values, marker="o", lw=2.5, ms=8, label=f"{s} shots")
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("Accuracy (after 20 updates)")
ax.set_title("Shots Don't Change Accuracy;\nLayers Still Lower It")
ax.set_ylim(0.8, 1.0)
ax.set_xticks(sorted(combined["D"].unique()))
ax.legend()
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Accuracy: Scaling All Three Together", fontsize=17)
save(fig, "5_together_accuracy.png")

print(f"Saved images to {OUT}/:")
for path in sorted(OUT.glob("*.png")):
    print("  " + path.name)
