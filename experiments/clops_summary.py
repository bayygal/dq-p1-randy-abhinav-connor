"""
Simple summary: how shots, QV layers and parameter updates affect CLOPS.

Shots and QV layers come from factorial_results.csv (factorial_clops.py).
Parameter updates come from parameter_updates_results.csv
(parameter_updates_clops.py, which times the training loop in
ParameterUpdatesTest.py).
"""
import pandas as pd
import matplotlib.pyplot as plt
from paths import DATA, FIGURES

df = pd.read_csv(DATA / "factorial_results.csv")
pu = pd.read_csv(DATA / "parameter_updates_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, col, label, color in [
    (axes[0, 0], "S", "Number of Shots", "tab:blue"),
    (axes[0, 1], "D", "Number of QV Layers", "tab:orange"),
]:
    avg = df.groupby(col)["clops"].mean() / 1e6
    ax.plot(avg.index, avg.values, marker="o", color=color)
    ax.set_xlabel(label)
    ax.set_ylabel("CLOPS (millions)")
    ax.set_title(f"{label} vs CLOPS")
    ax.set_ylim(bottom=0)

ax = axes[1, 0]
for shots, group in pu.groupby("shots"):
    ax.plot(group["updates"], group["clops"] / 1e3, marker="o", label=f"{shots} shots")
ax.set_xlabel("Number of Parameter Updates")
ax.set_ylabel("CLOPS (thousands)")
ax.set_title("Parameter Updates and Shots Together\n(ParameterUpdatesTest.py)")
ax.set_ylim(bottom=0)
ax.legend()

ax = axes[1, 1]
for shots, group in df.groupby("S"):
    avg = group.groupby("D")["clops"].mean() / 1e6
    ax.plot(avg.index, avg.values, marker="o", label=f"{shots} shots")
ax.set_xlabel("Number of QV Layers")
ax.set_ylabel("CLOPS (millions)")
ax.set_title("QV Layers and Shots Together")
ax.legend()

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "clops_summary.png", dpi=300)

table = df.pivot_table(index="D", columns="S", values="clops", aggfunc="mean") / 1e6
table.index.name = "QV layers"
table.columns.name = "shots"
print("Average CLOPS (millions) for each combination of QV layers and shots\n")
print(table.round(2).to_string())

table = pu.pivot_table(index="updates", columns="shots", values="clops") / 1e3
table.index.name = "updates"
print("\nCLOPS (thousands) for each combination of parameter updates and shots")
print("(from ParameterUpdatesTest.py)\n")
print(table.round(1).to_string())
print("\nGraph saved to clops_summary.png")
