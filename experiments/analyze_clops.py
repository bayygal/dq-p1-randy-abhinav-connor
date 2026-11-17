"""
Analysis of the factorial CLOPS experiment (run factorial_clops.py first).

1. Three-way ANOVA on log(CLOPS): main effects of D, S, K and all interactions.
2. Timing models. Every circuit run costs some time t_run. Two candidates:
     planned model   t_run = c0 + c1*S*D     (hardware-like: every shot runs every layer)
     additive model  t_run = c0 + a*D + b*S + c*S*D
   On a statevector simulator the circuit is simulated once and the shots are
   sampled afterwards, so the additive model is expected to fit better.
   CLOPS = S*D / t_run, since M and K cancel.
3. Plots: heatmaps, interaction plots, quality plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.optimize import curve_fit
from paths import DATA, FIGURES


def load():
    return (pd.read_csv(DATA / "factorial_results.csv"),
            pd.read_csv(DATA / "quality_results.csv"),
            pd.read_csv(DATA / "qubit_results.csv"))


def run_anova(df):
    model = smf.ols("np.log(clops) ~ C(D) * C(S) * C(K)", data=df).fit()
    table = anova_lm(model, typ=2)
    table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()
    print("Three-way ANOVA on log(CLOPS)")
    print("(eta_sq = share of total variation explained by each term)\n")
    print(table[["df", "sum_sq", "F", "PR(>F)", "eta_sq"]].to_string(float_format=lambda x: f"{x:.4g}"))
    print()
    return table


def r_squared(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def fit_timing_models(df):
    S, D = df["S"].to_numpy(float), df["D"].to_numpy(float)
    t_run = (df["time_s"] / (df["K"] * df["M"])).to_numpy()
    log_clops = np.log(df["clops"].to_numpy())

    def planned(x, c0, c1):
        s, d = x
        return c0 + c1 * s * d

    (c0, c1), _ = curve_fit(planned, (S, D), t_run, p0=[1e-3, 1e-7], bounds=(0, np.inf))
    planned_r2 = r_squared(log_clops, np.log(S * D / planned((S, D), c0, c1)))

    X = np.column_stack([np.ones_like(S), D, S, S * D])
    coef, *_ = np.linalg.lstsq(X, t_run, rcond=None)
    additive_r2 = r_squared(log_clops, np.log(S * D / (X @ coef)))

    print("Timing models (time per circuit run, fitted to all runs)")
    print(f"  planned   t = c0 + c1*S*D            c0 = {c0 * 1e3:.3f} ms, "
          f"c1 = {c1 * 1e9:.2f} ns     R^2 on log CLOPS = {planned_r2:.3f}")
    print(f"  additive  t = c0 + a*D + b*S + c*S*D c0 = {coef[0] * 1e3:.3f} ms, "
          f"a = {coef[1] * 1e6:.2f} us, b = {coef[2] * 1e6:.3f} us, "
          f"c = {coef[3] * 1e9:.2f} ns     R^2 on log CLOPS = {additive_r2:.3f}")
    print()
    return {"planned": (c0, c1), "additive": coef}


def summarize(df, anova):
    by = lambda col: df.groupby(col)["clops"].median()
    print("Median CLOPS by factor level")
    for col in ["D", "S", "K"]:
        levels = ", ".join(f"{lvl}: {v:,.0f}" for lvl, v in by(col).items())
        print(f"  {col}: {levels}")
    print()
    terms = anova.drop(index="Residual").sort_values("eta_sq", ascending=False)
    print("Largest effects: " + ", ".join(
        f"{t} ({e:.1%})" for t, e in terms["eta_sq"].head(4).items()))
    print()


def plot_heatmaps(df):
    ks = sorted(df["K"].unique())
    ds, ss = sorted(df["D"].unique()), sorted(df["S"].unique())
    med = df.groupby(["K", "D", "S"])["clops"].median()
    vmin, vmax = med.min(), med.max()

    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.2), sharey=True)
    for ax, k in zip(axes, ks):
        grid = np.array([[med[(k, d, s)] for s in ss] for d in ds])
        im = ax.imshow(grid, origin="lower", cmap="viridis", norm="log", vmin=vmin, vmax=vmax)
        for i in range(len(ds)):
            for j in range(len(ss)):
                ax.text(j, i, f"{grid[i, j] / 1e6:.2g}M", ha="center", va="center",
                        color="white", fontsize=8)
        ax.set_xticks(range(len(ss)), ss)
        ax.set_yticks(range(len(ds)), ds)
        ax.set_xlabel("Shots (S)")
        ax.set_title(f"K = {k} parameter updates")
    axes[0].set_ylabel("QV Layers (D)")
    fig.colorbar(im, ax=axes, label="Median CLOPS", shrink=0.85)
    fig.suptitle("CLOPS across QV layers x shots x parameter updates")
    plt.savefig(FIGURES / "clops_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def interaction_panel(ax, df, x, line, xlabel):
    for lvl, group in df.groupby(line):
        stats = group.groupby(x)["clops"].agg(["median", lambda v: v.quantile(0.25),
                                               lambda v: v.quantile(0.75)])
        stats.columns = ["median", "q1", "q3"]
        ax.errorbar(stats.index, stats["median"],
                    yerr=[stats["median"] - stats["q1"], stats["q3"] - stats["median"]],
                    marker="o", capsize=3, label=f"{line} = {lvl}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CLOPS (median, IQR bars)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def plot_interactions(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    interaction_panel(axes[0], df, "D", "S", "QV Layers (D)")
    axes[0].set_title("QV layers x shots")
    interaction_panel(axes[1], df, "S", "D", "Shots (S)")
    axes[1].set_title("Shots x QV layers")
    interaction_panel(axes[2], df[df["S"] == 1000], "K", "D", "Parameter Updates (K)")
    axes[2].set_title("Parameter updates x QV layers (S = 1000)")
    plt.tight_layout()
    plt.savefig(FIGURES / "clops_interactions.png", dpi=300)
    plt.close(fig)


def plot_quality(quality, qubits):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(quality["D"], quality["hop"], marker="^", color="tab:green")
    axes[0].set_xlabel("QV Layers (D)")
    axes[0].set_title(f"Quality vs QV layers ({quality['qubits'].iloc[0]} qubits, noisy)")

    q = qubits.groupby("qubits").agg(hop=("hop", "first"), clops=("clops", "median"))
    axes[1].plot(q.index, q["hop"], marker="^", color="tab:green")
    axes[1].set_xlabel("Qubits (D = qubits)")
    axes[1].set_title("Quality vs qubit count (noisy)")

    for ax in axes[:2]:
        ax.axhline(2 / 3, linestyle="--", color="gray", label="QV pass threshold (2/3)")
        ax.axhline(0.5, linestyle=":", color="red", label="Pure noise (0.5)")
        ax.set_ylabel("Heavy Output Probability")
        ax.legend()

    axes[2].plot(q.index, q["clops"], marker="o")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Qubits (D = qubits)")
    axes[2].set_ylabel("CLOPS (median)")
    axes[2].set_title("CLOPS vs qubit count (simulator)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "quality_vs_layers_qubits.png", dpi=300)
    plt.close(fig)


def main():
    df, quality, qubits = load()
    anova = run_anova(df)
    fit_timing_models(df)
    summarize(df, anova)

    print("Quality (heavy output probability, noisy)")
    print("  vs D:      " + ", ".join(f"D={d}: {h:.3f}" for d, h in zip(quality["D"], quality["hop"])))
    q = qubits.groupby("qubits").agg(hop=("hop", "first"), clops=("clops", "median"))
    print("  vs qubits: " + ", ".join(f"n={n}: HOP {r.hop:.3f}, CLOPS {r.clops:,.0f}"
                                     for n, r in q.iterrows()))
    print()

    plot_heatmaps(df)
    plot_interactions(df)
    plot_quality(quality, qubits)
    print("Plots saved: clops_heatmaps.png, clops_interactions.png, quality_vs_layers_qubits.png")


if __name__ == "__main__":
    main()
