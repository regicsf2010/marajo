"""011.5 — Ensemble 60+240 (cross-FPS).

Compara dia a dia o detector energy_0.5_2.0 entre os dois regimes:
- 240 FPS: energy_0.5_2.0_median (010, detector consolidado)
- 60 FPS:  energy_0.5_2.0_mean   (011, detector sobrevivente)

Como cada vídeo é coleta diferente, não dá pareamento por (dia, ângulo).
Pareamos por dia (média dos 4 ângulos). Plot scatter + correlação +
matriz de confusão "ambos passam / ambos falham / discordam".
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


JSON_240 = "out/experimentos/010/results_pixel.json"
JSON_60 = "out/experimentos/011/results_pixel.json"
OUT_DIR = "out/experimentos/011-5/"

FEATURE_240 = "energy_0.5_2.0_median"
FEATURE_60 = "energy_0.5_2.0_mean"


def aggregate_by_day(series):
    """series['x'] tem 32 valores (4/dia x 8 dias). Agrega média por dia."""
    by_day = defaultdict(list)
    for x, y in zip(series["x"], series["y"]):
        by_day[int(x)].append(y)
    days = sorted(by_day)
    means = [float(np.mean(by_day[d])) for d in days]
    return days, means


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_240) as f:
        data_240 = json.load(f)
    with open(JSON_60) as f:
        data_60 = json.load(f)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for ax, batch in zip(axes, ("february", "april")):
        s240 = data_240["series"][FEATURE_240][batch]
        s60 = data_60["series"][FEATURE_60][batch]
        d240, m240 = aggregate_by_day(s240)
        d60, m60 = aggregate_by_day(s60)
        # garantir que os dias batem (precisam coincidir)
        common = sorted(set(d240) & set(d60))
        if len(common) != len(d240):
            print(f"  AVISO {batch}: dias diferentes 240={d240} 60={d60}")
        idx_240 = {d: i for i, d in enumerate(d240)}
        idx_60 = {d: i for i, d in enumerate(d60)}
        m240_c = np.array([m240[idx_240[d]] for d in common])
        m60_c = np.array([m60[idx_60[d]] for d in common])

        if len(common) >= 3:
            r, p_corr = scipy_stats.pearsonr(m240_c, m60_c)
            r_sp, p_sp = scipy_stats.spearmanr(m240_c, m60_c)
        else:
            r, p_corr, r_sp, p_sp = float("nan"), float("nan"), float("nan"), float("nan")

        # Plot
        days_idx = np.arange(len(common))
        color = "#2E5BFF" if batch == "february" else "#FF8533"
        ax_b = ax
        ax_b.scatter(days_idx, m240_c, color=color, marker="o", s=80,
                     label="240 FPS (median)", edgecolor="black")
        ax2 = ax_b.twinx()
        ax2.scatter(days_idx, m60_c, color=color, marker="s", s=80, alpha=0.6,
                    label="60 FPS (mean)")
        for i in range(len(common)):
            ax_b.plot([days_idx[i], days_idx[i]], [m240_c[i], m240_c[i]], color=color)
        ax_b.set_xticks(days_idx)
        ax_b.set_xticklabels([str(d)[-4:] for d in common], rotation=45)
        ax_b.set_ylabel("energy 0.5-2 (240 FPS, median)", color="black")
        ax2.set_ylabel("energy 0.5-2 (60 FPS, mean)", color="gray")
        ax_b.set_title(
            f"{batch} — Pearson r={r:.2f} (p={p_corr:.3f}), "
            f"Spearman ρ={r_sp:.2f} (p={p_sp:.3f})",
            fontsize=10,
        )
        ax_b.grid(True, alpha=0.3)
        # Legend combinada
        h1, l1 = ax_b.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_b.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

        for d, v240, v60 in zip(common, m240_c, m60_c):
            rows.append({
                "batch": batch, "day": d,
                "energy_240_median": float(v240),
                "energy_60_mean": float(v60),
            })

    fig.suptitle("011.5 — energy_0.5_2.0 dia a dia: 240 FPS (median) × 60 FPS (mean)",
                 fontsize=12)
    out_path = os.path.join(OUT_DIR, "ensemble_60_240.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out_path}")

    csv_path = os.path.join(OUT_DIR, "ensemble_60_240.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] {csv_path}")

    # Plot 2: scatter direto (mean 60 vs median 240) com ambos batches
    fig, ax = plt.subplots(figsize=(7, 7))
    for batch in ("february", "april"):
        s240 = data_240["series"][FEATURE_240][batch]
        s60 = data_60["series"][FEATURE_60][batch]
        d240, m240 = aggregate_by_day(s240)
        d60, m60 = aggregate_by_day(s60)
        common = sorted(set(d240) & set(d60))
        idx_240 = {d: i for i, d in enumerate(d240)}
        idx_60 = {d: i for i, d in enumerate(d60)}
        m240_c = [m240[idx_240[d]] for d in common]
        m60_c = [m60[idx_60[d]] for d in common]
        color = "#2E5BFF" if batch == "february" else "#FF8533"
        ax.scatter(m240_c, m60_c, color=color, label=batch, s=90, alpha=0.7,
                   edgecolor="white", linewidth=0.8)
        for d, x, y in zip(common, m240_c, m60_c):
            ax.annotate(str(d)[-4:], (x, y), fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("240 FPS  ·  energy_0.5_2.0_median (mean por dia)")
    ax.set_ylabel("60 FPS  ·  energy_0.5_2.0_mean (mean por dia)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("011.5 — Cross-FPS: cada ponto é 1 dia (média dos 4 ângulos)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(OUT_DIR, "scatter_60_vs_240.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out_path}")


if __name__ == "__main__":
    main()
