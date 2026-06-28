"""Resumo visual dos sub-experimentos 011.* — uma figura.

Mostra side-by-side os detectores que apareceram em cada sub-experimento:
- 011 (base): 1 detector pixel
- 011.3 (bandas finas): + 2 phase
- 011.4 (downsample): + 2 phase
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "out/experimentos/011/followup_summary.png"
OUT_PDF = "out/experimentos/011/followup_summary.pdf"


# (experimento, frontend, feature, título curto, descrição)
DETECTORS = [
    ("011 base", "pixel", "energy_0.5_2.0_mean",
     "energy_0.5_2.0_mean", "banda baixa pixel"),
    ("011.3 fine", "phase", "peak_freq_1.0_1.5_median",
     "peak_freq_1.0_1.5_median", "freq dominante 1-1.5 Hz"),
    ("011.3 fine", "phase", "energy_3.0_5.0_mean",
     "energy_3.0_5.0_mean", "energia 3-5 Hz"),
    ("011.4 ds", "phase", "peak_freq_2.0_5.0_mean",
     "peak_freq_2.0_5.0_mean", "freq dominante 2-5 Hz"),
    ("011.4 ds", "phase", "peak_freq_2.0_5.0_median",
     "peak_freq_2.0_5.0_median", "(mediana)"),
]

JSON_PATHS = {
    ("011 base", "pixel"): "out/experimentos/011/results_pixel.json",
    ("011 base", "phase"): "out/experimentos/011/results_phase.json",
    ("011.3 fine", "pixel"): "out/experimentos/011-3/results_pixel.json",
    ("011.3 fine", "phase"): "out/experimentos/011-3/results_phase.json",
    ("011.4 ds", "pixel"): "out/experimentos/011-4/results_pixel.json",
    ("011.4 ds", "phase"): "out/experimentos/011-4/results_phase.json",
}

BATCH_COLOR = {"february": "#2E5BFF", "april": "#FF8533"}


def load(path):
    with open(path) as f:
        return json.load(f)


def get(data, feature, batch):
    s = data["series"][feature][batch]
    return np.asarray(s["x"]), np.asarray(s["y"])


def get_tests(data, feature, batch):
    for t in data["tests"]:
        if t["feature"] == feature and t["batch"] == batch:
            return t["tests"]
    return None


def plot_panel(ax, exp, frontend, feature, title, desc):
    data = load(JSON_PATHS[(exp, frontend)])
    for batch in ("april", "february"):
        x, y = get(data, feature, batch)
        unique = sorted(set(x.tolist()))
        idx = {d: i + 1 for i, d in enumerate(unique)}
        days = np.asarray([idx[v] for v in x])
        ax.scatter(days, y, color=BATCH_COLOR[batch], alpha=0.65, s=38,
                   edgecolor="white", linewidth=0.6, label=batch)
        t = get_tests(data, feature, batch)
        if t["linear"]["p_value"] < 0.05:
            xs = np.array([min(days), max(days)])
            slope = t["linear"]["slope"]
            intercept = np.mean(y) - slope * np.mean(days)
            ax.plot(xs, slope * xs + intercept, color=BATCH_COLOR[batch],
                    linewidth=1.8, alpha=0.85, linestyle="--")

    feb_mk = get_tests(data, feature, "february")["mann_kendall"]
    apr_mk = get_tests(data, feature, "april")["mann_kendall"]
    star = "★ " if feb_mk["p_value"] < 0.05 else ""

    ax.set_title(f"{star}[{exp}/{frontend}] {title}", fontsize=10, fontweight="bold")
    ax.set_xlabel("dia (1-8)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)

    txt = (
        f"feb MK p={feb_mk['p_value']:.3f}, τ={feb_mk['statistic']:+.2f}\n"
        f"apr MK p={apr_mk['p_value']:.3f}\n"
        f"{desc}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                      edgecolor="lightgray"))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes_flat = axes.reshape(-1)

    for ax, (exp, frontend, feature, title, desc) in zip(axes_flat, DETECTORS):
        plot_panel(ax, exp, frontend, feature, title, desc)

    # Card-resumo
    card = axes_flat[5]
    card.axis("off")
    card.set_xlim(0, 1); card.set_ylim(0, 1)
    card.text(0.5, 0.97, "Sub-experimentos 011.* — detectores cumulativos",
              ha="center", va="top", fontsize=12, fontweight="bold")
    lines = [
        ("011 (base, 60 FPS):", "#2E5BFF"),
        ("  • 1 detector pixel sobrevive (mean)", None),
        ("", None),
        ("011.1 (mean vs median):", "#222"),
        ("  • distribuição CPs tem skew positiva consistente", None),
        ("  • mean absorve cauda alta (sinal); mediana ignora", None),
        ("", None),
        ("011.2 (leave-one-day-out):", "#222"),
        ("  • dia 03/04 carrega quase todos confounds em april", None),
        ("  • dia 09/04 carrega o confound em energy_2.0_5.0", None),
        ("", None),
        ("011.3 (bandas finas) + 011.4 (downsample):", "#FF8533"),
        ("  • +4 detectores phase em variantes peak_freq", None),
        ("", None),
        ("Total 60 FPS: 5 detectores (1 pixel + 4 phase)", "#222"),
    ]
    y = 0.88
    for line, color in lines:
        weight = "bold" if color else "normal"
        c = color or "#222"
        card.text(0.03, y, line, fontsize=9, color=c, fontweight=weight,
                  ha="left", va="top")
        y -= 0.058

    fig.suptitle("011.* — abrir a rede em 60 FPS (resumo dos sub-experimentos)",
                 fontsize=13, fontweight="bold", y=1.02)

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[OK] {OUT}")
    print(f"[OK] {OUT_PDF}")


if __name__ == "__main__":
    main()
