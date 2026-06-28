"""Plot consolidado dos 5 detectores finais (3 pixel + 2 phase) pra apresentação.

Le os JSONs gerados pelo run_experiment_010.py e produz uma figura única em grid
2x3 com cada detector mostrando february vs april, anotando p-values dos 3 testes
e marcando significância. O 6º painel é um cartão-resumo textual.

Output: out/experimentos/010/detectores_consolidados.png + .pdf
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PIXEL_PATH = "out/experimentos/010/results_pixel.json"
PHASE_PATH = "out/experimentos/010/results_phase.json"
OUT_BASE = "out/experimentos/010/detectores_consolidados"


# (frontend, feature, título curto, descrição física)
DETECTORS = [
    ("pixel", "energy_0.5_2.0_median",
     "Energia em 0.5–2 Hz",
     "soma do PSD na banda baixa\n(detector original, 004.2)"),
    ("pixel", "frac_ratio_low_over_high_median",
     "Razão de frações 0.5–2 / 2–5",
     "redistribuição entre bandas\n(novo, 010)"),
    ("pixel", "centroid_0.5_5.0_median",
     "Centroide em 0.5–5 Hz",
     "deslocamento espectral macro\n(novo, 010)"),
    ("phase", "energy_2.0_5.0_mean",
     "Energia em 2–5 Hz (phase)",
     "movimento sub-pixel na banda média\n(detector original, 008)"),
    ("phase", "energy_2.0_5.0_median",
     "Energia em 2–5 Hz (phase, mediana)",
     "refinamento do detector phase\n(melhor p, 010)"),
]

BATCH_COLOR = {"february": "#2E5BFF", "april": "#FF8533"}
BATCH_LABEL = {"february": "february (estresse)", "april": "april (controle)"}


def load_series(path: str):
    with open(path) as f:
        d = json.load(f)
    return d


def get_series(data, feature):
    s = data["series"][feature]
    out = {}
    for batch in ("february", "april"):
        x = np.asarray(s[batch]["x"])
        y = np.asarray(s[batch]["y"])
        out[batch] = (x, y)
    # também busca os p-values
    tests = {}
    for t in data["tests"]:
        if t["feature"] == feature:
            tests[t["batch"]] = t["tests"]
    return out, tests


def plot_detector(ax, series, tests, title, description, ylabel="valor"):
    for batch in ("april", "february"):  # february por cima
        x, y = series[batch]
        # converte data YYYYMMDD em "dia ordinal dentro do batch" (1..N)
        unique = sorted(set(x.tolist()))
        day_idx = {d: i + 1 for i, d in enumerate(unique)}
        days = np.asarray([day_idx[v] for v in x])
        ax.scatter(
            days, y,
            color=BATCH_COLOR[batch], alpha=0.65, s=42,
            edgecolor="white", linewidth=0.7,
            label=BATCH_LABEL[batch],
        )
        # linha da regressão se há tendência
        ln = tests[batch]["linear"]
        if ln["p_value"] < 0.05:
            xs = np.array([min(days), max(days)])
            slope = ln["slope"]
            intercept = np.mean(y) - slope * np.mean(days)
            ax.plot(xs, slope * xs + intercept, color=BATCH_COLOR[batch],
                    linewidth=2.0, alpha=0.85, linestyle="--")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("dia ordinal (1–8)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # anotação com p-values
    feb_mk = tests["february"]["mann_kendall"]
    apr_mk = tests["april"]["mann_kendall"]
    fb_marker = "★" if feb_mk["p_value"] < 0.05 else " "
    feb_color = BATCH_COLOR["february"] if feb_mk["p_value"] < 0.05 else "gray"
    apr_color = BATCH_COLOR["april"] if apr_mk["p_value"] < 0.05 else "gray"

    txt = (
        f"$\\bf{{{fb_marker}\\ february:}}$ MK p={feb_mk['p_value']:.3f}, τ={feb_mk['statistic']:+.2f}\n"
        f"$\\it{{april:}}$ MK p={apr_mk['p_value']:.3f}, τ={apr_mk['statistic']:+.2f}\n"
        f"\n{description}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                  edgecolor="lightgray"),
    )


def make_summary_card(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "Critério de detector", ha="center", va="top",
            fontsize=12, fontweight="bold")
    ax.text(0.5, 0.86, "p < 0.05 em february  ∧  p ≥ 0.05 em april",
            ha="center", va="top", fontsize=10, style="italic", color="#555")

    summary_lines = [
        ("Pipeline pixel-grayscale (3 detectores)", "#2E5BFF"),
        ("• energy_0.5_2.0_median: cresce em february", None),
        ("• frac_ratio (0.5-2 / 2-5): cresce em february", None),
        ("• centroid_0.5_5.0_median: cai em february", None),
        ("", None),
        ("Pipeline phase-based (2 detectores)", "#FF8533"),
        ("• energy_2.0_5.0_mean: cai em february", None),
        ("• energy_2.0_5.0_median: cai em february", None),
        ("", None),
        ("Convergência entre frontends:", "#222"),
        ("fenômenos físicos distintos →", None),
        ("evidência independente, não redundante.", None),
    ]
    y = 0.75
    for line, color in summary_lines:
        weight = "bold" if color else "normal"
        c = color or "#222"
        ax.text(0.05, y, line, fontsize=9.5, color=c, fontweight=weight,
                ha="left", va="top")
        y -= 0.062


def main():
    pixel = load_series(PIXEL_PATH)
    phase = load_series(PHASE_PATH)
    sources = {"pixel": pixel, "phase": phase}

    fig, axes = plt.subplots(
        2, 3, figsize=(15.5, 9), constrained_layout=True,
    )
    axes_flat = axes.reshape(-1)

    for ax, (frontend, feat, title, desc) in zip(axes_flat[:5], DETECTORS):
        series, tests = get_series(sources[frontend], feat)
        plot_detector(ax, series, tests, title, desc)

    # legenda compartilhada (canto do 1º subplot)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # tira duplicatas mantendo ordem
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend(
            [h for h, _ in uniq], [l for _, l in uniq],
            loc="upper center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, 1.03), frameon=False,
        )

    # cartão-resumo na última célula
    make_summary_card(axes_flat[5])

    fig.suptitle(
        "Cinco detectores consolidados de estresse hídrico (mudas de açaí, n=32 por batch)",
        fontsize=13, fontweight="bold", y=1.06,
    )

    fig.savefig(f"{OUT_BASE}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    print(f"[OK] {OUT_BASE}.png")
    print(f"[OK] {OUT_BASE}.pdf")


if __name__ == "__main__":
    main()
