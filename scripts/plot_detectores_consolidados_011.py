"""Plot consolidado dos 5 detectores pra 60 FPS (experimento 011).

Versão do plot_detectores_consolidados.py apontando pros JSONs de 011 em vez
de 010. Saída em out/experimentos/011/.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PIXEL_PATH = "out/experimentos/011/results_pixel.json"
PHASE_PATH = "out/experimentos/011/results_phase.json"
OUT_BASE = "out/experimentos/011/detectores_consolidados"


DETECTORS = [
    ("pixel", "energy_0.5_2.0_mean",
     "Energia em 0.5–2 Hz (mean)",
     "variante mean do detector pixel\n(sobrevive em 60 FPS)"),
    ("pixel", "energy_0.5_2.0_median",
     "Energia em 0.5–2 Hz (median)",
     "detector original (004.2)\n(NÃO sobrevive em 60 FPS)"),
    ("pixel", "frac_ratio_low_over_high_median",
     "Razão de frações 0.5–2 / 2–5",
     "redistribuição entre bandas\n(novo, 010)"),
    ("phase", "energy_2.0_5.0_mean",
     "Energia em 2–5 Hz (phase)",
     "detector original 008 — ambos\nbatches caem juntos em 60 FPS"),
    ("phase", "energy_2.0_5.0_median",
     "Energia em 2–5 Hz (phase, mediana)",
     "refinamento phase do 010"),
]

BATCH_COLOR = {"february": "#2E5BFF", "april": "#FF8533"}
BATCH_LABEL = {"february": "february (estresse)", "april": "april (controle)"}


def load_series(path: str):
    with open(path) as f:
        return json.load(f)


def get_series(data, feature):
    s = data["series"][feature]
    out = {}
    for batch in ("february", "april"):
        x = np.asarray(s[batch]["x"])
        y = np.asarray(s[batch]["y"])
        out[batch] = (x, y)
    tests = {}
    for t in data["tests"]:
        if t["feature"] == feature:
            tests[t["batch"]] = t["tests"]
    return out, tests


def plot_detector(ax, series, tests, title, description, ylabel="valor"):
    for batch in ("april", "february"):
        x, y = series[batch]
        unique = sorted(set(x.tolist()))
        day_idx = {d: i + 1 for i, d in enumerate(unique)}
        days = np.asarray([day_idx[v] for v in x])
        ax.scatter(
            days, y,
            color=BATCH_COLOR[batch], alpha=0.65, s=42,
            edgecolor="white", linewidth=0.7,
            label=BATCH_LABEL[batch],
        )
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

    feb_mk = tests["february"]["mann_kendall"]
    apr_mk = tests["april"]["mann_kendall"]
    fb_marker = "★" if feb_mk["p_value"] < 0.05 else " "

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


def make_summary_card(ax, pixel_pass, phase_pass):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "60 FPS — síntese", ha="center", va="top",
            fontsize=12, fontweight="bold")
    ax.text(0.5, 0.86, "p < 0.05 em february  ∧  p ≥ 0.05 em april",
            ha="center", va="top", fontsize=10, style="italic", color="#555")

    summary_lines = [
        (f"Pipeline pixel: {pixel_pass}/3 detectores validados", "#2E5BFF"),
        (f"Pipeline phase: {phase_pass}/2 detectores validados", "#FF8533"),
        ("", None),
        ("Único sobrevivente:", "#222"),
        ("• energy_0.5_2.0_mean (pixel)", None),
        ("  feb MK p=0.042 τ=+0.27; apr p=0.267", None),
        ("", None),
        ("Mudança de regime:", "#222"),
        ("• 240 FPS slow-mo → 60 FPS real", None),
        ("• 720×1280 → 1080×1920 (feb)", None),
        ("• 800/3.3s → 600/10s; Δf 0.30→0.10 Hz", None),
        ("", None),
        ("Interpretação:", "#222"),
        ("o sinal de banda baixa em pixel persiste,", None),
        ("mas a maioria dos detectores depende do", None),
        ("regime 240 FPS.", None),
    ]
    y = 0.75
    for line, color in summary_lines:
        weight = "bold" if color else "normal"
        c = color or "#222"
        ax.text(0.05, y, line, fontsize=9.5, color=c, fontweight=weight,
                ha="left", va="top")
        y -= 0.062


def count_passing(data, features, alpha=0.05):
    n = 0
    for feat in features:
        feb_tests = next(t for t in data["tests"] if t["feature"] == feat and t["batch"] == "february")
        apr_tests = next(t for t in data["tests"] if t["feature"] == feat and t["batch"] == "april")
        feb_pass = any(v["p_value"] < alpha for v in feb_tests["tests"].values())
        apr_no = all(v["p_value"] >= alpha for v in apr_tests["tests"].values())
        if feb_pass and apr_no:
            n += 1
    return n


def main():
    pixel = load_series(PIXEL_PATH)
    phase = load_series(PHASE_PATH)
    sources = {"pixel": pixel, "phase": phase}

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), constrained_layout=True)
    axes_flat = axes.reshape(-1)

    for ax, (frontend, feat, title, desc) in zip(axes_flat[:5], DETECTORS):
        series, tests = get_series(sources[frontend], feat)
        plot_detector(ax, series, tests, title, desc)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend(
            [h for h, _ in uniq], [l for _, l in uniq],
            loc="upper center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, 1.03), frameon=False,
        )

    pixel_pass = count_passing(pixel, [d[1] for d in DETECTORS if d[0] == "pixel"])
    phase_pass = count_passing(phase, [d[1] for d in DETECTORS if d[0] == "phase"])
    make_summary_card(axes_flat[5], pixel_pass, phase_pass)

    fig.suptitle(
        "Detectores em 60 FPS (n=32 por batch; experimento 011)",
        fontsize=13, fontweight="bold", y=1.06,
    )

    fig.savefig(f"{OUT_BASE}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    print(f"[OK] {OUT_BASE}.png")
    print(f"[OK] {OUT_BASE}.pdf")


if __name__ == "__main__":
    main()
