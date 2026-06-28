"""011.1 — Diagnóstico mean vs median.

Re-roda só o pipeline pixel salvando os 10 valores individuais de
energy_0.5_2.0 por CP por vídeo. Pra cada vídeo, lista 10 numerinhos.

Saída:
- CSV com colunas: video, day, batch, cp, energy_0.5_2.0
- Plot boxplot: cada coluna = 1 dia, mostrando distribuição dos 10 valores
  por dia (concat 4 ângulos × 10 CPs = 40 pontos por dia).
- Plot da hipótese: mean por vídeo vs median por vídeo, anotando direção.
"""

from __future__ import annotations

import csv
import gc
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.pipelines.over_time import run_over_time
from marajo.modal.spectral_features import band_features


CONFIG = "configs/011-60fps.yaml"
PREPROC_DIR = "out/all_angles_60fps/"
OUT_DIR = "out/experimentos/011-1/"
BAND = (0.5, 2.0)


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> int:
    m = _DAY_RE.search(path)
    return int(m.group(1)) if m else 0


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = PipelineConfig.load(CONFIG)

    over_time = run_over_time(
        config=config,
        out_dir=PREPROC_DIR,
        do_preprocess=False,
        keep_fft_data=True,
    )

    rows: list[dict] = []
    for batch_name, paths in over_time.batches.items():
        for path in paths:
            day = day_from_path(path)
            res = over_time.per_video[path]
            for cp_id, comp in res.fft_data.items():
                bf = band_features(comp, band_low=BAND[0], band_high=BAND[1])
                rows.append({
                    "video": os.path.basename(path),
                    "day": day,
                    "batch": batch_name,
                    "cp": cp_id,
                    "energy": bf["energy"],
                    "energy_fraction": bf["energy_fraction"],
                    "peak_freq": bf["peak_freq"],
                    "centroid": bf["centroid"],
                })

    csv_path = os.path.join(OUT_DIR, "cp_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] {csv_path}  ({len(rows)} linhas)")

    # ---- Plot 1: boxplot por (batch, dia) com 40 pontos cada ----
    feb_days = sorted({r["day"] for r in rows if r["batch"] == "february"})
    apr_days = sorted({r["day"] for r in rows if r["batch"] == "april"})

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for ax, batch, days, color in zip(
        axes, ("february", "april"), (feb_days, apr_days), ("#2E5BFF", "#FF8533")
    ):
        data = []
        means = []
        medians = []
        for d in days:
            vals = [r["energy"] for r in rows if r["batch"] == batch and r["day"] == d]
            data.append(vals)
            means.append(np.mean(vals))
            medians.append(np.median(vals))
        positions = np.arange(len(days))
        bp = ax.boxplot(
            data, positions=positions, widths=0.6, patch_artist=True,
            showfliers=True, medianprops=dict(color="black", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.plot(positions, means, "o-", color="red", label="mean (por dia)", linewidth=2)
        ax.plot(positions, medians, "s--", color="green", label="median (por dia)", linewidth=2)
        ax.set_xticks(positions)
        ax.set_xticklabels(days, rotation=45)
        ax.set_title(f"{batch} — distribuição dos 10 CPs × 4 ângulos = 40 valores/dia")
        ax.set_ylabel("energy_0.5_2.0")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle("011.1 — Distribuição de energy_0.5_2.0 por dia (pixel, 60 FPS)", fontsize=13)
    out_path = os.path.join(OUT_DIR, "boxplot_per_day.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out_path}")

    # ---- Plot 2: scatter mean vs median por vídeo (cada ponto = 1 vídeo) ----
    videos = sorted({r["video"] for r in rows})
    means_per_vid = []
    medians_per_vid = []
    batches_per_vid = []
    for v in videos:
        vals = [r["energy"] for r in rows if r["video"] == v]
        means_per_vid.append(np.mean(vals))
        medians_per_vid.append(np.median(vals))
        batches_per_vid.append(next(r["batch"] for r in rows if r["video"] == v))

    fig, ax = plt.subplots(figsize=(8, 8))
    for batch, color in (("february", "#2E5BFF"), ("april", "#FF8533")):
        idx = [i for i, b in enumerate(batches_per_vid) if b == batch]
        ax.scatter(
            [means_per_vid[i] for i in idx],
            [medians_per_vid[i] for i in idx],
            color=color, label=batch, alpha=0.7, s=60, edgecolor="white",
        )
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("mean(energy_0.5_2.0) por vídeo")
    ax.set_ylabel("median(energy_0.5_2.0) por vídeo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("011.1 — mean vs median por vídeo (60 FPS)\n"
                 "Acima da diagonal: median > mean (raro). Abaixo: mean inflada por outlier(s).")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(OUT_DIR, "scatter_mean_vs_median.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out_path}")

    # ---- Quantificação: quantos CPs por vídeo são outliers (>3*IQR) ----
    outlier_summary = []
    for v in videos:
        vals = sorted(r["energy"] for r in rows if r["video"] == v)
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper = q3 + 3 * iqr
        n_out = sum(1 for x in vals if x > upper)
        outlier_summary.append({
            "video": v,
            "batch": next(r["batch"] for r in rows if r["video"] == v),
            "day": next(r["day"] for r in rows if r["video"] == v),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "n_outliers": n_out,
            "max_val": float(max(vals)),
            "max_over_median": float(max(vals) / np.median(vals)) if np.median(vals) > 0 else float("nan"),
        })
    sum_path = os.path.join(OUT_DIR, "outlier_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(outlier_summary[0].keys()))
        w.writeheader()
        w.writerows(outlier_summary)
    print(f"[CSV] {sum_path}")

    n_with_outliers = sum(1 for r in outlier_summary if r["n_outliers"] > 0)
    print(f"\nVídeos com pelo menos 1 CP outlier (>q3+3*IQR): {n_with_outliers}/{len(videos)}")
    print(f"Vídeo com maior max/median: {max(outlier_summary, key=lambda r: r['max_over_median'])}")

    gc.collect()


if __name__ == "__main__":
    main()
