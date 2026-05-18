"""Experimento 006 — Sensibilidade do detector aos hiperparâmetros do CP_alg.

Grid search sobre (num_pcs, short_half_life, long_half_life) em cima dos 64 vídeos
do 003. Pra cada combinação, recomputa CP + FFT + a feature `energy_0.5_2.0_median`
e aplica Mann-Kendall em february e april separadamente.

Otimização: PCA é caro mas independe dos parâmetros do CP — rodamos 1x por vídeo
com n_components = max(num_pcs), reusando o score nas combinações.

Saída: tabela CSV com (num_pcs, shf, lhf, p_february_MK, p_april_MK, tau_february) +
heatmap mostrando onde o detector persiste em p < 0.05.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from marajo.config import CPConfig, PipelineConfig
from marajo.decomposition.cp import run_cp_on_components
from marajo.decomposition.pca import compute_pca
from marajo.io.video import load_grayscale_dataset, video_status
from marajo.modal.fft import compute_fft_for_components
from marajo.modal.spectral_features import video_band_features
from marajo.modal.trends import mann_kendall


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> float:
    m = _DAY_RE.search(path)
    return float(m.group(1)) if m else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search dos hiperparâmetros do CP.")
    p.add_argument("--config", default="configs/003-all-angles.yaml")
    p.add_argument("--out-dir", default="out/all_angles/")
    p.add_argument("--plots-dir", default="out/experimentos/006/")
    p.add_argument("--num-pcs-grid", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--shf-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--lhf-grid", type=float, nargs="+", default=[1000.0, 100000.0, 900000.0])
    p.add_argument("--feature", default="energy_0.5_2.0_median")
    p.add_argument("--band", nargs=2, type=float, default=[0.5, 2.0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    batches = {"february": config.batches.february, "april": config.batches.april}
    all_videos = batches["february"] + batches["april"]

    max_num_pcs = max(args.num_pcs_grid)
    band_low, band_high = args.band

    print(f"Grid: num_pcs={args.num_pcs_grid}, shf={args.shf_grid}, lhf={args.lhf_grid}")
    print(f"Total combinações: {len(args.num_pcs_grid) * len(args.shf_grid) * len(args.lhf_grid)}")
    print(f"Vídeos: {len(all_videos)}")
    print()

    # Etapa 1: PCA por vídeo (1x cada)
    print("Computando PCA por vídeo (1x cada, n_components =", max_num_pcs, ")...")
    pca_per_video: dict[str, tuple] = {}
    fps_per_video: dict[str, int] = {}
    for i, video_path in enumerate(all_videos):
        pp = os.path.join(args.out_dir, os.path.basename(video_path))
        dataset = load_grayscale_dataset(pp)
        info = video_status(pp)
        pca = compute_pca(dataset, n_components=max_num_pcs)
        pca_per_video[video_path] = pca
        fps_per_video[video_path] = info.fps
        del dataset
        print(f"  [{i+1}/{len(all_videos)}] {os.path.basename(video_path)}")
    print()

    # Etapa 2: pra cada combinação, computar CP + FFT + feature
    rows: list[dict] = []
    combos = [
        (n, s, l)
        for n in args.num_pcs_grid
        for s in args.shf_grid
        for l in args.lhf_grid
    ]
    for idx, (n_pcs, shf, lhf) in enumerate(combos):
        cp_cfg = CPConfig(short_half_life=shf, long_half_life=lhf, max_mask_len=50, n_mask_horizon=10)
        feats_by_batch: dict[str, list[float]] = {"february": [], "april": []}
        x_by_batch: dict[str, list[float]] = {"february": [], "april": []}

        for batch_name, batch_videos in batches.items():
            for video_path in batch_videos:
                pca = pca_per_video[video_path]
                fps = fps_per_video[video_path]
                cp_result = run_cp_on_components(pca, n_pc=n_pcs, config=cp_cfg)
                fft_data = compute_fft_for_components(cp_result.unmixed, fps, range(n_pcs))
                feats = video_band_features(fft_data, [(band_low, band_high)])
                feats_by_batch[batch_name].append(feats[args.feature])
                x_by_batch[batch_name].append(day_from_path(video_path))

        mk_feb = mann_kendall(feats_by_batch["february"], x=x_by_batch["february"])
        mk_apr = mann_kendall(feats_by_batch["april"], x=x_by_batch["april"])
        row = {
            "num_pcs": n_pcs,
            "short_half_life": shf,
            "long_half_life": lhf,
            "tau_february": mk_feb.statistic,
            "p_february": mk_feb.p_value,
            "tau_april": mk_apr.statistic,
            "p_april": mk_apr.p_value,
            "detector": (mk_feb.p_value < 0.05) and (mk_apr.p_value >= 0.05),
        }
        rows.append(row)
        flag = "✅" if row["detector"] else "—"
        print(
            f"  [{idx+1:2d}/{len(combos)}] n_pcs={n_pcs} shf={shf} lhf={lhf:>9} | "
            f"feb p={mk_feb.p_value:.3f} (τ={mk_feb.statistic:+.2f}) | "
            f"apr p={mk_apr.p_value:.3f} | {flag}"
        )

    # Etapa 3: salva CSV
    csv_path = os.path.join(args.plots_dir, "grid_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV salvo: {csv_path}")

    # Etapa 4: heatmap de p_february por (num_pcs, lhf), uma matriz por shf
    fig, axes = plt.subplots(1, len(args.shf_grid), figsize=(5 * len(args.shf_grid), 5), constrained_layout=True)
    if len(args.shf_grid) == 1:
        axes = [axes]
    for ax, shf in zip(axes, args.shf_grid):
        m = np.full((len(args.num_pcs_grid), len(args.lhf_grid)), np.nan)
        for i, n in enumerate(args.num_pcs_grid):
            for j, l in enumerate(args.lhf_grid):
                row = next(r for r in rows if r["num_pcs"] == n and r["short_half_life"] == shf and r["long_half_life"] == l)
                m[i, j] = row["p_february"]
        im = ax.imshow(m, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=0.2)
        ax.set_xticks(range(len(args.lhf_grid)))
        ax.set_xticklabels([f"{l:g}" for l in args.lhf_grid], rotation=30)
        ax.set_yticks(range(len(args.num_pcs_grid)))
        ax.set_yticklabels(args.num_pcs_grid)
        ax.set_xlabel("long_half_life")
        ax.set_ylabel("num_pcs")
        ax.set_title(f"p_february (MK), shf={shf}")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i,j]:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Sensibilidade do detector `{args.feature}` (p_february < 0.05 = verde)", fontsize=12)
    fig.savefig(os.path.join(args.plots_dir, "pvalue_grid.png"), bbox_inches="tight")
    print(f"Heatmap salvo: {args.plots_dir}/pvalue_grid.png")

    # Estatística agregada
    n_detector = sum(1 for r in rows if r["detector"])
    print(f"\nDetector válido (p_feb<0.05 e p_apr>=0.05) em {n_detector}/{len(rows)} combinações.")
    feb_pvals = [r["p_february"] for r in rows]
    print(f"p_february — min: {min(feb_pvals):.4f}, median: {sorted(feb_pvals)[len(feb_pvals)//2]:.4f}, max: {max(feb_pvals):.4f}")
    apr_pvals = [r["p_april"] for r in rows]
    print(f"p_april — min: {min(apr_pvals):.4f}, median: {sorted(apr_pvals)[len(apr_pvals)//2]:.4f}, max: {max(apr_pvals):.4f}")


if __name__ == "__main__":
    main()
