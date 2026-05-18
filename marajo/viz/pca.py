"""Plot diagnóstico do PCA (autovalores + variância acumulada)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_pca_eigenvalues_with_threshold(
    latent: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    w: float = 15,
    h: float = 8,
) -> int:
    """Plota autovalores e variância acumulada com linha do limiar. Retorna nº de PCs no limiar."""
    V = np.asarray(latent, dtype=np.float64).ravel()
    if V.size == 0:
        raise ValueError("latent está vazio.")
    if np.any(V < 0):
        raise ValueError("Autovalores devem ser não negativos.")

    t = threshold / 100.0 if threshold > 1 else threshold
    if not (0 < t <= 1):
        raise ValueError("threshold deve estar em (0, 1] ou (0, 100].")

    explained_ratio = V / np.sum(V)
    cumulative = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cumulative, t) + 1)
    pc_idx = n_components - 1

    components = np.arange(1, len(V) + 1)

    fig, ax1 = plt.subplots(figsize=(w, h))
    ax1.plot(components, V, marker="o", linewidth=2, label="Autovalores")
    ax1.axvline(n_components, linestyle="--", linewidth=1.2, label=f"PC = {n_components}")
    ax1.plot(components[pc_idx], V[pc_idx], "ro")
    ax1.set_xlabel("Componente principal", fontsize=20)
    ax1.set_ylabel("Autovalor", fontsize=20)
    ax1.set_title("Autovalores do PCA e variância acumulada", fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=22)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, marker="s", color="r", linewidth=2, label="Variância acumulada")
    ax2.axhline(t, linestyle="--", linewidth=1.2, label=f"Limiar = {t:.2%}")
    ax2.plot(components[pc_idx], cumulative[pc_idx], "bs")
    ax2.set_ylabel("Variância acumulada", fontsize=20)
    ax2.text(
        n_components, cumulative[pc_idx],
        f"  PC {n_components}\n  acum = {cumulative[pc_idx]:.2%}",
        va="bottom", ha="right", fontsize=10,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
               fancybox=True, shadow=True, fontsize=20)
    ax1.tick_params(axis="both", labelsize=22)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return n_components
