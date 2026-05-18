"""Plots da evolução das frequências ao longo dos dias."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_components_over_time(
    freq_per_component: dict[int, list[float]],
    save_path: Optional[str] = None,
    w: float = 20,
    h: float = 10,
):
    """Uma curva por componente CP — frequência dominante ao longo dos dias."""
    n_days = len(next(iter(freq_per_component.values())))
    days = np.arange(1, n_days + 1)

    fig, ax = plt.subplots(figsize=(w, h))
    for comp_id, values in freq_per_component.items():
        ax.plot(days, values, marker="o", label=f"CP {comp_id}")
    ax.set_xlabel("Dia", fontsize=22)
    ax.set_ylabel("Frequência (Hz)", fontsize=22)
    ax.set_title("Evolução das frequências de maior intensidade ao longo dos dias", fontsize=22)
    ax.set_xticks(days)
    ax.legend(loc="upper right", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_mean_over_time(
    mean_freqs: Sequence[float],
    save_path: Optional[str] = None,
    w: float = 20,
    h: float = 10,
):
    """Frequência média (sobre componentes) por dia, com interpolação spline cúbica."""
    days = np.arange(1, len(mean_freqs) + 1)
    y = np.asarray(mean_freqs)

    days_smooth = np.linspace(days.min(), days.max(), 300)
    spline = make_interp_spline(days, y, k=3)
    y_smooth = spline(days_smooth)

    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(days_smooth, y_smooth, label="CP mean (interpolado)")
    ax.scatter(days, y, label="Pontos originais")
    ax.set_xlabel("Dia", fontsize=22)
    ax.set_ylabel("Frequência (Hz)", fontsize=22)
    ax.set_title("Evolução da frequência média de maior intensidade ao longo dos dias", fontsize=22)
    ax.set_xticks(days)
    ax.legend(loc="upper right", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_batches_over_time(
    mean_freqs_by_batch: dict[str, Sequence[float]],
    save_path: Optional[str] = None,
    title: str = "Evolução da frequência média ao longo dos dias por batch",
    w: float = 20,
    h: float = 10,
):
    """Uma curva interpolada por batch (ex.: february vs april).

    Lê o número de pontos de cada batch independentemente; se forem iguais,
    todos aparecem sobre o mesmo eixo de dias.
    """
    if not mean_freqs_by_batch:
        raise ValueError("mean_freqs_by_batch vazio.")

    fig, ax = plt.subplots(figsize=(w, h))
    max_n = 0
    for name, values in mean_freqs_by_batch.items():
        y = np.asarray(values)
        n = len(y)
        max_n = max(max_n, n)
        days = np.arange(1, n + 1)
        days_smooth = np.linspace(days.min(), days.max(), 300)
        spline = make_interp_spline(days, y, k=min(3, n - 1))
        ax.plot(days_smooth, spline(days_smooth), label=f"{name} (interpolado)")
        ax.scatter(days, y, label=f"{name} (pontos)")

    ax.set_xlabel("Dia", fontsize=22)
    ax.set_ylabel("Frequência (Hz)", fontsize=22)
    ax.set_title(title, fontsize=22)
    ax.set_xticks(np.arange(1, max_n + 1))
    ax.legend(loc="upper right", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
