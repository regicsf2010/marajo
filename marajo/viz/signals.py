"""Plots de séries temporais (sinais brutos / fontes separadas)."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(
    signal: np.ndarray,
    title: str = "Centroid of x position",
    xlabel: str = "frame",
    ylabel: str = "cx",
    save_path: Optional[str] = None,
    w: float = 19,
    h: float = 8,
):
    x = np.arange(len(signal)) + 1
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(x, signal, linestyle="-", color="k", linewidth=2, label="signal")
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.legend(loc="upper left", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_signals(
    signals: list[np.ndarray],
    names: list[str],
    title: str = "Centroid of x position",
    xlabel: str = "frame",
    ylabel: str = "cx",
    save_path: Optional[str] = None,
    w: float = 19,
    h: float = 8,
):
    x = np.arange(len(signals[0])) + 1
    colors = ["k", "r", "b", "g", "c"]
    fig, ax = plt.subplots(figsize=(w, h))
    for i, signal in enumerate(signals):
        ax.plot(x, signal, linestyle="-", color=colors[i % len(colors)], linewidth=2, label=names[i])
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.legend(loc="upper left", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
