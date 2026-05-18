"""Plots de espectros: FFT, PSD/phase, espectrograma."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from marajo.io.video import VideoInfo
from marajo.modal.fft import ComponentFFT
from marajo.modal.peaks import PeakInfo
from marajo.modal.spectrogram import SpectrogramResult


def plot_freq(
    freqs: np.ndarray,
    fft_vals: np.ndarray,
    title: str = "Amplitudes of frequencies",
    save_path: Optional[str] = None,
    w: float = 12,
    h: float = 8,
):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(freqs, fft_vals, linestyle="-", color="k", linewidth=2, label="signal")
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("frequency (Hz)", fontsize=22)
    ax.set_ylabel("amplitude", fontsize=22)
    ax.legend(loc="upper right", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_freqs(
    freqs_list: list[np.ndarray],
    fft_vals_list: list[np.ndarray],
    names: list[str],
    title: str = "Amplitudes of frequencies",
    save_path: Optional[str] = None,
    w: float = 19,
    h: float = 8,
):
    colors = ["k", "r", "b", "g", "c"]
    fig, ax = plt.subplots(figsize=(w, h))
    for i, (x, y) in enumerate(zip(freqs_list, fft_vals_list)):
        ax.plot(x, y, linestyle="-", color=colors[i % len(colors)], linewidth=2, label=names[i])
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("frequency (Hz)", fontsize=22)
    ax.set_ylabel("amplitude", fontsize=22)
    ax.legend(loc="upper right", fancybox=True, shadow=True, fontsize=20)
    ax.tick_params(axis="both", labelsize=22)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_source_psd_phase(
    fft_data: dict[int, ComponentFFT],
    peaks_info: dict[int, PeakInfo],
    video: VideoInfo,
    components: Sequence[int],
    save_path: Optional[str] = None,
    w: float = 20,
    h: float = 15,
):
    """Grid (n_components × 3): sinal temporal, PSD com top peaks, fase."""
    t = np.arange(video.frames) / video.fps

    fig, axes = plt.subplots(len(components), 3, figsize=(w, h), constrained_layout=True)
    if len(components) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, comp_id in enumerate(components):
        comp = fft_data[comp_id]
        psd = np.abs(comp.values) ** 2
        phase = np.angle(comp.values) ** 2

        mask = comp.freqs > 0
        freq_pos = comp.freqs[mask]
        psd_pos = psd[mask]
        phase_pos = phase[mask]

        info = peaks_info[comp_id]
        peak_text = "\n".join(
            f"freq: {f:.2f} Hz | amp: {a:.2f}"
            for f, a in zip(info.top_freqs, info.top_amps)
        )

        ax = axes[row, 0]
        ax.plot(t, comp.signal, linewidth=1.5)
        ax.set_title(f"Source {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

        ax = axes[row, 1]
        ax.plot(freq_pos, psd_pos, linewidth=1.5, color="k")
        ax.plot(info.top_freqs, info.top_amps, "ro", markersize=4)
        ax.text(
            0.5, 0.5, peak_text,
            transform=ax.transAxes, fontsize=9, ha="left", va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.set_title(f"PSD {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

        ax = axes[row, 2]
        ax.plot(freq_pos, phase_pos, linewidth=1.5, color="k")
        ax.set_title(f"Phase {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[-1, 2].set_xlabel("Frequency (Hz)")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_spectrogram(
    spec: SpectrogramResult,
    title: str = "Espectrograma",
    save_path: Optional[str] = None,
    w: float = 19,
    h: float = 10,
):
    fig, ax = plt.subplots(figsize=(w, h))
    pcm = ax.pcolormesh(spec.times, spec.freqs, spec.Sxx, shading="gouraud")
    ax.set_ylabel("Frequência (Hz)")
    ax.set_xlabel("Tempo (s)")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="Intensidade")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
