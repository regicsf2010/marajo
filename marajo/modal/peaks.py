"""Detecção dos maiores picos espectrais por componente."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from marajo.modal.fft import ComponentFFT


@dataclass
class PeakInfo:
    highest_freq: float
    highest_amp: float
    top_freqs: np.ndarray
    top_amps: np.ndarray
    top_indices: np.ndarray


def highest_peak_frequencies(
    fft_data: dict[int, ComponentFFT],
    n_peaks: int = 5,
) -> dict[int, PeakInfo]:
    """Para cada componente, identifica os n_peaks maiores picos do PSD nas frequências positivas.

    Mantém a heurística do baseline: se find_peaks não retornar nada, usa argmax do PSD positivo.
    """
    peaks: dict[int, PeakInfo] = {}
    for comp_id, comp in fft_data.items():
        psd = np.abs(comp.values) ** 2
        mask = comp.freqs > 0
        freq_pos = comp.freqs[mask]
        psd_pos = psd[mask]

        idx, _ = find_peaks(psd_pos)
        if idx.size == 0:
            idx = np.array([int(np.argmax(psd_pos))])

        sorted_idx = idx[np.argsort(psd_pos[idx])[::-1]]
        top_idx = sorted_idx[:n_peaks]

        peaks[comp_id] = PeakInfo(
            highest_freq=float(freq_pos[top_idx[0]]),
            highest_amp=float(psd_pos[top_idx[0]]),
            top_freqs=freq_pos[top_idx],
            top_amps=psd_pos[top_idx],
            top_indices=top_idx,
        )
    return peaks


def top_n_peaks(
    freqs: np.ndarray,
    fft_vals: np.ndarray,
    n: int = 3,
    min_freq: float = 0.5,
) -> list[tuple[float, float]]:
    """Versão escalar (não-PSD) do detector — usa magnitude direta e corta DC/baixa freq.

    Equivalente ao `get_top_n_peaks` do baseline (Functions.py).
    """
    mask = freqs > min_freq
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    peaks, _ = find_peaks(fft_vals)
    if peaks.size == 0:
        return []

    sorted_peaks = peaks[np.argsort(fft_vals[peaks])[::-1]]
    top = sorted_peaks[:n]
    return [(float(freqs[i]), float(fft_vals[i])) for i in top]
