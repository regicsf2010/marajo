"""Espectrograma de um sinal temporal (wrapper fino sobre scipy.signal.spectrogram)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import spectrogram


@dataclass
class SpectrogramResult:
    freqs: np.ndarray  # (n_freq,)
    times: np.ndarray  # (n_time,)
    Sxx: np.ndarray    # (n_freq, n_time)


def compute_spectrogram(
    signal: np.ndarray,
    fps: float,
    nperseg: int = 320,
    noverlap: int = 192,
    window: str = "hann",
) -> SpectrogramResult:
    f, t, Sxx = spectrogram(signal, fs=fps, window=window, nperseg=nperseg, noverlap=noverlap)
    return SpectrogramResult(freqs=f, times=t, Sxx=Sxx)
