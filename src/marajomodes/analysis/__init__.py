"""Módulo de análise: extração de sinal e análise espectral (FFT, picos)."""

from marajomodes.analysis.signal import extract_signal
from marajomodes.analysis.spectral import compute_fft, get_top_n_peaks

__all__ = [
    "extract_signal",
    "compute_fft",
    "get_top_n_peaks",
]
