"""Damping ratio (fator Q) estimado por half-power bandwidth no PSD.

Pra cada pico identificado no espectro:
  Q = f_peak / (f_upper_-3dB - f_lower_-3dB)
  ζ ≈ 1 / (2Q)

Onde f_upper/f_lower são as frequências onde o PSD cai pra metade do valor do
pico (equivalente a -3 dB em escala logarítmica).

Pressuposto: o pico é razoavelmente isolado e não é dominado por picos vizinhos
na região de busca. Se o decaimento não acontece dentro da janela, retorna NaN.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.signal import find_peaks

from marajo.modal.fft import ComponentFFT


def half_power_indices(
    psd: np.ndarray,
    peak_idx: int,
    half_value: Optional[float] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Encontra (lower_idx, upper_idx) onde PSD cai pra `half_value` (default: PSD[peak]/2)."""
    if peak_idx < 0 or peak_idx >= len(psd):
        return None, None
    peak_value = float(psd[peak_idx])
    if peak_value <= 0:
        return None, None
    target = peak_value / 2.0 if half_value is None else half_value

    # Caminha pra esquerda enquanto PSD > target (e enquanto não vira a ladeira pra cima de novo)
    lower = None
    prev = peak_value
    for i in range(peak_idx - 1, -1, -1):
        if psd[i] <= target:
            lower = i
            break
        if psd[i] > prev:  # virou pra cima — entrou em outro pico
            break
        prev = psd[i]

    upper = None
    prev = peak_value
    for i in range(peak_idx + 1, len(psd)):
        if psd[i] <= target:
            upper = i
            break
        if psd[i] > prev:
            break
        prev = psd[i]

    return lower, upper


def damping_ratio_at_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    peak_idx: int,
) -> float:
    """ζ ≈ 1/(2Q). NaN se não for possível medir half-power dos dois lados."""
    if peak_idx < 0 or peak_idx >= len(freqs):
        return float("nan")
    f_peak = float(freqs[peak_idx])
    if f_peak <= 0:
        return float("nan")

    lower, upper = half_power_indices(psd, peak_idx)
    if lower is None or upper is None:
        return float("nan")

    f_lower = float(freqs[lower])
    f_upper = float(freqs[upper])
    bw = f_upper - f_lower
    if bw <= 0:
        return float("nan")

    q = f_peak / bw
    return 1.0 / (2.0 * q)


def damping_for_component(
    fft: ComponentFFT,
    freq_min: float = 0.0,
    freq_max: float = float("inf"),
) -> float:
    """Damping ratio do maior pico do CP, dentro da banda permitida."""
    psd = np.abs(fft.values) ** 2
    mask = (fft.freqs > freq_min) & (fft.freqs <= freq_max)
    freqs_band = fft.freqs[mask]
    psd_band = psd[mask]
    if freqs_band.size < 3:
        return float("nan")

    idx, _ = find_peaks(psd_band)
    if idx.size == 0:
        idx = np.array([int(np.argmax(psd_band))])
    top = int(idx[int(np.argmax(psd_band[idx]))])
    return damping_ratio_at_peak(freqs_band, psd_band, top)


def video_damping_features(
    fft_data: dict[int, ComponentFFT],
    freq_min: float = 0.0,
    freq_max: float = float("inf"),
    components: Iterable[int] | None = None,
) -> dict[str, float]:
    """Damping ratio do top-peak por CP, agregado por mean/median (nan-aware)."""
    if components is None:
        components = list(fft_data.keys())

    per_cp = [damping_for_component(fft_data[i], freq_min, freq_max) for i in components]
    arr = np.asarray(per_cp, dtype=float)

    if np.all(np.isnan(arr)):
        return {"damping_mean": float("nan"), "damping_median": float("nan")}

    with np.errstate(all="ignore"):
        return {
            "damping_mean": float(np.nanmean(arr)),
            "damping_median": float(np.nanmedian(arr)),
        }
