"""Features espectrais aplicadas a cada `ComponentFFT`.

Motivação: o baseline (`highest_peak_frequencies`) reduz cada PSD a 1 número (a
freq do maior pico). Isso descarta a forma do espectro. Aqui ampliamos pra um
vetor de features que captura distribuição, concentração e localização de
energia, mantendo `highest_freq` na lista pra ancorar comparação com o 001.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks

from marajo.modal.fft import ComponentFFT


@dataclass
class SpectralFeatures:
    """Features escalares extraídas de um único PSD (uma fonte CP)."""

    # baseline (replicado do 001 pra comparação direta)
    highest_freq: float

    # localização / "primeiro momento" do espectro
    centroid: float            # média ponderada de f por amplitude
    spread: float              # bandwidth: desvio em torno do centroide

    # forma do espectro
    flatness: float            # geom_mean(psd) / arith_mean(psd) ∈ [0,1]; 1 = ruído branco, 0 = tom puro

    # concentração nos top-K picos
    top_k_freq_mean: float     # média das K maiores freqs (descarta DC via freq_min)
    energy_concentration: float  # fração da energia total contida nos K maiores picos


def _positive_band(fft: ComponentFFT, freq_min: float) -> tuple[np.ndarray, np.ndarray]:
    psd = np.abs(fft.values) ** 2
    mask = fft.freqs > freq_min
    return fft.freqs[mask], psd[mask]


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    total = np.sum(psd)
    return float(np.sum(freqs * psd) / total) if total > 0 else 0.0


def spectral_spread(freqs: np.ndarray, psd: np.ndarray, centroid: float | None = None) -> float:
    if centroid is None:
        centroid = spectral_centroid(freqs, psd)
    total = np.sum(psd)
    if total <= 0:
        return 0.0
    var = np.sum(((freqs - centroid) ** 2) * psd) / total
    return float(np.sqrt(var))


def spectral_flatness(psd: np.ndarray) -> float:
    """Medida de "ruidicidade" do espectro: 1 = ruído branco, 0 = tom puro."""
    psd = psd[psd > 0]  # evita log(0)
    if psd.size == 0:
        return 0.0
    geom = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(geom / arith) if arith > 0 else 0.0


def _top_k_peak_indices(psd: np.ndarray, k: int) -> np.ndarray:
    """Top-k picos LOCAIS por amplitude. Fallback pro argmax global se não achar locais."""
    idx, _ = find_peaks(psd)
    if idx.size == 0:
        idx = np.array([int(np.argmax(psd))])
    sorted_idx = idx[np.argsort(psd[idx])[::-1]]
    return sorted_idx[:k]


def top_k_freq_mean(freqs: np.ndarray, psd: np.ndarray, k: int) -> float:
    idx = _top_k_peak_indices(psd, k)
    return float(np.mean(freqs[idx]))


def energy_concentration(psd: np.ndarray, k: int) -> float:
    idx = _top_k_peak_indices(psd, k)
    total = np.sum(psd)
    return float(np.sum(psd[idx]) / total) if total > 0 else 0.0


def highest_freq(freqs: np.ndarray, psd: np.ndarray) -> float:
    idx = _top_k_peak_indices(psd, 1)
    return float(freqs[idx[0]])


def spectral_features(
    fft: ComponentFFT,
    freq_min: float = 0.0,
    top_k: int = 5,
) -> SpectralFeatures:
    """Calcula todas as features de um PSD; descarta DC (e opcionalmente baixas freqs)."""
    freqs, psd = _positive_band(fft, freq_min)
    if freqs.size == 0:
        return SpectralFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    centroid = spectral_centroid(freqs, psd)
    spread = spectral_spread(freqs, psd, centroid)
    flatness = spectral_flatness(psd)
    return SpectralFeatures(
        highest_freq=highest_freq(freqs, psd),
        centroid=centroid,
        spread=spread,
        flatness=flatness,
        top_k_freq_mean=top_k_freq_mean(freqs, psd, top_k),
        energy_concentration=energy_concentration(psd, top_k),
    )


FEATURE_NAMES: list[str] = [f.name for f in fields(SpectralFeatures)]


def video_features(
    fft_data: dict[int, ComponentFFT],
    freq_min: float = 0.0,
    top_k: int = 5,
    components: Iterable[int] | None = None,
) -> dict[str, float]:
    """Agrega features dos N CPs num vetor único por vídeo.

    Pra cada feature, gera duas variantes:
      - `{feature}_mean`: média aritmética sobre os CPs.
      - `{feature}_median`: mediana sobre os CPs (robusta a outlier).

    Convenção: a feature `highest_freq_mean` é a métrica exata do baseline (001),
    então o veredito do 001 sai dessa coluna.
    """
    if components is None:
        components = list(fft_data.keys())

    per_cp: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for cp_id in components:
        feats = spectral_features(fft_data[cp_id], freq_min=freq_min, top_k=top_k)
        for name in FEATURE_NAMES:
            per_cp[name].append(getattr(feats, name))

    aggregated: dict[str, float] = {}
    for name in FEATURE_NAMES:
        arr = np.asarray(per_cp[name])
        aggregated[f"{name}_mean"] = float(np.mean(arr))
        aggregated[f"{name}_median"] = float(np.median(arr))
    return aggregated
