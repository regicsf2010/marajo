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


def _positive_band(
    fft: ComponentFFT, freq_min: float, freq_max: float = float("inf")
) -> tuple[np.ndarray, np.ndarray]:
    psd = np.abs(fft.values) ** 2
    mask = (fft.freqs > freq_min) & (fft.freqs <= freq_max)
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
    freq_max: float = float("inf"),
    top_k: int = 5,
) -> SpectralFeatures:
    """Calcula todas as features de um PSD; descarta DC (e opcionalmente baixas e altas freqs)."""
    freqs, psd = _positive_band(fft, freq_min, freq_max)
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
    freq_max: float = float("inf"),
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
        feats = spectral_features(
            fft_data[cp_id], freq_min=freq_min, freq_max=freq_max, top_k=top_k
        )
        for name in FEATURE_NAMES:
            per_cp[name].append(getattr(feats, name))

    aggregated: dict[str, float] = {}
    for name in FEATURE_NAMES:
        arr = np.asarray(per_cp[name])
        aggregated[f"{name}_mean"] = float(np.mean(arr))
        aggregated[f"{name}_median"] = float(np.median(arr))
    return aggregated


def _band_mask(freqs: np.ndarray, band_low: float, band_high: float) -> np.ndarray:
    return (freqs >= band_low) & (freqs < band_high)


def band_features(fft: ComponentFFT, band_low: float, band_high: float) -> dict[str, float]:
    """Features dentro de uma banda específica [band_low, band_high) em Hz.

    Retorna:
      - `energy`: soma do PSD dentro da banda (absoluta).
      - `energy_fraction`: energia da banda / energia total (DC excluído).
      - `peak_freq`: freq do maior pico local dentro da banda (NaN se vazia).
      - `centroid`: centroide espectral local da banda (NaN se vazia).
    """
    psd_total = np.abs(fft.values) ** 2
    total_positive = float(np.sum(psd_total[fft.freqs > 0]))
    if total_positive <= 0:
        return {"energy": 0.0, "energy_fraction": 0.0, "peak_freq": float("nan"), "centroid": float("nan")}

    mask = _band_mask(fft.freqs, band_low, band_high)
    freqs_b = fft.freqs[mask]
    psd_b = psd_total[mask]

    if freqs_b.size == 0 or np.sum(psd_b) <= 0:
        return {"energy": 0.0, "energy_fraction": 0.0, "peak_freq": float("nan"), "centroid": float("nan")}

    energy = float(np.sum(psd_b))
    fraction = energy / total_positive

    peak_idx = _top_k_peak_indices(psd_b, 1)
    peak_f = float(freqs_b[peak_idx[0]])
    centroid_local = spectral_centroid(freqs_b, psd_b)

    return {
        "energy": energy,
        "energy_fraction": fraction,
        "peak_freq": peak_f,
        "centroid": centroid_local,
    }


def video_band_features(
    fft_data: dict[int, ComponentFFT],
    bands: list[tuple[float, float]],
    components: Iterable[int] | None = None,
) -> dict[str, float]:
    """Computa band_features pra cada banda × CP, agrega sobre CPs (mean/median).

    Os nomes ficam `{feature}_{low}_{high}_{mean|median}` (Hz com 1 decimal).
    NaN é tratado como missing (usa nanmean/nanmedian).
    """
    if components is None:
        components = list(fft_data.keys())

    per_band_per_cp: dict[str, list[float]] = {}
    feature_keys = ["energy", "energy_fraction", "peak_freq", "centroid"]
    for low, high in bands:
        for fk in feature_keys:
            key = f"{fk}_{low:.1f}_{high:.1f}"
            per_band_per_cp[key] = []

    for cp_id in components:
        for low, high in bands:
            feats = band_features(fft_data[cp_id], low, high)
            for fk in feature_keys:
                per_band_per_cp[f"{fk}_{low:.1f}_{high:.1f}"].append(feats[fk])

    aggregated: dict[str, float] = {}
    for key, vals in per_band_per_cp.items():
        arr = np.asarray(vals, dtype=float)
        with np.errstate(all="ignore"):
            mean_val = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else 0.0
            median_val = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else 0.0
        aggregated[f"{key}_mean"] = mean_val
        aggregated[f"{key}_median"] = median_val
    return aggregated
