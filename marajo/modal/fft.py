"""FFT dos sinais separados pelo Complexity Pursuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ComponentFFT:
    signal: np.ndarray  # série temporal da fonte
    freqs: np.ndarray   # eixo de frequências (rfft)
    values: np.ndarray  # FFT complexa (mantém fase pra plot)


def compute_fft(signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """rfft do sinal sem janela, sem normalização, sem remoção de DC.

    A normalização e janelamento ficam a cargo de quem consome (o baseline original
    mantém comentado pra preservar amplitudes brutas).
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    values = np.fft.rfft(signal)
    return freqs, values


def compute_fft_for_components(
    unmixed: np.ndarray,
    fps: float,
    components: Iterable[int] | None = None,
) -> dict[int, ComponentFFT]:
    """Aplica `compute_fft` em cada coluna selecionada de `unmixed` (n_frames, n_pc)."""
    if components is None:
        components = range(unmixed.shape[1])

    result: dict[int, ComponentFFT] = {}
    for i in components:
        signal = np.asarray(unmixed[:, i]).ravel()
        freqs, values = compute_fft(signal, fps)
        result[int(i)] = ComponentFFT(signal=signal, freqs=freqs, values=values)
    return result
