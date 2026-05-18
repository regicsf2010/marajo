"""Complexity Pursuit (CP_alg): BSS via filtros short/long half-life + autovalor generalizado.

Referência: implementação original em CompVis.CP_alg do baseline do Reginaldo,
inspirada no método de Stone (2001) — Blind Source Separation Using Temporal Predictability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig
from scipy.signal import lfilter

from marajo.config import CPConfig
from marajo.decomposition.pca import PCAResult


@dataclass
class CPResult:
    unmixed: np.ndarray  # (n_frames, n_pc) — fontes separadas
    W_mix: np.ndarray   # (n_pc, n_pc) — matriz de demistura (raw, antes do fliplr/inv)
    W_inv: np.ndarray   # (n_pc, n_pc) — fliplr(inv(W_mix)), usado pra mode shapes


def _build_half_life_mask(half_life: float, n_horizon: int, max_len: int) -> np.ndarray:
    t = int(n_horizon * half_life)
    t = min(t, max_len)
    t = max(t, 1)
    lam = 2.0 ** (-1.0 / half_life)
    temp = np.arange(0, t)
    mask = lam**temp
    mask[0] = 0.0
    s = np.sum(np.abs(mask))
    if s > 0:
        mask = mask / s
    mask[0] = -1.0
    return mask


def complexity_pursuit(mixtures: np.ndarray, config: CPConfig | None = None) -> CPResult:
    """Executa o Complexity Pursuit em `mixtures` (n_frames, n_pc)."""
    cfg = config or CPConfig()

    s_mask = _build_half_life_mask(cfg.short_half_life, cfg.n_mask_horizon, cfg.max_mask_len)
    l_mask = _build_half_life_mask(cfg.long_half_life, cfg.n_mask_horizon, cfg.max_mask_len)

    S = lfilter(s_mask, 1, mixtures, axis=0)
    L = lfilter(l_mask, 1, mixtures, axis=0)

    U = np.cov(S, rowvar=False, bias=True)
    V = np.cov(L, rowvar=False, bias=True)

    _, W = eig(V, U)
    W = np.real(W)

    ys = -(mixtures @ W)

    W_inv = np.fliplr(np.linalg.inv(W))
    unmixed = -np.fliplr(ys)

    return CPResult(unmixed=unmixed, W_mix=W, W_inv=W_inv)


def run_cp_on_components(pca: PCAResult, n_pc: int, config: CPConfig | None = None) -> CPResult:
    """Aplica o CP nos primeiros `n_pc` padrões temporais do PCA."""
    mixtures = pca.coeff[:, :n_pc]
    return complexity_pursuit(mixtures, config)
