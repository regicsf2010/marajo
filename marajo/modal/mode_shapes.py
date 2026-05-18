"""Reconstrução espacial dos modos a partir do score do PCA e do W inverso do CP."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_mode_shapes(
    W_inv: np.ndarray,
    pca_score: np.ndarray,
    num_pcs: int,
    srcs: Sequence[int],
) -> np.ndarray:
    """Calcula mode shapes selecionados.

    Equivalente ao MATLAB:
        mode_shapes = (Winvmix * W(:,1:numPC)')';
        mode_shapes = mode_shapes(:,srcs);

    Args:
        W_inv: (n_pc, n_pc) — `fliplr(inv(W))` do Complexity Pursuit.
        pca_score: (n_pixels, n_components) — `score` do PCA.
        num_pcs: número de componentes principais usados no CP.
        srcs: índices das fontes (0-based) a retornar.

    Returns:
        Array (n_pixels, len(srcs)) com cada coluna sendo um mode shape achatado.
    """
    mode_shapes = (W_inv @ pca_score[:, :num_pcs].T).T
    return mode_shapes[:, list(srcs)]
