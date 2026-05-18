"""Modal matching via Modal Assurance Criterion (MAC).

MAC(φ_a, φ_b) = |φ_a^T φ_b|² / ((φ_a^T φ_a) (φ_b^T φ_b)) ∈ [0, 1]
  - MAC=1: modos colineares (mesmo modo físico).
  - MAC=0: modos ortogonais (modos diferentes).

Uso típico:
  Pra cada par de vídeos (A, B), monta matriz MAC[i, j] de todos os modos.
  Hungarian algorithm encontra permutação ótima que maximiza soma dos MACs —
  associa CP_i de A ao CP_j de B que provavelmente são o MESMO modo físico.

Detalhe operacional: mode shapes são derivados de ROIs com dimensões
diferentes entre vídeos. Pra comparar via MAC, é necessário interpolar pra
tamanho espacial comum (`resize_modes`).
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment


def resize_modes(
    modes: np.ndarray,
    n_rows: int,
    n_cols: int,
    target_rows: int,
    target_cols: int,
) -> np.ndarray:
    """Interpola mode shapes pra dimensão espacial comum.

    `modes`: (n_pixels, n_modes) com n_pixels = n_rows × n_cols.
    Retorna: (target_rows × target_cols, n_modes) também achatado.
    """
    n_modes = modes.shape[1]
    out = np.empty((target_rows * target_cols, n_modes), dtype=modes.dtype)
    for i in range(n_modes):
        img = modes[:, i].reshape(n_rows, n_cols)
        resized = cv.resize(img, (target_cols, target_rows), interpolation=cv.INTER_LINEAR)
        out[:, i] = resized.reshape(-1)
    return out


def mac_matrix(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    """Matriz MAC entre dois conjuntos de mode shapes.

    `modes_a`: (n_pixels, n_a). `modes_b`: (n_pixels, n_b). Devem ter o MESMO n_pixels.
    Retorna: (n_a, n_b) com cada entrada em [0, 1].
    """
    if modes_a.shape[0] != modes_b.shape[0]:
        raise ValueError(
            f"modes_a e modes_b devem ter o mesmo n_pixels; got {modes_a.shape[0]} vs {modes_b.shape[0]}"
        )
    num = (modes_a.T @ modes_b) ** 2
    norm_a = np.sum(modes_a**2, axis=0)
    norm_b = np.sum(modes_b**2, axis=0)
    denom = np.outer(norm_a, norm_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        mac = np.where(denom > 0, num / denom, 0.0)
    return mac


def hungarian_match(mac: np.ndarray) -> np.ndarray:
    """Permutação ótima que maximiza soma dos MACs.

    Retorna `col_ind` tal que `col_ind[i]` é o melhor match em B para A[i].
    """
    row_ind, col_ind = linear_sum_assignment(-mac)
    return col_ind


def align_to_reference(
    modes_list: list[np.ndarray],
    reference_idx: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Alinha cada conjunto de modes ao conjunto de referência via Hungarian.

    Retorna:
      - permutations: lista de arrays, cada um sendo a permutação aplicada ao
        respectivo conjunto pra alinhar ao reference.
      - macs: lista de arrays com os MAC values escolhidos pelo matching
        (sinaliza quão bom é o alinhamento).
    """
    ref = modes_list[reference_idx]
    permutations: list[np.ndarray] = []
    macs: list[np.ndarray] = []
    for i, m in enumerate(modes_list):
        mac = mac_matrix(ref, m)
        perm = hungarian_match(mac)
        permutations.append(perm)
        macs.append(mac[np.arange(len(perm)), perm])
    return permutations, macs
