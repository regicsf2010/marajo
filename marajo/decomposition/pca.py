"""PCA aplicado em datasets de vídeo (frames × pixels)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    coeff: np.ndarray  # (n_frames, n_components) — padrões temporais (H)
    score: np.ndarray  # (n_pixels, n_components) — coeficientes espaciais (W)
    latent: np.ndarray  # (n_components,) — variância explicada (V)


def compute_pca(
    dataset: np.ndarray,
    remove_mean: bool = True,
    n_components: int | None = None,
) -> PCAResult:
    """Computa PCA sobre o dataset de frames.

    `dataset` tem shape (n_frames, n_pixels). Internamente roda PCA em `dataset.T`
    pra extrair padrões temporais como componentes principais.

    `n_components` cappa o resultado: passar o número de PCs que você realmente vai
    consumir (ex.: `decomposition.num_pcs`) reduz drasticamente a memória do `score`,
    que escala com `n_pixels × n_components`. Default None preserva todos
    (necessário pro plot diagnóstico de autovalores).
    """
    X = dataset.copy()
    if remove_mean:
        X -= X.mean(axis=0, keepdims=True)

    model = PCA(n_components=n_components)
    score = model.fit_transform(X.T)
    coeff = model.components_.T
    latent = model.explained_variance_

    return PCAResult(coeff=coeff, score=score, latent=latent)


def num_components_for_variance(latent: np.ndarray, threshold: float) -> int:
    """Quantos PCs são necessários pra atingir `threshold` da variância acumulada.

    `threshold` aceita (0, 1] ou (0, 100] (percentual).
    """
    V = np.asarray(latent, dtype=np.float64).ravel()
    if V.size == 0:
        raise ValueError("latent está vazio.")
    if np.any(V < 0):
        raise ValueError("Autovalores em latent devem ser não negativos.")

    t = threshold / 100.0 if threshold > 1 else threshold
    if not (0 < t <= 1):
        raise ValueError("threshold deve estar em (0, 1] ou (0, 100].")

    explained_ratio = V / np.sum(V)
    cumulative = np.cumsum(explained_ratio)
    return int(np.searchsorted(cumulative, t) + 1)
