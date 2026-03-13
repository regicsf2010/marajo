import numpy as np
from scipy.signal import lfilter
from scipy.linalg import eig


def cp_alg(
    mixtures: np.ndarray,
    n_short: int = 10,
    shf: float = 1.0,
    lhf: float = 900_000.0,
    max_mask_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separação cega de fontes por Complexity Pursuit (CP).

    Usa máscaras de curto e longo prazo para definir covariâncias U (longo)
    e V (curto) e resolve o problema generalizado de autovalores V·w = λ·U·w.
    Equivalente à função MATLAB CP_alg.m.

    Parâmetros
    ----------
    mixtures : np.ndarray
        Matriz (n_amostras, n_canais), cada coluna é uma mistura (ex.: scores PCA).
    n_short : int, opcional
        Parâmetro n das máscaras (padrão 10).
    shf : float, opcional
        Meia-vida curta para a máscara de curto prazo (padrão 1).
    lhf : float, opcional
        Meia-vida longa para a máscara de longo prazo (padrão 900000).
    max_mask_len : int, opcional
        Comprimento máximo da máscara de convolução (padrão 50).

    Retorna
    -------
    unmixed : np.ndarray
        Fontes separadas (n_amostras, n_canais), com convenção flip/neg como no MATLAB.
    W : np.ndarray
        Matriz de separação (n_canais, n_canais).
    """
    # Máscara de curto prazo
    h = shf
    t = min(int(n_short * h), max_mask_len)
    t = max(t, 1)
    lam = 2.0 ** (-1.0 / h)
    temp = np.arange(t, dtype=float)
    mask = np.power(lam, temp)
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1.0
    s_mask = mask

    # Máscara de longo prazo
    h = lhf
    t = min(int(n_short * h), max_mask_len)
    t = max(t, 1)
    lam = 2.0 ** (-1.0 / h)
    temp = np.arange(t, dtype=float)
    mask = np.power(lam, temp)
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1.0
    l_mask = mask

    # Filtrar cada coluna (eixo 0)
    S = lfilter(s_mask, [1.0], mixtures, axis=0)
    L = lfilter(l_mask, [1.0], mixtures, axis=0)

    # Covariância: entre colunas (canais); ddof=0 = população, igual a cov(., 1) no MATLAB
    U = np.cov(L, rowvar=False, ddof=0)
    V = np.cov(S, rowvar=False, ddof=0)

    # Garantir simetria (erros numéricos)
    U = (U + U.T) / 2
    V = (V + V.T) / 2

    # Autovalores generalizados: V·w = λ·U·w  (scipy: A·v = λ·B·v)
    eigs = eig(V, U)
    eigenvalues = eigs[0]
    W = np.real(eigs[1])

    # Fontes (equivalente a ys = -(mixtures*W) no MATLAB)
    ys = -(mixtures @ W)

    return ys, W
