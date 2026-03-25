from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.signal import lfilter


def _cuda_cp_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.is_available())
    except ImportError:
        return False


def _short_long_masks() -> tuple[np.ndarray, np.ndarray]:
    n = 10

    shf = 1
    lhf = 900000
    max_mask_len = 50

    h = shf
    t = n * h

    lam = 2 ** (-1 / h)
    temp = np.arange(t).reshape(-1, 1)

    mask = lam**temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    s_mask = mask.flatten()

    h = lhf
    t = n * h
    t = min(t, max_mask_len)
    t = max(t, 1)

    lam = 2 ** (-1 / h)
    temp = np.arange(t).reshape(-1, 1)

    mask = lam**temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    l_mask = mask.flatten()

    return s_mask.astype(np.float64), l_mask.astype(np.float64)


def _cp_alg_numpy(
    mixtures: np.ndarray, s_mask: np.ndarray, l_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    S = lfilter(s_mask, [1], mixtures, axis=0)
    L = lfilter(l_mask, [1], mixtures, axis=0)

    U = np.cov(S, rowvar=False, bias=True)
    V = np.cov(L, rowvar=False, bias=True)

    _, W = eigh(V, U)
    W = np.real(W)

    ys = -(mixtures @ W)
    return ys, W


def _cp_alg_cuda(
    mixtures: np.ndarray, s_mask: np.ndarray, l_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    print('Running cp with cuda')

    import cupy as cp

    # Importing cupyx.scipy.signal pulls in cupyx.jit, which emits an experimental-API warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from cupyx.scipy.signal import lfilter as cp_lfilter

    dtype = (
        mixtures.dtype
        if mixtures.dtype in (np.float32, np.float64)
        else np.float64
    )
    X = cp.asarray(mixtures, dtype=dtype)
    b_s = cp.asarray(s_mask, dtype=dtype)
    b_l = cp.asarray(l_mask, dtype=dtype)
    a = cp.asarray([1.0], dtype=dtype)

    S = cp_lfilter(b_s, a, X, axis=0)
    L = cp_lfilter(b_l, a, X, axis=0)

    U = cp.cov(S, rowvar=False, bias=True)
    V = cp.cov(L, rowvar=False, bias=True)

    _, W = eigh(cp.asnumpy(V), cp.asnumpy(U))
    W = np.real(W)

    Wg = cp.asarray(W, dtype=dtype)
    ys = -(X @ Wg)
    ys = cp.asnumpy(cp.real(ys))
    return ys, W


def cp_alg(
    mixtures: np.ndarray,
    *,
    use_cuda: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CP-style demixing: band-split via IIR masks, generalized eigenvectors, then projection.

    When CuPy is installed and a CUDA device is available, the heavy steps (filtering,
    covariances, final projection) run on the GPU unless ``use_cuda=False``. The
    generalized eigenproblem is solved on the CPU (small ``n_features × n_features``).

    Parameters
    ----------
    mixtures
        Shape ``(time, features)``.
    use_cuda
        If True, require CuPy + CUDA. If False, NumPy/SciPy on CPU only.
        If None (default), use CUDA when available.
    """
    mixtures = np.asarray(mixtures)
    s_mask, l_mask = _short_long_masks()

    if use_cuda is None:
        use_cuda = _cuda_cp_available()
    elif use_cuda and not _cuda_cp_available():
        raise RuntimeError(
            "cp_alg(use_cuda=True) but CuPy is not installed or no CUDA device is available. "
            "Install a matching wheel, e.g. `pip install cupy-cuda12x` for CUDA 12.x."
        )

    if use_cuda:
        return _cp_alg_cuda(mixtures, s_mask, l_mask)
    return _cp_alg_numpy(mixtures, s_mask, l_mask)
