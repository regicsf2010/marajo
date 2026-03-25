from __future__ import annotations

import numpy as np


def _cuda_svd_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.is_available())
    except ImportError:
        return False


def pca(
    matrix: np.ndarray,
    *,
    use_cuda: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD. When CuPy is installed and a CUDA device is available, the SVD
    runs on the GPU unless ``use_cuda=False``.

    Parameters
    ----------
    matrix
        Data matrix (same layout as before: features × observations or as you pass today).
    use_cuda
        If True, require CuPy + CUDA. If False, always use NumPy on CPU.
        If None (default), use CUDA when available, otherwise CPU.
    """
    if use_cuda is None:
        use_cuda = _cuda_svd_available()
    elif use_cuda and not _cuda_svd_available():
        raise RuntimeError(
            "pca(use_cuda=True) but CuPy is not installed or no CUDA device is available. "
            "Install a matching wheel, e.g. `pip install cupy-cuda12x` for CUDA 12.x."
        )

    if use_cuda:
        print('using cuda')
        import cupy as cp

        arr = np.asarray(matrix)
        dtype = arr.dtype if arr.dtype.kind in "fc" else np.float64
        X = cp.asarray(arr, dtype=dtype)
        X = X - cp.mean(X, axis=0, keepdims=True)
        U, S, Vt = cp.linalg.svd(X, full_matrices=False)
        U = cp.asnumpy(U)
        S = cp.asnumpy(S)
        Vt = cp.asnumpy(Vt)
        n_obs = int(X.shape[0])
    else:
        print('no cuda')
        X = matrix - np.mean(matrix, axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        n_obs = X.shape[0]

    coeff = Vt.T
    score = U * S
    latent = (S**2) / (n_obs - 1)

    return coeff, score, latent
