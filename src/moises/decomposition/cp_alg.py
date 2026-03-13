import numpy as np
from scipy.signal import lfilter
from scipy.linalg import eigh

def cp_alg(mixtures):

    n = 10

    ###################################
    # COMPUTE V AND U
    ###################################

    shf = 1
    lhf = 900000
    max_mask_len = 50

    # -------- Short-term mask --------
    h = shf
    t = n * h

    lam = 2 ** (-1 / h)
    temp = np.arange(t).reshape(-1, 1)

    mask = lam ** temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    s_mask = mask.flatten()

    # -------- Long-term mask --------
    h = lhf
    t = n * h
    t = min(t, max_mask_len)
    t = max(t, 1)

    lam = 2 ** (-1 / h)
    temp = np.arange(t).reshape(-1, 1)

    mask = lam ** temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    l_mask = mask.flatten()

    # -------- Filtering --------
    S = lfilter(s_mask, [1], mixtures, axis=0)
    L = lfilter(l_mask, [1], mixtures, axis=0)

    # -------- Covariance matrices --------
    U = np.cov(S, rowvar=False, bias=True)
    V = np.cov(L, rowvar=False, bias=True)

    ###################################
    # Generalized eigenvalue problem
    ###################################

    d, W = eigh(V, U)
    W = np.real(W)

    ys = -(mixtures @ W)

    return ys, W