import numpy as np


def pca(
    matrix: np.ndarray
):

    X = matrix - np.mean(matrix, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    coeff = Vt.T
    score = U * S
    latent = (S**2) / (X.shape[0] - 1)

    return coeff, score, latent