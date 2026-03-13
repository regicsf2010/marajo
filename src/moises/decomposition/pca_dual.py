import numpy as np
from sklearn.decomposition import PCA


def pca_dual(
    X0: np.ndarray,
    X90: np.ndarray,
):
    """
    Aplica PCA à parte real (X0) e à parte imaginária (X90) e combina por variância.

    Em MATLAB: pca(X') com X (T×M) trata linhas como observações; aqui usamos
    X.T (M×T) para obter scores espaciais (H) e loadings temporais (W).
    As variâncias de ambos os PCAs são concatenadas e usadas para ordenar
    os componentes combinados (H e W).

    Parâmetros
    ----------
    X0 : np.ndarray
        Parte real (n_frames, n_pixels).
    X90 : np.ndarray
        Parte imaginária (n_frames, n_pixels).
    n_components : int, opcional
        Número de componentes por PCA. Se None, usa min(n_observações-1, n_variáveis)
        como no MATLAB (todas as componentes).

    Retorna
    -------
    H : np.ndarray
        Scores combinados e ordenados (n_pixels, 2*n_components).
    W : np.ndarray
        Loadings combinados e ordenados (n_frames, 2*n_components).
    V : np.ndarray
        Variâncias explicadas combinadas e ordenadas (2*n_components,).
    """
    print('PCA over X0')
    pca0 = PCA()
    pca0.fit(X0)                        # removed .T
    H0 = pca0.transform(X0)            # scores: (400, n_components)
    W0 = pca0.components_.T            # loadings: (82944, n_components)
    V0 = pca0.explained_variance_ratio_

    print('PCA over X90')
    pca90 = PCA()
    pca90.fit(X90)                      # removed .T
    H90 = pca90.transform(X90)
    W90 = pca90.components_.T
    V90 = pca90.explained_variance_ratio_

    print('End of Dim. Red.')

    # Sorting
    V = np.concatenate([V0, V90])
    idx = np.argsort(V)[::-1]          # descending
    V = V[idx]

    H = np.hstack([H0, H90])           # (400, 800)
    W = np.hstack([W0, W90])           # (82944, 800)
    H = H[:, idx]
    W = W[:, idx] 
    
    return H, W, V
