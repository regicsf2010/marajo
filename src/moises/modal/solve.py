import numpy as np


def solve_modal(
    ys: np.ndarray,
    W_mix: np.ndarray,
    W_pca: np.ndarray,
    srcs: list[int] | np.ndarray,
    num_pc: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtém coordenadas modais e formas modais a partir das fontes CP e dos loadings PCA.

    Aplica a mesma convenção do MATLAB: Winvmix = flip(inv(W_mix)), unmixed = -fliplr(ys),
    seleciona as fontes em `srcs` como coordenadas modais e resolve as formas modais
    via Winvmix @ W_pca.

    Parâmetros
    ----------
    ys : np.ndarray
        Saída bruta do CP (n_amostras, n_canais), ex.: (n_pixels, 16).
    W_mix : np.ndarray
        Matriz de mistura do CP (n_canais, n_canais).
    W_pca : np.ndarray
        Loadings temporais do PCA dual (n_frames, n_components), ex.: (n_frames, 32).
    srcs : array-like de int
        Índices (0-based) das fontes a usar como modos, ex.: [0, 1, 8, 9, 13, 14].
    num_pc : int, opcional
        Número de componentes PCA usados no CP (ex.: 16). Se None, usa W_mix.shape[0].

    Retorna
    -------
    modal_coord : np.ndarray
        Coordenadas modais (n_pixels, n_srcs), já com convenção de sinal do script.
    mode_shapes : np.ndarray
        Formas modais temporais (n_frames, n_srcs).
    """
    num_pc = num_pc or W_mix.shape[0]
    srcs = np.asarray(srcs)

    Winvmix = np.linalg.inv(W_mix)
    Winvmix = np.flipud(Winvmix)
    unmixed = -np.fliplr(ys)

    # Coordenadas modais = fontes selecionadas (espaciais)
    modal_coord = -unmixed[:, srcs]  # (n_pixels, n_srcs)

    # Formas modais: Winvmix @ W_pca[:, :num_pc].T -> (n_canais, n_frames) -> (n_frames, n_canais)
    W_sub = W_pca[:, :num_pc]  # (n_frames, num_pc)
    mode_shapes = (Winvmix @ W_sub.T).T  # (n_frames, num_pc)
    mode_shapes = mode_shapes[:, srcs]   # (n_frames, n_srcs)

    return modal_coord, mode_shapes
