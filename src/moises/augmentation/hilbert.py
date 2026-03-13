import numpy as np
from scipy.signal import hilbert


def hilbert_augment(dataset: np.ndarray) :
    """
    Aumenta o dataset aplicando a transformada de Hilbert ao longo do tempo (eixo 0).

    A parte real mantém o sinal original; a parte imaginária corresponde a uma
    rotação de 90° em fase (descarta componentes de frequência negativa).
    Equivalente ao passo "Dataset augmentation" do script MATLAB.

    Parâmetros
    ----------
    dataset : np.ndarray
        Matriz (n_frames, n_pixels) com média já removida.

    Retorna
    -------
    X0 : np.ndarray
        Parte real (n_frames, n_pixels) — dados originais.
    X90 : np.ndarray
        Parte imaginária (n_frames, n_pixels) — dados com 90° de fase.
    """
    # Hilbert ao longo do eixo 0 (tempo)
    H = hilbert(dataset, axis=0)
    X0 = dataset
    X90 = np.imag(H)
    return X0, X90
