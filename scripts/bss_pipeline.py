import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import lfilter
from scipy.linalg import eig
from sklearn.decomposition import PCA


def run_pipeline(
    video_path: str,
    nPC: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa o pipeline completo de processamento do vídeo.

    Parameters
    ----------
    video_path : str
        Caminho para o vídeo pré-processado.
    nPC : int
        Número de componentes principais utilizadas na separação cega.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Uma tupla contendo:
        - sinais separados (unmixed);
        - matriz inversa de mistura (Winvmix);
        - componentes principais (W).

    Raises
    ------
    FileNotFoundError
        Se o vídeo não existir.
    """

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: '{video_path}'.")

    dataset = load_grayscale_dataset(video_path)

    H, W, _ = compute_pca(dataset)

    # H: autovetores
    unmixed, Winvmix = run_cp_on_components(H, nPC) 

    # W: dados originais projetados em TODOS os autovetores 
    # shape: 
    #        nº de pixels originais, 
    #        nº de autovetores = nº de frames originais
    return unmixed, Winvmix, W 


def load_grayscale_dataset(video_path: str) -> np.ndarray:
    """
    Carrega um vídeo em escala de cinza e o converte para uma matriz
    onde cada linha representa um frame vetorizado.

    Parameters
    ----------
    video_path : str
        Caminho para o vídeo.

    Returns
    -------
    np.ndarray
        Matriz de dimensão (n_frames, n_pixels).

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    IOError
        Se o vídeo não puder ser aberto.
    ValueError
        Se o vídeo não contiver frames.
    """

    video_file = Path(video_path)

    if not video_file.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: '{video_file}'.")

    cap = cv.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: '{video_file}'.")

    frames = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # gray
            if frame.ndim == 3:
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame[:, :, 0]
            
            # cada frame em uma linha
            # série temporal de pixels em colunas
            # PCA é aplicado na transposta de frames
            frames.append(frame.reshape(-1))

    finally:
        cap.release()

    if not frames:
        raise ValueError("O vídeo não contém frames.")

    return np.asarray(frames, dtype=np.float32)


def compute_pca(
    dataset: np.ndarray,
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a Análise de Componentes Principais (PCA).

    Parameters
    ----------
    dataset : np.ndarray
        Matriz de entrada.
    remove_mean : bool, default=True
        Remove a média antes da decomposição.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - principal_components: autovetores do PCA;
        - scores: projeção dos dados nas componentes principais;
        - explained_variance: variância explicada por cada componente.
    """

    X = dataset.copy()

    if remove_mean:
        X -= X.mean(axis=0, keepdims=True)

    pca = PCA()

    scores = pca.fit_transform(X.T)
    principal_components = pca.components_.T
    explained_variance = pca.explained_variance_

#     pca_model = PCA()
#     score_W = pca_model.fit_transform(X.T)
#     coeff_H = pca_model.components_.T
#     latent_V = pca_model.explained_variance_

#     return coeff_H, score_W, latent_V

    return principal_components, scores, explained_variance


def run_cp_on_components(H, n_pc):
    """
    Executa o CP_alg sobre os n_pc primeiros componentes principais.
    """
    mixtures = H[:, :n_pc]
    
     # Blind source separation
    unmixed, Wmix = CP_alg(mixtures)
    
    Winvmix = np.fliplr(np.linalg.inv(Wmix))
    
    unmixed = -np.fliplr(unmixed)
    return unmixed, Winvmix


def CP_alg(mixtures):

    n = 10

    ###################################
    # COMPUTE V AND U
    ###################################

    # Short and long half-lives
    shf = 1
    lhf = 900000

    # Max mask length
    max_mask_len = 50

    ###################################
    # Short-term mask
    ###################################

    h = shf
    t = int(n * h)

    lam = 2 ** (-1 / h)

    temp = np.arange(0, t)

    mask = lam ** temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    s_mask = mask

    ###################################
    # Long-term mask
    ###################################

    h = lhf
    t = int(n * h)
    t = min(t, max_mask_len)
    t = max(t, 1)

    lam = 2 ** (-1 / h)

    temp = np.arange(0, t)

    mask = lam ** temp
    mask[0] = 0
    mask = mask / np.sum(np.abs(mask))
    mask[0] = -1

    l_mask = mask

    ###################################
    # Filter each column of mixtures
    ###################################

    S = lfilter(s_mask, 1, mixtures, axis=0)
    L = lfilter(l_mask, 1, mixtures, axis=0)

    ###################################
    # Covariance matrices
    ###################################

    U = np.cov(S, rowvar=False, bias=True)
    V = np.cov(L, rowvar=False, bias=True)

    ###################################
    # Generalized eigenvalue problem
    ###################################

    eigvals, W = eig(V, U)

    W = np.real(W)

    ###################################
    # Extract sources
    ###################################

    ys = -(mixtures @ W)

    return ys, W


def compute_fft(signal, fps):
    # número de amostras
    N = len(signal)

    # remove offset DC
    # x = x - np.mean(x)

    # aplica janela de Hann
    # window = np.hanning(N)
    # x = x * window 

    # eixo de frequências
    freqs = np.fft.rfftfreq(N, d = 1/fps)

    # FFT
    # fft_vals = np.abs(np.fft.rfft(x)) # deixar para fazer o abs fora da função, pois precisamos também do angle
    fft_vals = np.fft.rfft(signal)

    # normalização da amplitude
    # fft_vals = (2 / N) * fft_vals

    return freqs, fft_vals


def compute_fft_for_components(unmixed, fps, nPC: list):
    fft_data = {}

    for i in nPC:
        freqs, fft_vals = compute_fft(unmixed[:, i], fps)

        fft_data[i] = {
            "f": np.asarray(freqs).ravel(),
            "v": np.asarray(fft_vals).ravel(),
            "signal": np.asarray(unmixed[:, i]).ravel()
        }

    return fft_data


def get_highest_peak_frequencies(fft_data, n_peaks=5):
    peaks_info = {}

    for comp_id, data in fft_data.items():
        freq_plot = data["f"]
        fft_vals = data["v"]

        psd = np.abs(fft_vals) ** 2

        mask = freq_plot > 0
        freq_pos = freq_plot[mask]
        psd_pos = psd[mask]

        peak_indices, _ = find_peaks(psd_pos)

        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(psd_pos)])

        sorted_peak_indices = peak_indices[np.argsort(psd_pos[peak_indices])[::-1]]
        top_peak_indices = sorted_peak_indices[:n_peaks]

        peaks_info[comp_id] = {
            "highest_freq": float(freq_pos[top_peak_indices[0]]),
            "highest_amp": float(psd_pos[top_peak_indices[0]]),
            "top_freqs": freq_pos[top_peak_indices],
            "top_amps": psd_pos[top_peak_indices],
            "top_indices": top_peak_indices
        }

    return peaks_info


def compute_mode_shapes(Winvmix, W, numPC, srcs):
    """
    Calcula os mode shapes.

    Equivalente MATLAB:
    --------------------
    mode_shapes = (Winvmix*W(:,1:numPC)')';
    mode_shapes = mode_shapes(:,srcs);

    Parâmetros
    ----------
    Winvmix : np.ndarray
        Matriz de mistura estimada/inversa.
    W : np.ndarray
        Score do PCA.
    numPC : int
        Número de componentes principais usados.
    srcs : list
        Índices dos modos desejados (em Python começa em 0).

    Retorna
    -------
    mode_shapes : np.ndarray
        Mode shapes selecionados.
    """

    # equivalente a:
    # (Winvmix*W(:,1:numPC)')'
    mode_shapes = (Winvmix @ W[:, :numPC].T).T

    # equivalente a:
    # mode_shapes(:,srcs)
    mode_shapes = mode_shapes[:, srcs]

    return mode_shapes