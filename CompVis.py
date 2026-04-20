import cv2 as cv
import numpy as np
from scipy.signal import lfilter
from scipy.linalg import eig
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def pre_processing(in_video_path, out_video_path, num_frames, fps = None, scale = .2, roi = None):
    # carrega o vídeo
    video = cv.VideoCapture(in_video_path)
    
    # lança uma exceção caso ocorra algum erro no carregamento
    if not video.isOpened():
        raise IOError("Erro ao abrir o vídeo")

    # especifica o FPS do vídeo de saída
    # duas possibilidades: 
    # 1. ou o usuário especifica o FPS ou
    # 2. usa o mesmo fps do vídeo de entrada
    if fps is None:
        fps = video.get(cv.CAP_PROP_FPS)
    
    # lógica para orientar o vídeo de saída
    if roi is not None:
        x, y, w, h = roi
        base_w, base_h = w, h
    else:
        base_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        base_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    width = int(base_w * scale)
    height = int(base_h * scale)
    
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(out_video_path, fourcc, fps, (width, height), isColor = False)
    
    count = 0
    
    while count < num_frames:
        ret, frame = video.read()
        if not ret:
            break
            
        # frame = cv.rotate(frame, cv.ROTATE_180) # precisa de mais refinamento, nem sempre será 180 graus
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # o código original transformava em escala de cinza
        
        if roi is not None:
            gray = gray[y : y + h, x : x + w] # recorta somente cena de interesse, por enquanto esse passo é manual
        
        gray_small = cv.resize(gray, (width, height), interpolation = cv.INTER_AREA)
        out.write(gray_small)
        count += 1

    video.release()
    out.release()
    cv.destroyAllWindows()
    
def load_grayscale_dataset(video_path):
    
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # grayscale
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        frames.append(frame.astype(np.float32).reshape(-1))

    cap.release()

    dataset = np.array(frames, dtype=np.float32)
    return dataset


def compute_pca(dataset, remove_mean=True):
    """
    Calcula o PCA e retorna:
    - coeff  -> componentes principais (autovetores)
    - score  -> projeção dos dados
    - latent -> variância explicada
    """
    X = dataset.copy()

    if remove_mean:
        X -= X.mean(axis=0, keepdims=True)

    pca_model = PCA()
    score_W = pca_model.fit_transform(X.T)
    coeff_H = pca_model.components_.T
    latent_V = pca_model.explained_variance_

    return coeff_H, score_W, latent_V


def run_cp_on_components(H, n_pc):
    """
    Executa o CP_alg sobre os n_pc primeiros componentes principais.
    """
    mixtures = H[:, :n_pc]
    unmixed, Wmix = CP_alg(mixtures)
    unmixed = -np.fliplr(unmixed)
    return unmixed, Wmix


def get_num_components_for_variance(V, x):
    V = np.asarray(V, dtype=np.float64).ravel()

    if V.size == 0:
        raise ValueError("V está vazio.")

    if np.any(V < 0):
        raise ValueError("Os autovalores em V devem ser não negativos.")

    if x > 1:
        x = x / 100.0

    if not (0 < x <= 1):
        raise ValueError("x deve estar em (0,1] ou em (0,100].")

    explained_ratio = V / np.sum(V)
    cumulative_ratio = np.cumsum(explained_ratio)

    n_components = np.searchsorted(cumulative_ratio, x) + 1
    return n_components
    
def la_pipeline(video_path, nPC):

    dataset = load_grayscale_dataset(video_path)

    H, W, V = compute_pca(dataset)

    unmixed, Wmix = run_cp_on_components(H, nPC)
    
    return unmixed



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



from scipy.signal import find_peaks
import numpy as np


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