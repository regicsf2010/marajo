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
    
def moses_code(video_path, nPC):

    cap = cv.VideoCapture(video_path)

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

    # remove média (fundo estático)
    dataset -= dataset.mean(axis=0, keepdims=True)

    # PCA
    pca_model = PCA()
    W = pca_model.fit_transform(dataset.T)     # score
    H = pca_model.components_.T                # coeff
    # V = pca_model.explained_variance_          # latent
    
    # usar apenas os primeiros PCs
    mixtures = H[:, :nPC]

    # Blind source separation
    unmixed, Wmix = CP_alg(mixtures)

    # reorganização final
    unmixed = -np.fliplr(unmixed)

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


def plot_pc(unmixed, video_features, nPC: list, n_peaks_text = 5, w = 20, h = 15):
    t = np.arange(video_features['frames']).reshape(-1, 1) / video_features['fps']
    t_plot = np.asarray(t).ravel()

    fig, axes = plt.subplots(len(nPC), 3, figsize = (w, h), constrained_layout = True)

    if len(nPC) == 1:
        axes = np.expand_dims(axes, axis=0)

    cax = 0 # contador para a linha do axis
    
    for i in nPC:
        signal = unmixed[:, i]

        # FFT do componente i
        freqs, fft_vals = compute_fft(unmixed[:, i], video_features['fps'])
        freq_plot = np.asarray(freqs).ravel()
        fft_signal = np.asarray(fft_vals).ravel()

        psd = np.abs(fft_vals) ** 2
        phase = np.angle(fft_vals) ** 2

        # Manter apenas frequências positivas
        mask = freq_plot > 0
        freq_pos = freq_plot[mask]
        psd_pos = psd[mask]
        phase_pos = phase[mask]

        # Encontrar picos reais da PSD
        peak_indices, _ = find_peaks(psd_pos)

        # Se não houver picos detectados, usa o maior valor global
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(psd_pos)])

        # Ordenar os picos pela amplitude, do maior para o menor
        sorted_peak_indices = peak_indices[np.argsort(psd_pos[peak_indices])[::-1]]

        # Selecionar os n primeiros
        top_peak_indices = sorted_peak_indices[:n_peaks_text]

        # Montar texto
        peak_lines = []
        for idx in top_peak_indices:
            peak_lines.append(f"freq: {freq_pos[idx]:.2f} Hz | amp: {psd_pos[idx]:.2f}")
        peak_text = "\n".join(peak_lines)

        # --- Source ---
        ax = axes[cax, 0]
        ax.plot(t_plot, signal, linewidth=1.5)
        ax.set_title(f"Source {i}", fontsize=10)
        ax.tick_params(labelsize=9)

        # --- PSD ---
        ax = axes[cax, 1]
        ax.plot(freq_pos, psd_pos, linewidth=1.5, color="k")

        # Marcar os picos selecionados
        ax.plot(freq_pos[top_peak_indices], psd_pos[top_peak_indices], "ro", markersize=4)

        # Texto centralizado no gráfico da PSD
        ax.text(
            0.5, 0.5, peak_text,
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_title(f"PSD {i}", fontsize=10)
        ax.tick_params(labelsize=9)

        # --- Phase ---
        ax = axes[cax, 2]
        ax.plot(freq_pos, phase_pos, linewidth=1.5, color="k")
        ax.set_title(f"Phase {i}", fontsize=10)
        ax.tick_params(labelsize=9)

        cax += 1
        
    # Labels finais
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")

    plt.show()