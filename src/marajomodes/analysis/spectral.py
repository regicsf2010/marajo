import numpy as np
from scipy.signal import find_peaks


def get_top_n_peaks(freqs, fft_vals, n=3, min_freq=0.5):
    """
    Retorna os n picos de maior amplitude no espectro de frequências.

    Ignora frequências abaixo de min_freq (ruído DC e muito baixas), detecta
    picos locais e ordena por amplitude em ordem decrescente.

    Parâmetros
    ----------
    freqs : array-like
        Array de frequências em Hz (ex.: saída de compute_fft).
    fft_vals : array-like
        Array de amplitudes do espectro (ex.: saída de compute_fft).
    n : int, opcional
        Número de picos a retornar (padrão 3).
    min_freq : float, opcional
        Frequência mínima em Hz; valores abaixo são ignorados (padrão 0.5).

    Retorna
    -------
    list of tuple (float, float)
        Lista de (frequência, amplitude) para cada pico, ordenada da maior
        para a menor amplitude. Lista vazia se não houver picos.
    """
    # Ignora frequências muito baixas (ruído DC)
    mask = freqs > min_freq
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    # Detecta picos locais
    peaks, _ = find_peaks(fft_vals)

    if len(peaks) == 0:
        return []

    # Ordena picos por amplitude (decrescente)
    sorted_peaks = peaks[np.argsort(fft_vals[peaks])[::-1]]

    # Seleciona os n maiores
    top_peaks = sorted_peaks[:n]

    results = [(freqs[i], fft_vals[i]) for i in top_peaks]

    return results

def compute_fft(signal, fps):
    """
    Calcula a FFT (transformada de Fourier) real do sinal e retorna o espectro
    de amplitudes em frequências positivas.

    Aplica janela de Hanning ao sinal para reduzir vazamento espectral nas bordas.
    As frequências são calculadas em Hz com base no FPS (taxa de amostragem).

    Parâmetros
    ----------
    signal : array-like
        Sinal 1D no domínio do tempo (ex.: saída de extract_signal).
    fps : float
        Taxa de amostragem em frames por segundo (FPS do vídeo).

    Retorna
    -------
    freqs : np.ndarray
        Array de frequências em Hz (metade do tamanho do sinal + 1, FFT real).
    fft_vals : np.ndarray
        Amplitudes (módulo) do espectro para cada frequência.
    """
    N = len(signal)

    # janela de Hanning (melhora muito o espectro)
    window = np.hanning(N)
    signal = signal * window # Multiplica o sinal por uma janela suave que começa e termina em zero. (Evita descontinuidade nas bordas → vazamento espectral (spectral leakage).)

    freqs = np.fft.rfftfreq(N, d = 1 / fps) # retorna só frequências positivas
    fft_vals = np.abs(np.fft.rfft(signal)) # aplica a fft, obtém valores complexos e depois a amplitude com o módulo

    return freqs, fft_vals