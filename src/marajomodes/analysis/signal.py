import cv2 as cv
import numpy as np

def extract_signal(video_path):
    """
    Extrai um sinal 1D de movimento lateral a partir de um vídeo.

    Para cada par de frames consecutivos, calcula a diferença absoluta (movimento),
    soma as diferenças por coluna e obtém a posição do centróide no eixo X. O
    resultado é uma série temporal: a posição horizontal do "centro de movimento"
    em cada frame. A média do sinal é removida (tendência DC).

    Útil para analisar oscilações ou modos de vibração que se manifestam como
    movimento horizontal na imagem (ex.: análise de frequência com compute_fft).

    Parâmetros
    ----------
    video_path : str
        Caminho do vídeo (geralmente o vídeo já pré-processado em escala de cinza).

    Retorna
    -------
    np.ndarray
        Sinal 1D de comprimento igual ao número de frames menos um, com média zero.

    Levanta
    -------
    RuntimeError
        Se o vídeo não puder ser aberto ou estiver vazio.
    """
    cap = cv.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Erro ao abrir vídeo")

    if prev.ndim == 3:
        prev = prev[:, :, 0]
        
    prev = prev.astype(np.float32)

    signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            frame = frame[:, :, 0]
            
        frame = frame.astype(np.float32)
        diff = cv.absdiff(frame, prev)
        col_sum = np.sum(diff, axis=0)
        x_positions = np.arange(len(col_sum))

        if np.sum(col_sum) > 0:
            cx = np.sum(x_positions * col_sum) / np.sum(col_sum)
        else:
            cx = 0

        signal.append(cx)

        prev = frame

    cap.release()

    signal = np.array(signal)

    # Remove tendência DC
    signal -= np.mean(signal)

    return signal