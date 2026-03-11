import cv2 as cv
from cv2.typing import MatLike
import numpy as np

from marajomodes.utils.conversions import frame_to_gray_array

def extract_signal_handle_frame(
    frame: MatLike,
    prev: np.ndarray[tuple, np.dtype[np.float32]]
) -> tuple[float, np.ndarray[tuple, np.dtype[np.float32]]]:
    """
    Extrai um sinal 1D de movimento lateral a partir de um frame.

    Para cada frame, calcula a diferença absoluta (movimento), soma as diferenças por coluna e obtém a posição do centróide no eixo X. O
    resultado é uma série temporal: a posição horizontal do "centro de movimento"
    em cada frame. A média do sinal é removida (tendência DC).
    

    Parâmetros
    ----------
    frame : np.ndarray
        Frame da imagem em escala de cinza.
    prev : np.ndarray
        Frame anterior em escala de cinza.

    Retorna
    -------
    float
        Posição horizontal do "centro de movimento" em cada frame.

    Levanta
    -------
    RuntimeError
        Se o frame não puder ser aberto ou estiver vazio.
    """

    frame = frame_to_gray_array(frame)

    diff = cv.absdiff(frame, prev)
    col_sum = np.sum(diff, axis=0)
    x_positions = np.arange(len(col_sum))

    if np.sum(col_sum) > 0:
        return np.sum(x_positions * col_sum) / np.sum(col_sum), frame

    return 0, frame

def extract_signal(video_path: str) -> np.ndarray[tuple, np.dtype[np.float32]]:
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
    if not ret: raise RuntimeError("Erro ao abrir vídeo")

    prev = frame_to_gray_array(prev)

    signal = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        cx, prev = extract_signal_handle_frame(frame, prev)
        signal.append(cx)

    cap.release()

    signal = np.array(signal)

    # Remove tendência DC
    signal -= np.mean(signal)

    return signal