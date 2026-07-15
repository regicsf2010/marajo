import cv2 as cv
import numpy as np
import subprocess
import json
from scipy.signal import find_peaks
from pathlib import Path

def get_video_info(video_path: str) -> dict:
    """
    Retorna informações sobre um vídeo.

    Parameters
    ----------
    video_path : str
        Caminho para o vídeo.

    Returns
    -------
    dict
        Informações do vídeo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    IOError
        Se o vídeo não puder ser aberto.
    """

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: '{video_path}'.")

    video = cv.VideoCapture(str(video_path))

    if not video.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo '{video_path}'.")

    try:
        fps = video.get(cv.CAP_PROP_FPS)
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        duration = None if fps <= 0 else round(frames / fps, 2)

        ret, frame = video.read()

        shape = frame.shape if ret else None

        return {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frames": frames,
            "duration": duration,
            "shape": shape,
        }

    finally:
        video.release()

        
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
    return int(n_components)

