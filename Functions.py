import cv2 as cv
import numpy as np
import subprocess
import json
from scipy.signal import find_peaks

def video_rotation(video_path):
    """
    Retorna a rotação do vídeo em graus (0, 90, 180, 270)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    data = json.loads(result.stdout)
    try:
        rotation = int(data["streams"][0]["side_data_list"][0]["rotation"])
    except (KeyError, IndexError):
        rotation = 0

    # Normalizar
    rotation = rotation % 360
    return rotation

def video_status(video_path, verbose = False):
    video = cv.VideoCapture(video_path)
    fps = round(video.get(cv.CAP_PROP_FPS))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = round(video.get(cv.CAP_PROP_FRAME_COUNT))
    duration = np.round(frames / fps, 2)
    pModes = {0: 'BGR', 1: 'RGB', 2: 'GRAY', 3: 'YUYV'}
    mode = pModes[int(video.get(cv.CAP_PROP_MODE))]
    shape = video.read()[1].shape
    rotation = video_rotation(video_path)
    
    out = {
        'vpath': video_path,
        'fps': fps,
        'width': width,
        'height': height,
        'frames': frames,
        'duration': duration,
        'mode': mode,
        'shape': shape,
        'rotation': rotation
    }
    
    if verbose:
        for k, v in out.items():
            print(f'{k}: {v}')            
        
    return out
    
def get_top_n_peaks(freqs, fft_vals, n = 3, min_freq = 0.5):
    
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