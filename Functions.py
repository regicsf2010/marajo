import cv2 as cv
import tkinter as tk
import numpy as np
import subprocess
import json
from scipy.signal import find_peaks

def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # não mostra janela
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h

def roi_selection(video_path):
    drawing = False
    ix, iy = -1, -1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, frame, frame_copy

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                frame = frame_copy.copy()
                cv.rectangle(frame, (ix, iy), (x, y), (0, 0, 255), 2)

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            w = abs(x - ix)
            h = abs(y - iy)
            print(f"x = {x0}, y = {y0}, w = {w}, h = {h} | {x0}, {y0}, {w}, {h}")
            cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

    # === Abrir vídeo ===
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Não foi possível ler o vídeo")
        
    frame_copy = frame.copy()
    
    win_name = "Selecione a ROI (ENTER ou ESC para sair)"

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    h, w = frame.shape[:2]
    MAX_W, MAX_H = 800, 800

    scale = min(MAX_W / w, MAX_H / h, 1.0)
    win_w, win_h = int(w * scale), int(h * scale)

    cv.resizeWindow(win_name, win_w, win_h)

    screen_w, screen_h = get_screen_size()
    cv.moveWindow(win_name, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    
    cv.setMouseCallback(win_name, mouse_callback)

    while cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow(win_name, frame)
        if cv.waitKey(1) in (13, 27):
            break

    cv.destroyAllWindows()
    
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

def video_status(video_path):
    video = cv.VideoCapture(video_path)
    print(f'PATH = {video_path}')
    
    fps = round(video.get(cv.CAP_PROP_FPS))
    print(f'FPS = {fps}')
    
    print(f'WIDTH, HEIGHT = {int(video.get(cv.CAP_PROP_FRAME_WIDTH))}, {int(video.get(cv.CAP_PROP_FRAME_HEIGHT))}')
    
    num_frames = round(video.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'FRAME COUNT = {num_frames}')
    
    duration = num_frames / fps
    print(f'DURATION (s) = {duration:.3}')   
    
    modes = {0: 'BGR', 1: 'RGB', 2: 'GRAY', 3: 'YUYV'}
    
    print(f'MODE = {modes[int(video.get(cv.CAP_PROP_MODE))]}')
    
    print(f'SHAPE = {video.read()[1].shape}')
    
    print(f'ROTATION = {video_rotation(video_path)}')
    
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