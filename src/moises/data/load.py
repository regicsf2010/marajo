import cv2 as cv
import numpy as np


def load_video_dataset(
    video_path: str,
    max_frames: int | None = None,
) :
    """
    Carrega um vídeo como matriz (frames × pixels) em escala de cinza e remove o DC.

    Cada linha é um frame (vetorizado por colunas); a média temporal por pixel é
    subtraída (background equalization), mantendo apenas as variações dinâmicas.

    Parâmetros
    ----------
    video_path : str
        Caminho do arquivo de vídeo.
    max_frames : int, opcional
        Número máximo de frames a carregar. Se None, carrega todo o vídeo.

    Retorna
    -------
    dataset : np.ndarray
        Matriz (n_frames, n_pixels) em float64, já com média removida (DC).
    fps : float
        Taxa de quadros (frames por segundo).
    shape : tuple (n_rows, n_cols)
        Dimensões espaciais do frame (altura, largura).
    mean : np.ndarray
        Média por pixel (1D, length n_pixels) para reconstrução do background.
    frames : np.ndarray
        Tensão (n_rows, n_cols, n_frames) em uint8 para visualização.

    Levanta
    -------
    IOError
        Se o vídeo não puder ser aberto.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erro ao abrir o vídeo: {video_path}")

    fps = int(cap.get(cv.CAP_PROP_FPS))
    n_rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    n_pixels = n_rows * n_cols

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # --- Compression settings ---
    scale = 0.5  # 0.5 = half resolution (75% memory reduction)
    new_cols = int(n_cols * scale)
    new_rows = int(n_rows * scale)
    n_pixels = new_rows * new_cols

    n_frames = (
        min(max_frames, total_frames)
        if max_frames is not None
        else total_frames
    )

    dataset = np.zeros((n_frames, n_pixels), dtype=np.float32)
    frames = np.zeros((new_rows, new_cols, n_frames), dtype=np.uint8)

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            dataset = dataset[:i]
            frames = frames[:, :, :i]
            n_frames = i
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        small = cv.resize(gray, (new_cols, new_rows), interpolation=cv.INTER_AREA)
    
        frames[:, :, i] = small
        dataset[i, :] = small.ravel().astype(np.float32)

        #frames[:, :, i] = gray
        #dataset[i, :] = gray.ravel().astype(np.float32)

    cap.release()

    mean = np.mean(dataset, axis=0, dtype=np.float32)
    dataset = dataset - mean

    return dataset, fps, (new_rows, new_cols), mean, frames
