import cv2 as cv
import numpy as np
from moises.data.video import Video


def load_video_dataset(
    video: Video,
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

    print("Loading video dataset...")
    print("Video shape:", video.width, video.height)

    dataset = np.zeros((video.frame_count, video.n_pixels), dtype=np.float32)
    frames = np.zeros((video.height, video.width, video.frame_count), dtype=np.uint8)

    cap = cv.VideoCapture(video.video_path)
    if not cap.isOpened():
        raise IOError(f"Erro ao abrir o vídeo: {video.video_path}")

    for i in range(video.frame_count):
        ret, frame = cap.read()
        if not ret:
            dataset = dataset[:i]
            frames = frames[:, :, :i]
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Resize frame to the (possibly downscaled) video dimensions
        small = cv.resize(gray, (video.width, video.height), interpolation=cv.INTER_AREA)

        frames[:, :, i] = small
        dataset[i, :] = small.ravel().astype(np.float32)

    cap.release()

    mean = np.mean(dataset, axis=0, dtype=np.float32)
    dataset = dataset - mean

    return dataset, mean
