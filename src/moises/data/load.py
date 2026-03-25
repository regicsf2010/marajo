from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2 as cv
import numpy as np

from moises.data.video import Video


def _load_video_chunk(
    video_path: str,
    start: int,
    count: int,
    width: int,
    height: int,
    n_pixels: int,
) -> tuple[int, np.ndarray]:
    """Read a contiguous frame range; each call uses its own VideoCapture."""
    dataset = np.zeros((count, n_pixels), dtype=np.float32)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Erro ao abrir o vídeo: {video_path}")

    if start > 0:
        cap.set(cv.CAP_PROP_POS_FRAMES, start)

    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            dataset = dataset[:i]
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        small = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
        dataset[i, :] = small.ravel().astype(np.float32)

    cap.release()
    return start, dataset


def _load_video_dataset_parallel(video: Video) -> tuple[np.ndarray, np.ndarray]:

    print('Using parallel load')
    n = video.frame_count
    w = min(video.workers, n)
    chunk_size = (n + w - 1) // w
    tasks: list[tuple[int, int]] = []
    for k in range(w):
        start = k * chunk_size
        count = min(chunk_size, n - start)
        if count <= 0:
            break
        tasks.append((start, count))

    results: list[tuple[int, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = [
            ex.submit(
                _load_video_chunk,
                video.video_path,
                start,
                count,
                video.width,
                video.height,
                video.n_pixels,
            )
            for start, count in tasks
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])
    dataset = np.vstack([r[1] for r in results])

    mean = np.mean(dataset, axis=0, dtype=np.float32)
    dataset = dataset - mean
    return dataset, mean


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

    if video.workers > 1 and video.frame_count > 0:
        return _load_video_dataset_parallel(video)

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
