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


def compute_source_spatial_maps(W, mixtures, unmixed, n_pc):
    """
    Calcula o mapa espacial associado a cada fonte separada.

    Parameters
    ----------
    W : np.ndarray
        Scores do PCA.
        Shape: (n_pixels, n_components)

    mixtures : np.ndarray
        Componentes principais fornecidos ao BSS.
        Shape: (n_frames, n_pc)

    unmixed : np.ndarray
        Fontes separadas pelo BSS.
        Shape: (n_frames, n_pc)

    n_pc : int
        Número de componentes principais utilizados no BSS.

    Returns
    -------
    spatial_maps : np.ndarray
        Peso de cada fonte em cada pixel.
        Shape: (n_pixels, n_pc)

    A : np.ndarray
        Matriz que reconstrói as misturas a partir das fontes.
        mixtures ~= unmixed @ A
    """

    W_selected = W[:, :n_pc]

    # Resolve:
    #
    # mixtures ≈ unmixed @ A
    #
    A, _, _, _ = np.linalg.lstsq(
        unmixed,
        mixtures,
        rcond=None,
    )

    # X.T ≈ W_selected @ mixtures.T
    #
    # mixtures ≈ unmixed @ A
    #
    # X.T ≈ W_selected @ A.T @ unmixed.T
    #
    spatial_maps = W_selected @ A.T

    return spatial_maps, A


def create_static_source_overlay_video(
    input_video,
    output_video,
    spatial_map,
    alpha=1,
    threshold=0.15,
):
    cap = cv.VideoCapture(str(input_video))

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    writer = cv.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # =========================================================
    # Mapa espacial da fonte
    # =========================================================

    source_map = spatial_map.reshape(height, width)

    # Intensidade espacial da fonte
    source_map = np.abs(source_map)

    # Normalização para [0, 1]
    source_map /= source_map.max() + 1e-12

    # =========================================================
    # Máscara espacial
    # =========================================================

    # Regiões abaixo do threshold não recebem o heatmap
    mask = source_map.copy()

    mask[mask < threshold] = 0.0

    # Reescala os valores restantes para [0, 1]
    mask = (mask - threshold) / (1.0 - threshold)
    mask = np.clip(mask, 0.0, 1.0)

    # Alpha varia espacialmente, mas não temporalmente
    alpha_mask = alpha * mask

    # (H, W) -> (H, W, 1)
    alpha_mask = alpha_mask[..., np.newaxis]

    # =========================================================
    # Heatmap
    # =========================================================

    heatmap_uint8 = (source_map * 255).astype(np.uint8)

    heatmap = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

    heatmap_float = heatmap.astype(np.float32)

    # =========================================================
    # Geração do vídeo
    # =========================================================

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_float = frame.astype(np.float32)

        overlay = (
            frame_float * (1.0 - alpha_mask)
            + heatmap_float * alpha_mask
        )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        writer.write(overlay)

    cap.release()
    writer.release()
    
    
def create_dynamic_source_overlay_video(
    input_video,
    output_video,
    spatial_map,
    source_signal,
    alpha=1,
    threshold=0.15,
):
    cap = cv.VideoCapture(str(input_video))

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    writer = cv.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # =========================================================
    # Mapa espacial
    # =========================================================

    source_map = spatial_map.reshape(height, width)

    source_map = np.abs(source_map)

    source_map /= source_map.max() + 1e-12

    # =========================================================
    # Máscara espacial
    # =========================================================

    mask = source_map.copy()

    mask[mask < threshold] = 0.0

    mask = (mask - threshold) / (1.0 - threshold)

    mask = np.clip(mask, 0.0, 1.0)

    # =========================================================
    # Normalização temporal da fonte
    # =========================================================

    source_signal = np.abs(source_signal)

    source_signal /= (source_signal.max() + 1e-12)

    # =========================================================
    # Geração do vídeo
    # =========================================================

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx >= len(source_signal):
            break

        # Amplitude instantânea da fonte
        temporal_amplitude = source_signal[frame_idx]

        # Mapa de excitação naquele instante
        excitation_map = (source_map * temporal_amplitude)

        heatmap_uint8 = (excitation_map * 255).astype(np.uint8)

        heatmap = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

        heatmap_float = heatmap.astype(np.float32)

        # Alpha também varia temporalmente
        alpha_mask = (alpha * mask * temporal_amplitude)

        alpha_mask = alpha_mask[..., np.newaxis]

        frame_float = frame.astype(np.float32)

        overlay = (
            frame_float
            * (1.0 - alpha_mask)
            + heatmap_float
            * alpha_mask
        )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        writer.write(overlay)

        frame_idx += 1

    cap.release()
    writer.release()
    

def create_video_grid(
    video_paths,
    output_path,
    n_cols=5,
    cell_width=320,
    titles=None
):
    caps = [cv.VideoCapture(str(path)) for path in video_paths]

    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError("Não foi possível abrir um ou mais vídeos.")

    fps = caps[0].get(cv.CAP_PROP_FPS)
    original_width = int(caps[0].get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(caps[0].get(cv.CAP_PROP_FRAME_HEIGHT))

    aspect_ratio = original_height / original_width

    cell_height = int(cell_width * aspect_ratio)

    n_videos = len(video_paths)

    n_rows = int(np.ceil(n_videos / n_cols))

    output_width = n_cols * cell_width
    output_height = n_rows * cell_height

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    writer = cv.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

    frame_idx = 0

    while True:
        frames = []

        for video_idx, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                frames = None
                break

            frame = cv.resize(frame, (cell_width, cell_height))

            cv.putText(frame, f"Source {video_idx}" if titles is None else f"Source {titles[video_idx]}", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255), 2, cv.LINE_AA)

            frames.append(frame)

        if frames is None:
            break

        # Completa a grade, caso necessário
        while len(frames) < n_rows * n_cols:
            blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

            frames.append(blank)

        rows = []

        for row_idx in range(n_rows):
            start = row_idx * n_cols
            end = start + n_cols

            row = np.hstack(frames[start:end])

            rows.append(row)

        grid = np.vstack(rows)

        writer.write(grid)

        frame_idx += 1

    for cap in caps:
        cap.release()

    writer.release()