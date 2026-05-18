"""Leitura e introspecção de vídeos."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import cv2 as cv
import numpy as np


PIXEL_MODES = {0: "BGR", 1: "RGB", 2: "GRAY", 3: "YUYV"}


@dataclass
class VideoInfo:
    path: str
    fps: int
    width: int
    height: int
    frames: int
    duration: float
    mode: str
    shape: tuple[int, int, int]
    rotation: int


def video_rotation(video_path: str) -> int:
    """Rotação do vídeo em graus (0/90/180/270) lida do metadado via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout) if result.stdout else {}
    try:
        rotation = int(data["streams"][0]["side_data_list"][0]["rotation"])
    except (KeyError, IndexError, TypeError):
        rotation = 0
    return rotation % 360


def video_status(video_path: str, verbose: bool = False) -> VideoInfo:
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = round(video.get(cv.CAP_PROP_FPS))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = round(video.get(cv.CAP_PROP_FRAME_COUNT))
    duration = float(np.round(frames / fps, 2)) if fps else 0.0
    mode = PIXEL_MODES.get(int(video.get(cv.CAP_PROP_MODE)), "UNKNOWN")
    ok, first_frame = video.read()
    if not ok:
        video.release()
        raise IOError(f"Vídeo abriu mas não foi possível ler o primeiro frame: {video_path}")
    shape = tuple(first_frame.shape)
    video.release()

    info = VideoInfo(
        path=video_path,
        fps=fps,
        width=width,
        height=height,
        frames=frames,
        duration=duration,
        mode=mode,
        shape=shape,
        rotation=video_rotation(video_path),
    )

    if verbose:
        for k, v in info.__dict__.items():
            print(f"{k}: {v}")

    return info


def load_grayscale_dataset(video_path: str) -> np.ndarray:
    """Lê todos os frames de um vídeo monocromático e retorna matriz (n_frames, n_pixels) float32.

    Espera-se que o vídeo já tenha sido pré-processado com `preprocess_video` (gray + crop + downscale).
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        frames.append(frame.astype(np.float32).reshape(-1))

    cap.release()
    return np.asarray(frames, dtype=np.float32)
