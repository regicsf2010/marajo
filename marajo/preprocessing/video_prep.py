"""Pré-processamento de vídeo: recorte da ROI, conversão para escala de cinza e downscale."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import cv2 as cv

from marajo.io.roi import ROI


@dataclass
class PreprocessConfig:
    num_frames: int = 800
    fps: Optional[int] = 240
    scale: float = 0.2


def preprocess_video(
    in_path: str,
    out_path: str,
    roi: Optional[Union[ROI, tuple[int, int, int, int]]] = None,
    config: Optional[PreprocessConfig] = None,
) -> str:
    """Lê `num_frames` do vídeo de entrada, recorta a ROI (opcional), converte pra gray e downscale.

    Salva como mp4 monocromático em `out_path`. Retorna o caminho de saída.
    """
    cfg = config or PreprocessConfig()

    video = cv.VideoCapture(in_path)
    if not video.isOpened():
        raise IOError(f"Erro ao abrir o vídeo: {in_path}")

    fps = cfg.fps if cfg.fps is not None else video.get(cv.CAP_PROP_FPS)

    if roi is not None:
        x, y, w, h = roi.as_tuple() if isinstance(roi, ROI) else roi
        base_w, base_h = w, h
    else:
        x = y = 0
        base_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        base_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    width = int(base_w * cfg.scale)
    height = int(base_h * cfg.scale)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(out_path, fourcc, fps, (width, height), isColor=False)

    count = 0
    while count < cfg.num_frames:
        ok, frame = video.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if roi is not None:
            gray = gray[y : y + base_h, x : x + base_w]
        gray_small = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
        writer.write(gray_small)
        count += 1

    video.release()
    writer.release()
    return out_path
