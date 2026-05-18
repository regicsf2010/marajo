"""Phase-based motion signals via steerable pyramid complexa (Wadhwa 2013, Yang 2017).

A ideia é completar o passo que o pipeline baseline pulou: extrair movimento
sub-pixel via a fase de uma decomposição complexa, em vez de operar diretamente
nos pixels grayscale.

Pra cada frame, o vídeo é decomposto em pirâmide steerable complexa. A fase
dos coeficientes complexos em cada (escala, orientação) carrega informação de
posição local. A diferença temporal de fase representa velocidade sub-pixel.

Cache em disco: cada vídeo gera um `.npz` com os sinais de fase de cada subband
— processar é caro, ler o cache é instantâneo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np
import pyrtools as pt


@dataclass
class PhaseConfig:
    n_scales: int = 3            # níveis da pirâmide (incluindo level 0 = full res)
    n_orientations: int = 2      # 2 = horizontal + vertical
    subsample_factor: int = 4    # downsample espacial extra antes de salvar (1 = sem)
    use_phase_velocity: bool = True  # se True, salva np.diff(phase, axis=0) — mais estável que fase absoluta


def _subsample_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def compute_phase_signals(
    video_path: str,
    config: Optional[PhaseConfig] = None,
) -> dict:
    """Retorna dict com chaves de subband e arrays (n_frames, n_pixels_eff).

    Estrutura do retorno:
        {
            "config": dataclass dict,
            "video_shape": (H, W),
            "subband_shapes": {key: (h, w)},
            "signals": {key: ndarray (n_frames-1 ou n_frames, n_pixels_eff)},
        }

    Cada subband é identificada por (scale, orientation).
    """
    cfg = config or PhaseConfig()

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        raise IOError(f"Vídeo sem frames legíveis: {video_path}")

    n_frames = len(frames)
    H, W = frames[0].shape

    # Inicializa pirâmide no primeiro frame pra descobrir shapes das subbands
    pyr_init = pt.pyramids.SteerablePyramidFreq(
        frames[0], height=cfg.n_scales, order=cfg.n_orientations - 1, is_complex=True
    )
    subband_keys = [(s, o) for s in range(cfg.n_scales) for o in range(cfg.n_orientations)]
    subband_shapes: dict = {}
    storage: dict[tuple[int, int], np.ndarray] = {}
    for k in subband_keys:
        h, w = pyr_init.pyr_coeffs[k].shape
        h_s, w_s = _subsample_2d(np.empty((h, w)), cfg.subsample_factor).shape
        subband_shapes[k] = (h_s, w_s)
        storage[k] = np.empty((n_frames, h_s * w_s), dtype=np.float32)

    # Loop por frame
    for t, frame in enumerate(frames):
        pyr = pt.pyramids.SteerablePyramidFreq(
            frame, height=cfg.n_scales, order=cfg.n_orientations - 1, is_complex=True
        )
        for k in subband_keys:
            phase_full = np.angle(pyr.pyr_coeffs[k]).astype(np.float32)
            phase_sub = _subsample_2d(phase_full, cfg.subsample_factor)
            storage[k][t] = phase_sub.ravel()

    if cfg.use_phase_velocity:
        # Velocidade: phase difference temporal com unwrapping leve.
        # diff_wrapped pega o "menor caminho" entre fases consecutivas no círculo.
        for k in subband_keys:
            phase = storage[k]
            diff = np.diff(phase, axis=0)
            # Unwrap simples: traz pra [-pi, pi]
            diff = np.angle(np.exp(1j * diff)).astype(np.float32)
            storage[k] = diff

    return {
        "config": {
            "n_scales": cfg.n_scales,
            "n_orientations": cfg.n_orientations,
            "subsample_factor": cfg.subsample_factor,
            "use_phase_velocity": cfg.use_phase_velocity,
        },
        "video_shape": (H, W),
        "subband_shapes": {str(k): v for k, v in subband_shapes.items()},
        "signals": {str(k): v for k, v in storage.items()},
    }


def save_phase_cache(data: dict, out_path: str) -> None:
    """Salva o dict de signals + metadados em .npz."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {f"signal__{k}": v for k, v in data["signals"].items()}
    payload["__config_keys"] = np.array(list(data["config"].keys()), dtype=object)
    payload["__config_vals"] = np.array(list(data["config"].values()), dtype=object)
    payload["__video_shape"] = np.array(data["video_shape"], dtype=np.int32)
    payload["__subband_keys"] = np.array(list(data["subband_shapes"].keys()), dtype=object)
    payload["__subband_shapes"] = np.array(list(data["subband_shapes"].values()), dtype=np.int32)
    np.savez_compressed(out_path, **payload)


def load_phase_cache(in_path: str) -> dict:
    raw = np.load(in_path, allow_pickle=True)
    config = dict(zip(raw["__config_keys"].tolist(), raw["__config_vals"].tolist()))
    subband_keys = raw["__subband_keys"].tolist()
    subband_shapes = {k: tuple(s) for k, s in zip(subband_keys, raw["__subband_shapes"].tolist())}
    signals = {k.replace("signal__", ""): raw[k] for k in raw.files if k.startswith("signal__")}
    return {
        "config": config,
        "video_shape": tuple(raw["__video_shape"].tolist()),
        "subband_shapes": subband_shapes,
        "signals": signals,
    }


def flatten_signals(data: dict) -> np.ndarray:
    """Concatena todas as subbands num único dataset (n_frames, total_features).

    Esse formato é o que o pipeline modal (PCA + CP) espera.
    """
    arrays = [data["signals"][k] for k in sorted(data["signals"].keys())]
    return np.concatenate(arrays, axis=1)


def compute_and_cache(
    video_path: str,
    cache_path: str,
    config: Optional[PhaseConfig] = None,
    force: bool = False,
) -> dict:
    """Computa + cacheia se não existe; carrega se existe."""
    if os.path.exists(cache_path) and not force:
        return load_phase_cache(cache_path)
    data = compute_phase_signals(video_path, config)
    save_phase_cache(data, cache_path)
    return data
