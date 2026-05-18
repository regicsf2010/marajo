"""Configuração do pipeline: dataclasses como fonte da verdade + loader opcional de YAML."""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreprocessConfig:
    num_frames: int = 800
    fps: int = 240
    scale: float = 0.2


@dataclass
class DecompositionConfig:
    num_pcs: int = 10
    variance_threshold: float = 0.90


@dataclass
class CPConfig:
    short_half_life: float = 1.0
    long_half_life: float = 900000.0
    max_mask_len: int = 50
    n_mask_horizon: int = 10


@dataclass
class ModalConfig:
    n_peaks: int = 5
    peak_min_freq: float = 0.5
    bands: list[list[float]] = field(default_factory=list)  # opcional: [[low, high], ...] em Hz
    freq_max: float | None = None  # opcional: descarta bins acima dessa freq nas features globais


@dataclass
class PathsConfig:
    rois_json: str = "rois/rois.json"
    out_dir: str = "out/"
    videos_root: str = ""  # se setado, paths relativos em over_time_videos resolvem contra ele


def resolve_video_path(path: str, videos_root: str) -> str:
    """Resolve um path de vídeo: absoluto fica como está, relativo é prependado com videos_root."""
    if os.path.isabs(path) or not videos_root:
        return path
    return os.path.join(videos_root, path)


@dataclass
class BatchesConfig:
    """Batches do experimento de aquisição.

    Nomes são neutros temporais (mês de captura), sem pressupor o efeito esperado.
    Convenção: february = sem rega; april = regada regularmente. A interpretação
    fisiológica (stressed vs healthy) fica nos plots/README, não no schema.
    """

    february: list[str] = field(default_factory=list)
    april: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    cp: CPConfig = field(default_factory=CPConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    batches: BatchesConfig = field(default_factory=BatchesConfig)

    def all_videos(self) -> list[tuple[str, str]]:
        """Retorna [(batch_name, video_path), ...] na ordem february → april."""
        return [("february", p) for p in self.batches.february] + [
            ("april", p) for p in self.batches.april
        ]

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return _from_dict(cls, raw)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data

    type_hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for f in fields(cls):
        if f.name not in data:
            continue
        target_type = type_hints.get(f.name)
        raw_value = data[f.name]
        if isinstance(target_type, type) and is_dataclass(target_type):
            kwargs[f.name] = _from_dict(target_type, raw_value or {})
        else:
            kwargs[f.name] = raw_value

    return cls(**kwargs)
