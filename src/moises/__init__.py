"""
Moises: pipeline de análise modal por vídeo (PCA + Hilbert + Complexity Pursuit).

Equivalente ao script MATLAB scriptCS_testing.m. Uso típico:

    from moises import load_video_dataset, hilbert_augment, pca, cp_alg, solve_modal
"""

from moises.data import load_video_dataset
from moises.augmentation import hilbert_augment
from moises.decomposition import pca, cp_alg
from moises.modal import solve_modal

__all__ = [
    "load_video_dataset",
    "hilbert_augment",
    "pca",
    "cp_alg",
    "solve_modal",
]
