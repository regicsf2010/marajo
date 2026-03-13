"""Redução de dimensionalidade (PCA) e separação de fontes (Complexity Pursuit)."""

from moises.decomposition.reduction import pca
from moises.decomposition.cp_alg import cp_alg

__all__ = ["pca", "cp_alg"]
