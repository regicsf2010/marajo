"""Visualização espacial dos mode shapes."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_mode_shapes(
    mode_shapes: np.ndarray,
    n_rows: int,
    n_cols: int,
    srcs: Sequence[int],
    save_path: Optional[str] = None,
    w: float = 20,
    h: float = 10,
):
    """Renderiza cada mode shape como heatmap. Shape esperado: (n_pixels, len(srcs))."""
    num_src = len(srcs)
    grid_rows = 2
    grid_cols = int(np.ceil(num_src / 2))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(w, h), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i in range(num_src):
        ax = axes[i]
        S = mode_shapes[:, i]
        img = S.reshape(n_rows, n_cols)
        im = ax.imshow(img, aspect="auto")
        ax.set_title(f"Mode Shape {srcs[i]}", fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax)

    for j in range(num_src, len(axes)):
        axes[j].axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes
