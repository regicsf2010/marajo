"""Visualizações da análise de tendência (experimento 002)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from marajo.pipelines.trend_analysis import TrendAnalysisResult


_BATCH_COLOR = {
    "february": "tab:blue",
    "april": "tab:orange",
}


def plot_feature_trends(
    result: TrendAnalysisResult,
    save_path: Optional[str] = None,
    alpha: float = 0.05,
    ncols: int = 3,
    w_per_col: float = 6.5,
    h_per_row: float = 4.0,
):
    """Grid com uma feature por subplot; cada subplot tem february e april sobrepostos.

    Anotações: p-values dos 3 testes por batch. Destaca em verde batches com p < alpha
    em algum teste (= revelaram tendência).
    """
    features = result.feature_names
    n = len(features)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_per_col * ncols, h_per_row * nrows),
                             constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, feat in zip(axes, features):
        series = result.series_by_feature[feat]
        annotations: list[str] = []
        for batch, values in series.by_batch.items():
            days = np.arange(1, len(values) + 1)
            color = _BATCH_COLOR.get(batch, None)
            ax.plot(days, values, marker="o", color=color, label=batch)

            trend = result.trend(feat, batch)
            mk = trend.tests["mann_kendall"]
            sp = trend.tests["spearman"]
            ln = trend.tests["linear"]
            any_trend = any(t.p_value < alpha for t in trend.tests.values())
            marker = "*" if any_trend else " "
            annotations.append(
                f"{marker}{batch}: MK p={mk.p_value:.3f} (τ={mk.statistic:+.2f}) | "
                f"Sp p={sp.p_value:.3f} (ρ={sp.statistic:+.2f}) | "
                f"Lin p={ln.p_value:.3f} (slope={ln.slope:+.3f})"
            )

        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("dia (no batch)", fontsize=9)
        ax.set_ylabel(feat, fontsize=9)
        ax.tick_params(labelsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.0, -0.35, "\n".join(annotations),
            transform=ax.transAxes, fontsize=7.5, va="top", family="monospace",
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Análise de tendência por feature (* = p < {alpha} em ao menos um teste)",
        fontsize=14,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_pvalue_heatmap(
    result: TrendAnalysisResult,
    save_path: Optional[str] = None,
    alpha: float = 0.05,
    w: float = 12,
    h: float = 8,
):
    """Heatmap dos p-values. Verde = passa em p < α (tendência detectada)."""
    features = result.feature_names
    batches = sorted({t.batch for t in result.trends})
    tests = ["mann_kendall", "spearman", "linear"]

    col_labels = [f"{b}\n{t}" for b in batches for t in tests]
    matrix = np.full((len(features), len(col_labels)), np.nan)

    for i, feat in enumerate(features):
        for jb, batch in enumerate(batches):
            tr = result.trend(feat, batch)
            for jt, test in enumerate(tests):
                matrix[i, jb * len(tests) + jt] = tr.tests[test].p_value

    fig, ax = plt.subplots(figsize=(w, h))
    cmap = plt.get_cmap("RdYlGn_r")  # vermelho = alto, verde = baixo
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.5)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_title(f"p-values por (feature × batch × teste) — verde = p < {alpha}", fontsize=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p = matrix[i, j]
            marker = "*" if p < alpha else ""
            ax.text(j, i, f"{p:.3f}{marker}", ha="center", va="center",
                    fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="p-value")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
