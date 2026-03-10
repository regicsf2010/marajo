"""
Marajomodes: análise de modos de vibração a partir de vídeo.

Uso típico:
    from marajomodes import (
        pre_processing,
        extract_signal,
        compute_fft,
        get_top_n_peaks,
        plot_signal,
        plot_freq,
        roi_selection,
        video_status,
    )
"""

from marajomodes.data import (
    pre_processing,
    roi_selection,
    video_rotation,
    video_status,
)
from marajomodes.analysis import (
    compute_fft,
    extract_signal,
    get_top_n_peaks,
)
from marajomodes.visualization import (
    plot_freq,
    plot_signal,
)

__all__ = [
    "pre_processing",
    "roi_selection",
    "video_rotation",
    "video_status",
    "extract_signal",
    "compute_fft",
    "get_top_n_peaks",
    "plot_signal",
    "plot_freq",
]
