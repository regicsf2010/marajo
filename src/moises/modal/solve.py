import numpy as np


def solve_modal(
    unmixed: np.ndarray,
    Winvmix: np.ndarray,
    W: np.ndarray,
    srcs: list[int] | np.ndarray,
    num_pc: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    modal_coord = -unmixed[:, srcs]

    mode_shapes = (Winvmix @ W[:, :num_pc].T).T
    mode_shapes = mode_shapes[:, srcs]

    return modal_coord, mode_shapes