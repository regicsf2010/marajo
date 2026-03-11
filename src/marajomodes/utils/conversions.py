from cv2.typing import MatLike
import numpy as np

def frame_to_gray_array(frame: MatLike) -> np.ndarray[tuple, np.dtype[np.float32]]:
    """
    Converte um frame para numpy array de escala de cinza.
    """
    if frame.ndim == 3:
        frame = frame[:, :, 0]

    return frame.astype(np.float32)