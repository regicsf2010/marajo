import cv2 as cv

class Video:

    fps: float
    width: int
    height: int
    n_pixels: int
    frame_count: int
    duration: float
    mode: str
    shape: tuple[int, int, int]
    scale: float

    def __init__(self, video_path: str, max_frames: int | None = None, scale: float = 1):

        self.video_path = video_path
        self.max_frames = max_frames
        self.scale = scale

    def load(self):
        cap = cv.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise IOError(f"Erro ao abrir o vídeo: {self.video_path}")

        self.fps = cap.get(cv.CAP_PROP_FPS)

        self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.n_pixels = int(self.width * self.height)

        if self.scale != 1:
            self.resize_frames()
        

        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if self.max_frames is not None:
            self.frame_count = min(self.frame_count, self.max_frames)

        self.duration = self.frame_count / self.fps
        self.mode = str(cap.get(cv.CAP_PROP_FORMAT))
        self.shape = (int(self.height), int(self.width), 3)
        cap.release()

    def resize_frames(self):
        new_cols = int(self.width * self.scale)
        new_rows = int(self.height * self.scale)
        self.n_pixels = new_rows * new_cols
        self.shape = (new_rows, new_cols, 3)
        self.width = new_cols
        self.height = new_rows
