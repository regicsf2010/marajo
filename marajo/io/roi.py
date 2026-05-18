"""Persistência e seleção interativa de ROI (region of interest) por vídeo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_list(cls, data: list[int]) -> "ROI":
        x, y, w, h = data
        return cls(int(x), int(y), int(w), int(h))

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def load_rois(json_path: str | Path = "rois/rois.json") -> dict[str, ROI]:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: ROI.from_list(coords) for name, coords in raw.items()}


def save_rois(rois: dict[str, ROI], json_path: str | Path = "rois/rois.json") -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    serializable = {name: roi.to_list() for name, roi in rois.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def get_roi_for_video(video_path: str, json_path: str | Path = "rois/rois.json") -> Optional[ROI]:
    return load_rois(json_path).get(os.path.basename(video_path))


def _get_screen_size() -> tuple[int, int]:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def _get_first_frame(video_path: str):
    cap = cv.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Não foi possível ler o vídeo: {video_path}")
    return frame


def select_roi_interactively(video_path: str, max_w: int = 800, max_h: int = 800) -> Optional[ROI]:
    """Abre uma janela cv com o primeiro frame pro usuário desenhar a ROI. ENTER confirma, ESC cancela."""
    drawing = False
    ix, iy = -1, -1
    selected: Optional[list[int]] = None

    frame = _get_first_frame(video_path)
    frame_copy = frame.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal ix, iy, drawing, frame, selected
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            frame = frame_copy.copy()
            cv.rectangle(frame, (ix, iy), (x, y), (0, 0, 255), 2)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            w = abs(x - ix)
            h = abs(y - iy)
            selected = [int(x0), int(y0), int(w), int(h)]
            frame = frame_copy.copy()
            cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

    win = "Selecione a ROI (ENTER confirma | ESC cancela)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    h_orig, w_orig = frame.shape[:2]
    scale = min(max_w / w_orig, max_h / h_orig, 1.0)
    win_w, win_h = int(w_orig * scale), int(h_orig * scale)
    cv.resizeWindow(win, win_w, win_h)
    screen_w, screen_h = _get_screen_size()
    cv.moveWindow(win, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    cv.setMouseCallback(win, on_mouse)

    confirmed = False
    while cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow(win, frame)
        key = cv.waitKey(1)
        if key == 13:
            confirmed = selected is not None
            break
        elif key == 27:
            break

    cv.destroyAllWindows()
    return ROI.from_list(selected) if confirmed and selected is not None else None


def annotate_and_save_roi(
    video_path: str,
    json_path: str | Path = "rois/rois.json",
    overwrite: bool = False,
) -> Optional[ROI]:
    rois = load_rois(json_path)
    name = os.path.basename(video_path)
    if name in rois and not overwrite:
        return rois[name]

    roi = select_roi_interactively(video_path)
    if roi is None:
        return None

    rois[name] = roi
    save_rois(rois, json_path)
    return roi
