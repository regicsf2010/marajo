"""Sanity check visual das ROIs 60 FPS: salva 4 frames com retângulo sobreposto.

Pega o "principal" 60 FPS de feb_d1, feb_d8, apr_d1, apr_d8 e desenha a ROI
sobre o primeiro frame. Saída em `out/experimentos/011/roi_check/`.
"""

from __future__ import annotations

import os
import sys

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.io.roi import load_rois

VIDEOS_ROOT = "/home/ygarasab/mestrado/tese/videos_caseiro_muda_acai"
SAMPLES = [
    ("20260207", "60"),
    ("20260214", "60"),
    ("20260403", "60"),
    ("20260410", "60"),
]


def main() -> None:
    out_dir = "out/experimentos/011/roi_check"
    os.makedirs(out_dir, exist_ok=True)

    rois = load_rois("rois/rois.json")
    for day, fps in SAMPLES:
        day_dir = os.path.join(VIDEOS_ROOT, day, fps)
        first = sorted(f for f in os.listdir(day_dir) if f.endswith(".mp4"))[0]
        roi = rois.get(first)
        if roi is None:
            print(f"sem ROI pra {first}")
            continue
        cap = cv.VideoCapture(os.path.join(day_dir, first))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"falha ao ler frame de {first}")
            continue
        x, y, w, h = roi.x, roi.y, roi.w, roi.h
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
        cv.putText(frame, f"{day} {fps}fps ROI {x},{y},{w},{h}",
                   (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        out_path = os.path.join(out_dir, f"{day}_{fps}fps.jpg")
        cv.imwrite(out_path, frame)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
