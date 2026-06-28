"""Gera ROIs pros vídeos 60 FPS escalando as ROIs anotadas pros vídeos 240 FPS
do mesmo dia.

Premissa geométrica: pra cada dia, a câmera permanece no tripé entre as 4 capturas
e entre as gravações 60/240 FPS. O frame raw que o OpenCV devolve é sempre portrait
(altura > largura). Difere apenas em resolução:
  - Motorola G60 (07-14/02 e 02/04): 60 FPS = 1080×1920, 240 FPS = 720×1280  → scale 1.5×
  - Motorola Signature (03-10/04):   60 FPS = 1080×1920, 240 FPS = 1080×1920 → scale 1.0×

Salva todas as ROIs derivadas no mesmo `rois/rois.json` (não destrutivo — só anexa).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.io.roi import ROI, load_rois, save_rois


VIDEOS_ROOT = "/home/ygarasab/mestrado/tese/videos_caseiro_muda_acai"

CANONICAL_DAYS = [
    "20260207", "20260208", "20260209", "20260210",
    "20260211", "20260212", "20260213", "20260214",
    "20260403", "20260404", "20260405", "20260406",
    "20260407", "20260408", "20260409", "20260410",
]


def first_frame_shape(video_path: str) -> tuple[int, int]:
    cap = cv.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"não foi possível ler frame de {video_path}")
    return frame.shape[0], frame.shape[1]  # (h, w)


def scale_roi(roi: ROI, scale: float) -> ROI:
    return ROI(
        x=int(round(roi.x * scale)),
        y=int(round(roi.y * scale)),
        w=int(round(roi.w * scale)),
        h=int(round(roi.h * scale)),
    )


def find_240_roi_for_day(day: str, rois: dict[str, ROI]) -> tuple[str, ROI] | None:
    dir_240 = os.path.join(VIDEOS_ROOT, day, "240")
    if not os.path.isdir(dir_240):
        return None
    for name in sorted(os.listdir(dir_240)):
        if name in rois:
            return name, rois[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera ROIs pros vídeos 60 FPS por escala.")
    parser.add_argument("--rois-json", default="rois/rois.json")
    parser.add_argument("--dry-run", action="store_true", help="Mostra sem salvar.")
    args = parser.parse_args()

    rois = load_rois(args.rois_json)
    print(f"ROIs carregadas: {len(rois)} entradas")

    new_entries: dict[str, ROI] = {}

    for day in CANONICAL_DAYS:
        match = find_240_roi_for_day(day, rois)
        if match is None:
            print(f"[{day}] sem ROI 240 anotada — pulando")
            continue
        ref_name, ref_roi = match

        dir_240 = os.path.join(VIDEOS_ROOT, day, "240")
        dir_60 = os.path.join(VIDEOS_ROOT, day, "60")
        if not os.path.isdir(dir_60):
            print(f"[{day}] sem pasta 60/ — pulando")
            continue

        ref_240_path = os.path.join(dir_240, ref_name)
        h_240, w_240 = first_frame_shape(ref_240_path)

        for vid_name in sorted(os.listdir(dir_60)):
            if not vid_name.endswith(".mp4"):
                continue
            vid_60_path = os.path.join(dir_60, vid_name)
            h_60, w_60 = first_frame_shape(vid_60_path)

            scale_h = h_60 / h_240
            scale_w = w_60 / w_240
            if abs(scale_h - scale_w) > 1e-6:
                print(f"[{day}] aspect mismatch ({scale_h:.3f} vs {scale_w:.3f}) — usando média")
            scale = 0.5 * (scale_h + scale_w)

            new_roi = scale_roi(ref_roi, scale)
            new_entries[vid_name] = new_roi
            print(f"  [{day}] {vid_name}: 240({w_240}x{h_240}) → 60({w_60}x{h_60}) "
                  f"scale={scale:.3f} ROI={new_roi.to_list()}")

    if args.dry_run:
        print(f"\nDRY-RUN: {len(new_entries)} ROIs novas (não salvas)")
        return

    rois.update(new_entries)
    save_rois(rois, args.rois_json)
    print(f"\nSalvou {len(new_entries)} ROIs novas em {args.rois_json} "
          f"(total agora: {len(rois)})")


if __name__ == "__main__":
    main()
