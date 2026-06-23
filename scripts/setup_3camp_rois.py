"""Setup de ROIs da 3ª campanha (`3_camp/`) com 2 desenhos só.

O rig é fixo (câmera no tripé, marcações de fita na mesa como fiducial), então
cada planta tem o MESMO enquadramento em todos os dias/períodos. Logo bastam
2 ROIs: uma pra planta _1 (controle) e uma pra _2 (estressada).

Fluxo:
  1. Abre o frame de referência de cada planta (último dia full m/n = 20260606,
     onde a planta está na maior extensão) e você desenha a ROI generosa.
  2. Propaga cada ROI pros 28 vídeos daquela planta em `rois/rois.json`.
  3. Gera montage de validação (dia 1 e dia 13, manhã e noite, com a ROI
     sobreposta) em `out/experimentos/012/roi_check/`.

Uso (de dentro de `marajo/`, com display):
    python scripts/setup_3camp_rois.py
    python scripts/setup_3camp_rois.py --plant 2      # redesenhar só uma planta
    python scripts/setup_3camp_rois.py --validate-only  # só refazer o montage
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.io.roi import ROI, load_rois, save_rois, select_roi_interactively

CAMP_ROOT = "/home/ygarasab/mestrado/tese/3_camp"
JSON_PATH = "rois/rois.json"
REF_DAY = "20260606"  # último dia com m/n (20260607_1 usa 240/) → planta na maior extensão
CHECK_DIR = "out/experimentos/012/roi_check"


def _first_mp4(d: str) -> str | None:
    if not os.path.isdir(d):
        return None
    vids = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
    return os.path.join(d, vids[0]) if vids else None


def plant_videos(plant: str) -> list[str]:
    """Todos os caminhos de vídeo da planta (_1 ou _2) em 3_camp."""
    out: list[str] = []
    for day in sorted(os.listdir(CAMP_ROOT)):
        day_dir = os.path.join(CAMP_ROOT, day)
        if not (os.path.isdir(day_dir) and day.isdigit()):
            continue
        for session in sorted(os.listdir(day_dir)):
            if not session.endswith(f"_{plant}"):
                continue
            sess_dir = os.path.join(day_dir, session)
            for per in sorted(os.listdir(sess_dir)):  # m, n, ou 240
                per_dir = os.path.join(sess_dir, per)
                v = _first_mp4(per_dir)
                if v:
                    out.append(v)
    return out


def ref_video(plant: str) -> str:
    """Frame de referência: 20260606/_{plant}/m."""
    d = os.path.join(CAMP_ROOT, REF_DAY, f"{REF_DAY}_{plant}", "m")
    v = _first_mp4(d)
    if not v:
        raise RuntimeError(f"Sem vídeo de referência em {d}")
    return v


def draw_and_propagate(plant: str, rois: dict[str, ROI]) -> None:
    ref = ref_video(plant)
    tag = "ESTRESSADA (_2)" if plant == "2" else "controle (_1)"
    print(f"\n=== Planta {tag} — desenhe a ROI generosa no frame de {REF_DAY} ===")
    print(f"    ref: {ref}")
    roi = select_roi_interactively(ref)
    if roi is None:
        print("    cancelado, nada propagado pra essa planta.")
        return
    vids = plant_videos(plant)
    for v in vids:
        rois[os.path.basename(v)] = roi
    save_rois(rois, JSON_PATH)
    print(f"    ROI {roi.to_list()} propagada pra {len(vids)} vídeos.")


def validate(plants: list[str]) -> None:
    os.makedirs(CHECK_DIR, exist_ok=True)
    rois = load_rois(JSON_PATH)
    checks = []
    for plant in plants:
        for day in ("20260525", "20260606"):
            for per in ("m", "n"):
                d = os.path.join(CAMP_ROOT, day, f"{day}_{plant}", per)
                v = _first_mp4(d)
                if v:
                    checks.append((plant, day, per, v))
    for plant, day, per, v in checks:
        roi = rois.get(os.path.basename(v))
        if roi is None:
            print(f"sem ROI pra {os.path.basename(v)}")
            continue
        cap = cv.VideoCapture(v)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        cv.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 0, 255), 6)
        label = f"p{plant} {day} {per}"
        cv.putText(frame, label, (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
        out = os.path.join(CHECK_DIR, f"p{plant}_{day}_{per}.jpg")
        cv.imwrite(out, frame)
        print(f"  {out}")
    print(f"\nConfira o enquadramento em {CHECK_DIR}/ — a ROI deve conter a planta nos 4 frames de cada planta.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", choices=["1", "2"], default=None, help="desenhar só uma planta")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    plants = [args.plant] if args.plant else ["1", "2"]

    if not args.validate_only:
        rois = load_rois(JSON_PATH)
        for plant in plants:
            draw_and_propagate(plant, rois)

    validate(["1", "2"])


if __name__ == "__main__":
    main()
