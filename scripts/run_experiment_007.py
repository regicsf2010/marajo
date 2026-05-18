"""Experimento 007 — Modal matching com MAC.

Hipótese: agregar features por mean/median sobre os 10 CPs mistura modos físicos
diferentes (permutation/sign ambiguity do BSS). Se associarmos CPs do MESMO modo
físico entre vídeos via Modal Assurance Criterion (MAC), podemos comparar
apples-to-apples e o sinal direcional pode emergir mais limpo.

Pipeline:
1. Pra cada vídeo: PCA + CP + mode_shapes.
2. Resize mode_shapes pra tamanho espacial comum.
3. Escolhe vídeo de referência (primeiro do batch february).
4. Alinha CPs de cada vídeo ao referência via Hungarian sobre MAC.
5. Pra cada CP alinhado k, extrai energy_0.5_2.0 daquela posição em todos os
   vídeos (em vez de mean/median sobre CPs).
6. Aplica testes de tendência por (CP_alinhado_k, batch).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from marajo.config import PipelineConfig
from marajo.decomposition.cp import run_cp_on_components
from marajo.decomposition.pca import compute_pca
from marajo.io.video import load_grayscale_dataset, video_status
from marajo.modal.fft import compute_fft_for_components
from marajo.modal.matching import align_to_reference, resize_modes
from marajo.modal.mode_shapes import compute_mode_shapes
from marajo.modal.spectral_features import band_features
from marajo.modal.trends import all_trend_tests


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> float:
    m = _DAY_RE.search(path)
    return float(m.group(1)) if m else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modal matching com MAC.")
    p.add_argument("--config", default="configs/003-all-angles.yaml")
    p.add_argument("--out-dir", default="out/all_angles/")
    p.add_argument("--plots-dir", default="out/experimentos/007/")
    p.add_argument("--target-rows", type=int, default=50)
    p.add_argument("--target-cols", type=int, default=50)
    p.add_argument("--band", nargs=2, type=float, default=[0.5, 2.0])
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    batches = {"february": list(config.batches.february), "april": list(config.batches.april)}
    all_videos = batches["february"] + batches["april"]
    num_pcs = config.decomposition.num_pcs
    band_low, band_high = args.band

    # Etapa 1: pra cada vídeo, computa PCA + CP + mode_shapes + FFT
    print(f"Processando {len(all_videos)} vídeos (PCA+CP+mode_shapes)...")
    per_video: dict[str, dict] = {}
    for i, video_path in enumerate(all_videos):
        pp = os.path.join(args.out_dir, os.path.basename(video_path))
        dataset = load_grayscale_dataset(pp)
        info = video_status(pp)
        pca = compute_pca(dataset, n_components=num_pcs)
        cp = run_cp_on_components(pca, n_pc=num_pcs, config=config.cp)
        mode_shapes_raw = compute_mode_shapes(cp.W_inv, pca.score, num_pcs, list(range(num_pcs)))
        # Resize ao tamanho comum
        n_rows = int(info.shape[0])
        n_cols = int(info.shape[1])
        # info.shape vem do primeiro frame BGR (h, w, 3); usar h,w
        # mas dataset reshape: 800 frames × (n_rows * n_cols). Let's recover n_rows*n_cols
        # do len(dataset[0]).
        # Mais robusto: usar VideoInfo width/height (que já é shape do vídeo pré-processado)
        n_rows = int(info.height)
        n_cols = int(info.width)
        if n_rows * n_cols != mode_shapes_raw.shape[0]:
            # fallback: tenta deduzir de num_pixels = mode_shapes_raw.shape[0]
            total = mode_shapes_raw.shape[0]
            # Procura fatoração que case com (h, w) próximos a info
            n_cols = int(info.width)
            n_rows = total // n_cols
        modes_resized = resize_modes(mode_shapes_raw, n_rows, n_cols, args.target_rows, args.target_cols)
        fft_data = compute_fft_for_components(cp.unmixed, info.fps, range(num_pcs))
        per_video[video_path] = {
            "modes_resized": modes_resized,
            "fft_data": fft_data,
            "fps": info.fps,
        }
        del dataset, pca, cp, mode_shapes_raw
        print(f"  [{i+1}/{len(all_videos)}] {os.path.basename(video_path)}")

    # Etapa 2: alinha modes ao primeiro vídeo de february (referência)
    reference_path = batches["february"][0]
    print(f"\nReferência: {os.path.basename(reference_path)}")
    modes_list = [per_video[v]["modes_resized"] for v in all_videos]
    ref_idx = all_videos.index(reference_path)
    permutations, macs_chosen = align_to_reference(modes_list, reference_idx=ref_idx)

    # Estatísticas dos MACs do matching (sanity check: alinhamento é bom?)
    mac_means = [m.mean() for m in macs_chosen]
    print(f"MAC médio do matching: min={min(mac_means):.3f}, mediana={sorted(mac_means)[len(mac_means)//2]:.3f}, max={max(mac_means):.3f}")
    print("  (valores próximos de 1 = matching forte; próximos de 0 = mode shapes muito diferentes)")

    # Etapa 3: pra cada CP alinhado k, extrai feature em cada vídeo
    rows: list[dict] = []
    feature_name = f"energy_{band_low:.1f}_{band_high:.1f}"
    for k in range(num_pcs):
        for batch_name, batch_videos in batches.items():
            values = []
            x_values = []
            for v in batch_videos:
                # Encontra qual CP de v está alinhado ao k-ésimo CP da referência
                video_idx = all_videos.index(v)
                perm = permutations[video_idx]
                cp_in_video = int(perm[k])
                fft = per_video[v]["fft_data"][cp_in_video]
                feats = band_features(fft, band_low, band_high)
                values.append(feats["energy"])
                x_values.append(day_from_path(v))

            tests = all_trend_tests(values, x=x_values)
            mk = tests["mann_kendall"]
            sp = tests["spearman"]
            ln = tests["linear"]
            rows.append({
                "cp_aligned": k,
                "batch": batch_name,
                "mk_tau": mk.statistic, "mk_p": mk.p_value,
                "sp_rho": sp.statistic, "sp_p": sp.p_value,
                "lin_slope": ln.slope, "lin_p": ln.p_value,
            })

    # Etapa 4: tabela markdown
    print("\n## Tabela de resultados (energy na banda {:.1f}-{:.1f} Hz por CP alinhado)\n".format(band_low, band_high))
    print("| CP_alinhado | batch | MK τ | MK p | Sp ρ | Sp p | Lin slope | Lin p | trend? |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        any_sig = (r["mk_p"] < args.alpha) or (r["sp_p"] < args.alpha) or (r["lin_p"] < args.alpha)
        flag = "✅" if any_sig else "—"
        print(
            f"| {r['cp_aligned']} | {r['batch']} | {r['mk_tau']:+.3f} | {r['mk_p']:.3f} | "
            f"{r['sp_rho']:+.3f} | {r['sp_p']:.3f} | {r['lin_slope']:+.4f} | {r['lin_p']:.3f} | {flag} |"
        )

    # Etapa 5: lista detectores válidos (feb sig, apr n.s.)
    print(f"\n## Detectores candidatos (CP alinhado) com α={args.alpha}\n")
    detectors = []
    for k in range(num_pcs):
        feb = next(r for r in rows if r["cp_aligned"] == k and r["batch"] == "february")
        apr = next(r for r in rows if r["cp_aligned"] == k and r["batch"] == "april")
        feb_sig = any(feb[c] < args.alpha for c in ("mk_p", "sp_p", "lin_p"))
        apr_ns = all(apr[c] >= args.alpha for c in ("mk_p", "sp_p", "lin_p"))
        if feb_sig and apr_ns:
            detectors.append((k, feb, apr))
            print(
                f"- CP {k}: feb MK p={feb['mk_p']:.3f} (τ={feb['mk_tau']:+.2f}); "
                f"apr MK p={apr['mk_p']:.3f}"
            )
    if not detectors:
        print("_Nenhum CP alinhado satisfaz o critério._")

    # Etapa 6: heatmap p-values
    fig, ax = plt.subplots(figsize=(10, 6))
    matrix = np.zeros((num_pcs, 6))
    col_labels = ["feb MK", "feb Sp", "feb Lin", "apr MK", "apr Sp", "apr Lin"]
    for k in range(num_pcs):
        feb = next(r for r in rows if r["cp_aligned"] == k and r["batch"] == "february")
        apr = next(r for r in rows if r["cp_aligned"] == k and r["batch"] == "april")
        matrix[k, 0] = feb["mk_p"]
        matrix[k, 1] = feb["sp_p"]
        matrix[k, 2] = feb["lin_p"]
        matrix[k, 3] = apr["mk_p"]
        matrix[k, 4] = apr["sp_p"]
        matrix[k, 5] = apr["lin_p"]
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(num_pcs))
    ax.set_yticklabels([f"CP {k}" for k in range(num_pcs)])
    for i in range(num_pcs):
        for j in range(6):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"p-values do detector `{feature_name}` por CP alinhado")
    fig.colorbar(im, ax=ax, label="p-value")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_dir, "pvalue_by_aligned_cp.png"), bbox_inches="tight")
    print(f"\nHeatmap salvo: {args.plots_dir}/pvalue_by_aligned_cp.png")

    # Persiste resultados brutos
    csv_path = os.path.join(args.plots_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(args.plots_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config_path": args.config,
            "alpha": args.alpha,
            "band": [band_low, band_high],
            "target_rows": args.target_rows,
            "target_cols": args.target_cols,
            "reference_video": reference_path,
            "mac_mean_min": float(min(mac_means)),
            "mac_mean_median": float(sorted(mac_means)[len(mac_means) // 2]),
            "mac_mean_max": float(max(mac_means)),
            "rows": rows,
        }, f, indent=2, default=float)
    print(f"[dados brutos] {csv_path}")
    print(f"[dados brutos] {json_path}")


if __name__ == "__main__":
    main()
