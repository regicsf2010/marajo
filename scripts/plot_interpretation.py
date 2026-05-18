"""Gera plots didáticos pra interpretação dos resultados:

B. Vinhetas dos 3 testes de tendência (Mann-Kendall, Spearman, regressão linear)
   usando dados reais do detector `energy_0.5_2.0_median` em february.

D. Grid 2x2 com evolução dos heatmaps de p-value (002 → 003 → 004 → 004.2).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy import stats

from marajo.config import PipelineConfig
from marajo.decomposition.cp import run_cp_on_components
from marajo.decomposition.pca import compute_pca
from marajo.io.video import load_grayscale_dataset, video_status
from marajo.modal.fft import compute_fft_for_components
from marajo.modal.spectral_features import band_features, video_band_features, video_features
from marajo.modal.trends import all_trend_tests, mann_kendall


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> int:
    m = _DAY_RE.search(path)
    return int(m.group(1)) if m else 0


def extract_feature_per_video(
    config: PipelineConfig,
    out_dir: str,
    band: tuple[float, float],
    aggregate: str = "median",
) -> tuple[dict[str, list[tuple[int, float, int]]], list[str]]:
    """Pra cada batch, retorna lista de (dia, valor da feature, idx_angulo_no_dia)."""
    batches = {"february": list(config.batches.february), "april": list(config.batches.april)}
    result: dict[str, list[tuple[int, float, int]]] = {b: [] for b in batches}
    feat_key = f"energy_{band[0]:.1f}_{band[1]:.1f}_{aggregate}"

    for batch_name, batch_videos in batches.items():
        day_counter: dict[int, int] = {}
        for vp in batch_videos:
            pp = os.path.join(out_dir, os.path.basename(vp))
            dataset = load_grayscale_dataset(pp)
            info = video_status(pp)
            pca = compute_pca(dataset, n_components=config.decomposition.num_pcs)
            cp = run_cp_on_components(pca, n_pc=config.decomposition.num_pcs, config=config.cp)
            fft_data = compute_fft_for_components(cp.unmixed, info.fps, range(config.decomposition.num_pcs))
            feats = video_band_features(fft_data, [band])
            value = feats[feat_key]

            day = day_from_path(vp)
            idx = day_counter.get(day, 0)
            day_counter[day] = idx + 1
            result[batch_name].append((day, value, idx))
            del dataset, pca, cp, fft_data

    return result, list(batches.keys())


def make_didactic_plot(feb_data: list[tuple[int, float, int]], out_path: str) -> None:
    """4 subplots: (1) série temporal; (2) MK pares destacados; (3) Spearman ranks;
    (4) linear regression + resíduos."""

    days = np.array([d[0] for d in feb_data])
    values = np.array([d[1] for d in feb_data])
    angles = np.array([d[2] for d in feb_data])

    # Mapeia dia YYYYMMDD pra ordinal 1..8
    unique_days = sorted(set(days))
    day_to_ord = {d: i + 1 for i, d in enumerate(unique_days)}
    x = np.array([day_to_ord[d] for d in days])

    n = len(values)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # ---- Subplot 1: série temporal com cor por ângulo ----
    ax = axes[0, 0]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # jitter pequeno em x pra separar os 4 ângulos
    for a in range(4):
        mask = angles == a
        jitter = (a - 1.5) * 0.08
        ax.scatter(x[mask] + jitter, values[mask], color=colors[a], label=f"ângulo {a+1}",
                   s=60, alpha=0.85, edgecolor="black", linewidth=0.4)
    # Linha de tendência (linregress sobre todos os pontos)
    slope, intercept, r, p, _ = stats.linregress(x, values)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, "k--", linewidth=1.5,
            label=f"regressão linear (slope={slope:+.2f})")
    ax.set_xticks(range(1, len(unique_days) + 1))
    ax.set_xlabel("dia ordinal do batch", fontsize=11)
    ax.set_ylabel("energy_0.5_2.0_median", fontsize=11)
    ax.set_title("(1) Série temporal — february (n=32)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 2: Mann-Kendall — pares concordantes vs discordantes ----
    ax = axes[0, 1]
    # Plota todos em cinza
    ax.scatter(x, values, color="lightgray", s=60, edgecolor="black", linewidth=0.4, zorder=1)
    # Destaca 2 pares: um concordante, um discordante
    # Concordante: encontra par (i, j) com x_i < x_j e y_j > y_i, idealmente o de maior diferença
    pairs_concord = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if x[i] < x[j] and values[j] > values[i]]
    pairs_discord = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if x[i] < x[j] and values[j] < values[i]]
    # Pega os "mais saudáveis" (maior delta)
    pc = max(pairs_concord, key=lambda p: (values[p[1]] - values[p[0]]) * (x[p[1]] - x[p[0]]))
    pd = max(pairs_discord, key=lambda p: (values[p[0]] - values[p[1]]) * (x[p[1]] - x[p[0]]))

    for (i, j), color, label in [(pc, "tab:green", "concordante (+1)"),
                                  (pd, "tab:red", "discordante (−1)")]:
        ax.scatter([x[i], x[j]], [values[i], values[j]], color=color, s=140, zorder=3,
                   edgecolor="black", linewidth=1.0)
        arr = FancyArrowPatch(
            (x[i], values[i]), (x[j], values[j]),
            arrowstyle="->", color=color, linewidth=2, mutation_scale=20, zorder=2,
        )
        ax.add_patch(arr)

    # Anotação
    n_pares_total = n * (n - 1) // 2
    ties_x = sum(1 for i in range(n) for j in range(i + 1, n) if x[i] == x[j])
    n_pares_validos = n_pares_total - ties_x
    n_concord = len(pairs_concord)
    n_discord = len(pairs_discord)
    s_val = n_concord - n_discord
    tau = s_val / n_pares_validos if n_pares_validos > 0 else 0.0
    ax.set_xticks(range(1, len(unique_days) + 1))
    ax.set_xlabel("dia ordinal", fontsize=11)
    ax.set_ylabel("energy_0.5_2.0_median", fontsize=11)
    ax.set_title("(2) Mann-Kendall — pares concordantes vs discordantes", fontsize=12, fontweight="bold")
    ax.text(
        0.02, 0.98,
        f"pares totais: {n_pares_total}\n"
        f"ties em x (mesmo dia): {ties_x}\n"
        f"pares válidos: {n_pares_validos}\n"
        f"concordantes: {n_concord}\n"
        f"discordantes: {n_discord}\n"
        f"S = {n_concord} − {n_discord} = {s_val}\n"
        f"τ = S / {n_pares_validos} = {tau:+.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    leg_handles = [
        plt.Line2D([0], [0], marker='o', color='tab:green', markersize=10, label='par concordante (y subiu)',
                   linewidth=0),
        plt.Line2D([0], [0], marker='o', color='tab:red', markersize=10, label='par discordante (y desceu)',
                   linewidth=0),
    ]
    ax.legend(handles=leg_handles, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 3: Spearman — original vs ranks ----
    ax = axes[1, 0]
    # Plota original como cinza claro no fundo + ranks como coloridos
    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(values)
    ax.scatter(ranks_x, ranks_y, color="tab:purple", s=60, edgecolor="black", linewidth=0.4, zorder=2,
               label="ranks (rank_x, rank_y)")
    # Reta y=x mostraria correlação perfeita
    rmax = max(ranks_x.max(), ranks_y.max())
    rho, p_sp = stats.spearmanr(x, values)
    # Reta de regressão dos ranks
    slope_r, intercept_r, _, _, _ = stats.linregress(ranks_x, ranks_y)
    xr = np.linspace(0, rmax, 100)
    ax.plot(xr, slope_r * xr + intercept_r, "k--", linewidth=1.5,
            label=f"reta dos ranks (ρ={rho:+.3f})")
    ax.set_xlabel("rank do dia (1..32)", fontsize=11)
    ax.set_ylabel("rank do valor (1..32)", fontsize=11)
    ax.set_title("(3) Spearman — correlação dos ranks", fontsize=12, fontweight="bold")
    ax.text(
        0.02, 0.98,
        "Spearman = Pearson aplicado\n"
        "aos RANKS de x e y.\n\n"
        f"ρ = {rho:+.3f}\n"
        f"p = {p_sp:.3f}\n\n"
        "Não assume linearidade nem\n"
        "distribuição normal — só ordem.",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 4: Linear regression + resíduos ----
    ax = axes[1, 1]
    slope, intercept, r, p_ln, stderr = stats.linregress(x, values)
    y_pred = slope * x + intercept
    # Pontos + resíduos
    for xi, yi, ypi in zip(x, values, y_pred):
        ax.plot([xi, xi], [yi, ypi], color="gray", linewidth=0.8, alpha=0.6, zorder=1)
    ax.scatter(x, values, color="tab:cyan", s=60, edgecolor="black", linewidth=0.4, zorder=3,
               label="observado")
    ax.scatter(x, y_pred, color="black", s=20, marker="x", zorder=4, label="predito (na reta)")
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, "k-", linewidth=1.8, zorder=2,
            label=f"y = {slope:+.3f}·x {intercept:+.2f}")
    ax.set_xticks(range(1, len(unique_days) + 1))
    ax.set_xlabel("dia ordinal", fontsize=11)
    ax.set_ylabel("energy_0.5_2.0_median", fontsize=11)
    ax.set_title("(4) Regressão linear — reta + resíduos", fontsize=12, fontweight="bold")
    ax.text(
        0.02, 0.98,
        f"slope = {slope:+.3f}\n"
        f"intercept = {intercept:+.2f}\n"
        f"r² = {r**2:.3f}\n"
        f"p = {p_ln:.3f}\n\n"
        "Resíduos (cinza vertical) = erro entre\n"
        "ponto observado e reta. A reta\n"
        "minimiza a soma dos resíduos².",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Como cada teste \"vê\" os mesmos dados — detector `energy_0.5_2.0_median` em february",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Vinheta didática salva em {out_path}")


def collect_pvalues_for_global(
    config: PipelineConfig,
    out_dir: str,
    freq_max: float | None,
    feature_names: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Re-roda análise pra um config, retorna dict[feature][batch] -> {mk, sp, lin}."""
    from marajo.pipelines.over_time import run_over_time
    from marajo.pipelines.trend_analysis import analyse_trends

    cfg = PipelineConfig.load("configs/004-lowpass.yaml" if freq_max == 30 else "configs/003-all-angles.yaml")
    cfg.modal.freq_max = freq_max
    cfg.modal.bands = []

    over_time = run_over_time(
        config=cfg, out_dir=out_dir, do_preprocess=False, keep_fft_data=True,
    )
    result = analyse_trends(over_time, cfg, x_extractor=lambda p: day_from_path(p))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for feat in feature_names:
        out[feat] = {}
        for batch in ("february", "april"):
            tr = result.trend(feat, batch)
            out[feat][batch] = {
                "mk": tr.tests["mann_kendall"].p_value,
                "sp": tr.tests["spearman"].p_value,
                "lin": tr.tests["linear"].p_value,
            }
    return out


def _heatmap(ax, p_data, feature_names, batches=("february", "april"), tests=("mk", "sp", "lin"), title=""):
    cols = [f"{b[:3]}\n{t.upper()}" for b in batches for t in tests]
    mat = np.full((len(feature_names), len(cols)), np.nan)
    for i, feat in enumerate(feature_names):
        c = 0
        for b in batches:
            for t in tests:
                if feat in p_data and b in p_data[feat]:
                    mat[i, c] = p_data[feat][b][t]
                c += 1
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    return im


def make_evolution_plot(out_dir_all: str, out_dir_eight: str, out_path: str) -> None:
    """Heatmaps lado a lado: 003 (n=32 sem filtro), 004 (n=32 freq_max=30), 004.2 (sub-bandas),
    + nota sobre o 002 (já está em out/experimentos/002/)."""
    from marajo.config import PipelineConfig as PC

    # Lista canônica de features pros painéis 003 e 004 (mesmas 12)
    global_features = [
        "highest_freq_mean", "highest_freq_median",
        "centroid_mean", "centroid_median",
        "spread_mean", "spread_median",
        "flatness_mean", "flatness_median",
        "top_k_freq_mean_mean", "top_k_freq_mean_median",
        "energy_concentration_mean", "energy_concentration_median",
    ]

    # 003: freq_max = None (espectro inteiro)
    print("Computando 003 (n=32, sem filtro)...")
    p003 = collect_pvalues_for_global(None, out_dir_all, freq_max=None, feature_names=global_features)

    # 004: freq_max = 30
    print("Computando 004 (n=32, freq_max=30)...")
    p004 = collect_pvalues_for_global(None, out_dir_all, freq_max=30.0, feature_names=global_features)

    # 004.2: precisa rodar com bands. Vamos chamar direto.
    print("Computando 004.2 (sub-bandas finas)...")
    from marajo.pipelines.over_time import run_over_time
    from marajo.pipelines.trend_analysis import analyse_trends
    cfg42 = PC.load("configs/004-2-fine-bands.yaml")
    ot42 = run_over_time(cfg42, out_dir=out_dir_all, do_preprocess=False, keep_fft_data=True)
    tr42 = analyse_trends(ot42, cfg42, x_extractor=lambda p: day_from_path(p))
    p042 = {}
    for feat in tr42.feature_names:
        p042[feat] = {}
        for batch in ("february", "april"):
            t = tr42.trend(feat, batch)
            p042[feat][batch] = {
                "mk": t.tests["mann_kendall"].p_value,
                "sp": t.tests["spearman"].p_value,
                "lin": t.tests["linear"].p_value,
            }

    # 002: n=8 (1 vídeo por dia). Vamos rodar uma config reduzida.
    print("Computando 002 (n=8)...")
    cfg002 = PC.load("configs/003-all-angles.yaml")
    # filtra batches só com o primeiro vídeo de cada dia
    def first_per_day(paths):
        seen = {}
        out = []
        for p in paths:
            d = day_from_path(p)
            if d not in seen:
                seen[d] = True
                out.append(p)
        return out
    cfg002.batches.february = first_per_day(cfg002.batches.february)
    cfg002.batches.april = first_per_day(cfg002.batches.april)
    cfg002.modal.freq_max = None
    cfg002.modal.bands = []
    ot002 = run_over_time(cfg002, out_dir=out_dir_all, do_preprocess=False, keep_fft_data=True)
    tr002 = analyse_trends(ot002, cfg002)  # sem x_extractor: usa índice (n=8 sem multiplicação)
    p002 = {}
    for feat in global_features:
        p002[feat] = {}
        for batch in ("february", "april"):
            t = tr002.trend(feat, batch)
            p002[feat][batch] = {
                "mk": t.tests["mann_kendall"].p_value,
                "sp": t.tests["spearman"].p_value,
                "lin": t.tests["linear"].p_value,
            }

    # Plot em grid 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 18), constrained_layout=True)

    im1 = _heatmap(axes[0, 0], p002, global_features, title="002 — n=8 (1 ângulo/dia), espectro inteiro")
    im2 = _heatmap(axes[0, 1], p003, global_features, title="003 — n=32 (4 ângulos/dia), espectro inteiro")
    im3 = _heatmap(axes[1, 0], p004, global_features, title="004 — n=32, freq_max=30 (sem banda alta)")
    # 004.2 tem 40 features — usa todas
    im4 = _heatmap(axes[1, 1], p042, tr42.feature_names, title="004.2 — n=32, 5 sub-bandas (0.5–30 Hz)")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.6, location="right", pad=0.02, label="p-value")

    fig.suptitle(
        "Evolução dos heatmaps de p-value — verde escuro = p < 0.05 (passa o teste)",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Evolução de heatmaps salva em {out_path}")

    return {
        "002_n8_full_spectrum": p002,
        "003_n32_full_spectrum": p003,
        "004_n32_freq_max_30": p004,
        "004_2_n32_fine_bands": p042,
    }


def _save_didactic_data(feb_data, apr_data, out_root: str) -> None:
    """Dump CSV/JSON dos dados que alimentam a vinheta didática."""
    csv_path = os.path.join(out_root, "detector_raw_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "day", "angle_idx", "energy_0.5_2.0_median"])
        for day, val, ang in feb_data:
            writer.writerow(["february", day, ang, f"{val:.6f}"])
        for day, val, ang in apr_data:
            writer.writerow(["april", day, ang, f"{val:.6f}"])
    print(f"[dados brutos] {csv_path}")


def _save_evolution_data(panels: dict, out_root: str) -> None:
    """Dump dos p-values por painel/feature/batch/teste pra o plot de evolução."""
    csv_path = os.path.join(out_root, "evolucao_pvalues.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "feature", "batch", "test", "p_value"])
        for exp_name, p_data in panels.items():
            for feat, batches in p_data.items():
                for batch, tests in batches.items():
                    for test_name, p in tests.items():
                        writer.writerow([exp_name, feat, batch, test_name, f"{p:.6f}"])
    json_path = os.path.join(out_root, "evolucao_pvalues.json")
    with open(json_path, "w") as f:
        json.dump(panels, f, indent=2, default=float)
    print(f"[dados brutos] {csv_path}")
    print(f"[dados brutos] {json_path}")


def main() -> None:
    out_root = "out/plots-interpretacao"
    os.makedirs(out_root, exist_ok=True)

    config_003 = PipelineConfig.load("configs/003-all-angles.yaml")
    out_dir_all = "out/all_angles/"

    print("Extraindo dados do detector em february e april...")
    feat_data, _ = extract_feature_per_video(
        config_003, out_dir_all, band=(0.5, 2.0), aggregate="median"
    )
    feb_data = feat_data["february"]
    apr_data = feat_data["april"]
    _save_didactic_data(feb_data, apr_data, out_root)

    print("\nGerando vinheta didática (B)...")
    make_didactic_plot(feb_data, os.path.join(out_root, "vinhetas_didaticas.png"))

    print("\nGerando evolução de heatmaps (D)...")
    panels = make_evolution_plot(out_dir_all, out_dir_all, os.path.join(out_root, "evolucao_heatmaps.png"))
    _save_evolution_data(panels, out_root)


if __name__ == "__main__":
    main()
