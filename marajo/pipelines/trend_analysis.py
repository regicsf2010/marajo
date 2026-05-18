"""Análise de tendência (experimentos 002 em diante).

Dado um OverTimeResult com `keep_fft_data=True`, extrai features espectrais
por vídeo e aplica testes de tendência separados em cada batch.

O sinal procurado: a feature deve ter tendência em `february` (p < α) e NÃO
ter tendência em `april` (p > α). Esse contraste é o detector de estresse.

Suporta x explícito (ex.: dia do calendário) pra quando há múltiplas observações
por dia (experimento 003 — 4 ângulos por dia).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from marajo.config import PipelineConfig
from marajo.modal.spectral_features import video_band_features, video_features
from marajo.modal.trends import TrendResult, all_trend_tests
from marajo.pipelines.over_time import CompactVideoResult, OverTimeResult


@dataclass
class FeatureSeries:
    """Série temporal de uma única feature ao longo dos dias de cada batch."""

    feature: str
    by_batch: dict[str, list[float]]                          # batch_name -> y (valores na ordem)
    x_by_batch: dict[str, list[float]] = field(default_factory=dict)  # batch_name -> x (ex.: dia)


@dataclass
class FeatureTrend:
    """Resultado dos 3 testes de tendência aplicados em UM batch de UMA feature."""

    feature: str
    batch: str
    values: list[float]
    x_values: list[float]
    tests: dict[str, TrendResult]


@dataclass
class TrendAnalysisResult:
    feature_names: list[str]
    series_by_feature: dict[str, FeatureSeries]
    trends: list[FeatureTrend]
    freq_min: float
    top_k: int

    def trend(self, feature: str, batch: str) -> FeatureTrend:
        for t in self.trends:
            if t.feature == feature and t.batch == batch:
                return t
        raise KeyError(f"trend não encontrada: feature={feature!r} batch={batch!r}")

    def separates(self, feature: str, alpha: float = 0.05) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for batch in {t.batch for t in self.trends}:
            t = self.trend(feature, batch)
            out[batch] = any(tr.p_value < alpha for tr in t.tests.values())
        return out


def analyse_trends(
    over_time_result: OverTimeResult,
    config: PipelineConfig,
    freq_min: float = 0.0,
    top_k: int = 5,
    x_extractor: Optional[Callable[[str], float]] = None,
) -> TrendAnalysisResult:
    """Roda a análise de tendência sobre um OverTimeResult com keep_fft_data=True.

    `x_extractor`: função que dado um video_path retorna o valor de x usado nos
    testes (ex.: dia do calendário). Se None, x é o índice 0..n-1 (default do 002).
    Útil pro 003 quando há múltiplos ângulos por dia.

    Se `config.modal.bands` está definido (lista não vazia), usa features POR
    BANDA em vez das features globais. Cada (banda × estatística) vira uma feature.
    """
    bands = config.modal.bands
    use_bands = bool(bands)
    if use_bands:
        band_tuples = [(float(low), float(high)) for low, high in bands]

    features_per_video: dict[str, dict[str, float]] = {}
    for video_path, result in over_time_result.per_video.items():
        fft_data = (
            result.fft_data
            if isinstance(result, CompactVideoResult)
            else getattr(result, "fft_data", None)
        )
        if fft_data is None:
            raise RuntimeError(
                f"Vídeo {video_path} não tem fft_data. Rode run_over_time com keep_fft_data=True."
            )
        if use_bands:
            features_per_video[video_path] = video_band_features(fft_data, band_tuples)
        else:
            features_per_video[video_path] = video_features(
                fft_data, freq_min=freq_min, top_k=top_k
            )

    feature_names = list(next(iter(features_per_video.values())).keys())

    # Para cada batch, monta x (default = índice) e y por vídeo
    series_by_feature: dict[str, FeatureSeries] = {}
    x_by_batch_global: dict[str, list[float]] = {}
    for batch_name, batch_videos in over_time_result.batches.items():
        if x_extractor is not None:
            x_by_batch_global[batch_name] = [float(x_extractor(v)) for v in batch_videos]
        else:
            x_by_batch_global[batch_name] = list(range(len(batch_videos)))

    for feat in feature_names:
        by_batch: dict[str, list[float]] = {}
        for batch_name, batch_videos in over_time_result.batches.items():
            by_batch[batch_name] = [features_per_video[v][feat] for v in batch_videos]
        series_by_feature[feat] = FeatureSeries(
            feature=feat,
            by_batch=by_batch,
            x_by_batch={b: list(x) for b, x in x_by_batch_global.items()},
        )

    trends: list[FeatureTrend] = []
    for feat, series in series_by_feature.items():
        for batch, values in series.by_batch.items():
            x_values = series.x_by_batch[batch]
            trends.append(
                FeatureTrend(
                    feature=feat,
                    batch=batch,
                    values=values,
                    x_values=x_values,
                    tests=all_trend_tests(values, x=x_values if x_extractor else None),
                )
            )

    return TrendAnalysisResult(
        feature_names=feature_names,
        series_by_feature=series_by_feature,
        trends=trends,
        freq_min=freq_min,
        top_k=top_k,
    )
