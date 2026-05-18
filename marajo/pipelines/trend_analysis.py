"""Análise de tendência (experimento 002).

Dado um OverTimeResult com `keep_fft_data=True`, extrai features espectrais
por vídeo e aplica testes de tendência separados em cada batch.

O sinal procurado: a feature deve ter tendência em `february` (p < α) e NÃO
ter tendência em `april` (p > α). Esse contraste é o detector de estresse.
"""

from __future__ import annotations

from dataclasses import dataclass

from marajo.config import PipelineConfig
from marajo.modal.spectral_features import video_features
from marajo.modal.trends import TrendResult, all_trend_tests
from marajo.pipelines.over_time import CompactVideoResult, OverTimeResult


@dataclass
class FeatureSeries:
    """Série temporal de uma única feature ao longo dos dias de cada batch."""

    feature: str
    by_batch: dict[str, list[float]]  # batch_name -> valores em ordem de dia


@dataclass
class FeatureTrend:
    """Resultado dos 3 testes de tendência aplicados em UM batch de UMA feature."""

    feature: str
    batch: str
    values: list[float]
    tests: dict[str, TrendResult]  # "mann_kendall"/"spearman"/"linear" -> resultado


@dataclass
class TrendAnalysisResult:
    feature_names: list[str]
    series_by_feature: dict[str, FeatureSeries]
    trends: list[FeatureTrend]  # uma entrada por (feature, batch)
    freq_min: float
    top_k: int

    def trend(self, feature: str, batch: str) -> FeatureTrend:
        for t in self.trends:
            if t.feature == feature and t.batch == batch:
                return t
        raise KeyError(f"trend não encontrada: feature={feature!r} batch={batch!r}")

    def separates(self, feature: str, alpha: float = 0.05) -> dict[str, bool]:
        """Por batch, indica se a feature mostra tendência sob algum dos 3 testes (p < α)."""
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
) -> TrendAnalysisResult:
    """Roda a análise de tendência sobre um OverTimeResult que tenha keep_fft_data=True."""
    # Extrai features de cada vídeo
    features_per_video: dict[str, dict[str, float]] = {}
    for video_path, result in over_time_result.per_video.items():
        if not isinstance(result, CompactVideoResult) or result.fft_data is None:
            # full result tem .fft_data atributo
            fft_data = getattr(result, "fft_data", None)
            if fft_data is None:
                raise RuntimeError(
                    f"Vídeo {video_path} não tem fft_data. Rode run_over_time com keep_fft_data=True."
                )
        else:
            fft_data = result.fft_data
        features_per_video[video_path] = video_features(
            fft_data, freq_min=freq_min, top_k=top_k
        )

    # Organiza features em séries temporais por batch
    feature_names = list(next(iter(features_per_video.values())).keys())
    series_by_feature: dict[str, FeatureSeries] = {}
    for feat in feature_names:
        by_batch: dict[str, list[float]] = {}
        for batch_name, batch_videos in over_time_result.batches.items():
            by_batch[batch_name] = [features_per_video[v][feat] for v in batch_videos]
        series_by_feature[feat] = FeatureSeries(feature=feat, by_batch=by_batch)

    # Aplica os 3 testes em cada (feature, batch)
    trends: list[FeatureTrend] = []
    for feat, series in series_by_feature.items():
        for batch, values in series.by_batch.items():
            trends.append(
                FeatureTrend(
                    feature=feat,
                    batch=batch,
                    values=values,
                    tests=all_trend_tests(values),
                )
            )

    return TrendAnalysisResult(
        feature_names=feature_names,
        series_by_feature=series_by_feature,
        trends=trends,
        freq_min=freq_min,
        top_k=top_k,
    )
