"""Testes de tendência aplicados a séries curtas (ex.: 8 dias de um batch).

Os 3 testes são complementares:
  - mann_kendall: não-paramétrico, detecta tendência monotônica (não assume linearidade).
  - spearman: também não-paramétrico, baseado em ranks; sensível a correlação ordinal.
  - linear_trend: paramétrico, mede slope linear (sensível mas menos robusto).

Convenção: convergência dos 3 = forte evidência de tendência (ou da ausência dela).
Divergência = motivo pra olhar o sinal com mais cuidado (outliers, não-monotonicidade).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class TrendResult:
    test: str            # "mann_kendall", "spearman", "linear"
    statistic: float     # tau, rho, ou slope conforme o teste
    p_value: float
    slope: float | None = None  # sempre preenchido pra linear; None pros outros

    def has_trend(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def mann_kendall(values: Sequence[float]) -> TrendResult:
    """Mann-Kendall: testa monotonicidade. tau ∈ [-1, 1]; sinal indica direção."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return TrendResult(test="mann_kendall", statistic=0.0, p_value=1.0)

    # S = soma de sinais
    S = 0
    for i in range(n - 1):
        S += int(np.sum(np.sign(x[i + 1 :] - x[i])))

    # Variância (sem ajuste de ties — ok pra floats sem repetição exata).
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Tau de Kendall.
    tau = 2.0 * S / (n * (n - 1))

    # Z com correção de continuidade.
    if S > 0:
        z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return TrendResult(test="mann_kendall", statistic=float(tau), p_value=float(p))


def spearman_trend(values: Sequence[float]) -> TrendResult:
    """Correlação de Spearman entre o índice (dia) e o valor."""
    x = np.arange(len(values))
    y = np.asarray(values, dtype=float)
    if len(y) < 3:
        return TrendResult(test="spearman", statistic=0.0, p_value=1.0)
    rho, p = stats.spearmanr(x, y)
    return TrendResult(test="spearman", statistic=float(rho), p_value=float(p))


def linear_trend(values: Sequence[float]) -> TrendResult:
    """Regressão linear; statistic = slope, p_value do slope ≠ 0."""
    x = np.arange(len(values))
    y = np.asarray(values, dtype=float)
    if len(y) < 3:
        return TrendResult(test="linear", statistic=0.0, p_value=1.0, slope=0.0)
    result = stats.linregress(x, y)
    return TrendResult(
        test="linear",
        statistic=float(result.slope),
        p_value=float(result.pvalue),
        slope=float(result.slope),
    )


def all_trend_tests(values: Sequence[float]) -> dict[str, TrendResult]:
    return {
        "mann_kendall": mann_kendall(values),
        "spearman": spearman_trend(values),
        "linear": linear_trend(values),
    }
