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


def _default_x(values: Sequence[float]) -> np.ndarray:
    return np.arange(len(values), dtype=float)


def mann_kendall(values: Sequence[float], x: Sequence[float] | None = None) -> TrendResult:
    """Mann-Kendall: testa monotonicidade. tau ∈ [-1, 1]; sinal indica direção.

    Quando `x` é fornecido e contém pares (x_i, x_j) com x_i == x_j (ex.: 4 ângulos
    do mesmo dia), esses pares são IGNORADOS na soma — não dá pra inferir
    monotonicidade entre amostras que compartilham o mesmo instante temporal.
    A variância é ajustada pelo número efetivo de pares válidos.
    """
    y = np.asarray(values, dtype=float)
    x_arr = _default_x(values) if x is None else np.asarray(x, dtype=float)
    n = len(y)
    if n < 3:
        return TrendResult(test="mann_kendall", statistic=0.0, p_value=1.0)

    # Soma de sinais, ignorando pares com mesmo x.
    S = 0
    n_pairs = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x_arr[i] == x_arr[j]:
                continue
            S += int(np.sign(y[j] - y[i]))
            n_pairs += 1

    if n_pairs == 0:
        return TrendResult(test="mann_kendall", statistic=0.0, p_value=1.0)

    # Variância: aproximação por nº de pares válidos.
    # (Para pares 'ignorados' por igualdade de x, eles funcionam como ties em x,
    # então a variância efetiva escala com n_pairs.)
    var_s = n_pairs * (2 * n + 5) / 9.0

    tau = S / n_pairs

    if S > 0:
        z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return TrendResult(test="mann_kendall", statistic=float(tau), p_value=float(p))


def spearman_trend(values: Sequence[float], x: Sequence[float] | None = None) -> TrendResult:
    """Correlação de Spearman entre x (default: índice) e o valor.

    Spearman lida nativamente com ties em x (média dos ranks).
    """
    y = np.asarray(values, dtype=float)
    x_arr = _default_x(values) if x is None else np.asarray(x, dtype=float)
    if len(y) < 3:
        return TrendResult(test="spearman", statistic=0.0, p_value=1.0)
    rho, p = stats.spearmanr(x_arr, y)
    return TrendResult(test="spearman", statistic=float(rho), p_value=float(p))


def linear_trend(values: Sequence[float], x: Sequence[float] | None = None) -> TrendResult:
    """Regressão linear; statistic = slope, p_value do slope ≠ 0.

    Aceita x explícito (default: índice). Funciona com valores repetidos em x.
    """
    y = np.asarray(values, dtype=float)
    x_arr = _default_x(values) if x is None else np.asarray(x, dtype=float)
    if len(y) < 3:
        return TrendResult(test="linear", statistic=0.0, p_value=1.0, slope=0.0)
    result = stats.linregress(x_arr, y)
    return TrendResult(
        test="linear",
        statistic=float(result.slope),
        p_value=float(result.pvalue),
        slope=float(result.slope),
    )


def all_trend_tests(
    values: Sequence[float],
    x: Sequence[float] | None = None,
) -> dict[str, TrendResult]:
    return {
        "mann_kendall": mann_kendall(values, x=x),
        "spearman": spearman_trend(values, x=x),
        "linear": linear_trend(values, x=x),
    }
