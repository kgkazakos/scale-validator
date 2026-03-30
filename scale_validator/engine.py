"""Scale validation engine — Cronbach's alpha, item analysis, factor retention."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from scale_validator.models import (
    FactorRetention,
    ItemStats,
    ScaleValidationResult,
)


def _cronbach_alpha(data: np.ndarray) -> float:
    """
    Compute Cronbach's alpha for a matrix of item responses.

    Parameters
    ----------
    data : shape (n_respondents, n_items)
    """
    n_items = data.shape[1]
    if n_items < 2:
        return float("nan")
    item_variances = np.var(data, axis=0, ddof=1)
    total_variance = np.var(np.sum(data, axis=1), ddof=1)
    if total_variance == 0:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1.0 - np.sum(item_variances) / total_variance)
    return float(alpha)


def _alpha_if_deleted(data: np.ndarray, item_idx: int) -> float:
    """Cronbach's alpha with item `item_idx` removed."""
    reduced = np.delete(data, item_idx, axis=1)
    return _cronbach_alpha(reduced)


def _item_total_correlation(data: np.ndarray, item_idx: int) -> float:
    """Pearson correlation between item score and total scale score (item excluded)."""
    item = data[:, item_idx]
    rest = np.sum(np.delete(data, item_idx, axis=1), axis=1)
    if np.std(item) == 0 or np.std(rest) == 0:
        return 0.0
    return float(np.corrcoef(item, rest)[0, 1])


def _alpha_interpretation(alpha: float) -> str:
    if alpha >= 0.90:
        return "excellent"
    if alpha >= 0.80:
        return "good"
    if alpha >= 0.70:
        return "adequate"
    if alpha >= 0.60:
        return "questionable"
    if alpha >= 0.50:
        return "poor"
    return "unacceptable"


def _overall_verdict(alpha: float, weak_count: int, n_items: int) -> str:
    interp = _alpha_interpretation(alpha)
    if weak_count == 0:
        return interp
    # Downgrade one level if there are weak items
    order = ["excellent", "good", "adequate", "questionable", "poor", "unacceptable"]
    idx = order.index(interp) if interp in order else 5
    return order[min(idx + 1, len(order) - 1)]


def _factor_retention(data: np.ndarray) -> FactorRetention:
    """
    Compute eigenvalue-based factor retention criteria.

    Uses PCA on the inter-item correlation matrix:
    - Kaiser criterion: retain factors with eigenvalue > 1
    - 95% variance criterion: minimum factors to explain 95% of variance
    """
    corr = np.corrcoef(data.T)
    eigenvalues, _ = np.linalg.eigh(corr)
    # eigh returns ascending order — reverse
    eigenvalues = eigenvalues[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)  # clip floating point negatives

    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total if total > 0 else eigenvalues * 0
    cumulative = np.cumsum(ratios)

    n_kaiser = int(np.sum(eigenvalues > 1.0))
    n_95pct = int(np.searchsorted(cumulative, 0.95) + 1)

    return FactorRetention(
        n_factors_kaiser=max(1, n_kaiser),
        n_factors_95pct=min(n_95pct, len(eigenvalues)),
        eigenvalues=eigenvalues.tolist(),
        explained_variance_ratios=ratios.tolist(),
    )


def validate_scale(
    df: pd.DataFrame,
    items: list[str],
    scale_name: str = "scale",
    llm_backend: Optional[str] = None,
) -> ScaleValidationResult:
    """
    Validate a Likert-type scale.

    Parameters
    ----------
    df:
        DataFrame with one row per respondent and one column per item.
    items:
        List of item column names.
    scale_name:
        Name for the scale (used in output).
    llm_backend:
        Optional LLM backend for narrative interpretation.

    Returns
    -------
    ScaleValidationResult with Cronbach's alpha, item statistics,
    inter-item correlations, and factor retention guidance.
    """
    if len(items) < 2:
        raise ValueError("Need at least 2 items to validate a scale.")

    subset = df[items].dropna()
    data = subset.to_numpy(dtype=float)
    n_respondents, n_items = data.shape

    alpha = _cronbach_alpha(data)
    alpha_interp = _alpha_interpretation(alpha)

    # Item-level statistics
    item_stats_list: list[ItemStats] = []
    for i, item in enumerate(items):
        itc = _item_total_correlation(data, i)
        aid = _alpha_if_deleted(data, i)
        item_stats_list.append(
            ItemStats(
                item=item,
                mean=float(np.mean(data[:, i])),
                std=float(np.std(data[:, i], ddof=1)),
                item_total_correlation=itc,
                alpha_if_deleted=aid,
            )
        )

    weak_items = [s.item for s in item_stats_list if s.flag in ("weak", "borderline")]

    # Inter-item correlation matrix
    corr_matrix = np.corrcoef(data.T)
    # Mean of off-diagonal elements
    mask = ~np.eye(n_items, dtype=bool)
    mean_iic = float(np.mean(corr_matrix[mask]))

    # Factor retention
    retention = _factor_retention(data) if n_items >= 3 else None

    verdict = _overall_verdict(alpha, len([s for s in item_stats_list if s.flag == "weak"]), n_items)

    result = ScaleValidationResult(
        scale_name=scale_name,
        n_items=n_items,
        n_respondents=n_respondents,
        cronbach_alpha=alpha,
        alpha_interpretation=alpha_interp,
        item_stats=item_stats_list,
        inter_item_correlation_mean=mean_iic,
        inter_item_correlation_matrix=corr_matrix.tolist(),
        factor_retention=retention,
        weak_items=weak_items,
        overall_verdict=verdict,
    )

    if llm_backend:
        try:
            from scale_validator.narrator import ScaleNarrator
            narrator = ScaleNarrator(backend=llm_backend)
            result.narrative = narrator.narrate(result.to_dict())
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("LLM narration failed: %s", exc)

    return result
