"""Core data models for scale-validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ItemStats:
    item: str
    mean: float
    std: float
    item_total_correlation: float
    alpha_if_deleted: float

    @property
    def flag(self) -> str:
        if self.item_total_correlation < 0.2:
            return "weak"
        if self.item_total_correlation < 0.3:
            return "borderline"
        return "ok"


@dataclass
class FactorRetention:
    n_factors_kaiser: int          # eigenvalues > 1
    n_factors_95pct: int           # cumulative variance > 95%
    eigenvalues: list[float] = field(default_factory=list)
    explained_variance_ratios: list[float] = field(default_factory=list)


@dataclass
class ScaleValidationResult:
    scale_name: str
    n_items: int
    n_respondents: int

    cronbach_alpha: float = 0.0
    alpha_interpretation: str = ""

    item_stats: list[ItemStats] = field(default_factory=list)
    inter_item_correlation_mean: float = 0.0
    inter_item_correlation_matrix: list[list[float]] = field(default_factory=list)

    factor_retention: Optional[FactorRetention] = None

    weak_items: list[str] = field(default_factory=list)
    overall_verdict: str = "adequate"   # "excellent" | "good" | "adequate" | "poor" | "unacceptable"

    narrative: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_name": self.scale_name,
            "n_items": self.n_items,
            "n_respondents": self.n_respondents,
            "cronbach_alpha": round(float(self.cronbach_alpha), 4),
            "alpha_interpretation": self.alpha_interpretation,
            "overall_verdict": self.overall_verdict,
            "inter_item_correlation_mean": round(float(self.inter_item_correlation_mean), 4),
            "weak_items": self.weak_items,
            "item_stats": [
                {
                    "item": s.item,
                    "mean": round(float(s.mean), 4),
                    "std": round(float(s.std), 4),
                    "item_total_correlation": round(float(s.item_total_correlation), 4),
                    "alpha_if_deleted": round(float(s.alpha_if_deleted), 4),
                    "flag": s.flag,
                }
                for s in self.item_stats
            ],
            "factor_retention": {
                "n_factors_kaiser": self.factor_retention.n_factors_kaiser,
                "n_factors_95pct": self.factor_retention.n_factors_95pct,
                "eigenvalues": [round(float(e), 4) for e in self.factor_retention.eigenvalues],
                "explained_variance_ratios": [
                    round(float(v), 4) for v in self.factor_retention.explained_variance_ratios
                ],
            } if self.factor_retention else None,
            "inter_item_correlation_matrix": [
                [round(float(v), 4) for v in row]
                for row in self.inter_item_correlation_matrix
            ],
            "narrative": self.narrative,
        }
