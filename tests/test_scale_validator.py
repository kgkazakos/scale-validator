"""Tests for scale-validator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scale_validator.engine import (
    _cronbach_alpha,
    _item_total_correlation,
    _alpha_if_deleted,
    _alpha_interpretation,
    _factor_retention,
    validate_scale,
)
from scale_validator.models import ScaleValidationResult
from scale_validator.narrator import ScaleNarrator


RNG = np.random.default_rng(42)


def make_good_scale(n=200, n_items=6):
    """Scale with high internal consistency — items share a strong common factor."""
    factor = RNG.normal(0, 1, n)
    data = {f"q{i+1}": factor + RNG.normal(0, 0.3, n) for i in range(n_items)}
    return pd.DataFrame(data)


def make_poor_scale(n=200, n_items=6):
    """Scale with low internal consistency — items are essentially uncorrelated."""
    data = {f"q{i+1}": RNG.integers(1, 6, n).astype(float) for i in range(n_items)}
    return pd.DataFrame(data)


def make_scale_with_weak_item(n=200):
    """5-item scale where q5 is random noise."""
    factor = RNG.normal(0, 1, n)
    data = {f"q{i+1}": factor + RNG.normal(0, 0.3, n) for i in range(4)}
    data["q5"] = RNG.normal(0, 1, n)  # no relationship to factor
    return pd.DataFrame(data)


# --- low-level helpers ---

def test_cronbach_alpha_perfect():
    data = np.tile(np.arange(1, 101), (5, 1)).T.astype(float)
    alpha = _cronbach_alpha(data)
    assert alpha == pytest.approx(1.0, abs=0.01)


def test_cronbach_alpha_two_identical_items():
    vals = np.arange(1, 51).astype(float)
    data = np.column_stack([vals, vals])
    alpha = _cronbach_alpha(data)
    assert alpha == pytest.approx(1.0, abs=0.01)


def test_cronbach_alpha_single_item_nan():
    data = np.arange(1, 51).astype(float).reshape(-1, 1)
    assert np.isnan(_cronbach_alpha(data))


def test_cronbach_alpha_good_scale():
    df = make_good_scale()
    data = df.to_numpy()
    alpha = _cronbach_alpha(data)
    assert alpha > 0.85


def test_cronbach_alpha_poor_scale():
    df = make_poor_scale()
    data = df.to_numpy()
    alpha = _cronbach_alpha(data)
    assert alpha < 0.5


def test_item_total_correlation_high_for_good_items():
    df = make_good_scale()
    data = df.to_numpy()
    for i in range(data.shape[1]):
        itc = _item_total_correlation(data, i)
        assert itc > 0.5


def test_alpha_if_deleted_less_than_original_for_good_items():
    df = make_good_scale()
    data = df.to_numpy()
    original = _cronbach_alpha(data)
    for i in range(data.shape[1]):
        aid = _alpha_if_deleted(data, i)
        assert aid <= original + 0.05  # deleting a good item shouldn't improve alpha much


def test_alpha_interpretation_levels():
    assert _alpha_interpretation(0.95) == "excellent"
    assert _alpha_interpretation(0.85) == "good"
    assert _alpha_interpretation(0.75) == "adequate"
    assert _alpha_interpretation(0.65) == "questionable"
    assert _alpha_interpretation(0.55) == "poor"
    assert _alpha_interpretation(0.40) == "unacceptable"


# --- factor retention ---

def test_factor_retention_single_factor():
    df = make_good_scale(n=300, n_items=6)
    data = df.to_numpy()
    retention = _factor_retention(data)
    assert retention.n_factors_kaiser >= 1
    assert len(retention.eigenvalues) == 6
    assert len(retention.explained_variance_ratios) == 6


def test_factor_retention_eigenvalues_descending():
    df = make_good_scale()
    data = df.to_numpy()
    retention = _factor_retention(data)
    evs = retention.eigenvalues
    assert all(evs[i] >= evs[i + 1] for i in range(len(evs) - 1))


def test_factor_retention_variance_sums_to_one():
    df = make_good_scale()
    data = df.to_numpy()
    retention = _factor_retention(data)
    assert abs(sum(retention.explained_variance_ratios) - 1.0) < 0.01


# --- validate_scale ---

def test_validate_scale_good_alpha():
    df = make_good_scale()
    result = validate_scale(df, items=[f"q{i+1}" for i in range(6)])
    assert result.cronbach_alpha > 0.85
    assert result.overall_verdict in ("excellent", "good")


def test_validate_scale_weak_item_flagged():
    df = make_scale_with_weak_item()
    result = validate_scale(df, items=[f"q{i+1}" for i in range(5)])
    assert "q5" in result.weak_items


def test_validate_scale_n_items_n_respondents():
    df = make_good_scale(n=150, n_items=5)
    result = validate_scale(df, items=[f"q{i+1}" for i in range(5)])
    assert result.n_items == 5
    assert result.n_respondents == 150


def test_validate_scale_correlation_matrix_shape():
    df = make_good_scale(n_items=4)
    result = validate_scale(df, items=[f"q{i+1}" for i in range(4)])
    mat = result.inter_item_correlation_matrix
    assert len(mat) == 4
    assert all(len(row) == 4 for row in mat)


def test_validate_scale_to_dict_serializable():
    import json
    df = make_good_scale()
    result = validate_scale(df, items=[f"q{i+1}" for i in range(6)])
    serialized = json.dumps(result.to_dict())
    parsed = json.loads(serialized)
    assert "cronbach_alpha" in parsed
    assert "item_stats" in parsed


def test_validate_scale_too_few_items_raises():
    df = pd.DataFrame({"q1": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least 2"):
        validate_scale(df, items=["q1"])


# --- Narrator ---

def test_narrator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        ScaleNarrator(backend="cohere")


def test_narrator_default_models():
    assert "gpt" in ScaleNarrator(backend="openai").model
    assert "claude" in ScaleNarrator(backend="anthropic").model
    assert "gemini" in ScaleNarrator(backend="gemini").model


def test_narrator_mocked():
    narrator = ScaleNarrator(backend="gemini")
    df = make_good_scale()
    result = validate_scale(df, items=[f"q{i+1}" for i in range(6)])
    with patch.object(narrator, "_call_gemini", return_value="The scale shows strong consistency."):
        narrative = narrator.narrate(result.to_dict())
    assert "scale" in narrative.lower()


def test_narrator_returns_empty_on_error():
    narrator = ScaleNarrator(backend="openai")
    df = make_good_scale()
    result = validate_scale(df, items=[f"q{i+1}" for i in range(6)])
    with patch.object(narrator, "_call_openai", side_effect=ConnectionError("down")):
        assert narrator.narrate(result.to_dict()) == ""
