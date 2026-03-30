# scale-validator

Cronbach's alpha, item-level analysis, and factor retention guidance for Likert-type research scales — before you report findings from a scale you haven't validated.

```
$ scale-validator validate survey.csv \
    --items "q1,q2,q3,q4,q5,q6" --scale-name "trust_scale"

  scale-validator · trust_scale
  α = 0.847 (good)  |  Verdict: GOOD  |  6 items · 312 respondents  |  Mean IIC = 0.481

  Item    Mean   SD     Item-Total r   α if deleted   Flag
  q1      3.82   0.91   0.612          0.821          ok
  q2      3.71   0.88   0.589          0.824          ok
  q3      3.90   0.94   0.071          0.891          weak
  q4      3.65   0.97   0.634          0.818          ok
  q5      3.78   0.89   0.601          0.822          ok
  q6      3.83   0.92   0.558          0.827          ok

  Factor retention → Kaiser: 1 factor(s)  |  95% variance: 2 factor(s)
  Items to review: q3
```

## The problem

Reporting findings from a Likert scale without checking its psychometric properties is a common and quietly damaging practice. A scale with poor internal consistency is measuring noise as much as signal. An item with a low item-total correlation is pulling in a different direction from the rest. A scale that loads on two factors is measuring two different things — and your summary score is meaningless. `scale-validator` surfaces all of this in seconds.

## What it computes

**Cronbach's alpha** — the standard measure of internal consistency, with plain-language interpretation: excellent (≥0.90), good (≥0.80), adequate (≥0.70), questionable (≥0.60), poor (≥0.50), unacceptable (<0.50).

**Item-total correlations** — Pearson correlation between each item and the total scale score (with that item excluded). Items below 0.20 are flagged as weak; 0.20–0.30 as borderline.

**Alpha-if-deleted** — what Cronbach's alpha would be if this item were removed. Items where deletion would substantially increase alpha are candidates for removal.

**Inter-item correlation matrix** — full pairwise correlation matrix plus the mean off-diagonal correlation. Values between 0.15 and 0.50 are generally healthy for a unidimensional scale.

**Factor retention guidance** — eigenvalue decomposition of the inter-item correlation matrix, with two retention criteria:
- Kaiser criterion: retain factors with eigenvalue > 1
- 95% variance criterion: minimum factors needed to explain 95% of scale variance

## Install

```bash
pip install scale-validator
```

With optional LLM narrative layer:

```bash
pip install "scale-validator[openai]"
pip install "scale-validator[anthropic]"
pip install "scale-validator[gemini]"
pip install "scale-validator[all]"
```

## Usage

### CLI

```bash
# Basic validation
scale-validator validate survey.csv \
  --items "q1,q2,q3,q4,q5,q6" \
  --scale-name "trust_scale"

# Save JSON report
scale-validator validate survey.csv \
  --items "trust1,trust2,trust3,trust4,trust5" \
  --scale-name "trust" \
  --output trust_validation.json

# With LLM narrative
scale-validator validate survey.csv \
  --items "q1,q2,q3,q4,q5" \
  --llm-backend anthropic
```

### Python API

```python
import pandas as pd
from scale_validator import validate_scale

df = pd.read_csv("survey.csv")

result = validate_scale(
    df,
    items=["q1", "q2", "q3", "q4", "q5", "q6"],
    scale_name="trust_scale",
)

print(f"Cronbach's α = {result.cronbach_alpha:.3f} ({result.alpha_interpretation})")
print(f"Overall verdict: {result.overall_verdict}")
print(f"Mean IIC: {result.inter_item_correlation_mean:.3f}")

for item in result.item_stats:
    print(f"  {item.item}: r={item.item_total_correlation:.3f}, α-if-deleted={item.alpha_if_deleted:.3f} [{item.flag}]")

if result.weak_items:
    print(f"Items to review: {result.weak_items}")

if result.factor_retention:
    print(f"Kaiser factors: {result.factor_retention.n_factors_kaiser}")

import json
print(json.dumps(result.to_dict(), indent=2))
```

## Item flag thresholds

| Flag | Item-total correlation | Meaning |
|------|----------------------|---------|
| `ok` | ≥ 0.30 | Item contributes well to the scale |
| `borderline` | 0.20–0.29 | Worth reviewing |
| `weak` | < 0.20 | Consider dropping or rewriting |

## Output (JSON)

```json
{
  "scale_name": "trust_scale",
  "n_items": 6,
  "n_respondents": 312,
  "cronbach_alpha": 0.847,
  "alpha_interpretation": "good",
  "overall_verdict": "good",
  "inter_item_correlation_mean": 0.481,
  "weak_items": ["q3"],
  "item_stats": [...],
  "factor_retention": {
    "n_factors_kaiser": 1,
    "n_factors_95pct": 2,
    "eigenvalues": [3.21, 0.87, 0.63, ...],
    "explained_variance_ratios": [0.535, 0.145, ...]
  }
}
```

## Part of the CPR Ecosystem

`scale-validator` is Tool 4.8 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools in this series:**

| Tool | What it does |
|------|-------------|
| [`survey-cleaner`](https://github.com/kgkazakos/survey-cleaner) | Remove speeders, straight-liners and gibberish open-ends |
| [`bayes-ab`](https://github.com/kgkazakos/bayes-ab) | Bayesian A/B testing with posteriors and ROPE decision rules |
| [`regression-diagnostics`](https://github.com/kgkazakos/regression-diagnostics) | OLS assumption checks, VIF, Cook's distance |
| [`power-analyst`](https://github.com/kgkazakos/power-analyst) | Sample size and power curves for common test types |

Part of the broader **Computational Product Research (CPR)** discipline. Core projects — CausalTrack, Synthetic User Council, Co-Reasoning Canvas — are developed separately and target CHI 2027/2028.

## License

MIT
