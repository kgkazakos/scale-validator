"""scale-validator — Cronbach's alpha, item analysis, and factor retention for Likert scales."""

from scale_validator.engine import validate_scale
from scale_validator.models import ScaleValidationResult

__version__ = "0.1.0"
__all__ = ["validate_scale", "ScaleValidationResult"]
