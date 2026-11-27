"""Import the stationary, switching and drifting generators from data."""

from .synthetic import DataConfig, generate_stationary_ar4, generate_switching_ar, generate_drifting_ar1

__all__ = ["DataConfig", "generate_stationary_ar4", "generate_switching_ar", "generate_drifting_ar1"]
