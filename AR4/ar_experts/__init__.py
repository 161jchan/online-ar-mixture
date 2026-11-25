"""Allow short imports such as:
    from ar_experts import OnlineARExperts

The calling program decides when to generate data, initialize the experts,
predict, and update.
"""

from .data import DataConfig, generate_stationary_ar4
from .experts import ExpertConfig, OnlineARExperts

__all__ = ["DataConfig", "generate_stationary_ar4", "ExpertConfig", "OnlineARExperts"]
