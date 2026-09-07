"""Shared odds and payout helpers.

These helpers are sport-agnostic. Sport-specific feature engineering and model
choices should stay under `sports/<sport>/`.
"""

import math


def american_to_implied_probability(american_odds: float) -> float:
    """Convert American odds to implied win probability."""
    odds = float(american_odds)
    if odds > 0:
        return 100.0 / (100.0 + odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def payout_profit_per_dollar(american_odds: float) -> float:
    """Return net profit per $1 staked if a bet wins."""
    odds = float(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def has_positive_edge(model_probability: float, american_odds: float, threshold: float = 0.0) -> bool:
    """Return whether model probability beats market implied probability by threshold."""
    try:
        odds = float(american_odds)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(odds) or odds == 0:
        return False
    implied_probability = american_to_implied_probability(american_odds)
    return float(model_probability) - implied_probability > threshold
