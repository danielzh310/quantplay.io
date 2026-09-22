"""Pregame sizing diagnostics, independent of which prediction features win.

Thresholds and discounts below are explicit risk-policy choices, not parameters
optimized on target-week profit. These are not individual confidence guarantees.
"""
import math

import pandas as pd

EXTREME_PROBABILITY = .75
LARGE_MARKET_GAP = .15
MIN_COMPARABLE_GAMES = 30


def extreme_probability_check(probability, market, reliability):
    """Downward-only correction from prior, comparable out-of-fold forecasts."""
    high = probability >= EXTREME_PROBABILITY
    disagreement = math.isfinite(market) and probability - market >= LARGE_MARKET_GAP
    if not high and not disagreement:
        return probability, 1., 0, "Ordinary estimate"
    # Include both orientations of each historical matchup. The high/gap masks
    # cannot select both sides of the same game at these thresholds.
    history = pd.concat([
        reliability[["p", "y", "market"]],
        reliability[["p", "y", "market"]].rsub(1),
    ], ignore_index=True)
    mask = pd.Series(True, index=history.index)
    if high:
        mask &= history.p.ge(EXTREME_PROBABILITY)
    if disagreement:
        mask &= history.p.sub(history.market).ge(LARGE_MARKET_GAP)
    if math.isfinite(market):
        mask &= history.market.notna() & history.market.ge(.5).eq(market >= .5)
    group = history[mask]
    n = len(group)
    # Smooth small samples toward no measured bias; separately discount stake
    # for weak support rather than pretending the estimate has been validated.
    overstatement = max(0., float((group.p - group.y).mean())) if n else 0.
    correction = overstatement * n / (n + MIN_COMPARABLE_GAMES)
    factor = .5 + .5 * min(1., n / MIN_COMPARABLE_GAMES)
    reason = "Extreme estimate: prior overstatement adjustment" if correction > 0 else "Extreme estimate: no measured overstatement"
    if n < MIN_COMPARABLE_GAMES:
        reason += "; limited comparable history"
    return max(0., probability - correction), factor, n, reason
