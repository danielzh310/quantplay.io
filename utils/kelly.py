from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class KellyResult:
    """Result for a single pick."""
    stake: float
    kelly_fraction: float
    is_no_bet: bool
    reason: str  # "", or explanation for no-bet / clamped stake


def american_to_b(american_odds: float) -> float:
    """
    Convert American odds to b, the net profit per 1 unit staked.

    +A => b = A/100
    -A => b = 100/A

    Example:
      +200 => b=2.0
      -150 => b=100/150=0.6667
    """
    if american_odds == 0:
        raise ValueError("American odds cannot be 0.")
    if american_odds > 0:
        return american_odds / 100.0
    return 100.0 / abs(american_odds)


def kelly_fraction(p: float, american_odds: float) -> float:
    """
    Full Kelly fraction f* for a single bet.

    f* = (b*p - (1-p)) / b, where b is net profit per 1 unit staked.
    If f* <= 0, optimal Kelly is to not bet.

    Returns:
      f* (can be <= 0). Caller typically clamps at 0.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1]. Got {p}")
    b = american_to_b(american_odds)
    q = 1.0 - p
    return (b * p - q) / b


def stake_from_fraction(
    bankroll: float,
    f: float,
    *,
    fraction_of_kelly: float = 0.25,
    cap_fraction_of_bankroll: Optional[float] = None,
    min_stake: float = 0.0,
) -> KellyResult:
    """
    Convert a Kelly fraction into a dollar stake with risk controls.

    Parameters:
      bankroll: total bankroll (user input)
      f: full Kelly fraction (can be <= 0)
      fraction_of_kelly: e.g., 0.25 for quarter-Kelly, 0.5 for half-Kelly
      cap_fraction_of_bankroll: e.g., 0.05 caps any single bet at 5% of bankroll
      min_stake: if >0, stakes below this become 0 unless you prefer rounding up

    Returns:
      KellyResult with stake and metadata.
    """
    if bankroll < 0:
        raise ValueError("bankroll must be non-negative.")
    if fraction_of_kelly <= 0:
        raise ValueError("fraction_of_kelly must be > 0.")

    if f <= 0 or bankroll == 0:
        return KellyResult(stake=0.0, kelly_fraction=max(0.0, f), is_no_bet=True, reason="no edge")

    raw_fraction = f * fraction_of_kelly
    stake = bankroll * raw_fraction

    reason_parts: List[str] = []
    # Cap
    if cap_fraction_of_bankroll is not None:
        cap_amount = bankroll * cap_fraction_of_bankroll
        if stake > cap_amount:
            stake = cap_amount
            reason_parts.append(f"capped at {cap_fraction_of_bankroll:.2%} of bankroll")

    # Min stake filter
    if stake < min_stake:
        return KellyResult(stake=0.0, kelly_fraction=f, is_no_bet=True, reason=f"below min_stake {min_stake}")

    return KellyResult(stake=float(stake), kelly_fraction=float(f), is_no_bet=False, reason="; ".join(reason_parts))


def normalize_stakes_to_bankroll(
    stakes: Sequence[float],
    bankroll_to_deploy: float,
    *,
    only_scale_positive: bool = True,
) -> List[float]:
    """
    Reduce stakes only when their sum exceeds bankroll_to_deploy.
    Kept under the original name for compatibility; never increases exposure.

    Notes:
    - If only_scale_positive=True, we scale only stakes > 0 and keep zeros at zero.
    - If sum(stakes) == 0, returns zeros.

    Returns:
      scaled stakes list
    """
    if bankroll_to_deploy < 0:
        raise ValueError("bankroll_to_deploy must be non-negative.")

    if only_scale_positive:
        pos_sum = sum(s for s in stakes if s > 0)
        if pos_sum <= 0:
            return [0.0 for _ in stakes]
        factor = min(1.0, bankroll_to_deploy / pos_sum)
        return [float(s * factor) if s > 0 else 0.0 for s in stakes]

    total = sum(stakes)
    if total <= 0:
        return [0.0 for _ in stakes]
    factor = min(1.0, bankroll_to_deploy / total)
    return [float(s * factor) for s in stakes]


def allocate_kelly(
    probs: Sequence[float],
    odds: Sequence[float],
    bankroll: float,
    *,
    fraction_of_kelly: float = 0.25,
    normalize_to_full_bankroll: bool = False,
    cap_fraction_of_bankroll: Optional[float] = None,
    min_stake: float = 0.0,
) -> Tuple[List[float], List[KellyResult]]:
    """
    Main helper: compute stakes for a slate.

    Inputs:
      probs: p for the side you are betting (same order as odds)
      odds: American odds for the side you are betting
      bankroll: user bankroll

    Options:
      fraction_of_kelly: 0.25 = quarter Kelly
      normalize_to_full_bankroll: legacy argument; upward normalization is disabled
      cap_fraction_of_bankroll: cap any single bet, e.g. 0.08
      min_stake: drop tiny bets to 0

    Returns:
      (stakes, results) where:
        stakes = final stakes after the total bankroll limit and cent rounding
        results = per-bet KellyResult matching the final stakes
    """
    if len(probs) != len(odds):
        raise ValueError("probs and odds must have the same length.")
    if bankroll < 0:
        raise ValueError("bankroll must be non-negative.")

    results: List[KellyResult] = []
    raw_stakes: List[float] = []

    for p, o in zip(probs, odds):
        f = kelly_fraction(p, o)
        r = stake_from_fraction(
            bankroll,
            f,
            fraction_of_kelly=fraction_of_kelly,
            cap_fraction_of_bankroll=cap_fraction_of_bankroll,
            min_stake=min_stake,
        )
        results.append(r)
        raw_stakes.append(r.stake)

    final_stakes = normalize_stakes_to_bankroll(raw_stakes, bankroll, only_scale_positive=True)
    final_stakes = [float(Decimal(str(s)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)) for s in final_stakes]
    final_stakes = [s if s >= min_stake else 0.0 for s in final_stakes]
    results = [replace(r, stake=s, is_no_bet=s <= 0,
                       reason=(r.reason + "; bankroll limit/rounding" if s != r.stake else r.reason))
               for r, s in zip(results, final_stakes)]
    return final_stakes, results

def profit_if_win(stake: float, american_odds: float) -> float:
    """Profit (not return) if the bet wins."""
    if stake <= 0:
        return 0.0
    if american_odds > 0:
        return stake * (american_odds / 100.0)
    return stake * (100.0 / abs(american_odds))


def profit_if_loss(stake: float) -> float:
    """Profit if the bet loses (negative stake)."""
    if stake <= 0:
        return 0.0
    return -stake
