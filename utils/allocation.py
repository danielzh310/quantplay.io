"""Percentage-first moneyline allocation, separate from the prediction models."""
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import math
import json
import pandas as pd
from utils.betting import american_to_implied_probability, payout_profit_per_dollar
from utils.kelly import kelly_fraction

INPUT_COLUMNS = ["game_id", "home_team", "away_team", "ml_pick", "ml_home_prob",
                 "ml_away_prob", "home_moneyline", "away_moneyline", "kelly_stake_ml"]
VERSION = "allocation-v8"

@dataclass(frozen=True)
class SizingSettings:
    bankroll: float = 160.0
    weekly_budget: float | None = None
    fractional_kelly: float = 0.25
    per_game_cap: float | None = None
    slate_cap: float = 1.0
    edge_threshold: float = 0.01
    minimum_stake: float = 0.01
    normalize_to_full_bankroll: bool = False
    max_bet_dollars: float | None = None
    selection_mode: str = "either_side"
    use_probability_bounds: bool = False  # research comparison; UI uses model probabilities

    def validate(self):
        if self.selection_mode not in ("predicted_winner", "either_side"):
            raise ValueError("Bet selection must be predicted_winner or either_side.")
        numeric = {key: value for key, value in asdict(self).items() if key != "selection_mode"}
        if any(not math.isfinite(v) or v < 0 for v in numeric.values() if v is not None):
            raise ValueError("Sizing inputs must be finite and non-negative.")
        if not 0 < self.fractional_kelly <= 1:
            raise ValueError("Kelly fraction must be above zero and no greater than one.")
        if any(v > 1 for v in (self.per_game_cap, self.slate_cap, self.edge_threshold) if v is not None):
            raise ValueError("Allocation caps and probability edge must be fractions from 0 to 1.")


def floor_cents(amount):
    return float(Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))


def normalize_stakes(weights, budget, cap, minimum):
    """Normalize proportionally; the largest stake controls the dollar ceiling.

    Only the explicit normalization option calls this. Removed small stakes
    remain passes; cents go to the largest rounding remainders deterministically.
    """
    active = [i for i, weight in enumerate(weights) if weight > 0]
    stakes = [0.0] * len(weights)
    while active:
        factor = min(budget / sum(weights[i] for i in active), cap / max(weights[i] for i in active))
        target = factor * sum(weights[i] for i in active)
        stakes = [0.0] * len(weights)
        for i in active:
            stakes[i] = factor * weights[i]
        too_small = [i for i in active if floor_cents(stakes[i]) < minimum or stakes[i] <= 0]
        if too_small:
            active = [i for i in active if i not in too_small]
            continue
        cents = [int(Decimal(str(floor_cents(value))) * 100) for value in stakes]
        leftover = int(Decimal(str(floor_cents(target))) * 100) - sum(cents)
        for i in sorted(active, key=lambda i: (-(stakes[i] * 100 - cents[i]), i)):
            if leftover > 0 and (cents[i] + 1) / 100 <= cap:
                cents[i] += 1
                leftover -= 1
        return [value / 100 for value in cents]
    return [0.0] * len(weights)


def size_saved_predictions(saved, settings):
    settings.validate()
    missing = set(INPUT_COLUMNS) - set(saved.columns)
    if missing:
        raise ValueError(f"Missing saved prediction fields: {sorted(missing)}")
    risk_columns = [f"ml_{side}_{field}" for side in ("home", "away")
                    for field in ("sizing_prob", "edge_risk_factor", "extreme_n", "extreme_reason")]
    bound_columns = [c for c in ("ml_home_prob_low", "ml_away_prob_low", "ml_reliability_n", *risk_columns) if c in saved]
    source = saved[INPUT_COLUMNS + bound_columns].copy()
    if source.empty or source.game_id.isna().any() or source.game_id.duplicated().any():
        raise ValueError("A nonempty slate with unique game IDs is required.")
    probabilities = source[["ml_home_prob", "ml_away_prob"]].apply(pd.to_numeric, errors="coerce")
    if (probabilities.isna().any().any() or not probabilities.ge(0).all().all()
            or not probabilities.le(1).all().all()
            or not probabilities.sum(axis=1).sub(1).abs().le(1e-6).all()):
        raise ValueError("Saved probabilities must be valid complementary win probabilities.")
    records = []
    for _, row in source.iterrows():
        choices = []
        for side in ("home", "away"):
            if settings.selection_mode == "predicted_winner" and side.upper() != row["ml_pick"]:
                continue
            odds = pd.to_numeric(row[f"{side}_moneyline"], errors="coerce")
            if not math.isfinite(odds) or abs(odds) < 100:
                continue
            probability = float(row[f"ml_{side}_prob"])
            sizing_probability = float(row.get(f"ml_{side}_sizing_prob", probability))
            if not math.isfinite(sizing_probability) or not 0 <= sizing_probability <= 1:
                raise ValueError("Risk-adjusted probability must be finite and between zero and one.")
            probability = min(probability, sizing_probability)
            # Production Kelly uses the model estimate after the extreme-risk
            # check. Empirical lower bounds remain an optional research setting.
            if settings.use_probability_bounds:
                bound = row.get(f"ml_{side}_prob_low", probability)
                probability = min(probability, float(bound)) if pd.notna(bound) else 0.0
            if "ml_reliability_n" in row:
                count = pd.to_numeric(row["ml_reliability_n"], errors="coerce")
                if pd.isna(count) or not math.isfinite(count) or count <= 0:
                    continue
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Sizing probability must be finite and between zero and one.")
            edge = probability - american_to_implied_probability(odds)
            if edge <= settings.edge_threshold:
                continue
            expected_return = probability * payout_profit_per_dollar(odds) - (1 - probability)
            factor = 1.
            reasons = []
            for key in (f"ml_{side}_edge_risk_factor",):
                value = float(row.get(key, 1.))
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Risk multipliers must be finite fractions from zero to one.")
                factor = min(factor, value)
            for key in (f"ml_{side}_extreme_reason",):
                if key in row and pd.notna(row[key]):
                    reasons.append(str(row[key]))
            choices.append((expected_return, side, kelly_fraction(probability, odds), edge, probability, factor, reasons))
        record = row.to_dict()
        record.update(bet_side="PASS", full_kelly=0.0, recommended_fraction=0.0,
                      allocation_fraction=0.0, expected_return_per_dollar=0.0,
                      probability_edge=0.0, sizing_probability=0.0, risk_multiplier=1.,
                      reason="No qualifying edge on predicted winner" if settings.selection_mode == "predicted_winner" else "No qualifying edge")
        if choices:
            expected, side, full, edge, probability, factor, reasons = max(choices, key=lambda choice: choice[0])
            recommended = max(0.0, full) * settings.fractional_kelly * factor
            record.update(bet_side=side.upper(), full_kelly=full,
                          recommended_fraction=recommended,
                          allocation_fraction=min(recommended, settings.per_game_cap) if settings.per_game_cap is not None else recommended,
                          expected_return_per_dollar=expected, probability_edge=edge,
                          sizing_probability=probability, risk_multiplier=factor,
                          reason=("Per-game cap" if settings.per_game_cap is not None and recommended > settings.per_game_cap else "Fractional Kelly")
                                 + ("; " + "; ".join(reasons) if reasons else ""))
        records.append(record)
    output = pd.DataFrame(records)
    # Kelly determines exposure by default. Optional dollar limits reduce bets;
    # only explicit normalization may increase them beyond Kelly's suggestion.
    budget = floor_cents(min(settings.bankroll, settings.bankroll * settings.slate_cap,
                             settings.weekly_budget if settings.weekly_budget is not None else settings.bankroll))
    cap_dollars = floor_cents(min(settings.bankroll,
                                 settings.bankroll * settings.per_game_cap if settings.per_game_cap is not None else settings.bankroll,
                                 settings.max_bet_dollars if settings.max_bet_dollars is not None else settings.bankroll))
    proposed_stakes = [f * settings.bankroll for f in output.allocation_fraction]
    if settings.max_bet_dollars is not None:
        capped = output.allocation_fraction * settings.bankroll > cap_dollars
        if capped.any():
            output.loc[output.bet_side.ne("PASS"), "reason"] += "; proportional dollar limit"
    proposed = sum(proposed_stakes)
    largest = max(proposed_stakes, default=0)
    scale = min(1.0, budget / proposed, cap_dollars / largest) if proposed else 1.0
    if scale < 1:
        output.loc[output.bet_side.ne("PASS"), "reason"] += "; slate limit"
    if settings.normalize_to_full_bankroll:
        # Apply risk discounts after normalization so it cannot cancel a common
        # historical reliability discount across the slate. Trimmed cash remains unused.
        weights = [f / r if r else 0. for f, r in zip(output.allocation_fraction, output.risk_multiplier)]
        normalized = normalize_stakes(weights, budget, cap_dollars, settings.minimum_stake)
        output["kelly_stake_ml"] = [floor_cents(v * r) for v, r in zip(normalized, output.risk_multiplier)]
        output.loc[output.bet_side.ne("PASS"), "reason"] += "; full-bankroll normalization"
    else:
        output["kelly_stake_ml"] = [floor_cents(stake * scale) for stake in proposed_stakes]
    no_bet = (output.kelly_stake_ml < settings.minimum_stake) | output.kelly_stake_ml.le(0)
    output.loc[no_bet & output.bet_side.ne("PASS"), "reason"] = "Below minimum or no available budget"
    output.loc[no_bet, "bet_side"] = "PASS"
    output.loc[no_bet, "kelly_stake_ml"] = 0.0
    output["allocation_fraction"] = output.kelly_stake_ml / settings.bankroll if settings.bankroll else 0.0
    output["kelly_no_bet_ml"] = no_bet
    output["home_ml_bet"] = output.bet_side.map(lambda side: "Bet" if side == "HOME" else "No Bet")
    output["away_ml_bet"] = output.bet_side.map(lambda side: "Bet" if side == "AWAY" else "No Bet")
    return output


def allocate_predictions(predictions, settings):
    """Keep model outputs intact while attaching one consistent bet decision."""
    source = predictions.copy()
    source["kelly_stake_ml"] = 0.0
    allocated = size_saved_predictions(source, settings)
    # Retired risk fields must not appear to influence a newly sized run.
    # Merely loading an old snapshot does not call this function or change it.
    out = predictions.drop(columns=["ml_starter_risk_factor", "ml_starter_risk_reason", "stake_before_concentration"], errors="ignore").copy()
    for column in allocated.columns:
        if column not in INPUT_COLUMNS or column == "kelly_stake_ml":
            out[column] = allocated[column].to_numpy()
    out["kelly_f_ml"] = out["full_kelly"]
    out["sizing_version"] = VERSION
    out["sizing_settings"] = json.dumps(asdict(settings), sort_keys=True)
    return out
