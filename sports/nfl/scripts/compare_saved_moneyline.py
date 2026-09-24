"""Compare fixed-weekly-bankroll backtests; never tune or activate a model.

Example: python -m sports.nfl.scripts.compare_saved_moneyline --baseline
outputs/kelly_weekly160_v6 --candidate outputs/experiment --output outputs/comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_run(directory):
    directory = Path(directory)
    weeks = pd.read_csv(directory / "weekly_results.csv")
    ledger = pd.read_csv(directory / "bet_ledger.csv")
    week_keys, bet_keys = ["season", "week", "kelly"], ["season", "week", "game_id", "kelly_fraction"]
    if weeks.duplicated(week_keys).any() or ledger.duplicated(bet_keys).any():
        raise ValueError("Duplicate weeks or wagers in comparison input")
    if weeks.empty or ledger.empty:
        raise ValueError("Comparison requires nonempty reports")
    for column in ("weekly_budget", "staked", "weekly_profit", "bets"):
        if not np.isfinite(weeks[column]).all():
            raise ValueError(f"Invalid weekly values: {column}")
    if (weeks.staked < 0).any() or (weeks.staked > weeks.weekly_budget + 1e-8).any():
        raise ValueError("Weekly stakes exceed the declared bankroll")
    grouped = ledger.groupby(["season", "week", "kelly_fraction"]).agg(
        net=("settled_profit", "sum"), stake=("kelly_stake_ml", "sum"),
        bets=("kelly_stake_ml", lambda values: values.gt(0).sum()))
    expected = weeks.set_index(week_keys).sort_index()
    grouped.index.names = week_keys
    if not grouped.index.sort_values().equals(expected.index):
        raise ValueError("Weekly report and ledger cover different slates")
    grouped = grouped.loc[expected.index]
    for actual, reported in (("net", "weekly_profit"), ("stake", "staked"), ("bets", "bets")):
        if not np.allclose(grouped[actual], expected[reported], atol=1e-7, rtol=0):
            raise ValueError("Weekly totals do not reconcile to wager ledger")
    return weeks, ledger


def summarize(weeks, evaluation_seasons):
    records = []
    groups = [("all", weeks), ("evaluation", weeks[weeks.season.isin(evaluation_seasons)])]
    groups += [(str(int(year)), group) for year, group in weeks.groupby("season")]
    for period, frame in groups:
        if frame.empty:
            raise ValueError("Requested evaluation seasons have no observations")
        for kelly, group in frame.sort_values(["season", "week"]).groupby("kelly"):
            net, stake = float(group.weekly_profit.sum()), float(group.staked.sum())
            curve = np.r_[0., group.weekly_profit.cumsum()]
            records.append(dict(period=period, kelly=kelly, net=round(net, 2), staked=round(stake, 2),
                                roi_pct=100*net/stake if stake > 0 else np.nan,
                                bets=int(group.bets.sum()),
                                max_drawdown=round(float((np.maximum.accumulate(curve)-curve).max()), 2)))
    return pd.DataFrame(records).set_index(["period", "kelly"]).sort_index()


def compare_runs(baseline_directory, candidate_directory, evaluation_seasons=(2024, 2025), minimum_bet_retention=.5):
    if not 0 < minimum_bet_retention <= 1:
        raise ValueError("Bet retention must be above zero and at most one")
    old_weeks, old_ledger = read_run(baseline_directory)
    new_weeks, new_ledger = read_run(candidate_directory)
    if not set(evaluation_seasons).issubset(set(old_weeks.season)):
        raise ValueError("An evaluation season is missing")
    keys = ["season", "week", "game_id", "kelly_fraction"]
    columns = ["home_team", "away_team", "home_moneyline", "away_moneyline", "home_score", "away_score"]
    try:
        pd.testing.assert_frame_equal(old_ledger.set_index(keys)[columns].sort_index(),
                                      new_ledger.set_index(keys)[columns].sort_index(), check_dtype=False)
        pd.testing.assert_frame_equal(old_weeks.set_index(["season", "week", "kelly"])[["weekly_budget"]].sort_index(),
                                      new_weeks.set_index(["season", "week", "kelly"])[["weekly_budget"]].sort_index(), check_dtype=False)
    except AssertionError as error:
        raise ValueError("Runs must use identical games, outcomes, odds, Kelly fractions and weekly bankrolls") from error
    old_settings = old_ledger.set_index(keys).sizing_settings.map(json.loads).sort_index()
    new_settings = new_ledger.set_index(keys).sizing_settings.map(json.loads).sort_index()
    if not old_settings.equals(new_settings):
        raise ValueError("Sizing settings differ; this is not an isolated model comparison")
    old = summarize(old_weeks, evaluation_seasons)
    new = summarize(new_weeks, evaluation_seasons)
    result = old.add_prefix("baseline_").join(new.add_prefix("candidate_"))
    result["bet_retention"] = result.candidate_bets / result.baseline_bets.replace(0, np.nan)
    result["passes"] = ((result.candidate_net >= result.baseline_net - 1e-8)
                        & (result.candidate_roi_pct >= result.baseline_roi_pct - 1e-8)
                        & (result.candidate_max_drawdown <= result.baseline_max_drawdown + 1e-8)
                        & result.bet_retention.ge(minimum_bet_retention))
    accepted = bool(result.loc[["all", "evaluation"], "passes"].all())
    decision = dict(passes_historical_gate=accepted, evaluation_seasons=list(evaluation_seasons),
                    minimum_bet_retention=minimum_bet_retention,
                    per_season_regressions=[dict(period=p, kelly=float(k)) for (p, k), r in result.iterrows()
                                            if p not in ("all", "evaluation") and not r["passes"]],
                    limitation="A historical gate cannot guarantee future profit. Previously inspected seasons are not untouched holdouts.")
    return result.reset_index(), decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-seasons", default="2024,2025")
    args = parser.parse_args()
    result, decision = compare_runs(args.baseline, args.candidate, tuple(map(int, args.evaluation_seasons.split(","))))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "comparison.csv", index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
