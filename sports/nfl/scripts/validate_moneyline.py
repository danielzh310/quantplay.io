"""Walk-forward validation of the production moneyline path on completed seasons.

Uses recorded closing prices, not a claim about fills available earlier in a
week. Does not grade/write any main app snapshot. Output includes losing weeks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from sports.nfl.data.loaders import load_weekly_data
from sports.nfl.data.preprocessing import build_features
from sports.nfl.data.moneyline_features import build_moneyline_features
from sports.nfl.data.efficiency import add_efficiency_features
from sports.nfl.models.moneyline import Moneyline
from sports.nfl.models.adaptive_moneyline import AdaptiveMoneyline
from utils.allocation import SizingSettings, allocate_predictions
from utils.betting import payout_profit_per_dollar


def settle(predictions, actual, bankroll, robust=True):
    frame = predictions.rename(columns={"home_win_prob": "ml_home_prob", "away_win_prob": "ml_away_prob",
                                        "moneyline_pick": "ml_pick"}).copy()
    frame = frame.merge(actual[["game_id", "home_team", "away_team", "home_score", "away_score"]], on="game_id")
    if not robust:
        frame = frame.drop(columns=["ml_home_prob_low", "ml_away_prob_low"], errors="ignore")
    sized = allocate_predictions(frame, SizingSettings(bankroll=bankroll, selection_mode="either_side", use_probability_bounds=robust))
    rows = []
    for _, row in sized.iterrows():
        profit = 0.
        flat_profit = 0.
        if row.bet_side in ("HOME", "AWAY") and row.kelly_stake_ml > 0:
            result = "PUSH" if row.home_score == row.away_score else "HOME" if row.home_score > row.away_score else "AWAY"
            flat_profit = 0 if result == "PUSH" else payout_profit_per_dollar(row[f"{row.bet_side.lower()}_moneyline"]) if result == row.bet_side else -1
            profit = flat_profit * row.kelly_stake_ml
        rows.append({**row.to_dict(), "net": round(profit, 2), "flat_net": flat_profit,
                     "flat_stake": int(row.kelly_stake_ml > 0)})
    return pd.DataFrame(rows)


def metrics(games, weeks, bankroll):
    non_ties = games[games.home_score.ne(games.away_score)]
    y = non_ties.home_score.gt(non_ties.away_score).astype(int)
    p = non_ties.ml_home_prob
    staked = games.kelly_stake_ml.sum()
    trace = np.r_[bankroll, weeks.bankroll.to_numpy()]
    peaks = np.maximum.accumulate(trace)
    flat_stakes = int(games.flat_stake.sum())
    return {"games": len(games), "bets": flat_stakes,
            "accuracy": float(np.mean((p > .5) == y)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "staked": float(staked),
            "net": float(games.net.sum()), "roi": float(games.net.sum()/staked) if staked else None,
            "flat_roi": float(games.flat_net.sum()/flat_stakes) if flat_stakes else None,
            "max_drawdown": float(np.min(trace/peaks-1)) if bankroll > 0 else 0.,
            "losing_weeks": int(weeks.net.lt(0).sum()), "weeks": len(weeks),
            "largest_stake": float(games.kelly_stake_ml.max())}


def validate(features, seasons, output, bankroll=160.):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*\n")
    reports, selections = [], []
    for season in seasons:
        balances = {"legacy_four_game": bankroll, "adaptive": bankroll, "adaptive_bound_sizing": bankroll}
        game_rows = {key: [] for key in balances}
        week_rows = {key: [] for key in balances}
        weeks = sorted(features.loc[(features.season == season) & features.season_type.eq("REG"), "week"].unique())
        for week in weeks:
            train = features[(features.season < season) | ((features.season == season) & features.season_type.eq("REG") & (features.week < week))]
            train = train[train.home_score.notna() & train.away_score.notna() & train.season_type.ne("PRE")]
            test = features[(features.season == season) & features.season_type.eq("REG") & (features.week == week)]
            test = test[test.home_score.notna() & test.away_score.notna()]
            if test.empty:
                continue
            baseline = Moneyline(weighting="legacy")
            baseline.train(train[train.home_score.ne(train.away_score)])
            baseline_predictions = baseline.predict(test)
            adaptive = AdaptiveMoneyline(weighting="auto").train(train)
            adaptive_predictions = adaptive.predict(test)
            selections.append({"season": int(season), "week": int(week), **adaptive.diagnostics})
            for arm, prediction in (("legacy_four_game", baseline_predictions), ("adaptive", adaptive_predictions),
                                    ("adaptive_bound_sizing", adaptive_predictions)):
                games = settle(prediction, test, balances[arm], robust=arm == "adaptive_bound_sizing")
                balances[arm] = max(0., balances[arm] + games.net.sum())
                games["season"], games["week"] = season, week
                game_rows[arm].append(games)
                week_rows[arm].append({"week": int(week), "net": float(games.net.sum()), "bankroll": balances[arm],
                                       "bets": int(games.kelly_stake_ml.gt(0).sum())})
            print(f"Validated {season} week {week}: {adaptive.selected}, training weights {adaptive.weighting}", flush=True)
        for arm in balances:
            if not game_rows[arm]:
                continue
            games, weeks = pd.concat(game_rows[arm], ignore_index=True), pd.DataFrame(week_rows[arm])
            report = {"season": season, "arm": arm, **metrics(games, weeks, bankroll)}
            early = games[games.week <= 4]
            if not early.empty:
                report["early_brier"] = float(brier_score_loss(early.home_score.gt(early.away_score), early.ml_home_prob))
                report["early_net"] = float(early.net.sum())
            reports.append(report)
            games.to_csv(output / f"{season}_{arm}_games.csv", index=False)
            weeks.to_csv(output / f"{season}_{arm}_weeks.csv", index=False)
        pd.DataFrame(reports).to_csv(output / "summary.csv", index=False)
        (output / "selections.json").write_text(json.dumps(selections, indent=2))
    return pd.DataFrame(reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--history-start", type=int, default=2016)
    parser.add_argument("--output-dir", default="outputs/model_validation/evaluation_v1")
    parser.add_argument("--feature-cache", default="outputs/model_validation/features_v2.pkl")
    args = parser.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]
    cache = Path(args.feature_cache)
    if cache.exists():
        features = pd.read_pickle(cache)
    else:
        years = list(range(args.history_start, max(seasons)+1))
        games = load_weekly_data(years)
        features = build_moneyline_features(build_features(games))
        cache.parent.mkdir(parents=True, exist_ok=True)
        (cache.parent / ".gitignore").write_text("*\n")
        features.to_pickle(cache)
    features = add_efficiency_features(features)
    report = validate(features, seasons, args.output_dir)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
