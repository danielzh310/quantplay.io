"""Compare weighting/calibration with identical sizing; never writes app snapshots.

Each week uses only earlier games. Bankroll is fixed at $160 for each slate to
isolate prediction changes; summed net is not a compounded bankroll simulation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from sports.nfl.models.adaptive_moneyline import AdaptiveMoneyline
from sports.nfl.models.calibration import reliability_report
from sports.nfl.scripts.validate_moneyline import settle

ARMS = {
    "legacy_standard": ("legacy", False),
    "equal_standard": ("none", False),
    "equal_favorite": ("none", True),
    "automatic": ("auto", True),
}


def compare(features, seasons, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*\n")
    summaries, selections, calibration = [], [], []
    for year in seasons:
        rows = {arm: [] for arm in ARMS}
        target_year = features[features.season.eq(year) & features.season_type.eq("REG")]
        for week in sorted(target_year.week.unique()):
            history = features[(features.season < year) | (features.season.eq(year) & features.season_type.eq("REG") & (features.week < week))]
            history = history[history.season_type.ne("PRE") & history.home_score.notna() & history.away_score.notna()]
            target = target_year[target_year.week.eq(week) & target_year.home_score.notna() & target_year.away_score.notna()]
            if target.empty:
                continue
            for arm, (weighting, favorite) in ARMS.items():
                model = AdaptiveMoneyline(weighting=weighting, favorite_calibration=favorite).train(history)
                prediction = model.predict(target)
                games = settle(prediction, target, bankroll=160., robust=False)
                games["season"], games["week"] = year, week
                games = games.merge(target[["game_id", "ml_market_home"]], on="game_id", validate="one_to_one")
                rows[arm].append(games)
                selections.append(dict(season=int(year), week=int(week), arm=arm, **model.diagnostics))
            print(f"Compared {year} week {week}", flush=True)
        for arm, frames in rows.items():
            if not frames:
                continue
            games = pd.concat(frames, ignore_index=True)
            non_ties = games[games.home_score.ne(games.away_score)]
            y = non_ties.home_score.gt(non_ties.away_score).astype(int)
            p = non_ties.ml_home_prob
            stake = float(games.kelly_stake_ml.sum())
            summaries.append(dict(season=year, arm=arm, games=len(games),
                                  brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                                  accuracy=float(np.mean((p > .5) == y)), bets=int(games.kelly_stake_ml.gt(0).sum()),
                                  staked=stake, net=float(games.net.sum()), roi=float(games.net.sum()/stake) if stake else None))
            calibration.extend(dict(season=year, arm=arm, **row) for row in reliability_report(p, y, non_ties.ml_market_home))
            games.to_csv(output / f"{year}_{arm}.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
        (output / "selections.json").write_text(json.dumps(selections, indent=2))
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2023,2024")
    parser.add_argument("--feature-cache", default="outputs/model_validation/features_v2.pkl")
    parser.add_argument("--output-dir", default="outputs/model_validation/equal-calibration-v1/evaluation")
    args = parser.parse_args()
    features = pd.read_pickle(args.feature_cache)
    print(compare(features, [int(s) for s in args.seasons.split(",")], args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
