"""Walk-forward backtest: profit-weighted vs. unweighted moneyline training,
and Kelly vs. flat staking, run side by side.

Two questions this answers:

1. Does sample-weighting Moneyline training by the payout of the realized
   winner (see `calculate_profit_weight` in `sports/nfl/models/moneyline.py`)
   help or hurt? Both arms are identical (features, train/predict split, edge
   threshold) except for that one training-time sample weight.

2. Is any apparent edge from the weighted model actually about pick quality,
   or is it a Kelly-staking artifact? Kelly sizes each bet in proportion to
   the model's confidence, so a model that is more (rightly or wrongly)
   confident about specific upsets gets bigger stakes on exactly those games.
   Flat staking bets the same amount on every qualifying pick, which isolates
   pick quality from that sizing effect. Both weighting arms are run under
   both staking modes, giving four arms total: weighted/unweighted x
   kelly/flat.

Odds source: by default this uses nflverse's recorded *closing* moneylines
for historical games. Pass `--odds-snapshot open` to instead use ESPN's
recorded *opening* line for each game -- the earliest snapshot ESPN retains
per provider, which approximates "the line early in the week" (there is no
timestamp attached, so it is not guaranteed to be exactly a Wednesday line;
an exact timestamped history would require a paid odds-history vendor this
repo does not currently integrate). Opening-line fetches hit ESPN live and
are cached to disk (`--odds-cache`) so repeat runs are fast.

Usage:
    python -m sports.nfl.scripts.backtest_moneyline_weighting --season 2025
    python -m sports.nfl.scripts.backtest_moneyline_weighting --season 2025 --odds-snapshot open
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from sports.nfl.data.loaders import load_weekly_data
from sports.nfl.data.odds import MARKET_COLUMNS, load_market_snapshots
from sports.nfl.data.preprocessing import build_features
from sports.nfl.models.moneyline import Moneyline
from utils.betting import american_to_implied_probability, has_positive_edge, payout_profit_per_dollar
from utils.kelly import kelly_fraction, stake_from_fraction

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs"
WEIGHTING_ARMS = {"weighted": True, "unweighted": False}
STAKING_MODES = ["kelly", "flat"]


def load_backtest_frame(season, history_start):
    seasons = list(range(history_start, season + 1))
    df = load_weekly_data(seasons=seasons, include_preseason=False)
    return build_features(df)


def apply_odds_snapshot(feats, snapshot, cache_path):
    """Overwrite home/away moneylines in-place with a fixed ESPN snapshot.

    No-op for `snapshot == "close"` since nflverse's recorded lines (already
    in `feats`) are themselves each game's closing market.
    """
    if snapshot == "close":
        return feats

    cache = pd.DataFrame(columns=["espn_event_id", *MARKET_COLUMNS])
    if cache_path.exists():
        cache = pd.read_csv(cache_path)

    have = set(cache["espn_event_id"].astype("Int64").dropna())
    all_ids = set(feats.loc[feats["espn_event_id"].notna(), "espn_event_id"].astype("Int64").dropna())
    missing = sorted(all_ids - have)

    if missing:
        print(f"Fetching '{snapshot}' odds for {len(missing)} games from ESPN (cached after this run)...")
        fetched = load_market_snapshots(missing, snapshot=snapshot)
        new_rows = pd.DataFrame([{"espn_event_id": event_id, **market} for event_id, market in fetched.items()])
        cache = pd.concat([cache, new_rows], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)

    lookup = cache.drop_duplicates("espn_event_id").set_index("espn_event_id")
    ids = feats["espn_event_id"].astype("Int64")
    out = feats.copy()
    for column in ["home_moneyline", "away_moneyline"]:
        out[column] = ids.map(lookup[column])
    return out


def predict_week(train_df, test_df, use_profit_weighting, edge_threshold):
    """Train once and score a week; staking is applied separately on top."""
    model = Moneyline(use_profit_weighting=use_profit_weighting)
    model.train(train_df)
    preds = model.predict(test_df)

    merged = test_df[
        ["game_id", "season", "week", "home_team", "away_team", "home_win", "home_moneyline", "away_moneyline"]
    ].merge(preds[["game_id", "home_win_prob", "away_win_prob", "moneyline_pick"]], on="game_id")

    merged["pick_prob"] = np.where(merged.moneyline_pick == "HOME", merged.home_win_prob, merged.away_win_prob)
    merged["pick_odds"] = np.where(merged.moneyline_pick == "HOME", merged.home_moneyline, merged.away_moneyline)
    merged["implied_prob"] = merged["pick_odds"].apply(american_to_implied_probability)
    merged["edge"] = merged["pick_prob"] - merged["implied_prob"]
    merged["has_edge"] = [
        has_positive_edge(p, o, edge_threshold) for p, o in zip(merged["pick_prob"], merged["pick_odds"])
    ]
    merged["actual_side"] = np.where(merged["home_win"] == 1, "HOME", "AWAY")
    return merged


def cap_weekly_stakes(stakes, bankroll):
    total = sum(stakes)
    if total <= bankroll or total <= 0:
        return stakes
    scale = bankroll / total
    return [s * scale for s in stakes]


def apply_staking(predicted, bankroll, staking_mode, fraction_of_kelly, cap_fraction, flat_fraction):
    games = predicted.copy()
    stakes = []
    for row in games.itertuples():
        if not row.has_edge:
            stakes.append(0.0)
            continue
        if staking_mode == "kelly":
            f = kelly_fraction(row.pick_prob, row.pick_odds)
            result = stake_from_fraction(
                bankroll, f, fraction_of_kelly=fraction_of_kelly, cap_fraction_of_bankroll=cap_fraction, min_stake=0.0
            )
            stakes.append(result.stake)
        else:  # flat: same stake on every qualifying pick, no confidence-based sizing
            stakes.append(flat_fraction * bankroll)
    games["stake"] = cap_weekly_stakes(stakes, bankroll)

    def pnl(row):
        if row["stake"] <= 0:
            return 0.0
        if row["moneyline_pick"] == row["actual_side"]:
            return row["stake"] * payout_profit_per_dollar(row["pick_odds"])
        return -row["stake"]

    games["pnl"] = games.apply(pnl, axis=1)
    games["bet_won"] = (games["moneyline_pick"] == games["actual_side"]) & (games["stake"] > 0)
    return games


def max_drawdown(bankroll_series):
    peak = bankroll_series.cummax()
    drawdown = (bankroll_series - peak) / peak
    return float(drawdown.min())


def summarize(games, bankroll_trace, starting_bankroll):
    final_bankroll = bankroll_trace["bankroll"].iloc[-1] if not bankroll_trace.empty else starting_bankroll
    staked = games["stake"].sum()
    placed = games[games["stake"] > 0]
    return {
        "games": len(games),
        "accuracy": float((games["moneyline_pick"] == games["actual_side"]).mean()),
        "log_loss": float(log_loss(games["home_win"], games["home_win_prob"], labels=[0, 1])),
        "brier_score": float(brier_score_loss(games["home_win"], games["home_win_prob"])),
        "bets_placed": int(len(placed)),
        "bet_rate": float(len(placed) / len(games)) if len(games) else float("nan"),
        "win_rate_on_bets": float(placed["bet_won"].mean()) if len(placed) else float("nan"),
        "total_staked": float(staked),
        "total_pnl": float(games["pnl"].sum()),
        "roi_on_staked_pct": float(games["pnl"].sum() / staked * 100) if staked else float("nan"),
        "starting_bankroll": float(starting_bankroll),
        "final_bankroll": float(final_bankroll),
        "roi_on_bankroll_pct": float((final_bankroll - starting_bankroll) / starting_bankroll * 100),
        "max_drawdown_pct": float(max_drawdown(bankroll_trace["bankroll"]) * 100) if not bankroll_trace.empty else float("nan"),
    }


def pooled_summary(pooled_games, per_season_summaries):
    """Aggregate across seasons: large-sample pick-quality stats pooled directly,
    bankroll-based risk/return stats averaged per-season (bankroll resets each
    season, so its trace can't just be concatenated)."""
    staked = pooled_games["stake"].sum()
    placed = pooled_games[pooled_games["stake"] > 0]
    season_rois = [s["roi_on_bankroll_pct"] for s in per_season_summaries]
    season_drawdowns = [s["max_drawdown_pct"] for s in per_season_summaries]
    return {
        "seasons": len(per_season_summaries),
        "games": len(pooled_games),
        "accuracy": float((pooled_games["moneyline_pick"] == pooled_games["actual_side"]).mean()),
        "log_loss": float(log_loss(pooled_games["home_win"], pooled_games["home_win_prob"], labels=[0, 1])),
        "brier_score": float(brier_score_loss(pooled_games["home_win"], pooled_games["home_win_prob"])),
        "bets_placed": int(len(placed)),
        "bet_rate": float(len(placed) / len(pooled_games)) if len(pooled_games) else float("nan"),
        "win_rate_on_bets": float(placed["bet_won"].mean()) if len(placed) else float("nan"),
        "total_staked": float(staked),
        "total_pnl": float(pooled_games["pnl"].sum()),
        "roi_on_staked_pct": float(pooled_games["pnl"].sum() / staked * 100) if staked else float("nan"),
        "avg_season_roi_pct": float(np.mean(season_rois)),
        "best_season_roi_pct": float(np.max(season_rois)),
        "worst_season_roi_pct": float(np.min(season_rois)),
        "profitable_seasons": int(sum(r > 0 for r in season_rois)),
        "worst_season_drawdown_pct": float(np.min(season_drawdowns)),
    }


def run_backtest(
    season, history_start, bankroll, fraction_of_kelly, cap_fraction, edge_threshold, flat_fraction,
    odds_snapshot, odds_cache, start_week=1,
):
    feats = load_backtest_frame(season, history_start)
    feats = apply_odds_snapshot(feats, odds_snapshot, odds_cache)
    weeks = sorted(
        feats.loc[(feats.season == season) & (feats.season_type == "REG") & (feats.week >= start_week), "week"]
        .unique()
    )

    arm_names = [f"{weighting}_{staking}" for weighting in WEIGHTING_ARMS for staking in STAKING_MODES]
    bankroll_state = {arm: bankroll for arm in arm_names}
    all_games = {arm: [] for arm in arm_names}
    traces = {arm: [] for arm in arm_names}

    for week in weeks:
        train_df = feats[(feats.season < season) | ((feats.season == season) & (feats.week < week))]
        train_df = train_df[train_df["home_score"].notna()]
        test_df = feats[(feats.season == season) & (feats.season_type == "REG") & (feats.week == week)]
        test_df = test_df[
            test_df["home_score"].notna() & test_df["home_moneyline"].notna() & test_df["away_moneyline"].notna()
        ]
        if train_df.empty or train_df["home_win"].nunique() < 2 or test_df.empty:
            continue

        for weighting, use_weighting in WEIGHTING_ARMS.items():
            predicted = predict_week(train_df, test_df, use_weighting, edge_threshold)
            for staking_mode in STAKING_MODES:
                arm = f"{weighting}_{staking_mode}"
                games = apply_staking(
                    predicted, bankroll_state[arm], staking_mode, fraction_of_kelly, cap_fraction, flat_fraction
                )
                bankroll_state[arm] += games["pnl"].sum()
                all_games[arm].append(games)
                traces[arm].append({"season": season, "week": week, "bankroll": bankroll_state[arm]})

    results = {arm: pd.concat(rows, ignore_index=True) for arm, rows in all_games.items() if rows}
    bankroll_traces = {arm: pd.DataFrame(trace) for arm, trace in traces.items()}
    return results, bankroll_traces


def parse_seasons(spec):
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2025, help="Single regular season to backtest.")
    parser.add_argument(
        "--seasons", type=str, default=None,
        help="Multiple seasons instead of --season, e.g. '2021-2025' or '2021,2022,2023'. "
             "Each season gets its own fresh bankroll; results are also pooled across all of them.",
    )
    parser.add_argument(
        "--start-week", type=int, default=1,
        help="Skip grading/betting weeks before this one each season (still used for training). "
             "E.g. --start-week 5 excludes weeks 1-4, when the 4-game rolling window is least mature.",
    )
    parser.add_argument(
        "--history-start", type=int, default=None,
        help="First season of training history (default per-season: min(2022, season - 1)).",
    )
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--fraction-of-kelly", type=float, default=0.25, help="0.25 = quarter Kelly.")
    parser.add_argument("--cap-fraction", type=float, default=0.05, help="Max Kelly stake as a fraction of bankroll.")
    parser.add_argument("--flat-fraction", type=float, default=0.02, help="Flat stake as a fraction of bankroll.")
    parser.add_argument("--edge-threshold", type=float, default=0.01, help="Min (model prob - implied prob) to bet.")
    parser.add_argument("--odds-snapshot", choices=["close", "open"], default="close",
                         help="'close' = nflverse recorded closing line. 'open' = ESPN's earliest recorded line.")
    parser.add_argument("--odds-cache", type=Path, default=OUTPUT_DIR / "nfl_opening_odds_cache.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons) if args.seasons else [args.season]
    pd.set_option("display.width", 140)

    per_season_results = {}
    per_season_summaries = {arm: [] for arm in [f"{w}_{s}" for w in WEIGHTING_ARMS for s in STAKING_MODES]}

    for season in seasons:
        history_start = args.history_start or min(2022, season - 1)
        print(f"\nBacktesting {season} regular season (weeks >= {args.start_week}, "
              f"training history from {history_start}, odds snapshot='{args.odds_snapshot}')...")
        results, traces = run_backtest(
            season, history_start, args.bankroll, args.fraction_of_kelly, args.cap_fraction,
            args.edge_threshold, args.flat_fraction, args.odds_snapshot, args.odds_cache, args.start_week,
        )
        if not results:
            print(f"  No completed weeks with usable odds found for {season}.")
            continue

        summaries = {arm: summarize(games, traces[arm], args.bankroll) for arm, games in results.items()}
        for arm, summary in summaries.items():
            per_season_summaries[arm].append({"season": season, **summary})
        per_season_results[season] = results

        print(pd.DataFrame(summaries).T.round(4).to_string())

    if not per_season_results:
        print("No seasons produced usable results.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(s) for s in seasons) if len(seasons) > 1 else str(seasons[0])
    suffix = tag if args.odds_snapshot == "close" else f"{tag}_{args.odds_snapshot}"

    for season, results in per_season_results.items():
        for arm, games in results.items():
            games.assign(arm=arm, season_run=season).to_csv(
                args.output_dir / f"nfl_backtest_{arm}_{season}{'' if args.odds_snapshot == 'close' else '_' + args.odds_snapshot}.csv",
                index=False,
            )

    per_season_df = pd.concat(
        [pd.DataFrame(rows) for rows in per_season_summaries.values() if rows],
        keys=[arm for arm, rows in per_season_summaries.items() if rows],
        names=["arm"],
    ).reset_index(level=0)
    per_season_df.to_csv(args.output_dir / f"nfl_backtest_per_season_{suffix}.csv", index=False)

    if len(seasons) > 1:
        pooled = {}
        for arm in per_season_summaries:
            season_summary_rows = per_season_summaries[arm]
            if not season_summary_rows:
                continue
            pooled_games = pd.concat(
                [per_season_results[row["season"]][arm] for row in season_summary_rows], ignore_index=True
            )
            pooled[arm] = pooled_summary(pooled_games, season_summary_rows)

        pooled_df = pd.DataFrame(pooled).T
        pooled_df.to_csv(args.output_dir / f"nfl_backtest_pooled_{suffix}.csv")
        print(f"\n=== Pooled across {len(seasons)} seasons ({tag}), weeks >= {args.start_week} ===")
        print(pooled_df.round(4).to_string())

    print(f"\nPer-season and pooled summaries, plus per-game detail, written to {args.output_dir}")


if __name__ == "__main__":
    main()
