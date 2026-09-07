from pathlib import Path

from sports.nfl.data.loaders import load_weekly_data
from sports.nfl.data.preprocessing import build_features
from sports.nfl.data.odds import MARKET_COLUMNS, refresh_game_odds

from sports.nfl.models.moneyline import Moneyline
from sports.nfl.models.spread import Spread
from sports.nfl.models.total import Total


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "nfl_weekly_picks.csv"


def run_weekly(
    season=2026,
    week=16,
    seasons=None,
    export=True,
    output_path=None,
    verbose=True,
    season_type=None,
):
    season_type = (season_type or ("POST" if week is not None and week >= 19 else "REG")).upper()
    if season_type not in {"PRE", "REG", "POST"}:
        raise ValueError("season_type must be PRE, REG, or POST")
    if verbose:
        print("Loading football data...")
    if seasons is None and season is not None:
        seasons = list(range(min(2022, int(season) - 1), int(season) + 1))
    df = load_weekly_data(seasons=seasons, include_preseason=season_type == "PRE")

    # Refresh the requested slate for every phase. Preseason history also needs
    # archived markets, which the scoreboard feed does not supply.
    target = df["season_type"] == season_type
    if season is not None:
        target &= df["season"] == season
    if week is not None:
        target &= df["week"] == week
    refresh = target | (df["season_type"] == "PRE")
    updated = refresh_game_odds(df.loc[refresh])
    df.loc[refresh, MARKET_COLUMNS] = updated[MARKET_COLUMNS]

    if verbose:
        print("Building rolling features...")
    feats = build_features(df)

    if season is not None and week is not None:
        phase = feats["season_type"].map({"PRE": 0, "REG": 1, "POST": 2})
        target_phase = {"PRE": 0, "REG": 1, "POST": 2}[season_type]
        train_df = feats[
            (feats["season"] < season) |
            ((feats["season"] == season) & (
                (phase < target_phase) | ((phase == target_phase) & (feats["week"] < week))
            ))
        ].copy()
        train_df = train_df[train_df["home_score"].notna()].copy()

        predict_df = feats[
            (feats["season"] == season) &
            (feats["week"] == week) & (feats["season_type"] == season_type)
        ].copy()
    else:
        train_df = feats[feats["home_score"].notna()].copy()
        predict_df = feats[feats["home_score"].isna() & (feats["season_type"] == season_type)].copy()

    if verbose:
        print(f"Training on {len(train_df)} completed games")

    if predict_df.empty:
        print("No games found for that season/week.")
        return None

    train_df = train_df.dropna(subset=["home_score", "away_score"])
    if train_df.empty or train_df["home_win"].nunique() < 2:
        raise ValueError("Not enough completed historical games to train the models.")

    if verbose:
        print(f"Predicting {len(predict_df)} upcoming games")

    ml = Moneyline()
    sp = Spread()
    tot = Total()

    if verbose:
        print("Training models...")
    ml.train(train_df)
    sp.train(train_df)
    tot.train(train_df)

    if verbose:
        print("Generating predictions...")
    ml_preds = ml.predict(predict_df)
    sp_preds = sp.predict(predict_df)
    tot_preds = tot.predict(predict_df)

    if verbose:
        print("Merging outputs...")
    base = predict_df[
        ["game_id", "home_team", "away_team", "spread_line", "total_line"]
    ].drop_duplicates("game_id")

    results = base.merge(ml_preds, on="game_id")
    results = results.merge(sp_preds, on="game_id")
    results = results.merge(tot_preds, on="game_id")

    if verbose:
        print("Cleaning and formatting output...")

    pretty = results.rename(columns={
        "moneyline_pick": "ml_pick",
        "home_win_prob": "ml_home_prob",
        "away_win_prob": "ml_away_prob",
        "predicted_margin": "model_spread_margin",
        "predicted_total": "projected_total",
        "total_edge": "vegas_total_edge",
    })

    round_cols = [
        "ml_home_prob",
        "ml_away_prob",
        "model_spread_margin",
        "spread_edge",
        "projected_total",
        "vegas_total_edge",
        "total_low_q",
        "total_high_q",
        "total_buffer",
    ]

    pretty[round_cols] = pretty[round_cols].round(2)

    ordered_cols = [
        "game_id",
        "home_team",
        "away_team",

        "ml_pick",
        "ml_home_prob",
        "ml_away_prob",
        "home_ml_bet",
        "away_ml_bet",
        "home_moneyline",
        "away_moneyline",

        "spread_line",
        "spread_pick",
        "model_spread_margin",
        "spread_edge",

        "total_line",
        "total_pick",
        "projected_total",
        "vegas_total_edge",

        "total_low_q",
        "total_high_q",
        "total_buffer",
    ]

    out_df = pretty[ordered_cols]

    if export:
        path = Path(output_path) if output_path else (
            REPO_ROOT / "outputs" / "nfl_preseason_weekly_picks.csv"
            if season_type == "PRE" else DEFAULT_OUTPUT_PATH
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(path, index=False)

        if verbose:
            print(f"Wrote predictions to {path}")

    if verbose:
        print("Weekly predictions generated.")

    return out_df
