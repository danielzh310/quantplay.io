from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from sports.nfl.data.loaders import load_weekly_data
from utils.betting import payout_profit_per_dollar


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs"


def prediction_snapshot_path(season: int, week: int, season_type="REG") -> Path:
    prefix = "nfl_preseason" if season_type == "PRE" else "nfl"
    return OUTPUT_DIR / f"{prefix}_predictions_{int(season)}_wk{int(week)}.csv"


def results_snapshot_path(season: int, week: int, season_type="REG") -> Path:
    prefix = "nfl_preseason" if season_type == "PRE" else "nfl"
    return OUTPUT_DIR / f"{prefix}_results_{int(season)}_wk{int(week)}.csv"


def save_prediction_snapshot(df: pd.DataFrame, season: int, week: int, season_type="REG") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = prediction_snapshot_path(season, week, season_type)
    # Preserve the previous snapshot and every new run before updating the
    # familiar latest-file path. Settings and original odds travel in the CSV.
    archive = OUTPUT_DIR / "prediction_history"
    archive.mkdir(exist_ok=True)
    (archive / ".gitignore").write_text("*\n", encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + uuid4().hex[:8]
    if path.exists():
        (archive / f"{path.stem}_{stamp}_previous.csv").write_bytes(path.read_bytes())
    df.to_csv(archive / f"{path.stem}_{stamp}.csv", index=False)
    df.to_csv(path, index=False)
    return path


def load_prediction_snapshot(season: int, week: int, season_type="REG") -> pd.DataFrame:
    path = prediction_snapshot_path(season, week, season_type)
    if not path.exists():
        raise FileNotFoundError(f"No saved prediction snapshot found at {path}")
    return pd.read_csv(path)


def _moneyline_result(row):
    if row["home_score"] > row["away_score"]:
        return "HOME"
    if row["away_score"] > row["home_score"]:
        return "AWAY"
    return "PUSH"


def _moneyline_net(row):
    stake = float(row.get("kelly_stake_ml", 0.0) or 0.0)
    side = row.get("bet_side", row["ml_pick"])
    if pd.isna(side):
        side = row["ml_pick"]
    if stake <= 0 or side == "PASS" or row["ml_result"] == "PUSH":
        return 0.0
    if side != row["ml_result"]:
        return -stake

    odds_col = "home_moneyline" if side == "HOME" else "away_moneyline"
    return stake * payout_profit_per_dollar(row[odds_col])


def grade_saved_predictions(season: int, week: int, season_type=None):
    season_type = (season_type or ("POST" if week >= 19 else "REG")).upper()
    predictions = load_prediction_snapshot(season, week, season_type)
    schedule = load_weekly_data(seasons=[int(season)], include_preseason=season_type == "PRE")
    actuals = schedule[
        (schedule["season"] == int(season)) &
        (schedule["week"] == int(week)) &
        (schedule["season_type"] == season_type) &
        schedule["home_score"].notna() &
        schedule["away_score"].notna()
    ].copy()

    if actuals.empty:
        return None, {
            "message": "No completed games found for that season/week yet.",
            "results_path": None,
        }

    actual_cols = ["game_id", "home_score", "away_score"]
    graded = predictions.merge(actuals[actual_cols], on="game_id", how="left")
    graded = graded[graded["home_score"].notna() & graded["away_score"].notna()].copy()

    if graded.empty:
        return None, {
            "message": "Saved predictions did not match any completed games.",
            "results_path": None,
        }

    graded["actual_margin"] = graded["home_score"] - graded["away_score"]
    graded["ml_result"] = graded.apply(_moneyline_result, axis=1)

    graded["ml_hit"] = graded["ml_pick"] == graded["ml_result"]
    bet_side = graded["bet_side"].fillna(graded["ml_pick"]) if "bet_side" in graded else graded["ml_pick"]
    funded = graded.get("kelly_stake_ml", pd.Series(0.0, index=graded.index)).gt(0) & bet_side.isin(["HOME", "AWAY"])
    graded["bet_hit"] = (bet_side == graded["ml_result"]).where(funded)
    graded["ml_net"] = graded.apply(_moneyline_net, axis=1).round(2)

    summary = {
        "games_completed": int(len(graded)),
        "ml_hits": int(graded["ml_hit"].sum()),
        "ml_accuracy": float(graded["ml_hit"].mean()),
        "ml_net": float(graded["ml_net"].sum().round(2)),
        "bets_placed": int(funded.sum()),
        "bets_won": int(graded["bet_hit"].fillna(False).sum()),
        "total_staked": float(graded.loc[funded, "kelly_stake_ml"].sum()) if "kelly_stake_ml" in graded else 0.0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = results_snapshot_path(season, week, season_type)
    graded.to_csv(results_path, index=False)
    summary["results_path"] = str(results_path)
    summary["message"] = "Results graded."

    return graded, summary
