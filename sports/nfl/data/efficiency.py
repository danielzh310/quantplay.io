"""Pregame EPA/success rates from completed prior games, with a small disk cache."""
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
import re
import warnings
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[3] / "outputs" / "efficiency_cache"
PBP_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "epa", "qb_kneel", "qb_spike"]
STAT_COLUMNS = ["game_id", "team", "off_plays", "off_epa", "off_success", "def_plays", "def_epa", "def_success"]
EPA_FEATURES = [f"ml_{side}_{unit}_epa" for side in ("home", "away") for unit in ("off", "def")]
SUCCESS_FEATURES = [f"ml_{side}_{unit}_success" for side in ("home", "away") for unit in ("off", "def")]
ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS"}


def aggregate_plays(pbp):
    """Scrimmage passes/runs (including sacks), excluding spikes and kneels."""
    valid = pbp.play_type.isin(["pass", "run"]) & pd.to_numeric(pbp.epa, errors="coerce").map(np.isfinite)
    for flag in ("qb_kneel", "qb_spike"):
        if flag in pbp:
            valid &= ~pbp[flag].eq(1)
    plays = pbp.loc[valid, ["game_id", "posteam", "defteam", "epa"]].dropna().copy()
    plays["epa"] = pd.to_numeric(plays.epa)
    plays["success"] = plays.epa.gt(0).astype(int)
    for team in ("posteam", "defteam"):
        plays[team] = plays[team].replace(ALIASES)
    offense = plays.groupby(["game_id", "posteam"]).agg(off_plays=("epa", "size"), off_epa=("epa", "sum"), off_success=("success", "sum")).reset_index().rename(columns={"posteam": "team"})
    defense = plays.groupby(["game_id", "defteam"]).agg(def_plays=("epa", "size"), def_epa=("epa", "sum"), def_success=("success", "sum")).reset_index().rename(columns={"defteam": "team"})
    return offense.merge(defense, on=["game_id", "team"], how="inner", validate="one_to_one")[STAT_COLUMNS]


def load_efficiency_history(history):
    """Fetch only years needed by completed pre-cutoff games; never cache failures."""
    eligible = history[history.season_type.ne("PRE") & history.home_score.notna() & history.away_score.notna()]
    eligible = eligible[eligible.game_id.astype(str).str.match(r"^\d{4}_\d{2}_[A-Z0-9]+_[A-Z0-9]+$")]
    frames, failed = [], []
    for year, season in eligible.groupby("season"):
        path = CACHE / f"team_efficiency_v1_{int(year)}.parquet"
        needed = set(season.game_id)
        cached = pd.DataFrame(columns=STAT_COLUMNS)
        if path.exists():
            try:
                cached = pd.read_parquet(path)[STAT_COLUMNS]
            except (OSError, ValueError, KeyError):
                pass
        complete = set(cached.groupby("game_id").team.nunique().loc[lambda counts: counts.eq(2)].index)
        if not needed.issubset(complete):
            try:
                url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{int(year)}.parquet"
                with urlopen(url, timeout=45) as response:
                    pbp = pd.read_parquet(BytesIO(response.read()), columns=PBP_COLUMNS)
                fresh = aggregate_plays(pbp)
                # Preserve previously cached games if a current release is partial.
                cached = (fresh if cached.empty else pd.concat([cached, fresh], ignore_index=True)).drop_duplicates(["game_id", "team"], keep="last")
                CACHE.mkdir(parents=True, exist_ok=True)
                cached.to_parquet(path, index=False)
            except (OSError, URLError, ValueError, KeyError) as exc:
                failed.append(str(int(year)))
        frames.append(cached[cached.game_id.isin(needed)])
    if failed:
        warnings.warn("EPA/success data unavailable for seasons " + ", ".join(failed) + "; scoring-only features remain available.", stacklevel=2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STAT_COLUMNS)


def add_efficiency_features(games, summaries=None, *, season=None, week=None, season_type="REG"):
    """Emit last-eight-game rates before updating a whole season/phase/week batch.

    Missing data remains NaN, not a zero performance estimate or a stake penalty.
    The estimator fits any imputation on its own earlier training fold only.
    """
    data = games.drop(columns=[c for c in games if c in EPA_FEATURES + SUCCESS_FEATURES or re.match(r"ml_(home|away)_efficiency_games$", c)]).copy()
    phase = data.season_type.map({"PRE": 0, "REG": 1, "POST": 2})
    before = pd.Series(True, index=data.index)
    if season is not None and week is not None:
        target_phase = {"PRE": 0, "REG": 1, "POST": 2}[season_type]
        before = data.season.lt(season) | (data.season.eq(season) & (phase.lt(target_phase) | (phase.eq(target_phase) & data.week.lt(week))))
    if summaries is None:
        summaries = load_efficiency_history(data[before])
    if summaries.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team efficiency summaries")
    stats = {(r.game_id, r.team): r for r in summaries.itertuples()}
    state = defaultdict(lambda: deque(maxlen=8))
    records = []
    data["_eff_phase"] = phase
    for _, batch in data.sort_values(["season", "_eff_phase", "week", "game_id"]).groupby(["season", "_eff_phase", "week"], sort=False):
        for game in batch.itertuples():
            record = {"game_id": game.game_id}
            for side in ("home", "away"):
                team = ALIASES.get(getattr(game, f"{side}_team"), getattr(game, f"{side}_team"))
                # Do not carry ancient team history across missing seasons.
                recent = [s for year, s in state[team] if year >= game.season - 1] if game.season_type != "PRE" else []
                record[f"ml_{side}_efficiency_games"] = len(recent)
                for unit in ("off", "def"):
                    plays = sum(getattr(r, f"{unit}_plays") for r in recent)
                    for metric in ("epa", "success"):
                        record[f"ml_{side}_{unit}_{metric}"] = sum(getattr(r, f"{unit}_{metric}") for r in recent) / plays if plays else np.nan
            records.append(record)
        for index, game in batch.iterrows():
            if not before.loc[index] or game.season_type == "PRE" or pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            for side in ("home", "away"):
                team = ALIASES.get(game[f"{side}_team"], game[f"{side}_team"])
                stat = stats.get((game.game_id, team))
                if stat is not None and stat.off_plays > 0 and stat.def_plays > 0:
                    state[team].append((game.season, stat))
    return data.drop(columns="_eff_phase").merge(pd.DataFrame(records), on="game_id", validate="one_to_one")
