"""Pregame moneyline features, independent of spread/total feature definitions.

History is updated only after a complete week/phase batch is emitted. Preseason
has separate team state. No actual starter or score from the target game is used.
"""
from collections import defaultdict
import math

import pandas as pd

HALF_LIVES = (4, 8, 12)
PRIOR_GAMES = 8.0  # shrinkage strength, explicitly fixed, not fitted to test P&L
PRIOR_STRENGTHS = (4, 8, 16)  # selected on earlier temporal validation, never target outcomes


def _number(value, default=float("nan")):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def market_probability(home_odds, away_odds):
    def implied(odds):
        odds = _number(odds)
        if not math.isfinite(odds) or abs(odds) < 100:
            return float("nan")
        return 100 / (100 + odds) if odds > 0 else -odds / (100 - odds)
    home, away = implied(home_odds), implied(away_odds)
    return home / (home + away) if math.isfinite(home + away) else float("nan")


def build_moneyline_features(games):
    data = games.copy()
    if data.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs in moneyline history")
    data["_phase"] = data.season_type.map({"PRE": 0, "REG": 1, "POST": 2})
    data = data.sort_values(["season", "_phase", "week", "game_id"])
    states = {}
    league = defaultdict(lambda: [0., 0])
    rows = []
    for (season, phase, week), batch in data.groupby(["season", "_phase", "week"], sort=False):
        pool = "PRE" if phase == 0 else "REG"
        league_mean = league[pool][0] / league[pool][1] if league[pool][1] else 22.0
        emitted = []
        for _, game in batch.iterrows():
            values = {"game_id": game.game_id}
            team_stats = {}
            for side, other in (("home", "away"), ("away", "home")):
                key = (pool, game[f"{side}_team"])
                state = states.setdefault(key, {"season": None, "for": [], "against": [], "date": None})
                if state["season"] != season:
                    old_for, old_against = state["for"], state["against"]
                    # Only carry the immediately prior season; missing seasons
                    # fall back to the league prior, not an obsolete roster.
                    continuous = state["season"] == season - 1
                    count = len(old_for) if continuous else 0
                    prior_for = (sum(old_for) + PRIOR_GAMES * league_mean) / (count + PRIOR_GAMES) if count else league_mean
                    prior_against = (sum(old_against) + PRIOR_GAMES * league_mean) / (count + PRIOR_GAMES) if count else league_mean
                    state.update(season=season, prior_for=prior_for, prior_against=prior_against,
                                 **{"for": [], "against": [], "date": None})
                    for strength in PRIOR_STRENGTHS:
                        for field, old_values in (("for", old_for), ("against", old_against)):
                            state[f"prior_{field}_k{strength}"] = ((sum(old_values) + strength * league_mean) / (count + strength)
                                                                    if count else league_mean)
                    for half_life in HALF_LIVES:
                        state[f"ew_for_{half_life}"] = prior_for
                        state[f"ew_against_{half_life}"] = prior_against
                    state["adjusted_for"] = 0.
                    state["adjusted_against"] = 0.
                count = len(state["for"])
                for field in ("for", "against"):
                    values[f"ml_{side}_{field}_prior"] = state[f"prior_{field}"]
                    values[f"ml_{side}_{field}_season"] = (sum(state[field]) + PRIOR_GAMES * state[f"prior_{field}"]) / (count + PRIOR_GAMES)
                    for half_life in HALF_LIVES:
                        values[f"ml_{side}_{field}_ew{half_life}"] = state[f"ew_{field}_{half_life}"]
                    values[f"ml_{side}_{field}_adjusted"] = state[f"adjusted_{field}"]
                    for strength in PRIOR_STRENGTHS:
                        prior = state[f"prior_{field}_k{strength}"]
                        values[f"ml_{side}_{field}_prior_k{strength}"] = prior
                        values[f"ml_{side}_{field}_season_k{strength}"] = (sum(state[field]) + strength * prior) / (count + strength)
                values[f"ml_{side}_history_n"] = count
                kickoff = pd.to_datetime(game.get("kickoff", game.get("gameday")), utc=True, errors="coerce")
                rest = _number(game.get(f"{side}_rest"))
                if not math.isfinite(rest) and state["date"] is not None and pd.notna(kickoff):
                    rest = (kickoff - state["date"]).total_seconds() / 86400
                values[f"ml_{side}_rest_known"] = float(math.isfinite(rest))
                values[f"ml_{side}_rest"] = min(21., max(0., rest)) if math.isfinite(rest) else 7.
                team_stats[side] = (state, kickoff)
            values["ml_neutral_site"] = float(str(game.get("location", "Home")).lower() == "neutral")
            values["ml_market_home"] = market_probability(game.get("home_moneyline"), game.get("away_moneyline"))
            values["ml_history_n"] = min(values["ml_home_history_n"], values["ml_away_history_n"])
            rows.append(values)
            emitted.append((game, values, team_stats))
        # Do not let an earlier game in this batch leak into another game's features.
        for game, values, team_stats in emitted:
            if not math.isfinite(_number(game.home_score)) or not math.isfinite(_number(game.away_score)):
                continue
            for side, other in (("home", "away"), ("away", "home")):
                state, kickoff = team_stats[side]
                scored, allowed = float(game[f"{side}_score"]), float(game[f"{other}_score"])
                state["for"].append(scored)
                state["against"].append(allowed)
                if pd.notna(kickoff):
                    state["date"] = kickoff
                for half_life in HALF_LIVES:
                    alpha = 1 - 2 ** (-1 / half_life)
                    state[f"ew_for_{half_life}"] += alpha * (scored - state[f"ew_for_{half_life}"])
                    state[f"ew_against_{half_life}"] += alpha * (allowed - state[f"ew_against_{half_life}"])
                alpha = 1 - 2 ** (-1 / 8)
                state["adjusted_for"] += alpha * (scored - values[f"ml_{other}_against_season"] - state["adjusted_for"])
                state["adjusted_against"] += alpha * (allowed - values[f"ml_{other}_for_season"] - state["adjusted_against"])
                league[pool][0] += scored
                league[pool][1] += 1
    extras = pd.DataFrame(rows)
    return games.merge(extras, on="game_id", validate="one_to_one")
