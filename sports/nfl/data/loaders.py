"""NFL schedules: nflverse regular/postseason plus ESPN preseason games."""
import json
from urllib.error import URLError
from urllib.request import urlopen

import nflreadpy as nfl
import pandas as pd

SCHEDULE_COLUMNS = [
    "game_id", "season", "week", "season_type", "gameday", "home_team", "away_team",
    "home_score", "away_score", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "espn_event_id", "kickoff", "game_started",
]
TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS"}


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _preseason_rows(payload, season):
    rows = []
    for event in payload.get("events", []):
        if event.get("season", {}).get("type") != 1 or event["season"]["year"] != season:
            continue
        competition = event["competitions"][0]
        sides = {c["homeAway"]: c for c in competition["competitors"]}
        home, away = sides["home"], sides["away"]
        completed = competition.get("status", event.get("status", {})).get("type", {}).get("completed", False)
        odds = (competition.get("odds") or [{}])[0]
        home_odds = odds.get("homeTeamOdds") or {}
        away_odds = odds.get("awayTeamOdds") or {}
        # ESPN week 1 is the Hall of Fame game, represented here as week 0.
        week = int(event["week"]["number"]) - 1
        home_team = TEAM_ALIASES.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
        away_team = TEAM_ALIASES.get(away["team"]["abbreviation"], away["team"]["abbreviation"])
        spread = abs(_number(odds.get("spread")))
        # nflverse uses a positive line when the home team is favored.
        if home_odds.get("favorite"):
            spread_line = spread
        elif away_odds.get("favorite"):
            spread_line = -spread
        else:
            spread_line = 0.0 if spread == 0 else float("nan")
        rows.append({
            "game_id": f"{season}_PRE{week:02d}_{away_team}_{home_team}",
            "season": season, "week": week, "season_type": "PRE",
            "gameday": event["date"], "home_team": home_team, "away_team": away_team,
            "home_score": _number(home.get("score")) if completed else float("nan"),
            "away_score": _number(away.get("score")) if completed else float("nan"),
            "spread_line": spread_line, "total_line": _number(odds.get("overUnder")),
            "home_moneyline": _number(home_odds.get("moneyLine")),
            "away_moneyline": _number(away_odds.get("moneyLine")),
            "espn_event_id": event.get("id"), "kickoff": event["date"],
            "game_started": completed or competition.get("status", event.get("status", {})).get("type", {}).get("state") == "in",
        })
    return rows


def load_preseason_data(seasons):
    rows = []
    for season in seasons:
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            f"?dates={season}0701-{season}0930&limit=1000"
        )
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.load(response)
            if "events" not in payload:
                raise ValueError("Schedule response has no events field")
            rows.extend(_preseason_rows(payload, int(season)))
        except (URLError, TimeoutError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Could not load ESPN preseason games for {season}: {exc}") from exc
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS).drop_duplicates("game_id")


def load_weekly_data(seasons=None, include_preseason=False):
    if seasons is None:
        seasons = list(range(2022, nfl.get_current_season() + 1))
    elif isinstance(seasons, int):
        seasons = [seasons]
    schedules = nfl.load_schedules(seasons=seasons).to_pandas()
    schedules["espn_event_id"] = schedules.get("espn", float("nan"))
    # nflverse kickoff times are Eastern local time; retain the actual instant.
    local_kickoff = pd.to_datetime(
        schedules["gameday"].astype(str) + " " + schedules["gametime"].fillna("00:00"),
        errors="coerce",
    )
    schedules["kickoff"] = local_kickoff.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    schedules["game_started"] = schedules["home_score"].notna()
    schedules["season_type"] = schedules["game_type"].map(
        {"REG": "REG", "WC": "POST", "DIV": "POST", "CON": "POST", "SB": "POST"}
    )
    playoff_week = schedules["game_type"].map({"WC": 19, "DIV": 20, "CON": 21, "SB": 22})
    schedules["week"] = playoff_week.fillna(schedules["week"]).astype(int)
    schedules = schedules[SCHEDULE_COLUMNS]
    if include_preseason:
        schedules = pd.concat([schedules, load_preseason_data(seasons)], ignore_index=True)
    schedules["gameday"] = pd.to_datetime(schedules["gameday"], utc=True, format="mixed")
    return schedules.sort_values(["season", "gameday", "game_id"]).reset_index(drop=True)
