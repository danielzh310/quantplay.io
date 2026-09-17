"""Read ESPN pregame markets, never in-game prices."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import warnings

import pandas as pd


MARKET_COLUMNS = ["home_moneyline", "away_moneyline", "spread_line", "total_line"]


def _price(value):
    if isinstance(value, dict):
        # `value` is a decimal payout, not an American price or point line.
        value = value.get("american", value.get("alternateDisplayValue"))
    try:
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def select_pregame_market(items, started):
    """Select one non-live provider; completed/in-progress games require close."""
    candidates = []
    for item in items:
        provider = item.get("provider") or {}
        if "live" in provider.get("name", "").lower() or str(provider.get("id")) == "59":
            continue
        snapshot = "close" if started else "current"
        home = item.get("homeTeamOdds") or {}
        away = item.get("awayTeamOdds") or {}
        home_snapshot = home.get(snapshot) or {}
        away_snapshot = away.get(snapshot) or {}
        market = {
            "home_moneyline": _price(home_snapshot.get("moneyLine")),
            "away_moneyline": _price(away_snapshot.get("moneyLine")),
            # Home handicap is the opposite sign of nflverse's expected margin.
            "spread_line": -_price(home_snapshot.get("pointSpread")),
            "total_line": _price((item.get(snapshot) or {}).get("total")),
        }
        if pd.isna(market["spread_line"]):
            market["spread_line"] = _price(away_snapshot.get("pointSpread"))
        if not started:
            # Legacy scalar fields are safe only while the game has not started.
            for side, source in [("home", home), ("away", away)]:
                column = f"{side}_moneyline"
                if pd.isna(market[column]):
                    market[column] = _price(source.get("moneyLine"))
            if pd.isna(market["total_line"]):
                market["total_line"] = _price(item.get("overUnder"))
            if pd.isna(market["spread_line"]):
                spread = abs(_price(item.get("spread")))
                if home.get("favorite"):
                    market["spread_line"] = spread
                elif away.get("favorite"):
                    market["spread_line"] = -spread
                elif spread == 0:
                    market["spread_line"] = 0.0
        for column in ["home_moneyline", "away_moneyline"]:
            if abs(market[column]) < 100:
                market[column] = float("nan")
        count = sum(pd.notna(value) for value in market.values())
        priority = _price(provider.get("priority", 999))
        if not math.isfinite(priority):
            priority = 999
        candidates.append(((-count, priority, str(provider.get("id", ""))), market))
    if not candidates:
        return {column: float("nan") for column in MARKET_COLUMNS}
    return min(candidates, key=lambda candidate: candidate[0])[1]


def load_game_odds(event_id, started):
    event_id = str(int(event_id))
    url = (
        "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
        f"events/{event_id}/competitions/{event_id}/odds?limit=100"
    )
    # Fetch anew on every query so upcoming prices cannot become stale in cache.
    with urlopen(url, timeout=15) as response:
        payload = json.load(response)
    return select_pregame_market(payload.get("items", []), started)


def select_market_snapshot(items, snapshot):
    """Select one non-live provider's fixed snapshot ('open', 'current', or 'close').

    Unlike `select_pregame_market`, this never depends on whether the game has
    started -- it always reads the same named snapshot, which is how a
    historical, already-completed game's opening line (as opposed to its
    closing line) can be recovered.
    """
    candidates = []
    for item in items:
        provider = item.get("provider") or {}
        if "live" in provider.get("name", "").lower() or str(provider.get("id")) == "59":
            continue
        home = item.get("homeTeamOdds") or {}
        away = item.get("awayTeamOdds") or {}
        home_snapshot = home.get(snapshot) or {}
        away_snapshot = away.get(snapshot) or {}
        market = {
            "home_moneyline": _price(home_snapshot.get("moneyLine")),
            "away_moneyline": _price(away_snapshot.get("moneyLine")),
            "spread_line": -_price(home_snapshot.get("pointSpread")),
            "total_line": _price((item.get(snapshot) or {}).get("total")),
        }
        if pd.isna(market["spread_line"]):
            market["spread_line"] = _price(away_snapshot.get("pointSpread"))
        for column in ["home_moneyline", "away_moneyline"]:
            if abs(market[column]) < 100:
                market[column] = float("nan")
        count = sum(pd.notna(value) for value in market.values())
        priority = _price(provider.get("priority", 999))
        if not math.isfinite(priority):
            priority = 999
        candidates.append(((-count, priority, str(provider.get("id", ""))), market))
    if not candidates:
        return {column: float("nan") for column in MARKET_COLUMNS}
    return min(candidates, key=lambda candidate: candidate[0])[1]


def load_game_market_snapshot(event_id, snapshot):
    event_id = str(int(event_id))
    url = (
        "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
        f"events/{event_id}/competitions/{event_id}/odds?limit=100"
    )
    with urlopen(url, timeout=15) as response:
        payload = json.load(response)
    return select_market_snapshot(payload.get("items", []), snapshot)


def load_market_snapshots(event_ids, snapshot="open", max_workers=6):
    """Fetch a fixed odds snapshot for many (typically historical) events at once.

    Used for backtesting against something other than the closing line, e.g.
    `snapshot="open"` to approximate the line as it stood early in game week,
    before injury reports and public/sharp money move it toward close.
    """
    unique_ids = sorted({int(event_id) for event_id in event_ids if pd.notna(event_id)})

    def fetch(event_id):
        try:
            return event_id, load_game_market_snapshot(event_id, snapshot), False
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return event_id, {column: float("nan") for column in MARKET_COLUMNS}, True

    results = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for event_id, market, failed in executor.map(fetch, unique_ids):
            results[event_id] = market
            failures += failed
    if failures:
        warnings.warn(f"ESPN '{snapshot}' odds unavailable for {failures}/{len(unique_ids)} events.", stacklevel=2)
    return results


def refresh_game_odds(games):
    """Refresh markets while preserving the schedule and unavailable-odds behavior."""
    if games.empty or "espn_event_id" not in games:
        return games.copy()
    out = games.copy()
    now = pd.Timestamp.now(tz="UTC")
    jobs = []
    for index, row in out.iterrows():
        kickoff = pd.to_datetime(row.get("kickoff", row.get("gameday")), utc=True)
        started = (
            bool(row.get("game_started", False))
            or pd.notna(row.get("home_score"))
            or (pd.notna(kickoff) and kickoff <= now)
        )
        jobs.append((index, row["espn_event_id"], started))

    def fetch(job):
        index, event_id, started = job
        if pd.isna(event_id):
            return index, {column: float("nan") for column in MARKET_COLUMNS}, True
        try:
            return index, load_game_odds(event_id, started), False
        except HTTPError as exc:
            if exc.code not in {404, 410}:
                return index, {column: float("nan") for column in MARKET_COLUMNS}, True
            return index, {column: float("nan") for column in MARKET_COLUMNS}, False
        except (URLError, TimeoutError, ValueError, OSError):
            return index, {column: float("nan") for column in MARKET_COLUMNS}, True

    failures = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, market, failed in executor.map(fetch, jobs):
            for column, value in market.items():
                out.at[index, column] = value
            failures += failed
    if failures:
        warnings.warn(f"ESPN odds unavailable for {failures} games; their markets were left blank.", stacklevel=2)
    return out
