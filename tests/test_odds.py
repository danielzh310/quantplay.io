import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd

from sports.nfl.controllers import results
from sports.nfl.data.odds import refresh_game_odds, select_pregame_market


def market():
    return {
        "provider": {"id": "58", "name": "ESPN BET", "priority": 0},
        "homeTeamOdds": {
            "moneyLine": 700,
            "current": {"moneyLine": {"american": "+140"}, "pointSpread": {"american": "+3.5"}},
            "close": {"moneyLine": {"american": "+225"}, "pointSpread": {"american": "+6.5"}},
        },
        "awayTeamOdds": {
            "moneyLine": -900,
            "current": {"moneyLine": {"american": "-160"}},
            "close": {"moneyLine": {"american": "-275"}},
        },
        "current": {"total": {"american": "40.5"}},
        "close": {"total": {"american": "35.5"}},
    }


class OddsTests(unittest.TestCase):
    def test_upcoming_current_and_completed_close(self):
        upcoming = select_pregame_market([market()], started=False)
        closed = select_pregame_market([market()], started=True)
        self.assertEqual(upcoming, {"home_moneyline": 140, "away_moneyline": -160, "spread_line": -3.5, "total_line": 40.5})
        self.assertEqual(closed, {"home_moneyline": 225, "away_moneyline": -275, "spread_line": -6.5, "total_line": 35.5})

    def test_live_provider_and_missing_close_cannot_supply_closing_odds(self):
        pregame = market()
        live = copy.deepcopy(pregame)
        live["provider"] = {"id": "59", "name": "ESPN Bet - Live Odds", "priority": -1}
        live["homeTeamOdds"]["close"]["moneyLine"]["american"] = "-999"
        self.assertEqual(select_pregame_market([live, pregame], True)["home_moneyline"], 225)
        del pregame["homeTeamOdds"]["close"]
        self.assertTrue(pd.isna(select_pregame_market([live, pregame], True)["home_moneyline"]))
        self.assertTrue(all(pd.isna(value) for value in select_pregame_market([live], False).values()))

    def test_decimal_payout_is_not_an_american_price(self):
        data = market()
        data["homeTeamOdds"]["close"]["moneyLine"] = {"value": 3.25}
        self.assertTrue(pd.isna(select_pregame_market([data], True)["home_moneyline"]))

    def test_refresh_uses_kickoff_and_fetches_again(self):
        games = pd.DataFrame({
            "espn_event_id": [1, 2, 3], "kickoff": ["2099-01-01T12:00Z", "2000-01-01T12:00Z", "2099-01-01T12:00Z"],
            "home_score": [float("nan"), float("nan"), 20], "game_started": [False, False, True],
        })
        with patch("sports.nfl.data.odds.load_game_odds", side_effect=lambda event, started: select_pregame_market([market()], started)) as fetch:
            first = refresh_game_odds(games)
            refresh_game_odds(games)
        self.assertEqual(fetch.call_count, 6)
        self.assertEqual(first.home_moneyline.tolist(), [140, 225, 225])

    def test_outage_does_not_keep_stale_prices(self):
        games = pd.DataFrame({"espn_event_id": [1], "kickoff": ["2099-01-01T12:00Z"], "home_moneyline": [140]})
        with patch("sports.nfl.data.odds.load_game_odds", side_effect=URLError("unavailable")), self.assertWarns(UserWarning):
            refreshed = refresh_game_odds(games)
        self.assertTrue(pd.isna(refreshed.iloc[0].home_moneyline))

    def test_grading_retains_saved_pregame_price(self):
        picks = pd.DataFrame({"game_id": ["game"], "ml_pick": ["HOME"], "home_moneyline": [140], "away_moneyline": [-160], "kelly_stake_ml": [10]})
        actuals = pd.DataFrame({"game_id": ["game"], "season": [2025], "week": [1], "season_type": ["PRE"],
                                "home_score": [24], "away_score": [16], "home_moneyline": [225]})
        with tempfile.TemporaryDirectory() as temp, patch.object(results, "OUTPUT_DIR", Path(temp)), patch.object(results, "load_weekly_data", return_value=actuals):
            results.save_prediction_snapshot(picks, 2025, 1, "PRE")
            graded, summary = results.grade_saved_predictions(2025, 1, "PRE")
        self.assertEqual(graded.iloc[0].home_moneyline, 140)
        self.assertEqual(summary["ml_net"], 14)


if __name__ == "__main__":
    unittest.main()
