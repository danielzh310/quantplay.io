import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from sports.nfl.controllers import predict, results
from sports.nfl.data.loaders import _preseason_rows, load_weekly_data
from sports.nfl.data.preprocessing import build_features
from sports.nfl.models.moneyline import calculate_profit_weight
from utils.betting import has_positive_edge


def schedules():
    rows = []
    for season in [2023, 2024, 2025]:
        for phase, weeks in [("PRE", range(4)), ("REG", range(1, 19))]:
            for week in weeks:
                rows.append({
                    "game_id": f"{season}_{phase}_{week}", "season": season,
                    "season_type": phase, "week": week, "home_team": "LA", "away_team": "WAS",
                    "home_score": 14 + week % 5 * 7, "away_score": 17 + week % 3 * 7,
                    "spread_line": 3.0 if phase == "REG" else np.nan,
                    "total_line": 42.0 if phase == "REG" else np.nan,
                    "home_moneyline": -150.0 if phase == "REG" else np.nan,
                    "away_moneyline": 130.0 if phase == "REG" else np.nan,
                })
    return pd.DataFrame(rows)


class PreseasonTests(unittest.TestCase):
    def setUp(self):
        teams = pl.DataFrame({"team_abbr": ["LA", "WAS"], "team_division": ["NFC West", "NFC East"]})
        self.teams_patch = patch("sports.nfl.data.preprocessing.nfl.load_teams", return_value=teams)
        self.teams_patch.start()
        self.addCleanup(self.teams_patch.stop)

    def test_missing_odds_never_signal_a_bet(self):
        for odds in [None, np.nan, np.inf, 0]:
            self.assertFalse(has_positive_edge(0.9, odds))
        self.assertTrue(has_positive_edge(0.9, -150))

    def test_espn_phase_weeks_scores_aliases_and_lines(self):
        event = {
            "season": {"year": 2025, "type": 1}, "week": {"number": 1},
            "date": "2025-08-01T00:00Z", "competitions": [{
                "status": {"type": {"completed": False}},
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "LAR"}, "score": "0"},
                    {"homeAway": "away", "team": {"abbreviation": "WSH"}, "score": "0"},
                ],
                "odds": [{"spread": -3, "overUnder": 35, "homeTeamOdds": {"favorite": False, "moneyLine": 130},
                          "awayTeamOdds": {"favorite": True, "moneyLine": -150}}],
            }],
        }
        regular = copy.deepcopy(event)
        regular["season"]["type"] = 2
        rows = _preseason_rows({"events": [event, regular]}, 2025)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game_id"], "2025_PRE00_WAS_LA")
        self.assertEqual(rows[0]["spread_line"], -3)
        self.assertTrue(np.isnan(rows[0]["home_score"]))
        event["week"]["number"] = 2
        event["competitions"][0]["status"]["type"]["completed"] = True
        event["competitions"][0].pop("odds")
        row = _preseason_rows({"events": [event]}, 2025)[0]
        self.assertEqual(row["week"], 1)
        self.assertEqual(row["home_score"], 0)
        self.assertTrue(np.isnan(row["home_moneyline"]))

    def test_loader_keeps_regular_and_preseason_week_one_separate(self):
        regular = schedules().query("season == 2025 and season_type == 'REG'").copy()
        regular["game_type"] = "REG"
        regular["gameday"] = "2025-09-07"
        regular["gametime"] = "13:00"
        preseason = schedules().query("season == 2025 and season_type == 'PRE'").copy()
        preseason["gameday"] = "2025-08-01T00:00Z"
        with patch("sports.nfl.data.loaders.nfl.load_schedules", return_value=pl.from_pandas(regular)), \
             patch("sports.nfl.data.loaders.load_preseason_data", return_value=preseason):
            loaded = load_weekly_data([2025], include_preseason=True)
        self.assertEqual(set(loaded.query("week == 1")["season_type"]), {"PRE", "REG"})
        self.assertEqual(loaded.iloc[0]["season_type"], "PRE")

    def test_features_do_not_use_other_teams_or_future_games(self):
        data = schedules()
        features = build_features(data)
        first = features.query("season == 2025 and season_type == 'PRE' and week == 0").iloc[0]
        self.assertEqual(first.home_off_season_avg, 0)
        week_one = features.query("season == 2025 and season_type == 'PRE' and week == 1").iloc[0]
        self.assertEqual(week_one.home_off_season_avg, 14)
        self.assertEqual(week_one.away_off_season_avg, 17)
        data.loc[(data.season == 2025) & ((data.season_type == "REG") | (data.week >= 1)), "home_score"] = 999
        changed = build_features(data).set_index("game_id")
        self.assertEqual(week_one.home_off_avg, changed.loc[week_one.game_id, "home_off_avg"])

    def test_actual_models_missing_odds_and_training_cutoff(self):
        data = schedules()
        with patch.object(predict, "load_weekly_data", return_value=data):
            output = predict.run_weekly(2025, 1, season_type="PRE", export=False, verbose=False)
            # Changing the target result and all later games must not change its predictions.
            altered = data.copy()
            later = (altered.season == 2025) & ((altered.season_type == "REG") | (altered.week >= 1))
            altered.loc[later, ["home_score", "away_score"]] = [100, 0]
            with patch.object(predict, "load_weekly_data", return_value=altered):
                repeated = predict.run_weekly(2025, 1, season_type="PRE", export=False, verbose=False)
        pd.testing.assert_frame_equal(output, repeated)
        self.assertEqual(output.game_id.tolist(), ["2025_PRE_1"])
        self.assertEqual(len(output.columns), 21)
        self.assertEqual(output.iloc[0].spread_pick, "PASS")
        self.assertEqual(output.iloc[0].total_pick, "PASS")
        self.assertEqual(output.iloc[0].home_ml_bet, "No Bet")
        self.assertEqual(output.iloc[0].away_ml_bet, "No Bet")
        self.assertTrue(np.isfinite(output.iloc[0].model_spread_margin))
        self.assertTrue(np.isfinite(output.iloc[0].projected_total))
        self.assertEqual(calculate_profit_weight({"home_win": 1, "home_moneyline": np.nan}), 1)

    def test_ui_preseason_prediction_and_grading(self):
        from streamlit.testing.v1 import AppTest

        data = schedules()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(results, "OUTPUT_DIR", Path(temp)), \
             patch.object(results, "load_weekly_data", return_value=data), \
             patch.object(predict, "load_weekly_data", return_value=data):
            app = AppTest.from_file("apps/web/app.py", default_timeout=30).run()
            next(s for s in app.selectbox if s.label == "Season").select(2025)
            next(s for s in app.selectbox if s.label == "Season phase").select("Preseason").run()
            next(s for s in app.selectbox if s.label == "Week").select("1").run()
            next(b for b in app.button if b.label == "Run predictions").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.dataframe[0].value.kelly_stake_ml.tolist(), [0.0])
            self.assertTrue(results.prediction_snapshot_path(2025, 1, "PRE").exists())
            self.assertFalse(results.prediction_snapshot_path(2025, 1).exists())
            next(b for b in app.button if b.label == "Check results").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(results.results_snapshot_path(2025, 1, "PRE").exists())
            next(s for s in app.selectbox if s.label == "Season phase").select("Regular season").run()
            next(s for s in app.selectbox if s.label == "Week").select("1").run()
            next(b for b in app.button if b.label == "Run predictions").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(results.prediction_snapshot_path(2025, 1).exists())


if __name__ == "__main__":
    unittest.main()
