import pandas as pd
import nflreadpy as nfl

ROLLING_WINDOW = 4


def build_features(df):
    """
    Build rolling offensive and defensive team features for football modeling
    moneyline, spread, and totals.
    """

    data = df.copy()

    # Preseason week numbers overlap regular-season weeks. Sort by phase first.
    data["phase_order"] = data.get("season_type", pd.Series("REG", index=data.index)).map({"PRE": 0, "REG": 1, "POST": 2})

    # Targets and helper columns
    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["margin"] = data["home_score"] - data["away_score"]
    data["total_pts"] = data["home_score"] + data["away_score"]

    # divisional game indicator
    teams_df = nfl.load_teams().to_pandas()
    team_divisions = teams_df.set_index('team_abbr')['team_division'].to_dict()
    home_division = data['home_team'].map(team_divisions)
    away_division = data['away_team'].map(team_divisions)
    data['is_division_game'] = (home_division == away_division).astype(int)


    home = data[
        ["game_id", "season", "phase_order", "week", "home_team", "home_score", "away_score"]
    ].copy()

    home.rename(
        columns={
            "home_team": "team",
            "home_score": "pts_for",
            "away_score": "pts_against",
        },
        inplace=True,
    )

    away = data[
        ["game_id", "season", "phase_order", "week", "away_team", "away_score", "home_score"]
    ].copy()

    away.rename(
        columns={
            "away_team": "team",
            "away_score": "pts_for",
            "home_score": "pts_against",
        },
        inplace=True,
    )

    teams = (
        pd.concat([home, away])
        .sort_values(["team", "season", "phase_order", "week"])
        .reset_index(drop=True)
    )

    # Compute windows within each group, using completed games only. Shift after
    # carrying history forward so future fixtures never erase the latest averages.
    for score, prefix in [("pts_for", "off"), ("pts_against", "def")]:
        teams[f"{prefix}_avg"] = teams.groupby("team")[score].transform(
            lambda values: values.dropna().rolling(ROLLING_WINDOW, min_periods=1)
            .mean().reindex(values.index).ffill().shift(1)
        ).fillna(0)
        teams[f"{prefix}_season_avg"] = teams.groupby(["team", "season"])[score].transform(
            lambda values: values.dropna().expanding().mean()
            .reindex(values.index).ffill().shift(1)
        ).fillna(0)

    home_feats = teams[
        ["game_id", "team", "off_avg", "def_avg", "off_season_avg", "def_season_avg"]
    ].rename(
        columns={
            "team": "home_team",
            "off_avg": "home_off_avg",
            "def_avg": "home_def_avg",
            "off_season_avg": "home_off_season_avg",
            "def_season_avg": "home_def_season_avg",
        }
    )

    away_feats = teams[
        ["game_id", "team", "off_avg", "def_avg", "off_season_avg", "def_season_avg"]
    ].rename(
        columns={
            "team": "away_team",
            "off_avg": "away_off_avg",
            "def_avg": "away_def_avg",
            "off_season_avg": "away_off_season_avg",
            "def_season_avg": "away_def_season_avg",
        }
    )

    data = data.merge(home_feats, on=["game_id", "home_team"])
    data = data.merge(away_feats, on=["game_id", "away_team"])

    return data
