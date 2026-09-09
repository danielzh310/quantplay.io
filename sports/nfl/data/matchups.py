"""Pregame display statistics; these do not change the prediction models."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from urllib.request import urlopen
from urllib.error import URLError
import warnings

import pandas as pd

# Key, display name, higher values are better.
COMPARISON_STATS = [
    ('points_for', 'Scoring offense', True),
    ('pass_yards', 'Passing offense', True),
    ('rush_yards', 'Rushing offense', True),
    ('giveaways', 'Ball security', False),
    ('points_against', 'Scoring defense', False),
    ('pass_allowed', 'Pass defense', False),
    ('rush_allowed', 'Run defense', False),
    ('takeaways', 'Takeaways', True),
]
ALIASES = {'LAR': 'LA', 'WSH': 'WAS'}


def parse_boxscore(payload):
    teams = {}
    for item in payload.get('boxscore', {}).get('teams', []):
        name = item['team']['abbreviation']
        stats = {}
        for stat in item.get('statistics', []):
            value = pd.to_numeric(stat.get('value'), errors='coerce')
            if pd.isna(value):
                value = pd.to_numeric(stat.get('displayValue'), errors='coerce')
            if pd.notna(value) and math.isfinite(value):
                stats[stat['name']] = float(value)
        teams[ALIASES.get(name, name)] = stats
    return teams


def load_boxscore(event_id):
    url = f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={int(event_id)}'
    with urlopen(url, timeout=20) as response:
        return parse_boxscore(json.load(response))


def add_matchup_stats(picks, history, season_type, meeting_history=None):
    # The caller provides only completed games before the selected week.
    history = history.dropna(subset=['home_score', 'away_score']).copy()
    history = history[history.season_type.eq('PRE') if season_type == 'PRE' else history.season_type.ne('PRE')]
    history['_phase'] = history.season_type.map({'PRE': 0, 'REG': 1, 'POST': 2})
    history = history.sort_values(['season', '_phase', 'week', 'game_id'])
    meetings_source = history.copy() if meeting_history is None else meeting_history.copy()
    meetings_source = meetings_source.dropna(subset=['home_score', 'away_score']).drop_duplicates('game_id')
    meetings_source = meetings_source[
        meetings_source.season_type.eq('PRE') if season_type == 'PRE'
        else meetings_source.season_type.ne('PRE')
    ].copy()
    # Keep relocated franchises together in the longer historical sample.
    for column in ('home_team', 'away_team'):
        meetings_source[column] = meetings_source[column].replace({**ALIASES, 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'})
    meetings_source['_phase'] = meetings_source.season_type.map({'PRE': 0, 'REG': 1, 'POST': 2})
    meetings_source = meetings_source.sort_values(['season', '_phase', 'week', 'game_id'])
    teams = sorted(set(history.home_team) | set(history.away_team))
    recent = {team: history[(history.home_team == team) | (history.away_team == team)].tail(5) for team in teams}
    needed = pd.concat(list(recent.values())).drop_duplicates('game_id') if recent else history

    def fetch(row):
        event_id = row.get('espn_event_id')
        if pd.isna(event_id):
            return row.game_id, {}
        try:
            return row.game_id, load_boxscore(event_id)
        except (URLError, TimeoutError, ValueError, OSError):
            return row.game_id, {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        boxes = dict(executor.map(fetch, (row for _, row in needed.iterrows())))
    if boxes and not any(boxes.values()):
        warnings.warn('Detailed box scores unavailable; passing/rushing and turnover comparisons will be omitted.', stacklevel=2)

    summaries = []
    for team, games in recent.items():
        observations = []
        box_count = 0
        for _, game in games.iterrows():
            side = 'home' if game.home_team == team else 'away'
            other = 'away' if side == 'home' else 'home'
            box = boxes.get(game.game_id, {})
            own = box.get(team, {})
            opponent = box.get(game[f'{other}_team'], {})
            box_count += bool(own and opponent)
            observations.append({
                'points_for': game[f'{side}_score'], 'points_against': game[f'{other}_score'],
                'pass_yards': own.get('netPassingYards'), 'rush_yards': own.get('rushingYards'),
                'giveaways': own.get('turnovers'), 'pass_allowed': opponent.get('netPassingYards'),
                'rush_allowed': opponent.get('rushingYards'), 'takeaways': opponent.get('turnovers'),
            })
        obs = pd.DataFrame(observations)
        summary = {'team': team, 'games': len(games), 'box_games': box_count}
        for key, _, _ in COMPARISON_STATS:
            column = pd.to_numeric(obs[key], errors='coerce')
            summary[key] = column.mean()
            summary[f'{key}_n'] = int(column.notna().sum())
        summaries.append(summary)
    league = pd.DataFrame(summaries)
    if league.empty:
        return picks.copy()
    league = league.set_index('team')
    for key, _, higher_better in COMPARISON_STATS:
        league[f'{key}_rating'] = league[key].rank(ascending=higher_better, pct=True) * 100
        league[f'{key}_pool'] = int(league[key].notna().sum())

    enriched = []
    for _, pick in picks.iterrows():
        record = pick.to_dict()
        record['comparison_version'] = 2
        for side in ['home', 'away']:
            team = pick[f'{side}_team']
            if team in league.index:
                for column, val in league.loc[team].items():
                    record[f'{side}_recent_{column}'] = val
        meetings = meetings_source[
            ((meetings_source.home_team == pick.home_team) & (meetings_source.away_team == pick.away_team)) |
            ((meetings_source.home_team == pick.away_team) & (meetings_source.away_team == pick.home_team))
        ].tail(10)
        record['matchup_history'] = json.dumps([
            {'game_id': game.game_id, 'season': int(game.season), 'week': int(game.week),
             'home_team': game.home_team, 'away_team': game.away_team,
             'home_score': float(game.home_score), 'away_score': float(game.away_score)}
            for _, game in meetings.iterrows()
        ])
        enriched.append(record)
    return pd.DataFrame(enriched)
