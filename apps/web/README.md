# NFL predictions

From the repository root, install dependencies and start the UI:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run apps/web/app.py
```

Select **NFL**, choose **Preseason** under **Season phase**, select a season and
week, and click **Run predictions**. The Hall of Fame game is preseason week 0;
weeks 1–3 cover the modern preseason (week 4 is also offered before 2021).
Choose **Regular season** for numbered regular-season weeks, or **Playoffs** for
Wildcard, Divisional, Conference Championship, and Super Bowl rounds. The week
dropdown updates to match the selected phase.

The table, column controls, Kelly settings, and CSV download use the same output
columns for all phases. **Check results** grades the saved moneyline picks once
games finish. Preseason snapshots use `outputs/nfl_preseason_predictions_<season>_wk<week>.csv`
and `outputs/nfl_preseason_results_<season>_wk<week>.csv`, so they cannot overwrite
regular-season snapshots. Running predictions again overwrites that phase/week's snapshot.

Preseason schedules and final scores come from ESPN's public scoreboard endpoint;
nflverse supplies regular-season and postseason history. Each prediction run fetches
the selected slate's game-specific ESPN odds for all NFL phases. Before kickoff,
it uses the provider's current pregame prices; once a game starts, it uses explicit
closing prices. Live-odds providers are excluded. Preseason training history also
gets these archived closing prices. Prices are fetched again on each run, rather
than cached, so upcoming markets can update. A single provider supplies each game's
markets, preferring complete data and then ESPN's provider priority.

Checking results retains the odds and stakes saved when predictions were generated;
it does not replace those prices with closing odds. Rerunning predictions does save
a new snapshot at the same path, using the newly fetched prices.

ESPN may omit individual markets or closing records. Missing moneylines mean no bet and a zero Kelly
stake. Without a spread line, a direct margin model supplies the projection and
the spread pick is `PASS`; the market edge remains blank. Total projections also
remain available without a total line, with a `PASS` pick and blank edge.

Preseason runs train on prior seasons plus earlier preseason weeks, including
available preseason history. They exclude the selected week and later games.
Regular-season runs continue to use regular/postseason history. Features use
completed games in chronological phase order and keep team histories separate.
The models use team scoring history; they do not yet model preseason lineup or
playing-time changes.

The Python entry point also supports preseason:

```python
from sports.nfl.controllers.predict import run_weekly

picks = run_weekly(season=2025, week=1, season_type="PRE", export=False)
```

Run the offline regression and UI checks with:

```powershell
python -m unittest discover -s tests -v
```
