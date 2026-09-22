# quantplay.io
### Sports Modeling Research for Pre-Game Prediction Signals

> DON'T GAMBLE KIDS
> This project is for educational purposes only and does not constitute financial or gambling advice.

---

## Project Overview

QuantPlay is a sports analytics research platform for exploring pre-game prediction signals across multiple sports.
The current working module lives in `sports/nfl` and focuses on football moneyline, spread, and total predictions.

The long-term goal is to keep QuantPlay as the umbrella project while adding sport-specific modules for baseball,
basketball, and other leagues without making the public app feel tied to one licensed sports brand.

---

## Current Module

### Football

- **Code path:** `sports/nfl`
- **Data source:** [nflreadpy](https://github.com/nflverse/nflreadpy)
- **Models:** moneyline classification, spread residual modeling, total projection
- **App:** Streamlit interface in `apps/web/app.py`

---

## Planned Modules

- `sports/mlb` for future baseball modeling
- `sports/nba` for future basketball modeling

These folders are scaffolded but not yet wired to production predictions.

---

## Project Layout

- `apps/web` contains the Streamlit UI.
- `sports/nfl` contains the current football prediction pipeline.
- `sports/mlb` and `sports/nba` are future sport modules.
- `utils` contains shared, sport-agnostic helpers such as odds math and Kelly staking.
- `sports/*/scripts` contains experiments, backtests, and model comparison commands.

The intent is to reuse the QuantPlay platform, not force one sport's feature math onto
another sport.

---

## Technical Scope

- **Tools & Libraries:** Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit
- **Techniques:** feature engineering, classification, regression, rolling-window modeling, Kelly-style stake sizing
- **Focus:** predictive sports analytics research, not gambling outcomes

---

## Run locally

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run apps/web/app.py
```

Select NFL, season, phase, and week, then **Run predictions**. Preseason,
regular season, and playoffs have separate week choices and saved files.
Preseason week 0 is the Hall of Fame game. **Check results** grades saved
betting sides and stakes using completed scores; it does not refresh the odds
or resize wagers. Rerunning predictions archives the previous snapshot before
replacing it. The picks view highlights graded results; full data is available
in the CSV tab and download.

## Current NFL model

The production moneyline model is `AdaptiveMoneyline` (`pregame-v5`), called
by `sports/nfl/controllers/predict.py`. Its inputs include recent scoring,
prior-season averages shrunk toward the league average, opponent-adjusted
scoring, rest, and neutral venues. It compares prior strengths of 4/8/16 games
and exponential decay half-lives of 4/8/12 games. Preseason team history is
separate from regular-season/playoff history. Player availability and injury
reports are not model inputs. Spread and total models remain separate.

Earlier games select equal or historical implied-probability training weights,
features, and raw/global-sigmoid/favorite-sigmoid calibration. Chronological
out-of-fold periods are split into calibration (first half), selection (next
quarter), and an independent reliability audit (last quarter), all before the
target slate. Candidate selection minimizes Brier loss, not target-week profit.

Favorite calibration uses which team is favored, with complementary favorite
and underdog probabilities. It requires at least 120 priced calibration games,
20 outcomes in each class, and a positive fitted slope. Missing or tied prices
fall back to global calibration or the raw forecast. Exact market probabilities
are not blended into the forecast; they provide a benchmark and identify large
model-versus-market gaps for reliability checks.

The risk layer checks estimates of at least 75% probability or a 15-percentage-point
advantage over the margin-adjusted market. Comparable earlier audit predictions
can correct overstatement downward, shrunk by `n/(n+30)`. Fewer than 30 comparable
games apply a stake multiplier from 0.5 to 1. Ordinary forecasts have no such
correction. These thresholds are policy choices, not guarantees of accuracy.

## Wager sizing

`utils/allocation.py` (`allocation-v8`) evaluates both moneyline sides and selects
the highest expected return that passes the 1-percentage-point edge threshold.
Fractional Kelly uses risk-adjusted model probabilities and offered odds;
quarter Kelly is the default. Insufficient historical validation or missing
moneylines means no funded bet.

The UI has no minimum-stake or per-bet cap controls. Stakes round down to cents,
zero stakes pass, and combined wagers cannot exceed the bankroll. Unused funds
remain unspent. **Normalize to use full bankroll** is optional, unchecked by
default, and warns that stakes may exceed Kelly sizing. Reliability discounts
still apply after normalization. Playoff wagering requires opt-in.

No player availability penalty, blanket Week 1/18 discount, or top-two
concentration rule applies. Historical snapshots keep their saved odds and
stakes until rerun. Python-only sizing options remain for existing research
callers; they do not introduce hidden limits in the UI.

## Data and comparisons

nflverse supplies regular/postseason history; ESPN supplies preseason schedules
and game-specific odds. Prediction runs refresh current pregame prices before
kickoff and explicit closing prices once games start. Missing market lines
produce a spread/total PASS while projections remain available.

Team radar charts use display-only ESPN box-score statistics from the last five
completed games before the selected matchup. Axes are team percentile ranks,
with defensive statistics reversed so farther out is better. Missing statistics
are not zero-filled. Previous-meeting views use up to ten games across fifteen
seasons. These charts do not change model inputs.

## Checks and historical evaluation

```powershell
python -m unittest discover -s tests -v
python -m sports.nfl.scripts.compare_moneyline_training --seasons 2023,2024
python -m sports.nfl.scripts.validate_moneyline --help
```

Tests are local and gitignored. Comparison outputs go under ignored
`outputs/model_validation/`; they do not overwrite app predictions or results.
The existing `backtest_moneyline_weighting.py` and the classic
`Moneyline(use_profit_weighting=True/False)` API remain supported.

The September 2026 comparison used the same sizing rules and a fresh $160 budget
per week, with recorded closing prices. Against historical-weight training with
standard calibration, the automatic model changed 2023 net from -$98.11 to
-$17.77 and 2024 net from -$147.09 to -$73.69. Probability accuracy improved;
neither season was profitable. These seasons had already been inspected, so
this is not a pristine holdout. ESPN UI prices can differ from the historical
cache; neither historical returns nor quoted prices guarantee future results
or executable fills. `RESULTS.md` retains the older research record.

## Disclaimer

This project is for educational and analytical purposes only.
It does not promote, encourage, or provide financial or gambling advice.

If you or someone you know has a gambling problem, please seek help:
**National Problem Gambling Helpline:** 1-800-522-4700
