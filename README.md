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

## Disclaimer

This project is for educational and analytical purposes only.
It does not promote, encourage, or provide financial or gambling advice.

If you or someone you know has a gambling problem, please seek help:
**National Problem Gambling Helpline:** 1-800-522-4700
