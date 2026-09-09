import sys
import math
import json
from html import escape
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sports.nfl.controllers.predict import run_weekly
from sports.nfl.controllers.results import (
    grade_saved_predictions,
    load_prediction_snapshot,
    prediction_snapshot_path,
    save_prediction_snapshot,
)

from utils.kelly import allocate_kelly
from sports.nfl.data.matchups import COMPARISON_STATS

st.set_page_config(page_title="QuantPlay", layout="wide")
st.title("quantplay.io")

sport = st.selectbox("Sport", options=["NFL", "NBA", "MLB"], index=0)


def render_placeholder(sport_name):
    st.divider()
    st.subheader(f"{sport_name} moneyline")
    st.info(f"{sport_name} models are coming soon.")


def add_kelly_columns(
    df,
    bankroll,
    fraction_of_kelly,
    normalize_to_full_bankroll,
    cap_fraction_of_bankroll,
):
    needed = ["ml_pick", "ml_home_prob", "ml_away_prob", "home_moneyline", "away_moneyline"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Kelly sizing skipped, missing columns: {missing}")

    probs = []
    odds = []
    for _, r in df.iterrows():
        pick = str(r["ml_pick"]).upper()
        if pick == "HOME":
            probs.append(float(r["ml_home_prob"]))
            odds.append(float(r["home_moneyline"]))
        elif pick == "AWAY":
            probs.append(float(r["ml_away_prob"]))
            odds.append(float(r["away_moneyline"]))
        else:
            probs.append(0.0)
            odds.append(100.0)

        if not math.isfinite(odds[-1]) or odds[-1] == 0:
            probs[-1] = 0.0
            odds[-1] = 100.0

    stakes, meta = allocate_kelly(
        probs=probs,
        odds=odds,
        bankroll=float(bankroll),
        fraction_of_kelly=float(fraction_of_kelly),
        normalize_to_full_bankroll=bool(normalize_to_full_bankroll),
        cap_fraction_of_bankroll=cap_fraction_of_bankroll,
        min_stake=0.0,
    )

    out = df.copy()
    out["kelly_stake_ml"] = [round(s, 2) for s in stakes]
    out["kelly_no_bet_ml"] = [m.is_no_bet for m in meta]
    out["kelly_f_ml"] = [round(float(m.kelly_fraction), 4) for m in meta]
    return out



FIELD_LABELS = {
    "game_id": "Game ID", "home_team": "Home team", "away_team": "Away team",
    "ml_pick": "Predicted winner", "ml_home_prob": "Home win probability",
    "ml_away_prob": "Away win probability", "home_ml_bet": "Home moneyline recommendation",
    "away_ml_bet": "Away moneyline recommendation", "home_moneyline": "Home moneyline odds",
    "away_moneyline": "Away moneyline odds", "spread_line": "Market spread (home expected margin)",
    "spread_pick": "Spread pick", "model_spread_margin": "Projected home winning margin",
    "spread_edge": "Spread edge (points)", "total_line": "Market total (points)",
    "total_pick": "Total pick", "projected_total": "Projected total (points)",
    "vegas_total_edge": "Total edge (points)", "total_low_q": "Lower total estimate (20th percentile)",
    "total_high_q": "Upper total estimate (80th percentile)", "total_buffer": "Total decision buffer (points)",
    "kelly_stake_ml": "Moneyline stake", "kelly_no_bet_ml": "No moneyline stake",
    "kelly_f_ml": "Full Kelly fraction", "ml_result": "Actual winner",
    "ml_hit": "Correct winner pick", "ml_net": "Moneyline net profit or loss",
    "home_score": "Home final score", "away_score": "Away final score",
    "actual_margin": "Actual home winning margin",
}


def readable_value(column, value, row):
    if pd.isna(value):
        return "Unavailable"
    if column in {"ml_pick", "spread_pick", "ml_result"}:
        if value in {"HOME", "AWAY"}:
            side = value.lower()
            return f"{row.get(side + '_team', side)} ({side})"
        if value == "PUSH":
            return "Tie (stake returned)"
    if value == "PASS":
        return "Pass (no bet)"
    if column in {"ml_home_prob", "ml_away_prob", "kelly_f_ml"}:
        return f"{float(value):.0%}"
    if column in {"kelly_stake_ml", "ml_net"}:
        return f"${float(value):,.2f}"
    if column in {"home_moneyline", "away_moneyline"}:
        return f"{float(value):+.0f}"
    if column in {"kelly_no_bet_ml", "ml_hit"}:
        return "Yes" if value else "No"
    return str(value)


def build_team_radar(row):
    # Only compare axes with real observations for both teams.
    axes = [
        (key, label, higher) for key, label, higher in COMPARISON_STATS
        if all(pd.notna(pd.to_numeric(row.get(f"{side}_recent_{key}_rating"), errors="coerce"))
               for side in ("home", "away"))
    ]
    if len(axes) < 3:
        return None, None
    figure = go.Figure()
    labels = [label for _, label, _ in axes]
    exact_rows = []
    for key, label, higher in axes:
        item = {"Statistic (per game)": label + (" (lower is better)" if not higher else "")}
        for side in ['home', 'away']:
            raw = row[f'{side}_recent_{key}']
            count = int(row[f'{side}_recent_{key}_n'])
            item[f"{row.get(side + '_team', side)} ({side})"] = f"{raw:.1f}"
        exact_rows.append(item)
    for side, color, fill, dash in [
        ("home", "#8b7cf8", "rgba(139,124,248,0.18)", "solid"),
        ("away", "#20aaa6", "rgba(32,170,166,0.18)", "dash"),
    ]:
        team = str(row.get(f"{side}_team", side.title()))
        ratings = [float(row[f'{side}_recent_{key}_rating']) for key, _, _ in axes]
        raw = [[float(row[f'{side}_recent_{key}']), int(row[f'{side}_recent_{key}_n']),
                int(row[f'{side}_recent_{key}_pool'])] for key, _, _ in axes]
        figure.add_trace(go.Scatterpolar(
            r=[*ratings, ratings[0]], theta=[*labels, labels[0]], customdata=[*raw, raw[0]],
            name=f"{team} ({side})", mode="lines+markers", fill="toself", fillcolor=fill,
            line=dict(color=color, width=3, dash=dash), marker=dict(size=7),
            hovertemplate="%{theta}<br>Average: %{customdata[0]:.1f} per game<br>Sample: %{customdata[1]} games<br>Percentile: %{r:.0f} (among %{customdata[2]} teams)<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=490, margin=dict(l=90, r=90, t=55, b=65),
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100]),
                   angularaxis=dict(rotation=90, direction="clockwise")),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.15),
        font=dict(size=13), dragmode=False,
    )
    return figure, pd.DataFrame(exact_rows).set_index("Statistic (per game)")


def summarize_meetings(row, meetings):
    games = sorted(meetings, key=lambda g: (g['season'], g['week'], g['game_id']), reverse=True)
    home = str(row['home_team'])
    home_scores = [float(g['home_score'] if g['home_team'] == home else g['away_score']) for g in games]
    away_scores = [float(g['away_score'] if g['home_team'] == home else g['home_score']) for g in games]
    margins = [h - a for h, a in zip(home_scores, away_scores)]
    totals = [h + a for h, a in zip(home_scores, away_scores)]
    n = len(games)
    return {
        'games': games, 'count': n,
        'home_wins': sum(m > 0 for m in margins), 'away_wins': sum(m < 0 for m in margins),
        'ties': sum(m == 0 for m in margins),
        'home_average': sum(home_scores) / n, 'away_average': sum(away_scores) / n,
        'average_margin': sum(margins) / n,
        'average_total': sum(totals) / n, 'median_total': float(pd.Series(totals).median()),
        'low_total': min(totals), 'high_total': max(totals),
        'one_score': sum(0 < abs(m) <= 8 for m in margins),
        'average_gap': sum(abs(m) for m in margins) / n,
    }


def render_head_to_head(row):
    try:
        meetings = json.loads(row.get('matchup_history', '[]'))
    except (ValueError, TypeError):
        meetings = []
    st.subheader("Matchup history at a glance")
    if not meetings:
        st.info("No previous meetings found in the loaded history before this game.")
        return
    stats = summarize_meetings(row, meetings)
    home, away = row['home_team'], row['away_team']
    years = [game['season'] for game in meetings]
    st.caption(f"Based on {stats['count']} prior meetings · {min(years)}–{max(years)} seasons. These describe past games, not a forecast.")
    if stats['count'] == 1:
        st.info("Only one prior meeting is available. These numbers describe that game, not an established trend.")

    record, margin = st.columns(2)
    with record:
        with st.container(border=True):
            st.subheader("Who has won?")
            st.write(f"**{home}: {stats['home_wins']} wins | {away}: {stats['away_wins']} wins**")
            st.write(f"Ties: {stats['ties']}")
            st.write(f"Average points scored: **{home} {stats['home_average']:.1f} | {away} {stats['away_average']:.1f}**")
    with margin:
        with st.container(border=True):
            st.subheader("Scoring advantage")
            diff = stats['average_margin']
            if abs(diff) < .05:
                st.write("**Even average scoring**")
            else:
                st.write(f"**{home if diff > 0 else away} +{abs(diff):.1f} points per meeting**")
            st.write("Average points scored minus points allowed against this opponent.")
            st.caption("This is a historical scoring margin, not a spread recommendation.")

    scoring, close = st.columns(2)
    with scoring:
        with st.container(border=True):
            st.subheader("Combined scoring")
            st.write(f"**{stats['average_total']:.1f} points per game**")
            st.write(f"Median: **{stats['median_total']:.1f}** | Range: **{stats['low_total']:.0f}-{stats['high_total']:.0f}**")
            st.caption("Both teams combined. Historical over/under results require each game's original line.")
    with close:
        with st.container(border=True):
            st.subheader("How competitive?")
            st.write(f"**{stats['one_score']} of {stats['count']} decided by one score**")
            st.write(f"Average final-score gap: **{stats['average_gap']:.1f} points**")
            st.caption("One score means a winning margin of 1-8 points. Ties are listed separately above.")

    with st.expander('See the games behind these stats'):
        for game in stats['games']:
            st.write(f"{game['season']} week {game['week']}: {game['away_team']} {game['away_score']:.0f} at {game['home_team']} {game['home_score']:.0f}")
    st.caption('Up to 10 prior meetings, looking back 15 seasons. Older games may involve different rosters and coaches. Preseason history stays separate from regular/playoff history.')


def render_game_summaries(df):
    for position, (_, row) in enumerate(df.iterrows()):
        matchup = f"{row.get('away_team', 'Away')} at {row.get('home_team', 'Home')}"
        with st.expander(f"{matchup} | Team comparison", expanded=position == 0):
            if row.get('comparison_version') != 2:
                st.info("Run predictions again to load offense, defense, and previous-meeting statistics for this saved run.")
                continue
            figure, exact = build_team_radar(row)
            st.subheader("Offense & defense matchup")
            if figure is None:
                st.info("Not enough box-score data for a complete team radar. Unavailable statistics are not replaced with zeros.")
            else:
                st.plotly_chart(figure, use_container_width=True,
                                config={"displayModeBar": False, "scrollZoom": False})
                st.caption("Last five completed games before this matchup, carrying across seasons. Each axis shows a percentile among teams with data in the loaded history. Farther out is better, including fewer yards/points allowed and fewer giveaways. These are comparisons, not win probabilities.")
                if len(exact) < len(COMPARISON_STATS):
                    st.info("Some axes are omitted because box-score data is missing for one or both teams.")
                with st.expander("View actual averages"):
                    st.table(exact)
            render_head_to_head(row)


def render_bet_summary(df):
    st.subheader("What to bet")
    st.write("Saved model picks with a positive stake. Zero-stake games are omitted.")
    bets = []
    for _, row in df.iterrows():
        pick = row.get("ml_pick")
        if pick not in {"HOME", "AWAY"}:
            continue
        side = pick.lower()
        stake = pd.to_numeric(row.get("kelly_stake_ml"), errors="coerce")
        odds = pd.to_numeric(row.get(f"{side}_moneyline"), errors="coerce")
        if pd.isna(stake) or pd.isna(odds) or not math.isfinite(stake) or not math.isfinite(odds):
            continue
        if stake <= 0 or odds == 0:
            continue
        bets.append((row, side, float(stake), float(odds)))

    if not bets:
        st.info("No bets to place from this saved run. All stakes are zero or the required odds are unavailable.")
        return

    st.write(f"{len(bets)} bets | Total stake: ${sum(bet[2] for bet in bets):,.2f}")
    cards = "".join(
        '<article class="pick-card">'
        f"<h4>{escape(str(row['away_team']))} at {escape(str(row['home_team']))}</h4>"
        '<p class="pick-label">TEAM TO WIN</p>'
        f"<p class='pick-team'>{escape(str(row[side + '_team']))}</p>"
        '<dl class="pick-numbers">'
        f'<div><dt>Stake</dt><dd>${stake:,.2f}</dd></div>'
        f'<div><dt>Moneyline odds</dt><dd>{odds:+.0f}</dd></div>'
        '</dl></article>'
        for row, side, stake, odds in bets
    )
    st.html("""
        <style>
        .pick-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:1rem;}
        .pick-card {border:1px solid color-mix(in srgb,currentColor 25%,transparent);border-radius:12px;padding:1.25rem;}
        .pick-card h4 {margin:0 0 1.25rem;font-size:1.1rem;}
        .pick-card .pick-label {margin:0;font-size:.85rem;letter-spacing:.04em;}
        .pick-card .pick-team {margin:.2rem 0 1.25rem;font-size:1.8rem;font-weight:700;}
        .pick-numbers {display:flex;flex-wrap:wrap;gap:1.5rem;margin:0;}
        .pick-numbers dt {font-size:.9rem;}
        .pick-numbers dd {margin:.25rem 0 0;font-size:1.35rem;font-weight:600;}
        </style>
    """ + f'<div class="pick-grid">{cards}</div>')



def render_nfl():
    st.markdown("[Skip to predictions](#nfl-predictions) | [Skip to graded results](#nfl-graded-results)")
    st.header("Choose games")
    phase_col, col1, col2 = st.columns(3)
    with phase_col:
        phase_label = st.selectbox("Season phase", ["Regular season", "Preseason", "Playoffs"])
    season_type = {"Preseason": "PRE", "Regular season": "REG", "Playoffs": "POST"}[phase_label]
    preseason = season_type == "PRE"
    with col1:
        seasons = list(range(2020, 2031))
        season = st.selectbox("Season", options=seasons, index=seasons.index(2026))
    with col2:
        if preseason:
            week_labels = ["Select week...", "Hall of Fame", "1", "2", "3"]
            if season < 2021:
                week_labels.append("4")
        elif season_type == "POST":
            week_labels = ["Select week...", "Wildcard", "Divisional", "Conference Championship", "Super Bowl"]
        else:
            week_labels = ["Select week..."] + [str(i) for i in range(1, 18 if season < 2021 else 19)]
        selected_week_label = st.selectbox("Week", options=week_labels, index=0, key=f"nfl_week_{season_type}")

        label_to_week = {str(i): i for i in range(1, 19)}
        label_to_week.update({
            "Hall of Fame": 0,
            "Wildcard": 19,
            "Divisional": 20,
            "Conference Championship": 21,
            "Super Bowl": 22,
        })

        week = label_to_week.get(selected_week_label, None)

    if preseason:
        st.caption("Preseason includes the Hall of Fame game. Missing odds produce no bet; missing market lines produce PASS while model projections remain available.")
    st.caption("Each run refreshes pregame odds before kickoff and uses closing odds once games start. Check results uses the odds saved with your picks.")


    ordered_cols = [
        "game_id",
        "home_team",
        "away_team",
        "ml_pick",
        "ml_home_prob",
        "ml_away_prob",
        "home_ml_bet",
        "away_ml_bet",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "spread_pick",
        "model_spread_margin",
        "spread_edge",
        "total_line",
        "total_pick",
        "projected_total",
        "vegas_total_edge",
        "total_low_q",
        "total_high_q",
        "total_buffer",
    ]

    default_on = [
        "game_id",
        "home_team",
        "away_team",
        "home_ml_bet",
        "away_ml_bet",
        "home_moneyline",
        "away_moneyline",
    ]

    if "nfl_selected_cols" not in st.session_state:
        st.session_state["nfl_selected_cols"] = default_on.copy()

    def set_defaults():
        st.session_state["nfl_selected_cols"] = default_on.copy()

    def select_all():
        st.session_state["nfl_selected_cols"] = ordered_cols.copy()

    def clear_all():
        st.session_state["nfl_selected_cols"] = []

    def selected_view(df):
        selected = st.session_state.get("nfl_selected_cols", default_on)
        if not selected:
            st.warning("No columns selected, showing defaults.")
            selected = default_on.copy()

        selected = [c for c in ordered_cols if c in set(selected)]
        view_df = df[selected].copy()

        for col in ["kelly_stake_ml", "kelly_no_bet_ml", "kelly_f_ml"]:
            if col in df.columns:
                view_df[col] = df[col]

        return view_df

    st.divider()
    kelly_column, display_column = st.columns(2, gap="large")
    with kelly_column:
        st.header("Kelly sizing (Moneyline)")

        k1, k2 = st.columns(2)
        with k1:
            bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=160.0, step=10.0)
        with k2:
            fraction_of_kelly = st.selectbox("Kelly fraction", options=[0.25, 0.5, 1.0], index=0)
        k3, k4 = st.columns(2)
        with k3:
            normalize_to_full_bankroll = st.checkbox("Normalize to use full bankroll", value=False)
        with k4:
            cap_pct = st.number_input(
                "Max bet cap (% of bankroll)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )

        cap_fraction_of_bankroll = None
        if cap_pct > 0:
            cap_fraction_of_bankroll = float(cap_pct) / 100.0

        if normalize_to_full_bankroll:
            st.warning("Using your full bankroll can undo fractional Kelly sizing and exceed your max bet cap.")

        st.caption(
            "Kelly uses model probability and sportsbook odds to size stakes. "
            "Only positive-edge picks get a stake. If normalization is on, stakes are scaled to sum to the bankroll."
        )

    with display_column:
        st.header("Columns to display")

        b1, b2, b3 = st.columns(3)
        with b1:
            st.button("Select defaults", on_click=set_defaults)
        with b2:
            st.button("Select all", on_click=select_all)
        with b3:
            st.button("Clear all", on_click=clear_all)
        st.caption("Defaults = game/team + ML bet + moneylines")

        with st.expander("Filter options", expanded=False):
            st.session_state["nfl_selected_cols"] = st.multiselect(
                "Columns",
                options=ordered_cols,
                default=st.session_state["nfl_selected_cols"],
            )

    st.divider()

    selection = (int(season), season_type, week)
    entries = st.session_state.setdefault("nfl_saved_views", {})
    settings = (bankroll, fraction_of_kelly, normalize_to_full_bankroll, cap_fraction_of_bankroll)
    slate_label = f"{season} {phase_label} - {selected_week_label}"
    if week is None:
        st.info("Select a week or playoff round to run predictions or check results.")

    run_col, load_col, results_col = st.columns(3)
    with run_col:
        run = st.button("Run predictions", type="primary", use_container_width=True)
    with load_col:
        load_saved = st.button("Load saved predictions", use_container_width=True)
    with results_col:
        check_results = st.button("Check results", use_container_width=True)

    # Keep a live region at a stable location so updates can be announced without
    # moving keyboard focus away from the user's action.
    status = st.empty()

    def announce(message):
        status.html(f'<div role="status" aria-live="polite" aria-atomic="true">{escape(message)}</div>')

    announce("")
    if (run or load_saved or check_results) and week is None:
        st.warning("Select a week or playoff round in Choose games, then try again.")
        announce("A week or playoff round is required.")
    elif run:
        announce(f"Loading odds and generating predictions for {slate_label}. Please wait.")
        try:
            with st.spinner("Loading odds and generating predictions..."):
                df = run_weekly(season=int(season), week=int(week), export=False,
                                verbose=False, season_type=season_type, include_team_stats=True)
            if df is None or df.empty:
                announce("No games found for this selection. Any previous results remain below.")
            else:
                df = add_kelly_columns(df, bankroll, fraction_of_kelly,
                                       normalize_to_full_bankroll, cap_fraction_of_bankroll)
                save_prediction_snapshot(df, int(season), int(week), season_type)
                entries[selection] = {"predictions": df, "settings": settings,
                                      "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                announce(f"Saved {len(df)} predictions for {slate_label}. Predictions are below.")
        except Exception as exc:
            st.error(f"Could not generate and save predictions. Try again. Details: {exc}")
            announce("Predictions could not be updated. Any previous results remain below.")
    elif load_saved:
        try:
            df = load_prediction_snapshot(int(season), int(week), season_type)
            saved_at = datetime.fromtimestamp(prediction_snapshot_path(int(season), int(week), season_type).stat().st_mtime)
            entries[selection] = {"predictions": df, "settings": None,
                                  "saved_at": saved_at.strftime("%Y-%m-%d %H:%M:%S")}
            announce(f"Loaded {len(df)} saved predictions for {slate_label}. Predictions are below.")
        except FileNotFoundError:
            st.warning("No saved predictions for this selection. Use Run predictions first.")
            announce("No saved predictions found.")
        except Exception as exc:
            st.error(f"Could not load saved predictions. Try again. Details: {exc}")
    elif check_results:
        announce(f"Checking completed games for {slate_label}. Please wait.")
        try:
            with st.spinner("Checking saved picks against completed games..."):
                graded, summary = grade_saved_predictions(int(season), int(week), season_type)
            if graded is None:
                announce(summary["message"])
            else:
                entry = entries.setdefault(selection, {})
                entry["graded"] = graded
                entry["summary"] = summary
                announce(f"Results ready for {summary['games_completed']} completed games. Graded results are below.")
        except FileNotFoundError:
            st.warning("No saved predictions for this selection. Use Run predictions first.")
            announce("No saved predictions found to grade.")
        except Exception as exc:
            st.error(f"Could not check results. Try again. Details: {exc}")
            announce("Results could not be updated. Any previous results remain below.")

    entry = entries.get(selection, {})
    display = "Data table"
    st.header("Predictions", anchor="nfl-predictions")
    if "predictions" in entry:
        df = entry["predictions"]
        st.write(f"{slate_label}. Saved {entry['saved_at']} (server time).")
        if entry.get("settings") is None:
            st.info("Saved stakes reflect the settings used when these picks were generated.")
        elif entry["settings"] != settings:
            st.warning("Sizing settings have changed. Run predictions again to update the saved stakes below.")
        picks_tab, csv_tab = st.tabs(["Simple picks", "Full table / CSV"])
        with picks_tab:
            render_bet_summary(df)
        with csv_tab:
            view_df = selected_view(df)
            st.dataframe(view_df, use_container_width=True)
            st.download_button("Download full predictions CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name=f"nfl_{'preseason_' if preseason else ''}weekly_picks_{season}_wk{week}.csv",
                               mime="text/csv")
            st.subheader("Explore each game")
            render_game_summaries(df)

    else:
        st.write("Choose games, then run predictions or load saved predictions to view them here.")

    st.header("Graded results", anchor="nfl-graded-results")
    if "graded" in entry:
        graded, summary = entry["graded"], entry["summary"]
        st.write(f"{slate_label}. {summary['games_completed']} completed games; "
                 f"{summary['ml_hits']} correct winner picks; net ${summary['ml_net']:.2f}.")
        if display == "Game summaries":
            render_game_summaries(graded)
        else:
            st.dataframe(graded, use_container_width=True)
        st.download_button("Download graded results CSV", data=graded.to_csv(index=False).encode("utf-8"),
                           file_name=f"nfl_results_{season}_{season_type}_wk{week}.csv", mime="text/csv")
    else:
        st.write("Use Check results to compare saved picks with completed games.")


if sport == "NFL":
    render_nfl()
else:
    render_placeholder(sport)
