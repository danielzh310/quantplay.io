import sys
import math
import json
import logging
import platform
from importlib.metadata import version as package_version
from html import escape
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sports.nfl.controllers.predict import run_weekly
from sports.nfl.models.adaptive_moneyline import VERSION as MONEYLINE_VERSION
from sports.nfl.controllers.results import (
    grade_saved_predictions,
    load_prediction_snapshot,
    prediction_snapshot_path,
    save_prediction_snapshot,
)

from utils.allocation import SizingSettings, allocate_predictions, VERSION as SIZING_VERSION
from sports.nfl.data.matchups import COMPARISON_STATS

st.set_page_config(page_title="QuantPlay", layout="wide")
st.title("quantplay.io")

sport = st.selectbox("Sport", options=["NFL", "NBA", "MLB"], index=0)


def render_placeholder(sport_name):
    st.divider()
    st.subheader(f"{sport_name} moneyline")
    st.info(f"{sport_name} models are coming soon.")


def add_kelly_columns(df, bankroll, fraction_of_kelly, *,
                      normalize_to_full_bankroll=False, betting_enabled=True,
                      selection_mode="either_side"):
    settings = SizingSettings(
        bankroll=float(bankroll),
        fractional_kelly=float(fraction_of_kelly),
        selection_mode=selection_mode,
        normalize_to_full_bankroll=normalize_to_full_bankroll,
        slate_cap=1.0 if betting_enabled else 0.0, minimum_stake=0.0,
    )
    return allocate_predictions(df, settings)



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
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                range=[0, 100], tickvals=[25, 50, 75, 100],
                tickfont=dict(color="#172033", size=14),
                gridcolor="#94a3b8", linecolor="#64748b",
            ),
            angularaxis=dict(
                rotation=90, direction="clockwise",
                tickfont=dict(color="#172033", size=14),
                gridcolor="#94a3b8", linecolor="#64748b",
            ),
        ),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.15,
                    font=dict(color="#172033")),
        font=dict(size=14, color="#172033"), dragmode=False,
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#172033")),
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


def render_bet_summary(df, graded=None):
    st.subheader("What to bet")
    st.write("Saved model picks with a positive stake. Zero-stake games are omitted.")
    bets = []
    for _, row in df.iterrows():
        pick = row.get("bet_side", row.get("ml_pick"))
        if pd.isna(pick):
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
        details = {}
        if not df.empty and "ml_model_details" in df:
            try:
                details = json.loads(df.iloc[0]["ml_model_details"])
            except (ValueError, TypeError):
                pass
        if details.get("status") == "insufficient_temporal_validation":
            st.info("Predictions are available, but there is not enough prior validation data to size bets reliably.")
        else:
            st.info("No funded bets from this run. Check available odds, estimated edges, and bankroll settings.")
        return

    st.write(f"{len(bets)} bets | Total stake: ${sum(bet[2] for bet in bets):,.2f}")
    results = {} if graded is None else {str(row['game_id']): row for _, row in graded.iterrows()}
    cards = []
    for row, side, stake, odds in bets:
        notes = []
        if (row.get(f"ml_{side}_sizing_prob", row.get(f"ml_{side}_prob", 0)) < row.get(f"ml_{side}_prob", 0)
                or row.get(f"ml_{side}_edge_risk_factor", 1.) < 1):
            notes.append("Large estimate adjusted for reliability")
        risk_html = "".join(f'<p class="pick-score">{escape(note)}.</p>' for note in notes)
        result = results.get(str(row['game_id']))
        outcome, result_html = "", ""
        if graded is not None:
            outcome = "pending"
            result_html = '<p class="pick-result">Pending result</p>'
            if result is not None:
                selected = result.get("bet_side", result.get("ml_pick"))
                if pd.isna(selected):
                    selected = result.get("ml_pick")
                # Only annotate the exact saved bet that was graded.
                if (selected == side.upper() and result.get("kelly_stake_ml") == stake
                        and result.get(f"{side}_moneyline") == odds):
                    outcome = "push" if result["ml_result"] == "PUSH" else "win" if selected == result["ml_result"] else "loss"
                    net = float(result["ml_net"])
                    net_label = f"{'-' if net < 0 else '+' if net > 0 else ''}${abs(net):,.2f}"
                    result_html = (
                        f'<p class="pick-result">{outcome.upper()} · Net {net_label}</p>'
                        f'<p class="pick-score">Final: {escape(str(row["away_team"]))} {result["away_score"]:.0f}'
                        f' – {escape(str(row["home_team"]))} {result["home_score"]:.0f}</p>'
                    )
        cards.append(
        f'<article class="pick-card {outcome}">'
        f"<h4>{escape(str(row['away_team']))} at {escape(str(row['home_team']))}</h4>"
        '<p class="pick-label">BET TO WIN</p>'
        f"<p class='pick-team'>{escape(str(row[side + '_team']))}</p>"
        '<dl class="pick-numbers">'
        f'<div><dt>Stake</dt><dd>${stake:,.2f}</dd></div>'
        f'<div><dt>Moneyline odds</dt><dd>{odds:+.0f}</dd></div>'
        f'</dl>{risk_html}{result_html}</article>'
        )
    st.html("""
        <style>
        .pick-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:1rem;}
        .pick-card {border:1px solid color-mix(in srgb,currentColor 25%,transparent);border-radius:12px;padding:1.25rem;}
        .pick-card h4 {margin:0 0 1.25rem;font-size:1.1rem;color:inherit;}
        .pick-card .pick-label {margin:0;font-size:.85rem;letter-spacing:.04em;}
        .pick-card .pick-team {margin:.2rem 0 1.25rem;font-size:1.8rem;font-weight:700;}
        .pick-numbers {display:flex;flex-wrap:wrap;gap:1.5rem;margin:0;}
        .pick-numbers dt {font-size:.9rem;}
        .pick-numbers dd {margin:.25rem 0 0;font-size:1.35rem;font-weight:600;}
        .pick-card.win {background:#ecfdf5;color:#064e3b;border:2px solid #047857;}
        .pick-card.loss {background:#fff1f2;color:#881337;border:2px solid #be123c;}
        .pick-card.push {background:#f1f5f9;color:#1e293b;border:2px solid #64748b;}
        .pick-card .pick-result {margin:1.25rem 0 0;font-weight:700;}
        .pick-card .pick-score {margin:.4rem 0 0;}
        </style>
    """ + f'<div class="pick-grid">{"".join(cards)}</div>')



def bet_results_view(graded):
    """Summarize only settled, funded moneyline picks, including legacy files."""
    side = graded["bet_side"].fillna(graded["ml_pick"]) if "bet_side" in graded else graded["ml_pick"]
    stakes = pd.to_numeric(graded.get("kelly_stake_ml", pd.Series(0.0, index=graded.index)), errors="coerce")
    funded = graded.loc[stakes.gt(0) & stakes.map(lambda value: pd.notna(value) and math.isfinite(value))
                        & side.isin(["HOME", "AWAY"])].copy()
    funded["selected_side"] = side.loc[funded.index]
    rows = []
    counts = {"Won": 0, "Lost": 0, "Push": 0}
    for _, row in funded.iterrows():
        chosen = row["selected_side"].lower()
        outcome = "Push" if row["ml_result"] == "PUSH" else "Won" if row["selected_side"] == row["ml_result"] else "Lost"
        counts[outcome] += 1
        net = float(row["ml_net"])
        rows.append({
            "Game": f"{row['away_team']} at {row['home_team']}",
            "Bet on": row[f"{chosen}_team"], "Stake": f"${float(row['kelly_stake_ml']):.2f}",
            "Saved odds": f"{float(row[f'{chosen}_moneyline']):+.0f}",
            "Final score (away–home)": f"{row['away_score']:.0f}–{row['home_score']:.0f}",
            "Result": outcome, "Net": f"{'-' if net < 0 else ''}${abs(net):.2f}",
        })
    stats = {"bets": len(funded), "wins": counts["Won"], "losses": counts["Lost"], "pushes": counts["Push"],
             "staked": float(funded.kelly_stake_ml.sum()) if "kelly_stake_ml" in funded else 0.0,
             "net": float(funded.ml_net.sum())}
    return pd.DataFrame(rows), stats


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
        "bet_side",
        "reason",
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
        "bet_side",
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

        selected = [c for c in ordered_cols if c in set(selected) and c in df.columns]
        view_df = df[selected].copy()

        for col in ["kelly_stake_ml", "kelly_no_bet_ml", "kelly_f_ml"]:
            if col in df.columns:
                view_df[col] = df[col]

        return view_df

    st.divider()
    kelly_column, display_column = st.columns(2, gap="large")
    with kelly_column:
        st.header("Kelly sizing (Moneyline)")

        moneyline_weighting = "auto"
        selection_mode = "either_side"

        k1, k2 = st.columns(2)
        with k1:
            bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=160.0, step=10.0,
                                       help="Total betting funds. This is the base used to calculate allocation percentages, not a spending target.")
        with k2:
            fraction_of_kelly = st.selectbox("Kelly fraction", options=[0.25, 0.5, 1.0], index=0)
        st.caption("Automatic sizing: Kelly stakes adjusted for reliability.")
        normalize_to_full_bankroll = st.checkbox("Normalize to use full bankroll", value=False)
        if normalize_to_full_bankroll:
            st.warning("Raises stakes beyond Kelly sizing. You could lose your entire bankroll.")
            st.caption("Risk reductions still apply, so some funds may remain unspent.")
        betting_enabled = st.checkbox("Enable playoff stakes", value=False) if season_type == "POST" else True

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
    settings = (bankroll, fraction_of_kelly, normalize_to_full_bankroll, betting_enabled,
                moneyline_weighting, selection_mode)
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
                                verbose=False, season_type=season_type, include_team_stats=True,
                                moneyline_weighting=moneyline_weighting)
            if df is None or df.empty:
                announce("No games found for this selection. Any previous results remain below.")
            else:
                df = add_kelly_columns(df, bankroll, fraction_of_kelly,
                                       normalize_to_full_bankroll=normalize_to_full_bankroll, betting_enabled=betting_enabled,
                                       selection_mode=selection_mode)
                save_prediction_snapshot(df, int(season), int(week), season_type)
                entries[selection] = {"predictions": df, "settings": settings,
                                      "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                announce(f"Saved {len(df)} predictions for {slate_label}. Predictions are below.")
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Prediction failed: season=%s phase=%s week=%s; Python=%s; dependencies=%s",
                season, season_type, week, platform.python_version(),
                {name: package_version(name) for name in ("pandas", "numpy", "scikit-learn", "scipy")},
            )
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
                # Display the saved picks checked by the grader, including games
                # still pending, even if predictions were not loaded first.
                df = load_prediction_snapshot(int(season), int(week), season_type)
                entry = {"predictions": df, "settings": None}
                entry["graded"] = graded
                entry["summary"] = summary
                entries[selection] = entry
                announce("Results ready. Wins and losses are highlighted on your saved bet cards below.")
        except FileNotFoundError:
            st.warning("No saved predictions for this selection. Use Run predictions first.")
            announce("No saved predictions found to grade.")
        except Exception as exc:
            st.error(f"Could not check results. Try again. Details: {exc}")
            announce("Results could not be updated. Any previous results remain below.")

    entry = entries.get(selection, {})
    st.header("Predictions", anchor="nfl-predictions")
    if "predictions" in entry:
        df = entry["predictions"]
        st.caption(slate_label)
        if "ml_model_version" in df and not df["ml_model_version"].eq(MONEYLINE_VERSION).all():
            st.warning("These picks use an older model. Click Run predictions to update them.")
        elif "sizing_version" in df and not df["sizing_version"].eq(SIZING_VERSION).all():
            st.warning("These picks use older wager sizing. Click Run predictions to update them.")
        if entry.get("settings") is not None and entry["settings"] != settings:
            st.warning("Model or sizing settings have changed. Run predictions again to update the saved picks below.")
        picks_tab, csv_tab = st.tabs(["Simple picks", "Full table / CSV"])
        with picks_tab:
            st.html('<div id="nfl-graded-results"></div>')
            if "graded" in entry:
                st.subheader("Bet results")
                bet_table, bet_stats = bet_results_view(entry["graded"])
                record_col, stake_col, net_col = st.columns(3)
                record_col.metric("Wins / Losses / Pushes", f"{bet_stats['wins']} / {bet_stats['losses']} / {bet_stats['pushes']}")
                stake_col.metric("Settled stakes", f"${bet_stats['staked']:,.2f}")
                net = bet_stats["net"]
                net_col.metric("Net profit / loss", f"{'-' if net < 0 else ''}${abs(net):,.2f}")
                st.caption(f"{bet_stats['bets']} settled bets. Results use final scores and your saved stakes and odds.")
                if bet_table.empty:
                    st.info("No funded moneyline bets have settled for this saved run.")
            else:
                st.caption("Use Check results to highlight wins and losses on your picks.")
            render_bet_summary(df, entry.get("graded"))
        with csv_tab:
            view_df = selected_view(df)
            st.dataframe(view_df, use_container_width=True)
            st.download_button("Download full predictions CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name=f"nfl_{'preseason_' if preseason else ''}weekly_picks_{season}_wk{week}.csv",
                               mime="text/csv")
            if "graded" in entry:
                with st.expander("Settled bet details"):
                    if not bet_table.empty:
                        st.table(bet_table.set_index("Game"))
                    st.download_button("Download graded results CSV", data=entry["graded"].to_csv(index=False).encode("utf-8"),
                                       file_name=f"nfl_results_{season}_{season_type}_wk{week}.csv", mime="text/csv")
            st.subheader("Explore each game")
            render_game_summaries(df)

    else:
        st.write("Choose games, then run predictions or load saved predictions to view them here.")

if sport == "NFL":
    render_nfl()
else:
    render_placeholder(sport)
