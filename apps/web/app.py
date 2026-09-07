import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sports.nfl.controllers.predict import run_weekly
from sports.nfl.controllers.results import (
    grade_saved_predictions,
    prediction_snapshot_path,
    save_prediction_snapshot,
)

from utils.kelly import allocate_kelly

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


def render_nfl():
    phase_label = st.selectbox("Season phase", ["Regular season", "Preseason", "Playoffs"])
    season_type = {"Preseason": "PRE", "Regular season": "REG", "Playoffs": "POST"}[phase_label]
    preseason = season_type == "PRE"
    col1, col2 = st.columns([1, 1])
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

    st.divider()

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

    st.subheader("Columns to display")

    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    with b1:
        st.button("Select defaults", on_click=set_defaults)
    with b2:
        st.button("Select all", on_click=select_all)
    with b3:
        st.button("Clear all", on_click=clear_all)
    with b4:
        st.caption("Defaults = game/team + ML bet + moneylines")

    with st.expander("Filter options", expanded=False):
        st.session_state["nfl_selected_cols"] = st.multiselect(
            "Columns",
            options=ordered_cols,
            default=st.session_state["nfl_selected_cols"],
        )

    st.divider()
    st.subheader("Kelly sizing (Moneyline)")

    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=160.0, step=10.0)
    with k2:
        fraction_of_kelly = st.selectbox("Kelly fraction", options=[0.25, 0.5, 1.0], index=0)
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

    st.divider()

    run_col, results_col = st.columns([1, 1])
    with run_col:
        run = st.button("Run predictions", type="primary", use_container_width=True)
    with results_col:
        check_results = st.button("Check results", use_container_width=True)

    if run:
        if week is None:
            st.warning("Please select a week before running predictions.")
        else:
            try:
                with st.spinner("Running pipeline..."):
                    df = run_weekly(
                        season=int(season),
                        week=int(week),
                        export=False,
                        verbose=False,
                        season_type=season_type,
                    )
            except Exception as exc:
                st.error(f"Could not run predictions: {exc}")
                return

            if df is None or len(df) == 0:
                st.warning("No games found for that season/week.")
            else:
                try:
                    df = add_kelly_columns(
                        df,
                        bankroll=bankroll,
                        fraction_of_kelly=fraction_of_kelly,
                        normalize_to_full_bankroll=normalize_to_full_bankroll,
                        cap_fraction_of_bankroll=cap_fraction_of_bankroll,
                    )
                    st.caption("Added columns: kelly_stake_ml, kelly_no_bet_ml, kelly_f_ml")
                except Exception as e:
                    st.warning(f"Kelly sizing failed: {e}")

                snapshot = save_prediction_snapshot(df, int(season), int(week), season_type)
                view_df = selected_view(df)

                st.success(f"Generated and saved {len(df)} picks to {snapshot}.")
                st.dataframe(view_df, use_container_width=True)

                csv_bytes = view_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download displayed CSV",
                    data=csv_bytes,
                    file_name=f"nfl_{'preseason_' if preseason else ''}weekly_picks_{int(season)}_wk{int(week)}.csv",
                    mime="text/csv",
                )

    if check_results:
        if week is None:
            st.warning("Please select a week before checking results.")
        else:
            snapshot = prediction_snapshot_path(int(season), int(week), season_type)
            if not snapshot.exists():
                st.warning(f"No saved predictions found yet for {int(season)} week {int(week)}.")
            else:
                try:
                    with st.spinner("Checking saved picks against completed games..."):
                        graded, summary = grade_saved_predictions(int(season), int(week), season_type)
                except Exception as exc:
                    st.error(f"Could not check results: {exc}")
                    return

                if graded is None:
                    st.warning(summary["message"])
                else:
                    st.success(summary["message"])
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Completed games", summary["games_completed"])
                    m2.metric("ML hits", f'{summary["ml_hits"]}/{summary["games_completed"]}')
                    m3.metric("ML net", f'${summary["ml_net"]:.2f}')

                    result_cols = [
                        "game_id",
                        "home_team",
                        "away_team",
                        "ml_pick",
                        "ml_result",
                        "ml_hit",
                        "kelly_stake_ml",
                        "ml_net",
                        "home_score",
                        "away_score",
                    ]
                    result_cols = [c for c in result_cols if c in graded.columns]
                    st.dataframe(graded[result_cols], use_container_width=True)
                    st.caption(f'Saved graded results to {summary["results_path"]}')


if sport == "NFL":
    render_nfl()
else:
    render_placeholder(sport)
