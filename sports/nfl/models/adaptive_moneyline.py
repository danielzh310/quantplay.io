"""Temporal model selection, calibration and empirical uncertainty for moneylines.

Only completed games supplied by the caller are used. The prediction slate is
never used for fitting, parameter selection, or calibration. This is not a
guarantee of profit or a joint-outcome portfolio optimizer.
"""
import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer

from sports.nfl.models.moneyline import FEATURES, calculate_legacy_weight, calculate_profit_weight
from sports.nfl.models.moneyline_risk import extreme_probability_check
from sports.nfl.data.moneyline_features import market_probability
from sports.nfl.models.calibration import FavoriteCalibrator, OddsAwareCalibrator, apply_calibration, logit, reliability_report, price_gap_report
from sports.nfl.data.efficiency import EPA_FEATURES, SUCCESS_FEATURES

VERSION = "pregame-v6"
PRIOR = [f"ml_{s}_{f}_{kind}" for s in ("home", "away") for f in ("for", "against") for kind in ("prior", "season")]
CONTEXT = ([f"ml_{s}_{f}_adjusted" for s in ("home", "away") for f in ("for", "against")]
           + [f"ml_{s}_{f}" for s in ("home", "away") for f in
              ("rest", "rest_known")]
           + ["ml_neutral_site"])


def candidate_features():
    candidates = {"four_game": FEATURES, "season_prior": PRIOR + ["is_division_game"]}
    for strength in (4, 16):
        candidates[f"season_prior_{strength}"] = [column + f"_k{strength}" for column in PRIOR] + ["is_division_game"]
    for half_life in (4, 8, 12):
        candidates[f"decay_{half_life}"] = PRIOR + [
            f"ml_{s}_{f}_ew{half_life}" for s in ("home", "away") for f in ("for", "against")
        ] + ["is_division_game"]
    candidates["opponent_rest"] = candidates["decay_8"] + CONTEXT
    # Ablations compare simpler scoring signals with the overlapping averages.
    candidates["season_only"] = [f"ml_{s}_{f}_season" for s in ("home", "away") for f in ("for", "against")] + ["is_division_game"]
    candidates["recent_only"] = [f"ml_{s}_{f}_ew8" for s in ("home", "away") for f in ("for", "against")] + ["is_division_game"]
    # Add each efficiency family separately; avoid an unconstrained feature pile.
    candidates["recent_epa"] = candidates["recent_only"] + EPA_FEATURES
    candidates["recent_success"] = candidates["recent_only"] + SUCCESS_FEATURES
    return candidates


def training_window(data, window):
    if window == "all_history" or data.empty:
        return data
    if window != "recent_3_seasons":
        raise ValueError("Unknown training history window")
    return data[data.season.ge(int(data.season.max()) - 2)]


def _fit(data, columns, weighting):
    # Imputation is fitted only on this past training fold, never the slate.
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    weights = None
    if weighting in ("legacy", "profit"):
        function = calculate_legacy_weight if weighting == "legacy" else calculate_profit_weight
        weights = data.apply(function, axis=1).to_numpy()
        # Pandas Copy-on-Write can expose a read-only NumPy view.
        weights = weights / weights.mean()
    model.fit(data[columns], data.home_win.astype(int), logisticregression__sample_weight=weights)
    return model


class AdaptiveMoneyline:
    def __init__(self, weighting="auto", *, favorite_calibration=True,
                 history_windows=("all_history", "recent_3_seasons")):
        if weighting not in ("auto", "legacy", "profit", "none"):
            raise ValueError("Unknown moneyline weighting")
        self.requested_weighting = weighting
        self.weighting = "none" if weighting == "auto" else weighting
        self.favorite_calibration = favorite_calibration
        if not history_windows or any(w not in ("all_history", "recent_3_seasons") for w in history_windows):
            raise ValueError("Unknown training history window")
        self.history_windows = tuple(history_windows)

    def train(self, history):
        data = history.loc[history.home_score.notna() & history.away_score.notna()
                           & history.home_score.ne(history.away_score)].copy()
        data["_period"] = data.season * 100 + data.season_type.map({"PRE": 0, "REG": 30, "POST": 60}) + data.week
        data = data.sort_values(["_period", "game_id"]).reset_index(drop=True)
        if len(data) < 2 or data.home_win.nunique() < 2:
            raise ValueError("Not enough historical outcomes for moneyline training")
        available = {name: cols for name, cols in candidate_features().items() if set(cols).issubset(data.columns)}
        for name in ("recent_epa", "recent_success"):
            if name in available and data[available[name]].notna().all(axis=1).sum() < 64:
                available.pop(name)
        if data.season_type.eq("PRE").all():
            available.pop("four_game", None)  # legacy windows mix phases
        if not available:
            raise ValueError("No complete moneyline feature set")
        periods = np.array_split(data._period.unique(), 4)
        oof = {}
        weightings = ("none", "legacy") if self.requested_weighting == "auto" else (self.requested_weighting,)
        self.weighting = weightings[0]
        for weighting in weightings:
            for window in self.history_windows:
                for name, columns in available.items():
                    predictions = pd.Series(np.nan, index=data.index)
                    for block in periods[1:]:
                        if not len(block):
                            continue
                        train = training_window(data[data._period < block[0]], window)
                        test = data[data._period.isin(block)]
                        if len(train) < 64 or train.home_win.nunique() < 2 or test.empty:
                            continue
                        model = _fit(train, columns, weighting)
                        predictions.loc[test.index] = model.predict_proba(test[columns])[:, 1]
                    oof[(weighting, window, name)] = predictions
        # Every candidate is evaluated on the same earlier games.
        valid = pd.concat(oof.values(), axis=1).notna().all(axis=1)
        validation = data.loc[valid]
        self.calibrator = None
        self.history_window = self.history_windows[0]
        self.columns = available.get("season_prior", available[next(iter(available))])
        self.selected = "season_prior" if "season_prior" in available else next(iter(available))
        self.reliability = pd.DataFrame(columns=["p", "y", "early", "market"])
        self.diagnostics = {"version": VERSION, "requested_weighting": self.requested_weighting,
                            "weighting": self.weighting, "favorite_calibration_enabled": self.favorite_calibration,
                            "training_games": len(data), "candidate_scores": {},
                            "history_windows": list(self.history_windows),
                            "efficiency_candidates_available": [n for n in available if n in ("recent_epa", "recent_success")],
                            "status": "insufficient_temporal_validation", "validation_games": 0}
        if len(validation) >= 120 and validation._period.nunique() >= 3:
            # Three chronological blocks: calibration, model selection, then
            # a separate risk audit which cannot influence model selection.
            blocks = validation._period.unique()
            split = max(1, len(blocks) // 2)
            cutoff = blocks[split]
            audit_cutoff = blocks[min(len(blocks)-1, max(split+1, 3*len(blocks)//4))]
            calibrate = validation[validation._period < cutoff]
            evaluate = validation[(validation._period >= cutoff) & (validation._period < audit_cutoff)]
            audit = validation[validation._period >= audit_cutoff]
            best = None
            fit_market = calibrate.get("ml_market_home", pd.Series(np.nan, index=calibrate.index)).to_numpy()
            eval_market = evaluate.get("ml_market_home", pd.Series(np.nan, index=evaluate.index)).to_numpy()
            for (weighting, window, name), forecast in oof.items():
                columns = available[name]
                raw_fit = forecast.loc[calibrate.index].to_numpy()
                raw_eval = forecast.loc[evaluate.index].to_numpy()
                if not np.isfinite(raw_fit).all() or not np.isfinite(raw_eval).all():
                    continue
                transforms = [("raw", None, raw_eval)]
                global_calibrator = None
                if len(calibrate) >= 60 and calibrate.home_win.nunique() == 2:
                    calibrator = LogisticRegression(C=1.0, max_iter=2000)
                    calibrator.fit(logit(raw_fit), calibrate.home_win.astype(int))
                    if calibrator.coef_[0, 0] > 0:
                        global_calibrator = calibrator
                        transforms.append(("sigmoid", calibrator, apply_calibration(calibrator, raw_eval, eval_market)))
                if self.favorite_calibration:
                    favorite = FavoriteCalibrator(fallback=global_calibrator)
                    if favorite.fit(raw_fit, calibrate.home_win.to_numpy(), fit_market):
                        transforms.append(("favorite_sigmoid", favorite, favorite.predict_home(raw_eval, eval_market)))
                    priced = OddsAwareCalibrator(fallback=global_calibrator)
                    if priced.fit(raw_fit, calibrate.home_win.to_numpy(), fit_market):
                        transforms.append(("odds_aware_sigmoid", priced, priced.predict_home(raw_eval, eval_market)))
                for calibration_name, calibrator, probabilities in transforms:
                    score = float(brier_score_loss(evaluate.home_win, probabilities))
                    key = f"{weighting}/{window}/{name}/{calibration_name}"
                    self.diagnostics["candidate_scores"][key] = score
                    if best is None or score < best[0] - 1e-12:
                        best = (score, weighting, window, name, columns, calibrator, probabilities, calibration_name)
            if best is not None:
                score, self.weighting, self.history_window, self.selected, self.columns, self.calibrator, probabilities, calibration_name = best
                audit_raw = oof[(self.weighting, self.history_window, self.selected)].loc[audit.index].to_numpy()
                audit_market = audit.get("ml_market_home", pd.Series(np.nan, index=audit.index)).to_numpy()
                audit_p = apply_calibration(self.calibrator, audit_raw, audit_market)
                self.reliability = pd.DataFrame({"p": audit_p, "y": audit.home_win.to_numpy(),
                                                "market": audit.get("ml_market_home", pd.Series(np.nan, index=audit.index)).to_numpy(),
                                                "early": audit.get("ml_history_n", pd.Series(0, index=audit.index)).lt(4).to_numpy()})
                self.diagnostics.update(status="temporal_validation", validation_games=len(evaluate),
                                        validation_brier=score, calibration=calibration_name,
                                        calibration_games=len(calibrate), validation_last_period=int(evaluate._period.max()),
                                        risk_audit_games=len(audit), risk_audit_first_period=int(audit._period.min()),
                                        risk_audit_last_period=int(audit._period.max()),
                                        calibration_audit=reliability_report(audit_p, audit.home_win.to_numpy(), audit_market),
                                        price_gap_audit=price_gap_report(audit_p, audit.home_win.to_numpy(), audit_market))
                # A benchmark for comparison. The optional odds-aware calibrator
                # also uses price magnitude, fitted exclusively on earlier games.
                market = evaluate.get("ml_market_home", pd.Series(np.nan, index=evaluate.index)).to_numpy()
                priced = np.isfinite(market)
                if priced.any():
                    self.diagnostics.update(
                        market_benchmark_games=int(priced.sum()),
                        market_benchmark_brier=float(brier_score_loss(evaluate.home_win.to_numpy()[priced], market[priced])),
                        model_benchmark_brier=float(brier_score_loss(evaluate.home_win.to_numpy()[priced], probabilities[priced])))
        final_training = training_window(data, self.history_window)
        self.model = _fit(final_training, self.columns, self.weighting)
        self.diagnostics.update(weighting=self.weighting, selected_features=self.selected,
                                underlying_features=self.selected, training_window=self.history_window,
                                fitted_training_games=len(final_training),
                                fitted_first_season=int(final_training.season.min()))
        return self

    def _lower_bounds(self, p, early):
        # Empirical calibration uncertainty, not a confidence guarantee for an
        # individual game. One-sided 90% Wilson bounds (z=1.28155), with at
        # least 30 prior evaluation games in a bin or a wider fallback pool.
        if self.reliability.empty:
            return 0., 0., 0
        history = self.reliability
        bin_id = min(int(p * 5), 4)
        bins = np.minimum((history.p * 5).astype(int), 4)
        group = history[(bins == bin_id) & history.early.eq(early)]
        if len(group) < 30:
            group = history[bins == bin_id]
        if len(group) < 30:
            group = history
        n = len(group)
        z = 1.2815515655446004
        bounds = []
        for probability, outcomes, predictions in ((p, group.y, group.p), (1-p, 1-group.y, 1-group.p)):
            rate = float(outcomes.mean())
            denominator = 1 + z*z/n
            lower = (rate + z*z/(2*n) - z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))) / denominator
            overstatement = max(0., float(predictions.mean()) - rate)
            bounds.append(max(0., probability - overstatement - (rate - lower)))
        return *bounds, n

    def predict(self, games):
        raw = self.model.predict_proba(games[self.columns])[:, 1]
        markets = np.array([market_probability(h, a) for h, a in zip(games.home_moneyline, games.away_moneyline)])
        p = apply_calibration(self.calibrator, raw, markets)
        bounds = [self._lower_bounds(value, count < 4)
                  for value, count in zip(p, games.get("ml_history_n", [0]*len(games)))]
        output = pd.DataFrame({
            "game_id": games.game_id.to_numpy(), "home_win_prob": p, "away_win_prob": 1-p,
            "moneyline_pick": np.where(p > .5, "HOME", "AWAY"),
            "home_ml_bet": "No Bet", "away_ml_bet": "No Bet",  # allocator owns actionable picks
            "home_moneyline": games.home_moneyline.to_numpy(), "away_moneyline": games.away_moneyline.to_numpy(),
            "ml_home_prob_low": [b[0] for b in bounds], "ml_away_prob_low": [b[1] for b in bounds],
            "ml_reliability_n": [b[2] for b in bounds], "ml_raw_home_prob": raw,
            "ml_model_version": VERSION, "ml_model_details": json.dumps(self.diagnostics, sort_keys=True),
        })
        risks = []
        for probability, (_, game) in zip(p, games.iterrows()):
            market = market_probability(game.home_moneyline, game.away_moneyline)
            record = {}
            for side, estimate, price in (("home", probability, market), ("away", 1-probability, 1-market)):
                adjusted, factor, n, reason = extreme_probability_check(estimate, price, self.reliability)
                record.update({f"ml_{side}_sizing_prob": adjusted, f"ml_{side}_edge_risk_factor": factor,
                               f"ml_{side}_extreme_n": n, f"ml_{side}_extreme_reason": reason})
            risks.append(record)
        return pd.concat([output, pd.DataFrame(risks)], axis=1)
