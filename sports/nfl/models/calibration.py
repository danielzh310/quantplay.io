"""Coherent favorite/underdog calibration and held-out reliability reports."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def logit(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), .00001, .99999)
    return np.log(p / (1 - p)).reshape(-1, 1)


def orientation(market):
    market = np.asarray(market, dtype=float)
    valid = np.isfinite(market) & (market > 0) & (market < 1)
    return np.where(valid & (market > .5), 1, np.where(valid & (market < .5), -1, 0))


def apply_calibration(calibrator, probabilities, market):
    if calibrator is None:
        return np.asarray(probabilities, dtype=float)
    if isinstance(calibrator, FavoriteCalibrator):
        return calibrator.predict_home(probabilities, market)
    return calibrator.predict_proba(logit(probabilities))[:, 1]


class FavoriteCalibrator:
    """Fit one favorite curve; underdog probability is its complement.

    Uses only which side is favored, not the market probability's magnitude.
    Unpriced/tied markets retain the global calibrator (or raw forecast).
    """
    def __init__(self, fallback=None):
        self.fallback = fallback

    def fit(self, probabilities, outcomes, market):
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=int)
        direction = orientation(market)
        valid = direction != 0
        favorite_p = np.where(direction > 0, p, 1-p)[valid]
        favorite_y = np.where(direction > 0, y, 1-y)[valid]
        # Fixed support safeguards, never tuned on the prediction slate.
        if len(favorite_y) < 120 or min(np.bincount(favorite_y, minlength=2)) < 20:
            return False
        self.model = LogisticRegression(C=1., max_iter=2000)
        self.model.fit(logit(favorite_p), favorite_y)
        return bool(self.model.coef_[0, 0] > 0)

    def predict_home(self, probabilities, market):
        p = np.asarray(probabilities, dtype=float)
        out = apply_calibration(self.fallback, p, market).copy()
        direction = orientation(market)
        valid = direction != 0
        if valid.any():
            favorite_p = np.where(direction > 0, p, 1-p)[valid]
            calibrated = self.model.predict_proba(logit(favorite_p))[:, 1]
            out[valid] = np.where(direction[valid] > 0, calibrated, 1-calibrated)
        return out


class OddsAwareCalibrator(FavoriteCalibrator):
    """Regularized favorite curve using model confidence and the market gap.

    Unlike FavoriteCalibrator, this explicitly uses the magnitude of the
    margin-adjusted market probability. Both sides still sum to one. Its fitted
    adjustment is a candidate, never a forced market replacement or bet veto.
    """
    @staticmethod
    def _matrix(probabilities, market):
        return np.column_stack([logit(probabilities).ravel(), logit(market).ravel()])

    def fit(self, probabilities, outcomes, market):
        p, y, m = np.asarray(probabilities), np.asarray(outcomes, dtype=int), np.asarray(market)
        direction = orientation(m)
        valid = direction != 0
        favorite_p = np.where(direction > 0, p, 1-p)[valid]
        favorite_m = np.where(direction > 0, m, 1-m)[valid]
        favorite_y = np.where(direction > 0, y, 1-y)[valid]
        if len(favorite_y) < 240 or min(np.bincount(favorite_y, minlength=2)) < 40:
            return False
        self.model = LogisticRegression(C=.1, max_iter=2000)
        self.model.fit(self._matrix(favorite_p, favorite_m), favorite_y)
        # Reject a reversed confidence relationship rather than extrapolating it.
        return bool(self.model.coef_[0, 0] > 0 and self.model.coef_[0, 1] >= 0)

    def predict_home(self, probabilities, market):
        p, m = np.asarray(probabilities), np.asarray(market)
        out = apply_calibration(self.fallback, p, m).copy()
        direction = orientation(m)
        valid = direction != 0
        if valid.any():
            favorite_p = np.where(direction > 0, p, 1-p)[valid]
            favorite_m = np.where(direction > 0, m, 1-m)[valid]
            calibrated = self.model.predict_proba(self._matrix(favorite_p, favorite_m))[:, 1]
            out[valid] = np.where(direction[valid] > 0, calibrated, 1-calibrated)
        return out


def price_gap_report(probabilities, outcomes, market):
    """Audit ordinary as well as extreme estimates by market range and model gap.

    This is descriptive only: never fit or select using this final audit block.
    Market gap is versus no-vig probability, not executable bookmaker edge.
    """
    p, y, m = (np.asarray(values, dtype=float) for values in (probabilities, outcomes, market))
    frame = pd.DataFrame({"p": np.r_[p, 1-p], "y": np.r_[y, 1-y], "market": np.r_[m, 1-m]})
    frame = frame[np.isfinite(frame.market) & frame.market.gt(0) & frame.market.lt(1)].copy()
    frame["price_band"] = pd.cut(frame.market, [0, .25, .5, .75, 1], include_lowest=True)
    frame["gap_band"] = pd.cut(frame.p-frame.market, [-1, -.1, -.03, .03, .1, 1], include_lowest=True)
    rows = []
    for (price, gap), group in frame.groupby(["price_band", "gap_band"], observed=True):
        predicted, observed = float(group.p.mean()), float(group.y.mean())
        rows.append(dict(price_band=str(price), market_gap_band=str(gap), observations=len(group),
                         predicted=predicted, observed=observed, overstatement=predicted-observed,
                         brier=float(((group.p-group.y)**2).mean())))
    return rows


def reliability_report(probabilities, outcomes, market):
    """Side-level calibration bins; favorites/underdogs each count once/game.

    Pick'em/unknown groups count both sides. These are diagnostics, not evidence
    of a betting edge or independent observations across opposite sides.
    """
    p, y, m = (np.asarray(values, dtype=float) for values in (probabilities, outcomes, market))
    frame = pd.DataFrame({"p": np.r_[p, 1-p], "y": np.r_[y, 1-y], "market": np.r_[m, 1-m]})
    direction = orientation(frame.market)
    frame["role"] = np.where(direction > 0, "favorite", np.where(direction < 0, "underdog",
                          np.where(frame.market.eq(.5), "pickem", "unpriced")))
    frame["bin"] = np.minimum((frame.p * 5).astype(int), 4)
    rows = []
    for (role, bin_id), group in frame.groupby(["role", "bin"], sort=True):
        predicted, observed = float(group.p.mean()), float(group.y.mean())
        rows.append(dict(role=role, probability_bin=f"{bin_id*.2:.1f}-{(bin_id+1)*.2:.1f}",
                         observations=len(group), predicted=predicted, observed=observed,
                         overstatement=predicted-observed, brier=float(((group.p-group.y)**2).mean())))
    return rows
