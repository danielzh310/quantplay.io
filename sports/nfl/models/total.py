from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]

EDGE_BUFFER = 2.0


class Total:
    """
    Gradient-boosted regression model for football over/under totals
    with bias correction and confidence-based betting.
    """

    def __init__(self):
        self.mid_model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        )

        self.low_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.2,
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        )

        self.high_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.8,
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        )

        self.residual_mean_ = 0.0
        self.residual_std_ = EDGE_BUFFER

    def _build_model_matrix(self, df):
        X = df[FEATURES].copy()

        X["pace"] = X["home_off_avg"] + X["away_off_avg"]
        X["def_total"] = X["home_def_avg"] + X["away_def_avg"]

        X["home_attack_adv"] = X["home_off_avg"] - X["away_def_avg"]
        X["away_attack_adv"] = X["away_off_avg"] - X["home_def_avg"]

        X["pace_def_interaction"] = X["pace"] * X["def_total"]

        X["home_boost"] = X["home_off_avg"] * 0.15

        return X

    def train(self, df):
        X = self._build_model_matrix(df)
        y = df["total_pts"]

        self.mid_model.fit(X, y)
        self.low_model.fit(X, y)
        self.high_model.fit(X, y)

        book_lines = df["total_line"].values[: len(y)]
        train_mid = self.mid_model.predict(X)

        residuals = pd.Series(train_mid - book_lines).dropna()
        if not residuals.empty:
            self.residual_mean_ = float(residuals.mean())
            self.residual_std_ = max(float(residuals.std(ddof=0)), EDGE_BUFFER)

    def predict(self, df):
        X = self._build_model_matrix(df)

        mid_raw = self.mid_model.predict(X)
        low_raw = self.low_model.predict(X)
        high_raw = self.high_model.predict(X)
        
        mid_preds = mid_raw - self.residual_mean_
        low_preds = low_raw - self.residual_mean_
        high_preds = high_raw - self.residual_mean_

        book_lines = df["total_line"].values[: len(mid_preds)]
        edges = mid_preds - book_lines

        buffer = max(EDGE_BUFFER, self.residual_std_)

        picks = []
        for lo, mid, hi, line in zip(low_preds, mid_preds, high_preds, book_lines):
            if hi > line + buffer and mid > line:
                picks.append("OVER")
            elif lo < line - buffer and mid < line:
                picks.append("UNDER")
            else:
                picks.append("PASS")

        return pd.DataFrame({
            "game_id": df["game_id"].values[: len(mid_preds)],
            "predicted_total": mid_preds,
            "total_low_q": low_preds,
            "total_high_q": high_preds,
            "total_edge": edges,
            "total_buffer": buffer,
            "total_pick": picks,
        })
