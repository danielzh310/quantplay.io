from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]


class Spread:
    def __init__(self):
        # tuned lasso on residual target (edge)
        self.model = Lasso(alpha=1.0, max_iter=10000)
        self.scaler = StandardScaler()
        self.margin_model = make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10000))
        self.has_market_training = False

    def _build_model_matrix(self, df):
        X = df[FEATURES].copy()

        # aggregated features:
        
        # spread line for residual modeling
        X["spread_line"] = df["spread_line"].values

        # matchup advantages
        X["home_attack_adv"] = X["home_off_avg"] - X["away_def_avg"]
        X["away_attack_adv"] = X["away_off_avg"] - X["home_def_avg"]
        X["net_advantage"] = X["home_attack_adv"] - X["away_attack_adv"]

        # overall pace and defensive context
        X["pace"] = X["home_off_avg"] + X["away_off_avg"]
        X["def_total"] = X["home_def_avg"] + X["away_def_avg"]
        X["def_diff"] = X["away_def_avg"] - X["home_def_avg"]

        # interactions
        X["pace_def_interaction"] = X["pace"] * X["def_total"]
        X["attack_def_interaction"] = X["net_advantage"] * X["def_diff"]

        # home-field scaling
        X["home_boost"] = X["home_off_avg"] * 0.12

        return X

    def train(self, df):
        X = self._build_model_matrix(df)
        # A direct margin fallback keeps projections available without market lines.
        self.margin_model.fit(X.drop(columns="spread_line"), df["margin"])
        available = df["spread_line"].notna()
        self.has_market_training = bool(available.any())
        if self.has_market_training:
            Xs = self.scaler.fit_transform(X.loc[available])
            self.model.fit(Xs, (df["margin"] - df["spread_line"])[available])

    def predict(self, df):
        X = self._build_model_matrix(df)
        margin = pd.Series(self.margin_model.predict(X.drop(columns="spread_line")), index=df.index)
        edges = pd.Series(float("nan"), index=df.index)
        picks = pd.Series("PASS", index=df.index)
        available = df["spread_line"].notna()
        if self.has_market_training and available.any():
            edges.loc[available] = self.model.predict(self.scaler.transform(X.loc[available]))
            margin.loc[available] = edges[available] + df.loc[available, "spread_line"]
        elif available.any():
            edges.loc[available] = margin[available] - df.loc[available, "spread_line"]
        picks.loc[edges > 0] = "HOME"
        picks.loc[edges <= 0] = "AWAY"
        return pd.DataFrame({
            "game_id": df["game_id"].values,
            "predicted_margin": margin.values,
            "spread_edge": edges.values,
            "spread_pick": picks.values,
        })
