"""Points model: average of XGBoost and ridge regression, plus a calibrated error model.

Over/under probabilities come from held-out errors scaled by sqrt(projection):
a 28-point scorer's range is wider than an 8-point bench player's, so the model
is more confident (in points) about low projections and less about high ones.

    python model.py      # train on everything through last night, save to artifacts/
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from xgboost import Booster, DMatrix, XGBRegressor

import data
import features as F

ARTIFACTS = Path(__file__).parent / "artifacts"
XGB_PATH, RIDGE_PATH, META_PATH = (ARTIFACTS / n for n in ("points_xgb.json", "points_ridge.json", "points_meta.json"))

PARAMS = {"learning_rate": 0.03, "max_depth": 4, "min_child_weight": 20, "subsample": 0.8,
          "colsample_bytree": 0.8, "objective": "reg:squarederror"}
MAX_TREES, EARLY_STOPPING = 3000, 100


class Ridge:
    """Ridge on standardized features with mean imputation, stored as plain JSON."""

    def __init__(self, means, scales, coef, intercept, alpha):
        self.means, self.scales = np.asarray(means), np.asarray(scales)
        self.coef, self.intercept, self.alpha = np.asarray(coef), float(intercept), float(alpha)

    @classmethod
    def fit(cls, X, y):
        raw = X.to_numpy(float)
        means = np.nanmean(raw, axis=0)
        filled = np.where(np.isnan(raw), means, raw)
        scales = filled.std(axis=0)
        scales[scales == 0] = 1
        r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit((filled - means) / scales, y)
        return cls(means, scales, r.coef_, r.intercept_, r.alpha_)

    def predict(self, X):
        raw = X.to_numpy(float)
        filled = np.where(np.isnan(raw), self.means, raw)
        return (filled - self.means) / self.scales @ self.coef + self.intercept

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


class Ensemble:
    def __init__(self, xgb, ridge):
        self.xgb, self.ridge = xgb, ridge

    def predict(self, rows):
        X = rows[F.FEATURES]
        if isinstance(self.xgb, Booster):
            gb = self.xgb.predict(DMatrix(X, feature_names=F.FEATURES))
        else:
            gb = self.xgb.predict(X)
        return np.clip((gb + self.ridge.predict(X)) / 2, 0, None)


def _scale(projection):
    return np.sqrt(np.maximum(projection, 4.0))


def _metrics(y, pred):
    err = np.asarray(y) - np.asarray(pred)
    return {"mae": round(float(np.abs(err).mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "n": int(len(err))}


def train(box):
    """Validate on a time split (last season = valid, current season = test), then refit on all."""
    rows = F.eligible(F.build(box))
    rows = rows[rows[F.TARGET].notna()]
    seasons = sorted(rows["season"].unique())
    test_season = seasons[-1] if (rows["season"] == seasons[-1]).sum() >= 3000 else seasons[-2]
    valid_season = test_season - 1
    tr, va, te = (rows[rows["season"] < valid_season], rows[rows["season"] == valid_season],
                  rows[rows["season"] >= test_season])

    xgb = XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOPPING, **PARAMS)
    xgb.fit(tr[F.FEATURES], tr[F.TARGET], eval_set=[(va[F.FEATURES], va[F.TARGET])], verbose=False)
    n_trees = xgb.best_iteration + 1
    valid_pred = Ensemble(xgb, Ridge.fit(tr[F.FEATURES], tr[F.TARGET])).predict(va)
    trva = rows[rows["season"] < test_season]
    test_pred = Ensemble(xgb, Ridge.fit(trva[F.FEATURES], trva[F.TARGET])).predict(te)

    # Scaled held-out errors from both seasons the models never trained on.
    holdout = np.concatenate([(va[F.TARGET] - valid_pred) / _scale(valid_pred),
                              (te[F.TARGET] - test_pred) / _scale(test_pred)])

    final_xgb = XGBRegressor(n_estimators=int(n_trees * 1.1), **PARAMS).fit(rows[F.FEATURES], rows[F.TARGET])
    final = Ensemble(final_xgb, Ridge.fit(rows[F.FEATURES], rows[F.TARGET]))

    last = rows["tipoff"].max()
    meta = {
        "model_type": "average of XGBoost and ridge regression",
        "features": F.FEATURES,
        "params": PARAMS,
        "n_trees": int(final_xgb.n_estimators),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": last.tz_convert("America/New_York").strftime("%Y-%m-%d"),
        "split": {"train": f"{int(tr['season'].min())}-{int(tr['season'].max())}", "valid": int(valid_season),
                  "test": f"{int(test_season)}+"},
        "test_metrics": _metrics(te[F.TARGET], test_pred),
        "baseline_metrics": _metrics(te[F.TARGET], te["pts_10"].fillna(tr[F.TARGET].mean())),
        "calibration_check": _calibration(te[F.TARGET].to_numpy(), test_pred, holdout),
        # Percentiles of (actual - projection) / sqrt(projection) on held-out seasons.
        "scaled_residual_quantiles": [round(float(q), 4) for q in
                                      np.quantile(holdout, np.linspace(0.0025, 0.9975, 200))],
        "importance": dict(sorted(zip(F.FEATURES, final_xgb.feature_importances_.astype(float).round(4)),
                                  key=lambda kv: -kv[1])),
    }
    return final, meta


def _calibration(y, pred, holdout):
    """How often actual > projection + k*sqrt(projection) vs. what the error model says."""
    out = {}
    for offset in (-4, -2, 0, 2, 4):
        line = pred + offset
        said = np.mean([_p_over(holdout, p, l) for p, l in zip(pred, line)])
        out[f"line = proj{offset:+d}"] = {"model_said": round(float(said), 3),
                                         "actual": round(float(np.mean(y > line)), 3)}
    return out


def _p_over(q, projection, line):
    return float(np.mean(np.asarray(q) > (line - projection) / _scale(projection)))


def prob_over(meta, projection, line):
    """Model probability that the player scores more than `line` points."""
    return _p_over(meta["scaled_residual_quantiles"], projection, line)


def save(model, meta):
    ARTIFACTS.mkdir(exist_ok=True)
    model.xgb.get_booster().save_model(XGB_PATH)
    RIDGE_PATH.write_text(json.dumps(model.ridge.to_dict()))
    META_PATH.write_text(json.dumps(meta, indent=2))


def load():
    booster = Booster()
    booster.load_model(XGB_PATH)
    ridge = Ridge(**json.loads(RIDGE_PATH.read_text()))
    return Ensemble(booster, ridge), json.loads(META_PATH.read_text())


if __name__ == "__main__":
    m, meta = train(data.load_box())
    save(m, meta)
    print(json.dumps({k: meta[k] for k in ("trained_through", "split", "n_trees", "test_metrics",
                                           "baseline_metrics", "calibration_check")}, indent=2))
