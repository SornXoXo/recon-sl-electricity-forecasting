"""Model factories, validation-set tuning, time-ordered stacking and per-household ARIMA.

All tree models are trained on the raw (capped) kWh target with squared-error
loss, because RMSE is the headline metric and CEB planning cares about total
kWh, which a log-target model would systematically under-predict.
"""
import itertools
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from . import config
from .evaluation import rmse

# ---------------------------------------------------------------------------
# Search spaces. Fixed values are justified in the notebook; the grids are kept
# small and equal-ish in size across models so baselines get a fair budget.
# Boosters use learning_rate 0.05 with early stopping (max 3000 rounds) on the
# validation set, so the number of trees is chosen by data rather than guessed.
# ---------------------------------------------------------------------------
SEARCH_SPACES = {
    "RandomForest": {
        "n_estimators": [300],
        "max_depth": [12, None],
        "min_samples_leaf": [10, 40],
        "max_features": [0.4, 0.7],
    },
    "XGBoost": {
        "max_depth": [4, 6, 8],
        "min_child_weight": [5, 20],
        "reg_lambda": [1.0, 10.0],
    },
    "LightGBM": {
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 100],
        "reg_lambda": [0.0, 10.0],
    },
    "CatBoost": {
        "depth": [4, 6, 8],
        "l2_leaf_reg": [3.0, 10.0],
    },
}

BOOSTER_FIXED = {"learning_rate": 0.05, "max_rounds": 3000, "early_stopping": 100}


def make_model(name, params, n_estimators=None):
    p = dict(params)
    if name == "RandomForest":
        return RandomForestRegressor(**p, bootstrap=True, n_jobs=-1, random_state=config.SEED)
    n = n_estimators or BOOSTER_FIXED["max_rounds"]
    lr = BOOSTER_FIXED["learning_rate"]
    if name == "XGBoost":
        return XGBRegressor(n_estimators=n, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", objective="reg:squarederror", n_jobs=-1,
                            random_state=config.SEED, **p)
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, objective="regression", n_jobs=-1,
                                 random_state=config.SEED, verbose=-1, **p)
    if name == "CatBoost":
        return CatBoostRegressor(iterations=n, learning_rate=lr, loss_function="RMSE",
                                 random_seed=config.SEED, verbose=0, thread_count=-1,
                                 allow_writing_files=False, **p)
    raise ValueError(name)


def _fit(name, model, X, y, X_val=None, y_val=None):
    """Fit; boosters early-stop on (X_val, y_val) when given. Returns best #rounds."""
    es = BOOSTER_FIXED["early_stopping"]
    if name == "RandomForest" or X_val is None:
        model.fit(X, y)
        return None
    if name == "XGBoost":
        model.set_params(early_stopping_rounds=es)
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return int(model.best_iteration) + 1
    if name == "LightGBM":
        model.fit(X, y, eval_X=(X_val,), eval_y=(y_val,), callbacks=[lgb.early_stopping(es, verbose=False)])
        return int(model.best_iteration_)
    if name == "CatBoost":
        model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=es, use_best_model=True)
        return int(model.get_best_iteration()) + 1


class ChangeTargetRegressor:
    """Predicts next month as lag_1 + f(x), where f is trained on (target - lag_1).

    Persistence is built in, so the trees only learn adjustments to last month and
    never have to reproduce absolute kWh levels they cannot extrapolate.
    """

    def __init__(self, model, base_feature="lag_1"):
        self.model = model
        self.base_feature = base_feature

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y, float) - X[self.base_feature].values)
        self.base_mean_ = float(X[self.base_feature].mean())   # used by the exact SHAP offset
        return self

    def predict(self, X):
        return X[self.base_feature].values + self.model.predict(X)


def tune(name, X_tr, y_tr, X_val, y_val_actual, y_val_fit, target="change"):
    """Grid search scored by validation RMSE against real (uncapped) readings.
    y_val_fit (cleaned) is used only for booster early stopping.
    target='change' fits on (y - lag_1) and adds lag_1 back for scoring."""
    space = SEARCH_SPACES[name]
    keys = list(space)
    off_tr = X_tr["lag_1"].values if target == "change" else 0.0
    off_val = X_val["lag_1"].values if target == "change" else 0.0
    rows, candidates = [], []
    for combo in itertools.product(*space.values()):
        params = dict(zip(keys, combo))
        t0 = time.time()
        model = make_model(name, params)
        n_best = _fit(name, model, X_tr, np.asarray(y_tr) - off_tr, X_val, np.asarray(y_val_fit) - off_val)
        score = rmse(y_val_actual, model.predict(X_val) + off_val)
        candidates.append((score, params, n_best))
        rows.append({**params, "best_rounds": n_best, "val_RMSE": score, "fit_seconds": time.time() - t0})
    log = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    # take params from the original dicts: a pandas row would turn ints into floats
    _, best_params, best_rounds = min(candidates, key=lambda c: c[0])
    return best_params, best_rounds, log


def fit_final(name, params, rounds, X, y, target="change"):
    model = make_model(name, params, n_estimators=rounds)
    if target == "change":
        model = ChangeTargetRegressor(model)
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# Stacking with time-ordered out-of-fold predictions
# ---------------------------------------------------------------------------
class TimeStackingRegressor:
    """Base learners -> Ridge meta-learner.

    OOF predictions come from an expanding window: for each block of target
    months, base learners are fitted on all earlier months only and predict the
    block. The Ridge meta-learner is fitted on those OOF predictions, then base
    learners are refitted on all data. Because the meta-learner is linear, the
    stack's SHAP values are exactly sum_i w_i * SHAP_i of the base learners.
    """

    # Base predictions are in the 10^2 kWh range over ~10^4 OOF rows, so X'X is ~10^8:
    # alphas below ~10^5 barely regularise, hence the wide grid.
    def __init__(self, base_specs, alphas=(1.0, 1e3, 1e5, 1e6, 1e7), n_blocks=4, block_months=2,
                 target="change"):
        self.base_specs = base_specs          # {name: (params, rounds)}
        self.alphas = alphas
        self.n_blocks = n_blocks
        self.block_months = block_months
        self.target = target

    def _oof(self, X, y, months):
        uniq = np.sort(months.unique())
        blocks = [uniq[len(uniq) - (b + 1) * self.block_months: len(uniq) - b * self.block_months]
                  for b in reversed(range(self.n_blocks))]
        oof, ys, block_id = [], [], []
        for i, blk in enumerate(blocks):
            tr, te = months < blk.min(), months.isin(blk)
            preds = [fit_final(n, p, r, X[tr], y[tr], self.target).predict(X[te])
                     for n, (p, r) in self.base_specs.items()]
            oof.append(np.column_stack(preds)); ys.append(y[te]); block_id.append(np.full(te.sum(), i))
        self.oof_blocks_ = [(pd.Timestamp(b.min()), pd.Timestamp(b.max())) for b in blocks]
        return np.vstack(oof), np.concatenate(ys), np.concatenate(block_id)

    def fit(self, X, y, months):
        y = np.asarray(y, float)
        P, Y, B = self._oof(X, y, months.reset_index(drop=True))
        # choose alpha by forward validation over the OOF blocks. positive=True: base
        # predictions are highly collinear and negative weights would just offset each other.
        scores = {}
        for a in self.alphas:
            errs = [rmse(Y[B == k], Ridge(alpha=a, positive=True).fit(P[B < k], Y[B < k]).predict(P[B == k]))
                    for k in range(1, self.n_blocks)]
            scores[a] = float(np.mean(errs))
        self.alpha_scores_ = scores
        self.alpha_ = min(scores, key=scores.get)
        self.meta_ = Ridge(alpha=self.alpha_, positive=True).fit(P, Y)
        self.bases_ = {n: fit_final(n, p, r, X, y, self.target) for n, (p, r) in self.base_specs.items()}
        return self

    def base_predictions(self, X):
        return np.column_stack([m.predict(X) for m in self.bases_.values()])

    def predict(self, X):
        return self.meta_.predict(self.base_predictions(X))

    @property
    def weights_(self):
        return {k: float(v) for k, v in zip(self.bases_, self.meta_.coef_)}, float(self.meta_.intercept_)


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------
def rolling_backtest(X, y_fit, y_true, months, origins, specs, stack_bases=None, target="change"):
    """For each origin month m: fit every model on target months < m, forecast month m.

    specs: {name: (params, rounds)} with hyperparameters fixed in advance.
    Returns a long DataFrame: month, model, RMSE, MAE, n.
    """
    X, y_fit = X.reset_index(drop=True), np.asarray(y_fit, float)
    y_true, months = np.asarray(y_true, float), months.reset_index(drop=True)
    rows = []
    for m in origins:
        tr, te = (months < m).values, (months == m).values
        preds = {"Naive (last month)": X.loc[te, "lag_1"].values}
        for name, (params, rounds) in specs.items():
            preds[name] = fit_final(name, params, rounds, X[tr], y_fit[tr], target).predict(X[te])
        if stack_bases:
            st = TimeStackingRegressor({n: specs[n] for n in stack_bases}, target=target)
            st.fit(X[tr].reset_index(drop=True), y_fit[tr], months[tr])
            preds["Stacking"] = st.predict(X[te])
        for name, p in preds.items():
            err = y_true[te] - p
            rows.append({"month": m, "model": name, "RMSE": float(np.sqrt(np.mean(err ** 2))),
                         "MAE": float(np.mean(np.abs(err))), "bias": float(err.mean()), "n": int(te.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ARIMA baseline (per household)
# ---------------------------------------------------------------------------
ARIMA_ORDERS = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]


def _arima_one(hh, s, forecast_months, select_end):
    from statsmodels.tsa.arima.model import ARIMA

    fit_s = s[s.index <= select_end].values
    best, best_aic = (1, 0, 0), np.inf
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDERS:
            try:
                aic = ARIMA(fit_s, order=order, trend="c" if order[1] == 0 else "n").fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best, best_aic = order, aic
        for m in forecast_months:
            hist = s[s.index < m].values
            try:
                fc = ARIMA(hist, order=best, trend="c" if best[1] == 0 else "n").fit().forecast(1)[0]
            except Exception:
                fc = hist[-1]   # fall back to naive if the refit fails to converge
            out.append({"household_ID": hh, "month": m, "pred": max(float(fc), 0.0), "order": str(best)})
    return out


def arima_forecasts(series_by_hh, forecast_months, select_end, n_jobs=-1):
    """For each household: choose (p,d,q) by AIC on data <= select_end, then for
    each forecast month refit on all earlier months and forecast one step.
    No seasonal term: 22-24 points cannot identify a 12-month seasonal ARIMA."""
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs)(
        delayed(_arima_one)(hh, s, forecast_months, select_end) for hh, s in series_by_hh.items())
    return pd.DataFrame([r for rows in results for r in rows])
