"""SHAP explanations for the project's tree models, change-target wrappers and the Ridge stack."""
import numpy as np
import shap

from .models import ChangeTargetRegressor, TimeStackingRegressor


def explain(model, X):
    """Return an exact, additive shap.Explanation (kWh) for rows X.

    * Tree model: TreeSHAP (tree_path_dependent).
    * ChangeTargetRegressor, g(x) = lag_1 + f(x): SHAP is linear in the model, so
      phi_g = phi_f plus, for lag_1, the exact SHAP of the linear term (lag_1 - E[lag_1]);
      the base value gains E[lag_1].
    * TimeStackingRegressor, s(x) = b + sum_i w_i g_i(x): phi_s = sum_i w_i phi_i and
      base = b + sum_i w_i base_i.
    """
    if isinstance(model, TimeStackingRegressor):
        weights, intercept = model.weights_
        values, base = np.zeros(X.shape), intercept
        for name, base_model in model.bases_.items():
            e = explain(base_model, X)
            values += weights[name] * e.values
            base += weights[name] * float(np.ravel(e.base_values)[0])
        return _make(values, base, X)
    if isinstance(model, ChangeTargetRegressor):
        e = explain(model.model, X)
        values = e.values.copy()
        j = list(X.columns).index(model.base_feature)
        values[:, j] += X[model.base_feature].values - model.base_mean_
        return _make(values, float(np.ravel(e.base_values)[0]) + model.base_mean_, X)
    e = shap.TreeExplainer(model)(X, check_additivity=False)
    return _make(e.values, float(np.ravel(e.base_values)[0]), X)


def explain_lgbm(model, X):
    """Same output as explain() for a LightGBM model (optionally inside ChangeTargetRegressor),
    but from LightGBM's own pred_contrib=True instead of shap.TreeExplainer.

    pred_contrib returns path-dependent TreeSHAP values with the base value in the last
    column, so it does not depend on which booster versions shap.TreeExplainer supports.
    The lag_1 persistence term is added exactly as in explain().
    """
    inner = model.model if isinstance(model, ChangeTargetRegressor) else model
    contrib = _lgbm_contrib(inner, X)
    values, base = contrib[:, :-1].copy(), float(contrib[0, -1])
    if isinstance(model, ChangeTargetRegressor):
        j = list(X.columns).index(model.base_feature)
        values[:, j] += X[model.base_feature].values - model.base_mean_
        base += model.base_mean_
    return _make(values, base, X)


def _lgbm_contrib(inner, X):
    """pred_contrib via the sklearn wrapper (the same call path as .predict), then via the
    raw Booster on a plain float array, which bypasses LightGBM's pandas handling.
    Raises with every attempt's error if none works."""
    booster = getattr(inner, "booster_", inner)
    attempts = {
        "LGBMRegressor.predict(DataFrame)": lambda: inner.predict(X, pred_contrib=True),
        "Booster.predict(ndarray)": lambda: booster.predict(X.to_numpy(dtype=float), pred_contrib=True),
    }
    errors = []
    for name, attempt in attempts.items():
        try:
            contrib = np.asarray(attempt(), dtype=float)
            if contrib.shape == (len(X), X.shape[1] + 1):
                return contrib
            errors.append(f"{name}: unexpected shape {contrib.shape}")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("LightGBM pred_contrib failed. " + " | ".join(errors))


def _make(values, base, X):
    return shap.Explanation(values=values, base_values=np.full(len(X), base),
                            data=X.values, feature_names=list(X.columns))
