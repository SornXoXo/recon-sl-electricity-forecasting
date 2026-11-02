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
    if _is(model, TimeStackingRegressor):
        weights, intercept = model.weights_
        values, base = np.zeros(X.shape), intercept
        for name, base_model in model.bases_.items():
            e = explain(base_model, X)
            values += weights[name] * e.values
            base += weights[name] * float(np.ravel(e.base_values)[0])
        return _make(values, base, X)
    if _is(model, ChangeTargetRegressor):
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
    inner = model.model if _is(model, ChangeTargetRegressor) else model
    contrib = getattr(inner, "booster_", inner).predict(X, pred_contrib=True)
    values, base = contrib[:, :-1].copy(), float(contrib[0, -1])
    if _is(model, ChangeTargetRegressor):
        j = list(X.columns).index(model.base_feature)
        values[:, j] += X[model.base_feature].values - model.base_mean_
        base += model.base_mean_
    return _make(values, base, X)


def _is(obj, cls):
    """isinstance that survives module reloads: when Streamlit re-imports src.models after a
    code change, a cached model is still an instance of the *old* class object, so a plain
    isinstance against the new class is False. Match on the qualified class name as well."""
    return isinstance(obj, cls) or (type(obj).__module__, type(obj).__qualname__) == (cls.__module__, cls.__qualname__)


def _make(values, base, X):
    return shap.Explanation(values=values, base_values=np.full(len(X), base),
                            data=X.values, feature_names=list(X.columns))
