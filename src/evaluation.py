"""Regression metrics used throughout the project."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def mape(y_true, y_pred, min_actual=config.MAPE_MIN_ACTUAL):
    """MAPE (%) over actuals >= min_actual. Readings of a few kWh (vacant homes,
    partial-month meters) make the percentage error explode and dominate the mean."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    keep = y_true >= min_actual
    return float(np.mean(np.abs((y_true[keep] - y_pred[keep]) / y_true[keep])) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
        "n": int(len(y_true)),
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
