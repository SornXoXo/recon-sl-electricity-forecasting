"""Leakage-free feature construction.

Each output row is one (household, target month). Lags are counted back from the
target month, so lag_1 is the most recent known month and lag_12 is the same
calendar month one year earlier. Every feature is computed from a series that
has been shifted by at least one month, so the target month never leaks in.
"""
import numpy as np
import pandas as pd

from . import config

FEATURES = (
    [f"lag_{k}" for k in config.LAGS]
    + [f"roll_mean_{w}" for w in config.ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in config.ROLL_WINDOWS]
    + ["hh_mean_past", "month_of_year", "month_sin", "month_cos"]
)


def build_features(df: pd.DataFrame, min_target_index=config.MIN_TARGET_INDEX) -> pd.DataFrame:
    """df needs household_ID, month, consumption (cleaned). Optional columns
    consumption_raw / is_imputed are carried through as the target's true value."""
    df = df.sort_values(["household_ID", "month"]).reset_index(drop=True)
    g = df.groupby("household_ID")["consumption"]
    past = g.shift(1)                       # last month known at forecast time
    past_g = past.groupby(df["household_ID"])

    out = df[["household_ID", "month"]].copy()
    for k in config.LAGS:
        out[f"lag_{k}"] = g.shift(k)
    for w in config.ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = past_g.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        out[f"roll_std_{w}"] = past_g.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    out["hh_mean_past"] = past_g.transform(lambda s: s.expanding().mean())

    m = df["month"].dt.month
    out["month_of_year"] = m
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)

    out["target"] = df["consumption"]
    if "consumption_raw" in df:
        out["target_actual"] = df["consumption_raw"]
        out["target_imputed"] = df["is_imputed"]
    out["t_index"] = g.cumcount()
    return out[out["t_index"] >= min_target_index].reset_index(drop=True)


def features_for_next_month(history: pd.Series, target_month) -> pd.DataFrame:
    """Single-row feature frame for forecasting `target_month` from `history`
    (a Series of cleaned monthly readings indexed by month-end dates, oldest first).
    Uses the exact same code path as training via a placeholder target row."""
    target_month = pd.Timestamp(target_month) + pd.offsets.MonthEnd(0)
    months = pd.to_datetime(history.index) + pd.offsets.MonthEnd(0)
    expected = pd.date_range(end=target_month - pd.offsets.MonthEnd(1), periods=len(history), freq="ME")
    if len(history) < max(config.LAGS) or not (months == expected).all():
        raise ValueError(f"history must be >= {max(config.LAGS)} consecutive months ending "
                         f"the month before {target_month:%Y-%m}")
    hist = pd.DataFrame({"household_ID": "query", "month": pd.to_datetime(history.index),
                         "consumption": history.values.astype(float)})
    hist = pd.concat([hist, pd.DataFrame({"household_ID": ["query"], "month": [target_month],
                                          "consumption": [np.nan]})], ignore_index=True)
    feats = build_features(hist, min_target_index=0)
    return feats.iloc[[-1]][FEATURES].reset_index(drop=True)
