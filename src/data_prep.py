"""Loading and cleaning of the RECON-SL monthly consumption data.

Every step appends a record to a cleaning log so the report can quote exact
counts. Statistics (medians, cap) are fitted on months <= CLEANING_FIT_END only.
"""
import numpy as np
import pandas as pd

from . import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the CSV with consumption kept as text so nothing is silently coerced."""
    df = pd.read_csv(path, dtype={"household_ID": str, "consumption": str})
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["household_ID", "month"]).reset_index(drop=True)


def parse_consumption(s: pd.Series) -> pd.Series:
    """'1,159' -> 1159.0, '' -> NaN. Anything else unparseable also becomes NaN."""
    return pd.to_numeric(s.str.strip().str.replace(",", "", regex=False), errors="coerce")


def clean(raw: pd.DataFrame, outlier_method="household", fit_end=config.CLEANING_FIT_END):
    """Return (clean_df, log, params). outlier_method: 'household' | 'global' | 'none'.

    clean_df columns:
      consumption_raw  - parsed original reading (NaN if blank/zero)
      is_imputed       - True where the reading was missing or zero
      consumption      - capped + imputed series used for features/targets
    """
    log = []
    df = raw.copy()
    n_rows, n_hh = len(df), df["household_ID"].nunique()
    log.append({"step": "load", "rows": n_rows, "households": n_hh})

    text = df["consumption"].fillna("").str.strip()
    n_blank = int((text == "").sum())
    n_comma = int(text.str.contains(",", regex=False).sum())
    df["consumption_raw"] = parse_consumption(df["consumption"].fillna(""))
    n_unparsed = int(df["consumption_raw"].isna().sum()) - n_blank
    log.append({"step": "parse numeric", "blank_to_nan": n_blank,
                "thousands_separator_recovered": n_comma, "other_unparseable": n_unparsed})

    zero = df["consumption_raw"] == 0
    df.loc[zero, "consumption_raw"] = np.nan
    log.append({"step": "zeros treated as missing", "zeros": int(zero.sum()),
                "households_affected": int(df.loc[zero, "household_ID"].nunique())})

    valid_counts = df.groupby("household_ID")["consumption_raw"].count()
    sparse = valid_counts[valid_counts < config.MIN_VALID_MONTHS].index
    df = df[~df["household_ID"].isin(sparse)].copy()
    log.append({"step": f"drop households with < {config.MIN_VALID_MONTHS} valid months",
                "households_dropped": len(sparse), "rows_dropped": len(sparse) * 25,
                "households_remaining": df["household_ID"].nunique(), "rows_remaining": len(df)})

    fit_mask = df["month"] <= pd.Timestamp(fit_end)
    hh_median = df.loc[fit_mask].groupby("household_ID")["consumption_raw"].median()
    global_median = float(df.loc[fit_mask, "consumption_raw"].median())
    own_median = df["household_ID"].map(hh_median).fillna(global_median)

    global_cap = float(df.loc[fit_mask, "consumption_raw"].quantile(config.CAP_QUANTILE))
    if outlier_method == "global":
        # one threshold for everyone: also clips households that are heavy every month
        upper = pd.Series(global_cap, index=df.index)
        desc = f"global cap at p{config.CAP_QUANTILE * 100:g} = {global_cap:.0f} kWh"
    elif outlier_method == "household":
        # only readings far above the household's own typical level are treated as spikes
        upper = config.SPIKE_MULTIPLE * own_median
        desc = f"clip readings > {config.SPIKE_MULTIPLE:g}x household median"
    elif outlier_method == "none":
        upper = pd.Series(np.inf, index=df.index)
        desc = "no outlier treatment"
    else:
        raise ValueError(outlier_method)
    n_capped = int((df["consumption_raw"] > upper).sum())
    capped = df["consumption_raw"].clip(upper=upper)
    log.append({"step": f"outliers: {desc} (stats fit on months <= {fit_end})",
                "readings_clipped": n_capped,
                "households_affected": int(df.loc[df["consumption_raw"] > upper, "household_ID"].nunique())})

    fill = own_median.clip(upper=upper)
    df["is_imputed"] = df["consumption_raw"].isna()
    df["consumption"] = capped.fillna(fill)
    n_hh_fallback = int(df.loc[df["is_imputed"], "household_ID"].map(hh_median).isna().sum())
    log.append({"step": "impute missing with household median (train months)",
                "imputed": int(df["is_imputed"].sum()),
                "used_global_median_fallback": n_hh_fallback,
                "global_median": global_median})

    params = {"outlier_method": outlier_method, "global_cap": global_cap,
              "spike_multiple": config.SPIKE_MULTIPLE, "global_median": global_median,
              "household_median": hh_median.to_dict(), "dropped_households": list(sparse)}
    return df, log, params
