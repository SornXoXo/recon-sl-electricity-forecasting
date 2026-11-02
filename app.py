"""Streamlit app: next-month household electricity forecast with a SHAP explanation.

Run with:  streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src import config
from src.explain import explain
from src.features import features_for_next_month

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Household electricity forecast", page_icon="⚡", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(config.MODELS_DIR / "best_model.joblib")


@st.cache_data
def load_clean_data():
    df = pd.read_csv(config.PROCESSED_DIR / "clean_consumption.csv", parse_dates=["month"])
    return df.sort_values(["household_ID", "month"])


def clean_user_readings(values, spike_multiple):
    """Same rules as training: zeros/blank -> missing, impute with the household's own
    median, clip spikes above spike_multiple x median."""
    s = pd.Series(values, dtype=float).replace(0, np.nan)
    med = s.median()
    notes = []
    if s.isna().any():
        notes.append(f"{int(s.isna().sum())} zero/blank month(s) replaced by your median ({med:.0f} kWh)")
    s = s.fillna(med)
    over = s > spike_multiple * med
    if over.any():
        notes.append(f"{int(over.sum())} spike(s) above {spike_multiple:g}x your median clipped")
    return s.clip(upper=spike_multiple * med), notes


def month_end(year, month):
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


try:
    bundle = load_bundle()
    data = load_clean_data()
except FileNotFoundError:
    st.error("Model or processed data not found. Run `notebooks/electricity_forecasting.ipynb` first.")
    st.stop()

model, features = bundle["model"], bundle["features"]
spike_multiple = bundle["cleaning"].get("spike_multiple", config.SPIKE_MULTIPLE)

st.title("⚡ Next-month household electricity forecast")
st.caption(f"RECON-SL Sri Lankan households · model: **{bundle['model_name']}** · "
           f"test MAE {bundle['test_metrics']['MAE']:.1f} kWh, RMSE {bundle['test_metrics']['RMSE']:.1f} kWh")

mode = st.sidebar.radio("Input", ["Pick a household from the dataset", "Enter my own readings"])
actual = None

if mode == "Pick a household from the dataset":
    households = data["household_ID"].unique()
    hh = st.sidebar.selectbox("Household ID", households)
    series = data[data["household_ID"] == hh].set_index("month")
    first_target = series.index[0] + pd.offsets.MonthEnd(max(config.LAGS))
    targets = pd.date_range(first_target, series.index[-1] + pd.offsets.MonthEnd(1), freq="ME")
    target = st.sidebar.selectbox("Target month", targets[::-1], format_func=lambda d: f"{d:%B %Y}")
    history = series.loc[series.index < target, "consumption"]
    if target in series.index and not series.loc[target, "is_imputed"]:
        actual = float(series.loc[target, "consumption_raw"])
    notes = []
else:
    c1, c2 = st.sidebar.columns(2)
    today = pd.Timestamp.today()
    years = list(range(2023, today.year + 2))
    year = c1.selectbox("Target year", years, index=years.index(today.year))
    month = c2.selectbox("Target month", range(1, 13), format_func=lambda m: MONTH_NAMES[m - 1],
                         index=today.month - 1)
    target = month_end(year, month)
    hist_months = pd.date_range(end=target - pd.offsets.MonthEnd(1), periods=12, freq="ME")
    st.subheader("Your last 12 monthly readings (kWh)")
    st.caption("Enter 0 for a month you don't know — it will be filled with your median, as in training.")
    cols = st.columns(6)
    default = [110, 105, 100, 95, 100, 120, 135, 130, 115, 105, 100, 105]
    values = [cols[i % 6].number_input(f"{m:%b %Y}", min_value=0.0, max_value=10000.0,
                                       value=float(default[i]), step=1.0, key=f"m{i}")
              for i, m in enumerate(hist_months)]
    if sum(v > 0 for v in values) < 6:
        st.warning("Please enter at least 6 non-zero months.")
        st.stop()
    cleaned, notes = clean_user_readings(values, spike_multiple)
    history = pd.Series(cleaned.values, index=hist_months)

X = features_for_next_month(history, target)[features]
pred = float(model.predict(X)[0])

left, right = st.columns([1, 2])
with left:
    st.metric(f"Forecast for {target:%B %Y}", f"{pred:,.0f} kWh",
              delta=f"{pred - history.iloc[-1]:+,.0f} kWh vs last month", delta_color="off")
    if actual is not None:
        st.metric("Actual reading (backtest)", f"{actual:,.0f} kWh", delta=f"{actual - pred:+,.0f} kWh error",
                  delta_color="off")
    for n in notes:
        st.info(n)
    if mode == "Enter my own readings":
        st.caption("Note: the 'household mean' feature uses only the 12 months entered here; "
                   "in training it used all available history.")

with right:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    h = history.iloc[-24:]
    ax.plot(h.index, h.values, marker="o", label="history (cleaned)")
    ax.plot([h.index[-1], target], [h.values[-1], pred], "--", color="#C44E52")
    ax.scatter([target], [pred], color="#C44E52", s=80, zorder=3, label="forecast")
    if actual is not None:
        ax.scatter([target], [actual], marker="x", color="k", s=80, zorder=3, label="actual")
    ax.set_ylabel("kWh"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

st.subheader("Why this forecast? (SHAP waterfall)")
st.caption("Starts from the model's average forecast (bottom) and adds each feature's contribution in kWh. "
           "Red pushes the forecast up, blue pulls it down.")
with st.spinner("Computing SHAP values…"):
    sv = explain(model, X)
shap.plots.waterfall(sv[0], max_display=12, show=False)
st.pyplot(plt.gcf(), clear_figure=True)

with st.expander("Feature values used"):
    st.dataframe(X.T.rename(columns={0: "value"}).round(2))
