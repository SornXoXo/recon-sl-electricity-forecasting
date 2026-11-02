# Next-month household electricity forecasting (RECON-SL, Sri Lanka)

MSc Applied Machine Learning final assignment. The project forecasts each Sri Lankan household's
**next-month electricity consumption (kWh)** from its own past readings. Use cases:
- **Ceylon Electricity Board (CEB) demand planning:** household forecasts aggregate to feeder and region level.
- **Bill anticipation:** households can see what's coming under the block tariff.

## Data

- **Source:** `monthly_consumption.csv` from the RECON-SL dataset of Sri Lankan household electricity consumption.
  **TODO: add the dataset citation / URL.**
- **Contents:** 101,575 rows, 4,063 households × 25 months (Oct 2022 – Oct 2024). Columns: `household_ID`, `month`, `consumption`.
- **Data issues found and handled** (full log in `results/cleaning_log.csv`):

| Issue | Count | Handling |
|---|---|---|
| Blank readings | 590 | treated as missing |
| Thousands-formatted strings (`"1,159"`) | 57 | parsed, because they are genuine readings, not junk |
| Zero readings (faulty or estimated meters) | 2,332 | treated as missing |
| Households with < 12 valid months | 64 | dropped (imputation would invent most of their series) |
| Isolated spikes (> 5× the household's own median) | 316 | clipped to 5× median |
| Missing values after the above | 1,616 | household median from training months |

A global 99.5th-percentile cap was tested and **rejected**: 74% of the readings it clips belong to households that are heavy *every* month, and it raised validation RMSE by 38% (`results/outlier_method_comparison.csv`).

## Method

- **Features (no leakage):**
  - lags 1, 2, 3, 6 and 12 months;
  - 3- and 6-month rolling mean and standard deviation;
  - the household's expanding past mean;
  - target month (number, sin, cos).

  Every feature uses only months before the target month. The notebook asserts this rather than just stating it.
- **Target:** the change from last month (prediction = `lag_1` + model output), chosen over the level on validation.
- **Split (by target month):**
  - train: 2023-04 → 2024-04 (13 months)
  - validation: 2024-05 → 2024-07
  - test: 2024-08 → 2024-10
- **Baselines:** naive, seasonal naive, per-household ARIMA (200-household sample, orders chosen by AIC), Random Forest, XGBoost.
- **Proposed models:** LightGBM; CatBoost; a stacking ensemble (LightGBM + CatBoost + RF → non-negative Ridge, trained on **time-ordered** out-of-fold predictions).
- **Hyperparameters:** grid search on validation with a similar budget for every model, and early stopping for boosters (`results/tuning_*.csv`, `results/summary.json`).
- **Main model:** best *proposed* model on validation RMSE. RF and XGBoost were taught in lectures and are baselines only.
- **Explainability:** exact SHAP values (TreeSHAP; extended exactly to the change-target wrapper and the linear stack) and a month-of-year PDP with centred ICE curves.

## Results

Test set, Aug–Oct 2024 (11,790 household-months; scored against real readings):

| Model | RMSE | MAE | R² | MAPE* | RMSE vs naive |
|---|---|---|---|---|---|
| Naive (last month) | **46.7** | **18.6** | 0.879 | 16.2% | – |
| Random Forest (baseline) | 48.4 | 21.1 | 0.869 | 16.8% | −3.8% |
| **LightGBM (main model)** | 48.5 | 20.8 | 0.869 | 16.6% | −3.9% |
| XGBoost (baseline) | 48.7 | 21.3 | 0.868 | 17.1% | −4.3% |
| CatBoost | 49.3 | 21.7 | 0.865 | 17.0% | −5.7% |
| Stacking | 49.7 | 22.5 | 0.863 | 17.4% | −6.4% |
| Seasonal naive | 72.0 | 36.9 | 0.712 | 33.5% | −54.3% |

\*MAPE excludes actual readings below 10 kWh: near-vacant months make the percentage explode.

On the 200-household ARIMA sample, ARIMA is 10.6% worse than naive.

The **rolling-origin backtest over 9 months** (2024-02 → 2024-10) gives a more reliable picture than the 3-month test:

| Model | Mean RMSE skill vs naive | Months beating naive (RMSE) |
|---|---|---|
| Random Forest | +3.9% | 6 / 9 |
| Stacking | +3.1% | 6 / 9 |
| XGBoost | +3.0% | 7 / 9 |
| LightGBM | +2.8% | 6 / 9 |
| CatBoost | +2.1% | 5 / 9 |

On MAE, naive is slightly better on average (by 1.6–4%).

**Honest summary:**
- Monthly household consumption is dominated by persistence. The learned models add a small, unstable improvement on RMSE: +7% on validation, −4% on the 3 test months, +3% averaged over 9 backtest months.
- The test months were hurt by changes shared by all households, which household history cannot anticipate. In addition, month features learned a one-off October 2023 drop as "seasonal".
- R² ≈ 0.87 is mostly *between-household* variance (naive scores 0.88 too). Use skill vs naive, not R², to judge the models.
- **SHAP:** the model is "last month + a rebound correction (`lag_2`, household mean) + a calendar effect". It shows a rise into March–April, consistent with Sri Lanka's hottest inter-monsoon period.

See the notebook's *Conclusions and limitations* for the full discussion, including the level vs change target decision.

## Repository layout

```
data/raw/                 original CSV
data/processed/           cleaned panel (written by the notebook; used by the app)
notebooks/                electricity_forecasting.ipynb: main analysis, runs top to bottom (~15 min)
src/                      config, cleaning, features, models (tuning, stacking, ARIMA, backtest), metrics, SHAP
models/                   best_model.joblib (model + features + cleaning params), feature_list.json
figures/                  all plots
results/                  metrics CSVs, tuning logs, cleaning log, summary.json
app.py                    Streamlit app
```

## How to run

Requires Python 3.13. All versions are pinned in `requirements.txt`.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. reproduce the analysis (writes data/processed, models, figures, results)
jupyter nbconvert --to notebook --execute --inplace notebooks/electricity_forecasting.ipynb

# 2. launch the app
streamlit run app.py
```

The app has two input modes:
- **Pick a household:** choose a household ID and a target month. For past months, it also shows the real reading as a backtest.
- **Enter your own readings:** type in 12 months of readings. Zeros are filled with your median and spikes are clipped, the same rules as training.

Both modes show the forecast and a SHAP waterfall explaining it.

> **Note for macOS users:** keep the project out of iCloud-synced folders (e.g. `~/Documents` with "Optimize Mac Storage"). iCloud offloads files, and reads then time out (`Errno 60`).
