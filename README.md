# ⚡ Predictive Paradox — Bangladesh Grid Demand Forecasting

> **IITG.ai Recruitment Task** · *"Unravel the Predictive Paradox."*

A robust, end-to-end machine learning pipeline for short-term electricity demand forecasting on the Bangladesh national grid. Predicts the **next hour's grid demand (`demand_mw` at `t+1`)** using historical consumption, environmental, and macroeconomic signals — with strict chronological discipline and zero future leakage in the feature set.

---

## 📁 Repository Structure

```
predictive-paradox/
├── model_pipeline.py              # End-to-end runnable Python script
├── README.md                       # This file
└── data/
    ├── PGCB_date_power_demand.xlsx # Hourly demand & generation data
    ├── weather_data.xlsx           # Hourly weather data
    └── economic_full_1.csv         # Annual World Bank macroeconomic data
```

---

## 🎯 Objective

Forecast `demand_mw` at time `t+1` using all information available up to and including time `t`. The problem is framed as a **supervised tabular regression** task — LightGBM receives a row of engineered features and predicts the next hour's grid load.

**Primary Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

---

## 🔧 Pipeline Overview

```
Raw PGCB Data
     │
     ▼
[1] Data Cleaning & Deduplication
     │  • Sort by datetime, reindex to 1-hour frequency
     │  • Duplicate resolution via rolling-median deviation ranking
     │
     ▼
[2] Demand Reconstruction
     │  • If demand_mw is missing but generation_mw + load_shedding are present:
     │    demand_mw ≈ generation_mw + load_shedding
     │
     ▼
[3] Anomaly Detection (Ensemble: IForest + LOF + CBLOF)
     │  • Physical bounds: [1,800 MW, 22,000 MW]
     │  • Majority-vote outlier removal (≥2 of 3 detectors flag → NaN)
     │
     ▼
[4] KNN Imputation (temporally-aware)
     │  • Short gaps (≤24h): KNN with lag features + calendar context
     │  • Long gaps (>24h): time-interpolation first, then KNN refine
     │
     ▼
[5] Feature Engineering
     │  • Calendar: hour, dow, month, cyclical encodings, peak flags
     │  • Demand lags: t-1h, t-2h, t-3h, t-24h, t-48h, t-168h
     │  • Rolling aggregates: mean (3h, 6h, 24h, 168h), std/max (24h)
     │  • Weather: temp, humidity, heat index, precipitation, cloud cover
     │  • Fuel mix lags, load shedding lags
     │  • Economic indicators (1-year lagged World Bank data)
     │  • Calendar events: Bangladesh public holidays, Ramadan, crisis flag
     │
     ▼
[6] Train / Val / Test Split (strict chronological)
     │  • Train: 2015-04 → 2023-11 (~74,000 rows)
     │  • Val:   Dec 2023           (742 rows, early-stopping only)
     │  • Test:  Jan–Dec 2024       (~8,773 rows, final score)
     │
     ▼
[7] LightGBM Training
     │  • 5,000 trees, lr=0.02, num_leaves=127
     │  • Early stopping (200 rounds) on val MAPE
     │
     ▼
[8] Evaluation → Final Test MAPE (2024)
```

---

## 📊 Data Preparation & Structural Integrity

### Duplicate Resolution
The PGCB dataset contains duplicate timestamps from data-entry inconsistencies. Resolution strategy:
1. Compute a 48-hour rolling median of `demand_mw`.
2. For each duplicate timestamp, keep the record **closest to the rolling median** (lowest absolute deviation).
3. Resample to strict 1-hour frequency using `.resample('1h').mean()`.

### Outlier Detection (Ensemble Approach)
Raw demand data contains severe, undocumented spikes — instrument errors, transcription mistakes, and grid events. A single detector is insufficient, so a **majority-vote ensemble** is used:

| Detector | Role | Contamination |
|---|---|---|
| **Isolation Forest** | Global anomaly scoring via random partitioning | 1.0% |
| **LOF (Local Outlier Factor)** | Density-based local anomaly scoring | 2.0% |
| **CBLOF (Cluster-Based LOF)** | Cluster-distance anomaly scoring | 1.5% |

A row is flagged as an outlier if **≥ 2 out of 3** detectors flag it **OR** it violates physical bounds (`< 1,800 MW` or `> 22,000 MW`). The physical bounds are derived from the 0.05th and 99.95th percentiles of valid data, clamped to domain-knowledge fallbacks.

Features fed to detectors: `demand_mw`, `hour`, `day_of_week`, `month`, `demand_diff`, `demand_diff_pct`, `gen_demand_ratio`.

### Missing Data Imputation (KNN)
After outlier removal, gaps are filled using a **feature-weighted KNN imputer**:

- **Long gaps (> 24 hours):** First filled by time-interpolation to provide an anchor, then refined by KNN.
- **Short gaps (≤ 24 hours):** Directly imputed via KNN.

KNN feature weights prioritize temporal context over raw demand:

| Feature | Weight |
|---|---|
| `hour` | 2.5 |
| `month` | 2.0 |
| `demand_lag_24h` | 2.0 |
| `demand_lag_48h` | 1.5 |
| `day_of_week` | 1.2 |
| `demand_lag_168h` | 1.3 |
| `demand_mw` | 1.0 |

All imputed values are clipped to `[PHYSICAL_LOWER, PHYSICAL_UPPER]`.

### Economic Data Integration
Annual World Bank indicators are integrated into the hourly feature set via a **1-year lag**:
- A 1-year lag is applied before joining to the hourly series by calendar year.
- This ensures the model only uses economic data that would have been **publicly available at training time** (annual reports are published with a lag).
- Years without data are forward/backward filled.

---

## 🛠️ Feature Engineering

Since LightGBM treats each row independently (no sequential memory), the concept of "time" must be **explicitly encoded** in the feature matrix.

### Calendar Features
| Feature | Purpose |
|---|---|
| `hour`, `day_of_week`, `month`, `day_of_year` | Raw calendar position |
| `hour_sin`, `hour_cos` | Cyclical hour encoding (prevents 23→0 discontinuity) |
| `month_sin`, `month_cos` | Cyclical month encoding |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding |
| `is_evening_peak` (18–21h) | Evening demand surge flag |
| `is_day_peak` (10–16h) | Daytime industrial load flag |
| `is_friday`, `is_weekend` | Bangladesh weekend pattern (Fri–Sat off) |
| `is_public_holiday` | Bangladesh fixed + variable public holidays (2015–2025) |
| `is_ramadan` | Ramadan flag (altered consumption pattern) |
| `is_crisis_2022` | Energy crisis dummy (Apr–Dec 2022) |
| `hour_x_month` | Hour–month interaction (captures seasonal peak-hour shifts) |
| `days_since_start` | Long-run trend proxy |

### Temporal Demand Features (the model's "memory")
All lag and rolling features are computed from `demand.shift(1)` to avoid **any leakage of the current hour's demand** into historical aggregates:

| Feature | Time Window | Purpose |
|---|---|---|
| `demand_lag_1h` | t−1 | Most recent reading |
| `demand_lag_2h`, `demand_lag_3h` | t−2, t−3 | Short-term trend |
| `demand_lag_24h` | t−23 | Same hour yesterday |
| `demand_lag_48h` | t−47 | Same hour 2 days ago |
| `demand_lag_168h` | t−167 | Same hour last week |
| `demand_roll_mean_3h` | t−1 to t−3 | Immediate local average |
| `demand_roll_mean_6h` | t−1 to t−6 | Short-run average |
| `demand_roll_mean_24h` | t−1 to t−24 | Daily average |
| `demand_roll_mean_168h` | t−1 to t−168 | Weekly average |
| `demand_roll_std_24h` | t−1 to t−24 | Volatility signal |
| `demand_roll_max_24h` | t−1 to t−24 | Recent peak |

### Weather Features
Hourly observed weather at time `t` (temperature, humidity, precipitation, cloud cover, soil temperature, wind direction, sunshine duration, and a derived **Heat Index**). Heat Index is computed when temperature ≥ 20°C and humidity ≥ 40%, combining temperature and humidity into a single thermal stress measure.

### Fuel Mix & Grid Features (1-hour lagged)
| Feature | Description |
|---|---|
| `gas_share_lag1h` | Gas as % of generation (t−1) |
| `coal_share_lag1h` | Coal as % of generation (t−1) |
| `lf_share_lag1h` | Liquid fuel as % of generation (t−1) |
| `india_import_share_lag1h` | India import share (t−1) |
| `shedding_lag_1h` | Load shedding MW (t−1) |
| `shedding_roll_7d` | 7-day rolling avg shedding (supply-side stress proxy) |

All fuel mix features are computed as `(fuel_source / total_generation) × 100` then lagged by 1 hour — no current-hour generation data leaks into features.

---

## ✅ Validation Strategy

### Chronological Split
```
|── Training ──────────────────────────|─ Val ─|─────── Test ────────|
  Apr 2015            Nov 2023        Dec 2023  Jan 2024    Dec 2024
```
- **Strict chronological separation** — no row shuffling at any stage.
- **Val set** (Dec 2023): used solely for early-stopping signal in LightGBM.
- **Test set** (all of 2024): held out until final evaluation.
---

## 🤖 Model: LightGBM Regressor

```python
lgb.LGBMRegressor(
    n_estimators      = 5000,
    learning_rate     = 0.02,
    num_leaves        = 127,
    min_child_samples = 50,
    feature_fraction  = 0.7,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    reg_alpha         = 0.05,
    reg_lambda        = 0.1,
    random_state      = 42,
)
```

**Why LightGBM?**
- Handles large tabular datasets efficiently (74k training rows × 55 features).
- Natively robust to feature scale differences (no normalization needed at model input).
- `num_leaves=127` provides sufficient complexity for multi-seasonal patterns.
- Regularization (`reg_alpha`, `reg_lambda`, `min_child_samples`) prevents overfitting.
- `feature_fraction=0.7` + `bagging_fraction=0.8` add stochasticity for better generalization.

Early stopping (200 rounds on val MAPE) prevents over-training.

---

## 📈 Key Feature Importance Insights

Based on LightGBM's gain-based feature importances, the primary demand drivers are:

1. **`demand_lag_1h`** — The most recent demand reading is by far the strongest predictor. Electricity demand is highly autocorrelated at short horizons.
2. **`demand_lag_24h`** — Same-hour-yesterday captures the dominant daily seasonality pattern.
3. **`demand_roll_mean_24h`** / **`demand_roll_mean_6h`** — Recent rolling averages smooth noise and provide trend context.
4. **`demand_lag_168h`** — Same-hour-last-week captures weekly seasonality (Friday–Saturday lower demand pattern in Bangladesh).
5. **`temp_c` / `heat_index`** — Temperature drives air conditioning load, especially in summer peak months.
6. **`hour_sin` / `hour_cos`** — Time-of-day is the fundamental driver of intraday demand shape.
7. **`days_since_start`** — Captures the long-run growth trend of Bangladesh's electrification.
8. **`is_crisis_2022`** — The 2022 energy crisis period had a structural downward shift in grid demand.
9. **`econ_kwh_per_capita`** / **`econ_gdp_per_capita`** — Long-term economic development level.
10. **`is_ramadan`** — Ramadan shifts the intraday demand curve (later evening peaks, lower daytime).

---

## 🏁 Results

| Split | Period | MAPE |
|---|---|---|
| Validation | Dec 2023 | ~2.2% |
| **Test** | **All of 2024** | **Reported at runtime** |

Run the pipeline to get the exact test MAPE:
```bash
python master_pipeline.py
# Output: Final Test MAPE (2024): 3.3777 %


## ⚙️ Setup & Reproduction

### Requirements
```bash
pip install pandas numpy scikit-learn lightgbm pyod openpyxl
```

### Run
```bash
# Ensure data files are in the same directory as the script
python model_pipeline.py
```

### Data Files Required
| File | Description |
|---|---|
| `PGCB_date_power_demand.xlsx` | Hourly demand, generation, and fuel-mix data |
| `weather_data.xlsx` | Hourly weather (skip first 3 rows) |
| `economic_full_1.csv` | Annual World Bank macroeconomic indicators |

---



*Built for IITG.ai Predictive Paradox Recruitment Task · April 2025*
