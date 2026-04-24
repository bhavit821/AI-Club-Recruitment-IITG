"""
=============================================================================
  PREDICTIVE PARADOX — Full Pipeline (Preprocessing + LightGBM Model)
=============================================================================

INPUT FILES:
  PATH_PGCB    → PGCB_date_power_demand.xlsx
  PATH_WEATHER → weather_data.xlsx
  PATH_ECON    → economic_full_1.csv

HOW TO RUN:
-----------
  pip install lightgbm scikit-learn pyod matplotlib pandas numpy openpyxl
  python final_pipeline.py

OUTPUT:
  - Final MAPE printed to console
  - Prediction plots + feature importance saved to current directory
  - predictions_2024.csv saved to current directory

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  [CHANGE THESE] File Paths
# ─────────────────────────────────────────────────────────────────────────────

PATH_PGCB    = "PGCB_date_power_demand.xlsx"
PATH_WEATHER = "weather_data.xlsx"
PATH_ECON    = "economic_full_1.csv"
PATH_OUTPUT_DIR = "."

# ─────────────────────────────────────────────────────────────────────────────
#  [OPTIONALLY CHANGE THESE] Parameters
# ─────────────────────────────────────────────────────────────────────────────

PHYSICAL_LOWER_FALLBACK = 1800
PHYSICAL_UPPER_FALLBACK = 22000

CONTAMINATION_IFOREST   = 0.01
CONTAMINATION_LOF       = 0.02
CONTAMINATION_CBLOF     = 0.015

LOF_N_NEIGHBORS         = 48
KNN_NEIGHBORS           = 7
LONG_GAP_THRESHOLD      = 24

MISSING_COL_THRESHOLD   = 0.15
TEST_YEAR               = 2024


# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from sklearn.impute      import KNNImputer
from sklearn.preprocessing import StandardScaler

from pyod.models.iforest import IForest
from pyod.models.lof     import LOF
from pyod.models.cblof   import CBLOF

import lightgbm as lgb


# =============================================================================
#  HOLIDAY CALENDAR — Bangladesh Public Holidays + Ramadan (2015–2025)
# =============================================================================

def build_bangladesh_holiday_calendar(years: list):
    fixed_holidays = [
        (2, 21), (3, 26), (4, 14), (5,  1),
        (8, 15), (12, 16), (12, 25),
    ]

    variable_holidays = {
        2015: [(3,21),(3,22),(3,23),(9,24),(9,25),(9,26),(10,22),(5,4)],
        2016: [(7,6),(7,7),(7,8),(9,12),(9,13),(9,14),(10,11),(5,21)],
        2017: [(6,25),(6,26),(6,27),(9,1),(9,2),(9,3),(9,30),(5,10)],
        2018: [(6,15),(6,16),(6,17),(8,21),(8,22),(8,23),(10,19),(4,30)],
        2019: [(6,4),(6,5),(6,6),(8,11),(8,12),(8,13),(10,8),(5,18)],
        2020: [(5,24),(5,25),(5,26),(7,31),(8,1),(8,2),(10,26),(5,7)],
        2021: [(5,13),(5,14),(5,15),(7,20),(7,21),(7,22),(10,15),(5,26)],
        2022: [(5,2),(5,3),(5,4),(7,9),(7,10),(7,11),(10,5),(5,16)],
        2023: [(4,21),(4,22),(4,23),(6,28),(6,29),(6,30),(10,24),(5,5)],
        2024: [(4,10),(4,11),(4,12),(6,16),(6,17),(6,18),(10,13),(5,23)],
        2025: [(3,30),(3,31),(4,1),(6,6),(6,7),(6,8),(10,2),(5,12)],
    }

    ramadan_starts = {
        2015: pd.Timestamp('2015-06-18'),
        2016: pd.Timestamp('2016-06-06'),
        2017: pd.Timestamp('2017-05-27'),
        2018: pd.Timestamp('2018-05-16'),
        2019: pd.Timestamp('2019-05-05'),
        2020: pd.Timestamp('2020-04-24'),
        2021: pd.Timestamp('2021-04-13'),
        2022: pd.Timestamp('2022-04-02'),
        2023: pd.Timestamp('2023-03-23'),
        2024: pd.Timestamp('2024-03-11'),
        2025: pd.Timestamp('2025-03-01'),
    }

    all_holiday_dates = []
    ramadan_dates     = []

    for year in years:
        for m, d in fixed_holidays:
            try:
                all_holiday_dates.append(pd.Timestamp(year=year, month=m, day=d))
            except ValueError:
                pass
        for m, d in variable_holidays.get(year, []):
            try:
                all_holiday_dates.append(pd.Timestamp(year=year, month=m, day=d))
            except ValueError:
                pass
        if year in ramadan_starts:
            start = ramadan_starts[year]
            ramadan_dates += pd.date_range(start, periods=30, freq='D').tolist()

    holiday_index = pd.DatetimeIndex(all_holiday_dates).normalize().unique()
    ramadan_index = pd.DatetimeIndex(ramadan_dates).normalize().unique()

    return holiday_index, ramadan_index


# =============================================================================
#  STEP 1 — Load & Initial Clean of PGCB Demand Data
# =============================================================================

def load_pgcb(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    numeric_cols = [
        'generation_mw', 'demand_mw', 'load_shedding',
        'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind',
        'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('datetime').reset_index(drop=True)

    n_dups = df['datetime'].duplicated().sum()
    if n_dups > 0:
        df['_rolling_med'] = (
            df['demand_mw']
            .rolling(window=48, center=True, min_periods=1)
            .median()
        )
        df['_deviation'] = (df['demand_mw'] - df['_rolling_med']).abs()
        df = df.sort_values(['datetime', '_deviation'])
        df = df.drop_duplicates(subset='datetime', keep='first')
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop(columns=['_rolling_med', '_deviation'])

    df = df.set_index('datetime')
    keep_cols = [c for c in numeric_cols if c in df.columns]
    df_hourly = df[keep_cols].resample('1h').mean()

    demand_valid = df_hourly['demand_mw'].dropna()
    global PHYSICAL_LOWER, PHYSICAL_UPPER
    PHYSICAL_LOWER = max(
        PHYSICAL_LOWER_FALLBACK,
        int(demand_valid.quantile(0.0005))
    )
    PHYSICAL_UPPER = min(
        PHYSICAL_UPPER_FALLBACK,
        int(demand_valid.quantile(0.9995) * 1.05)
    )

    return df_hourly


# =============================================================================
#  STEP 2 — Physics-Based Demand Reconstruction
# =============================================================================

def reconstruct_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mask = (
        df['demand_mw'].isna() &
        df['generation_mw'].notna() &
        df['load_shedding'].notna()
    )

    if mask.sum() > 0:
        df.loc[mask, 'demand_mw'] = (
            df.loc[mask, 'generation_mw'] + df.loc[mask, 'load_shedding']
        )
        df.loc[mask, 'demand_mw'] = df.loc[mask, 'demand_mw'].clip(
            lower=PHYSICAL_LOWER_FALLBACK,
            upper=PHYSICAL_UPPER_FALLBACK
        )

    return df


# =============================================================================
#  STEP 3 — Outlier Detection with PyOD Ensemble
# =============================================================================

def detect_outliers_pyod(df: pd.DataFrame) -> pd.Series:
    series = df['demand_mw'].copy()

    feat = pd.DataFrame({'demand_mw': series})
    feat['hour']            = series.index.hour
    feat['day_of_week']     = series.index.dayofweek
    feat['month']           = series.index.month
    feat['demand_diff']     = series.diff().abs()
    feat['demand_diff_pct'] = series.pct_change().abs().clip(0, 0.5)

    if 'generation_mw' in df.columns:
        feat['gen_demand_ratio'] = (
            df['generation_mw'] / series.replace(0, np.nan)
        ).clip(0.5, 1.5)

    valid_mask = feat['demand_mw'].notna() & feat['demand_diff'].notna()
    feat_valid = feat[valid_mask].fillna(feat[valid_mask].median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat_valid)

    iso = IForest(contamination=CONTAMINATION_IFOREST, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    iso_labels = iso.predict(X_scaled)

    lof = LOF(n_neighbors=LOF_N_NEIGHBORS, contamination=CONTAMINATION_LOF, n_jobs=-1)
    lof.fit(X_scaled)
    lof_labels = lof.predict(X_scaled)

    cblof = CBLOF(contamination=CONTAMINATION_CBLOF, check_estimator=False, random_state=42)
    cblof.fit(X_scaled)
    cblof_labels = cblof.predict(X_scaled)

    votes              = iso_labels + lof_labels + cblof_labels
    outlier_mask_valid = votes >= 2

    phys_low_mask  = series < PHYSICAL_LOWER
    phys_high_mask = series > PHYSICAL_UPPER

    outlier_full = pd.Series(False, index=series.index)
    outlier_full.loc[feat_valid.index[outlier_mask_valid]] = True
    outlier_full = outlier_full | phys_low_mask | phys_high_mask

    cleaned = series.copy()
    cleaned[outlier_full] = np.nan

    return cleaned


# =============================================================================
#  STEP 4 — Temporally-Aware Weighted KNN Imputation
# =============================================================================

def build_impute_features(series: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=series.index)
    feat['demand_mw']       = series
    feat['hour']            = series.index.hour
    feat['day_of_week']     = series.index.dayofweek
    feat['month']           = series.index.month
    feat['demand_lag_24h']  = series.shift(24)
    feat['demand_lag_48h']  = series.shift(48)
    feat['demand_lag_168h'] = series.shift(168)
    feat['demand_lag_336h'] = series.shift(336)
    return feat


FEATURE_WEIGHTS = {
    'demand_mw'      : 1.0,
    'hour'           : 2.5,
    'day_of_week'    : 1.2,
    'month'          : 2.0,
    'demand_lag_24h' : 2.0,
    'demand_lag_48h' : 1.5,
    'demand_lag_168h': 1.3,
    'demand_lag_336h': 1.1,
}


def knn_impute_demand(series: pd.Series,
                      n_neighbors: int = KNN_NEIGHBORS,
                      long_gap_threshold: int = LONG_GAP_THRESHOLD
                      ) -> pd.Series:
    result = series.copy()

    missing        = result.isna()
    gap_id         = (missing != missing.shift()).cumsum()
    gap_sizes      = missing.groupby(gap_id).transform('sum')
    long_gap_mask  = missing & (gap_sizes >  long_gap_threshold)
    short_gap_mask = missing & (gap_sizes <= long_gap_threshold)

    if long_gap_mask.any():
        result[long_gap_mask] = (
            result.interpolate(method='time', limit_direction='both')[long_gap_mask]
        )

    feat_df    = build_impute_features(result)
    scaler     = StandardScaler()
    fit_mask   = feat_df['demand_mw'].notna()
    scaler.fit(feat_df[fit_mask].fillna(0))
    X_scaled   = scaler.transform(feat_df.fillna(0))

    feat_names = list(feat_df.columns)
    for i, col in enumerate(feat_names):
        X_scaled[:, i] *= FEATURE_WEIGHTS.get(col, 1.0)

    demand_col_idx = feat_names.index('demand_mw')
    X_scaled[result.isna().values, demand_col_idx] = np.nan

    imputer   = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    X_imputed = imputer.fit_transform(X_scaled)

    demand_weight  = FEATURE_WEIGHTS['demand_mw']
    demand_mean    = scaler.mean_[demand_col_idx]
    demand_scale   = scaler.scale_[demand_col_idx]
    imputed_demand = (
        (X_imputed[:, demand_col_idx] / demand_weight) * demand_scale + demand_mean
    )

    result[short_gap_mask] = imputed_demand[short_gap_mask.values]
    result = result.clip(lower=PHYSICAL_LOWER, upper=PHYSICAL_UPPER)

    return result


# =============================================================================
#  STEP 5 — Load & Process Weather Data
# =============================================================================

def load_weather(path: str) -> pd.DataFrame:
    weather = pd.read_excel(path, skiprows=3, header=0)

    rename_map = {
        'time'                          : 'datetime',
        'temperature_2m (°C)'           : 'temp_c',
        'relative_humidity_2m (%)'      : 'humidity_pct',
        'apparent_temperature (°C)'     : 'apparent_temp_c',
        'precipitation (mm)'            : 'precipitation_mm',
        'dew_point_2m (°C)'             : 'dew_point_c',
        'soil_temperature_0_to_7cm (°C)': 'soil_temp_c',
        'wind_direction_10m (°)'        : 'wind_dir_deg',
        'cloud_cover (%)'               : 'cloud_cover_pct',
        'sunshine_duration (s)'         : 'sunshine_s',
    }
    weather = weather.rename(columns=rename_map)
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    weather = weather.set_index('datetime').sort_index()

    for col in weather.columns:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')

    weather = weather.resample('1h').mean()

    for col in ['apparent_temp_c', 'dew_point_c']:
        if col in weather.columns:
            weather = weather.drop(columns=[col])

    if 'temp_c' in weather.columns and 'humidity_pct' in weather.columns:
        T = weather['temp_c']
        H = weather['humidity_pct']
        hi = -8.78 + 1.611*T + 2.338*H - 0.1461*T*H
        weather['heat_index'] = np.where((T >= 20) & (H >= 40), hi, T)

    miss_pct  = weather.isnull().mean()
    drop_cols = miss_pct[miss_pct > MISSING_COL_THRESHOLD].index.tolist()
    if drop_cols:
        weather = weather.drop(columns=drop_cols)

    n_missing = weather.isna().sum().sum()
    if n_missing > 0:
        weather = weather.interpolate(method='time', limit_direction='both')

    return weather


# =============================================================================
#  STEP 6 — Load & Process Economic Data
# =============================================================================

def load_economic(path: str, target_years: list) -> pd.DataFrame:
    ECON_LAG = 1

    econ = pd.read_csv(path)

    useful_indicators = [
        'GDP growth (annual %)',
        'Population, total',
        'Urban population',
        'Inflation, consumer prices (annual %)',
        'Access to electricity (% of population)',
        'Electric power consumption (kWh per capita)',
        'GDP per capita (current US$)',
        'Industry (including construction), value added (% of GDP)',
        'Energy imports, net (% of energy use)',
    ]

    econ_filtered = econ[econ['Indicator Name'].isin(useful_indicators)].copy()

    year_cols       = [str(y) for y in range(1960, 2026)]
    available_years = [c for c in year_cols if c in econ_filtered.columns]

    econ_wide = (
        econ_filtered
        .set_index('Indicator Name')[available_years]
        .T
        .reset_index()
        .rename(columns={'index': 'year'})
    )
    econ_wide['year'] = econ_wide['year'].astype(int)
    econ_wide = econ_wide[econ_wide['year'].isin(target_years)].copy()

    econ_wide = econ_wide.sort_values('year')
    econ_wide[useful_indicators] = econ_wide[useful_indicators].ffill().bfill()

    if ECON_LAG > 0:
        econ_wide['year'] = econ_wide['year'] + ECON_LAG
        min_econ_year = econ_wide['year'].min()
        for fill_year in range(min(target_years), min_econ_year):
            fill_row         = econ_wide[econ_wide['year'] == min_econ_year].copy()
            fill_row['year'] = fill_year
            econ_wide        = pd.concat([fill_row, econ_wide], ignore_index=True)
        econ_wide = econ_wide.sort_values('year').reset_index(drop=True)

    col_map = {
        'GDP growth (annual %)'                                     : 'econ_gdp_growth',
        'Population, total'                                         : 'econ_population',
        'Urban population'                                          : 'econ_urban_pop',
        'Inflation, consumer prices (annual %)'                     : 'econ_inflation',
        'Access to electricity (% of population)'                   : 'econ_elec_access',
        'Electric power consumption (kWh per capita)'               : 'econ_kwh_per_capita',
        'GDP per capita (current US$)'                              : 'econ_gdp_per_capita',
        'Industry (including construction), value added (% of GDP)' : 'econ_industry_pct_gdp',
        'Energy imports, net (% of energy use)'                     : 'econ_energy_imports_pct',
    }
    econ_wide = econ_wide.rename(columns=col_map)

    return econ_wide


# =============================================================================
#  STEP 7 — Merge All Datasets + Feature Engineering
# =============================================================================

def merge_and_engineer(pgcb_df: pd.DataFrame,
                        demand_clean: pd.Series,
                        weather: pd.DataFrame,
                        econ: pd.DataFrame,
                        holiday_index: pd.DatetimeIndex,
                        ramadan_index: pd.DatetimeIndex) -> pd.DataFrame:

    master = pd.DataFrame({'demand_mw': demand_clean})
    master = master[master.index >= '2015-01-01']

    pgcb_features = ['generation_mw', 'load_shedding', 'gas', 'liquid_fuel',
                     'coal', 'hydro', 'solar', 'india_bheramara_hvdc',
                     'india_tripura', 'india_adani']
    pgcb_avail = [c for c in pgcb_features if c in pgcb_df.columns]
    master = master.join(pgcb_df[pgcb_avail], how='left')

    gen_t = master['generation_mw'].replace(0, np.nan)

    master['gas_share_t']          = (master['gas']          / gen_t * 100).clip(0, 100)
    master['coal_share_t']         = (master['coal']         / gen_t * 100).clip(0, 100)
    master['lf_share_t']           = (master['liquid_fuel']  / gen_t * 100).clip(0, 100)
    master['india_import_total_t'] = (
        master.get('india_bheramara_hvdc', pd.Series(0, index=master.index)).fillna(0) +
        master.get('india_tripura',        pd.Series(0, index=master.index)).fillna(0) +
        master.get('india_adani',          pd.Series(0, index=master.index)).fillna(0)
    )
    master['india_import_share_t'] = (
        master['india_import_total_t'] / gen_t * 100
    ).clip(0, 30)

    for col in ['gas_share_t', 'coal_share_t', 'lf_share_t',
                'india_import_total_t', 'india_import_share_t']:
        master[col.replace('_t', '_lag1h')] = master[col].shift(1)
        master = master.drop(columns=[col])

    master['shedding_lag_1h']  = master['load_shedding'].shift(1)
    master['shedding_roll_7d'] = (
        master['load_shedding'].shift(1).rolling(7*24, min_periods=1).mean()
    )

    drop_raw_gen = ['generation_mw', 'load_shedding', 'gas', 'liquid_fuel',
                    'coal', 'hydro', 'solar',
                    'india_bheramara_hvdc', 'india_tripura', 'india_adani']
    master = master.drop(columns=[c for c in drop_raw_gen if c in master.columns])

    master = master.join(weather, how='left')
    weather_cols = [c for c in weather.columns if c in master.columns]
    n_w_nan = master[weather_cols].isna().sum().sum()
    if n_w_nan > 0:
        master[weather_cols] = master[weather_cols].interpolate(
            method='time', limit_direction='both'
        )

    master['year'] = master.index.year
    master = master.merge(econ, on='year', how='left')
    master.index = demand_clean.index[demand_clean.index >= '2015-01-01']
    master.index.name = 'datetime'

    idx = master.index

    master['hour']        = idx.hour
    master['day_of_week'] = idx.dayofweek
    master['month']       = idx.month
    master['year']        = idx.year
    master['day_of_year'] = idx.dayofyear
    master['week_of_year']= idx.isocalendar().week.astype(int)

    master['hour_sin']    = np.sin(2 * np.pi * idx.hour  / 24)
    master['hour_cos']    = np.cos(2 * np.pi * idx.hour  / 24)
    master['month_sin']   = np.sin(2 * np.pi * idx.month / 12)
    master['month_cos']   = np.cos(2 * np.pi * idx.month / 12)
    master['dow_sin']     = np.sin(2 * np.pi * idx.dayofweek / 7)
    master['dow_cos']     = np.cos(2 * np.pi * idx.dayofweek / 7)

    master['is_evening_peak'] = idx.hour.isin(range(18, 22)).astype(int)
    master['is_day_peak']     = idx.hour.isin(range(10, 17)).astype(int)

    master['is_friday']   = (idx.dayofweek == 4).astype(int)
    master['is_saturday'] = (idx.dayofweek == 5).astype(int)
    master['is_weekend']  = (idx.dayofweek.isin([4, 5])).astype(int)

    date_only = idx.normalize()
    master['is_public_holiday'] = date_only.isin(holiday_index).astype(int)
    master['is_ramadan']        = date_only.isin(ramadan_index).astype(int)

    master['is_crisis_2022'] = (
        (idx >= '2022-04-01') & (idx <= '2022-12-31')
    ).astype(int)

    # Interaction term: hour × month (captures seasonal hour demand patterns)
    master['hour_x_month'] = idx.hour * idx.month

    # Trend feature: days since dataset start (captures grid growth over years)
    master['days_since_start'] = (idx - idx.min()).days

    demand = master['demand_mw']
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        master[f'demand_lag_{lag}h'] = demand.shift(lag)

    demand_shifted = demand.shift(1)
    for win in [3, 6, 24, 168]:
        master[f'demand_roll_mean_{win}h'] = demand_shifted.rolling(win, min_periods=1).mean()
    master['demand_roll_std_24h'] = demand_shifted.rolling(24, min_periods=1).std()
    master['demand_roll_max_24h'] = demand_shifted.rolling(24, min_periods=1).max()

    miss_pct  = master.isnull().mean()
    drop_cols = miss_pct[miss_pct > MISSING_COL_THRESHOLD].index.tolist()
    drop_cols = [c for c in drop_cols if c != 'demand_mw']
    if drop_cols:
        master = master.drop(columns=drop_cols)

    lag_cols = [c for c in master.columns if 'lag' in c or 'roll' in c]
    master   = master.dropna(subset=lag_cols)

    # Fill any remaining start-of-series lag NaN with bfill
    lag_roll_cols = [c for c in master.columns if 'lag' in c or 'roll' in c]
    master[lag_roll_cols] = master[lag_roll_cols].bfill()

    return master


# =============================================================================
#  STEP 8 — Train / Test Split (no file save)
# =============================================================================

def split_master(master: pd.DataFrame):
    train = master[master.index.year <  TEST_YEAR]
    test  = master[master.index.year == TEST_YEAR]
    return train, test


# =============================================================================
#  STEP 9 — LightGBM Feature Engineering on Master DataFrame
# =============================================================================

def build_features_from_df(master: pd.DataFrame) -> pd.DataFrame:
    """
    Additional feature engineering applied on top of the preprocessed master
    DataFrame before model training.
    """
    df = master.copy()

    # Ensure heat_index is present (may already exist from preprocessing)
    if 'heat_index' not in df.columns:
        if 'temp_c' in df.columns and 'humidity_pct' in df.columns:
            T = df['temp_c']
            H = df['humidity_pct']
            df['heat_index'] = np.where(
                (T >= 20) & (H >= 40),
                -8.78 + 1.611*T + 2.338*H - 0.1461*T*H,
                T
            )

    # Target: next-hour demand
    df['target'] = df['demand_mw'].shift(-1)

    # Drop last row (no next-hour label)
    df = df.dropna(subset=['target'])

    return df


# =============================================================================
#  STEP 10 — Train / Validation / Test Split for Model
# =============================================================================

def make_splits(df: pd.DataFrame):
    EXCLUDE = {'demand_mw', 'target', 'year'}
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    TRAIN_END = '2023-11-30 23:00:00'
    VAL_START = '2023-12-01 00:00:00'
    VAL_END   = '2023-12-31 23:00:00'
    TEST_START= '2024-01-01 00:00:00'
    TEST_END  = '2024-12-31 23:00:00'

    train = df[df.index <= TRAIN_END]
    val   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    X_train = train[feature_cols];  y_train = train['target']
    X_val   = val[feature_cols];    y_val   = val['target']
    X_test  = test[feature_cols];   y_test  = test['target']

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =============================================================================
#  STEP 11 — Train LightGBM
# =============================================================================

def train_lgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators      = 5000,
        learning_rate     = 0.02,
        num_leaves        = 127,
        max_depth         = -1,
        min_child_samples = 50,
        feature_fraction  = 0.7,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        reg_alpha         = 0.05,
        reg_lambda        = 0.1,
        n_jobs            = -1,
        random_state      = 42,
        verbose           = -1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train, y_train,
        eval_set    = [(X_val, y_val)],
        eval_metric = 'mape',
        callbacks   = callbacks,
    )

    return model


# =============================================================================
#  STEP 12 — Evaluate on 2024 Test Set
# =============================================================================

def evaluate(model, X_test, y_test, feature_cols):
    predictions = model.predict(X_test)
    y_true      = np.array(y_test)
    y_pred      = np.array(predictions)

    mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    test_index = X_test.index
    err_df = pd.DataFrame({
        'actual': y_true,
        'pred'  : y_pred,
        'hour'  : test_index.hour,
        'month' : test_index.month,
    })
    err_df['ape'] = np.abs(err_df['actual'] - err_df['pred']) / err_df['actual'] * 100

    return predictions, mape, mae, rmse, err_df


# =============================================================================
#  STEP 13 — Feature Importance Plot
# =============================================================================

def plot_feature_importance(model, feature_cols, out_dir):
    imp = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    top30 = imp.head(30)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#2196F3' if 'lag' in f or 'roll' in f
              else '#4CAF50' if any(w in f for w in ['hour','month','dow','peak','friday','crisis'])
              else '#FF9800'
              for f in top30.index]

    ax.barh(top30.index[::-1], top30.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=11)
    ax.set_title('LightGBM — Top 30 Feature Importances\n'
                 '(Blue=Lag/Roll, Green=Calendar, Orange=Weather/Economic)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return imp


# =============================================================================
#  STEP 14 — Prediction Plot (January 2024)
# =============================================================================

def plot_predictions(y_test, predictions, test_index, out_dir):
    jan = test_index.month == 1

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(test_index[jan], y_test[jan],
            color='black', lw=1.0, label='Actual', alpha=0.9)
    ax.plot(test_index[jan], predictions[jan],
            color='#2196F3', lw=1.0, label='LightGBM Predicted',
            linestyle='--', alpha=0.85)
    ax.fill_between(test_index[jan], y_test[jan], predictions[jan],
                    alpha=0.12, color='#2196F3')
    ax.set_ylabel('Demand (MW)', fontsize=11)
    ax.set_title('Actual vs Predicted Demand — January 2024', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate()
    plt.tight_layout()

    path = os.path.join(out_dir, 'predictions_jan2024.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
#  STEP 15 — Error Analysis Plot
# =============================================================================

def plot_error_analysis(err_df, mape, out_dir):
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'MAPE Error Analysis — 2024 Test Set  '
                 f'(Overall MAPE = {mape:.4f}%)',
                 fontsize=12, fontweight='bold')

    by_hour  = err_df.groupby('hour')['ape'].mean()
    by_month = err_df.groupby('month')['ape'].mean()

    axes[0].bar(by_hour.index, by_hour.values, color='#2196F3', alpha=0.8)
    axes[0].axhline(mape, color='red', linestyle='--', lw=1.2,
                    label=f'Overall {mape:.2f}%')
    axes[0].set_xlabel('Hour of Day');  axes[0].set_ylabel('MAPE (%)')
    axes[0].set_title('MAPE by Hour of Day')
    axes[0].legend();  axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(by_month.index, by_month.values, color='#4CAF50', alpha=0.8)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_names, fontsize=8)
    axes[1].axhline(mape, color='red', linestyle='--', lw=1.2,
                    label=f'Overall {mape:.2f}%')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('MAPE by Month (2024)')
    axes[1].legend();  axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'error_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
#  STEP 16 — Save Predictions CSV
# =============================================================================

def save_predictions(y_test, predictions, test_index, mape, out_dir):
    results = pd.DataFrame({
        'actual_demand_mw': y_test,
        'predicted_mw'    : predictions,
        'abs_error_mw'    : np.abs(y_test - predictions),
        'ape_pct'         : np.abs(y_test - predictions) / y_test * 100,
    }, index=test_index)
    results.index.name = 'datetime'

    path = os.path.join(out_dir, 'predictions_2024.csv')
    results.to_csv(path)

    return results


# =============================================================================
#  MAIN — Full Pipeline
# =============================================================================

def run():
    os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

    # ── Preprocessing ────────────────────────────────────────────────────────
    pgcb_df      = load_pgcb(PATH_PGCB)
    pgcb_df      = reconstruct_demand(pgcb_df)
    demand_no_outliers = detect_outliers_pyod(pgcb_df)
    demand_clean = knn_impute_demand(
        demand_no_outliers,
        n_neighbors=KNN_NEIGHBORS,
        long_gap_threshold=LONG_GAP_THRESHOLD
    )
    weather      = load_weather(PATH_WEATHER)
    target_years = list(pgcb_df.index.year.unique())
    econ         = load_economic(PATH_ECON, target_years)
    holiday_index, ramadan_index = build_bangladesh_holiday_calendar(target_years)

    master = merge_and_engineer(
        pgcb_df, demand_clean, weather, econ,
        holiday_index, ramadan_index
    )

    # ── Model preparation ────────────────────────────────────────────────────
    df = build_features_from_df(master)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = make_splits(df)

    # ── Train ────────────────────────────────────────────────────────────────
    model = train_lgbm(X_train, y_train, X_val, y_val)

    # ── Evaluate on 2024 ─────────────────────────────────────────────────────
    predictions, mape, mae, rmse, err_df = evaluate(
        model, X_test, y_test, feature_cols
    )

    # ── Plots & outputs ──────────────────────────────────────────────────────
    plot_feature_importance(model, feature_cols, PATH_OUTPUT_DIR)
    plot_predictions(np.array(y_test), predictions, X_test.index, PATH_OUTPUT_DIR)
    plot_error_analysis(err_df, mape, PATH_OUTPUT_DIR)
    save_predictions(np.array(y_test), predictions, X_test.index, mape, PATH_OUTPUT_DIR)

    # ── Final result ─────────────────────────────────────────────────────────
    print(f"MAPE (2024 Test Set): {mape:.4f} %")

    return model, predictions, mape


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, predictions, mape = run()
