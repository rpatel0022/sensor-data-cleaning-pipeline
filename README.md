# Industrial Sensor Data Cleaning Pipeline

A data cleaning pipeline built for **2 million+ rows** of household electric power consumption data. Handles missing sensor readings with time-series-aware interpolation, standardizes measurement units, detects anomalies using dual statistical methods (IQR + rolling z-score), and generates publication-quality visualizations.

Built specifically to demonstrate cleaning skills relevant to **power systems and industrial sensor data**.

## Problem

Sensor data from power monitoring systems has unique challenges that generic data cleaning doesn't handle well:
- **Missing readings** from sensor outages need time-series-aware imputation (not just mean/median)
- **Natural spikes** (oven turns on, AC kicks in) look like outliers but are normal — detection must be calibrated for the domain
- **Units vary** across systems (kW vs W, different voltage standards)
- **Scale** — 2M+ minute-by-minute readings spanning 4 years requires efficient processing

## Dataset

**Source:** [UCI Household Electric Power Consumption (Kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) — individual household electric power consumption measured every minute for ~4 years.

**Specifications:**
- **2,075,259 rows** (one per minute)
- **Date range:** December 16, 2006 to November 26, 2010
- **Delimiter:** Semicolon (`;`) — not comma
- **Missing value marker:** `?` character (not standard NaN)
- **7 measurement columns:**

| Original Column | Description | Unit |
|----------------|-------------|------|
| `Global_active_power` | Total active power consumed | kilowatt (kW) |
| `Global_reactive_power` | Total reactive power | kilowatt (kW) |
| `Voltage` | Average voltage | volt (V) |
| `Global_intensity` | Average current intensity | ampere (A) |
| `Sub_metering_1` | Kitchen (dishwasher, oven, microwave) | watt-hour (Wh) |
| `Sub_metering_2` | Laundry (washing machine, dryer, light) | watt-hour (Wh) |
| `Sub_metering_3` | Electric water heater + AC | watt-hour (Wh) |

## Pipeline Architecture

```
Raw .txt file ──> Parse ──> Handle Missing ──> Standardize Units ──> Detect Anomalies ──> Visualize
    (;-delimited)   │           │                    │                      │                │
    (? = NA)        ▼           ▼                    ▼                      ▼                ▼
               DateTime     Interpolate         Rename cols           IQR + Rolling       4 PNG
               index        / ffill /            Add units            Z-Score              plots
               2M+ rows     flag                 Derive kW→W          Flag anomalies
```

### Step 1: Load & Parse

- Auto-detects semicolon delimiter (tries `;` first, falls back to `,`)
- Recognizes `?` as missing value marker
- Combines `Date` + `Time` columns into a `datetime` index using `dd/mm/yyyy HH:MM:SS` format
- Converts all measurement columns to numeric (handles any remaining string values)
- Sets datetime as index and sorts chronologically
- **Result:** 2,075,259 rows, 7 numeric columns, datetime-indexed

### Step 2: Handle Missing Values (181,853 total)

Three strategies based on gap size, chosen specifically for time-series sensor data:

#### Strategy 1: Linear Interpolation (for <5% missing)
```python
df[col].interpolate(method="linear", limit=10)
```
- **Why linear?** Power consumption changes smoothly minute-to-minute — linear interpolation between known points is the most reasonable assumption
- **Why `limit=10`?** Prevents interpolating across long outages (>10 consecutive minutes) where true values are unknowable. Without this limit, you'd draw a straight line across multi-hour gaps, creating misleading data.

#### Strategy 2: Forward-Fill + Back-Fill (for 5-20% missing)
```python
df[col].ffill(limit=30).bfill(limit=30)
```
- **Why forward-fill?** Assumes the sensor reading stays roughly constant over short periods — reasonable for power data where appliances run in steady state
- **Why `limit=30`?** Caps propagation at 30 minutes. Beyond that, the assumption of "same as last reading" breaks down.

#### Strategy 3: Flag Only (for >20% missing)
```python
df[f"{col}_missing_flag"] = df[col].isna().astype(int)
```
- **Why not impute?** Inventing data for >20% of readings would be misleading. Better to create a binary flag column so downstream analysis can decide how to handle these periods.

**Result for this dataset:** All 7 columns had 1.25% missing → all handled with linear interpolation.

### Step 3: Unit Standardization

Renames columns to include units for clarity and adds derived measurements:

| Before | After | Unit |
|--------|-------|------|
| `Global_active_power` | `active_power_kw` | kW |
| `Global_reactive_power` | `reactive_power_kvar` | kVAR |
| `Voltage` | `voltage_v` | V |
| `Global_intensity` | `current_a` | A |
| `Sub_metering_1` | `sub_meter_1_wh` | Wh |
| `Sub_metering_2` | `sub_meter_2_wh` | Wh |
| `Sub_metering_3` | `sub_meter_3_wh` | Wh |

**Derived columns:**
- `active_power_w` = `active_power_kw × 1000` (kW to W conversion for comparability with sub-meter Wh readings)

**Voltage standard detection:** Computes median voltage. If >200V → EU 230V standard. If <200V → US 120V standard. This dataset: **EU 230V** (median ~240V).

### Step 4: Anomaly Detection (512,925 anomalies flagged)

Two complementary statistical methods, each catching different anomaly types:

#### Method 1: IQR (Inter-Quartile Range)
```
lower_bound = Q1 - 3 × IQR
upper_bound = Q3 + 3 × IQR
```

- Catches **global outliers** — values that are extreme relative to the entire dataset
- **Why 3x instead of the standard 1.5x?** Sensor data has natural spikes. When you turn on an oven, active power jumps from 0.5kW to 5kW — that's normal usage, not an error. With 1.5x IQR, ~25% of readings would be flagged. With 3x, only truly extreme values get caught (~4%).

#### Method 2: Rolling Z-Score
```
z = |value - rolling_mean(60)| / rolling_std(60)
anomaly if z > 3
```

- Catches **sudden local deviations** — readings that are normal globally but abnormal relative to their recent context
- **Window = 60 minutes:** Captures hourly usage patterns without being thrown off by day/night cycles (a 24-hour window would miss evening spikes because daytime is so different)
- **Threshold = 3:** Corresponds to ~0.3% false positive rate under normal distribution. Standard choice for anomaly detection.
- **`min_periods=10`:** Requires at least 10 valid readings in the window before computing z-scores. Prevents unreliable statistics from sparse windows.

#### Why Both Methods?

| Scenario | IQR catches it? | Rolling Z catches it? |
|----------|------------------|-----------------------|
| Value of 50kW (meter malfunction) | Yes (way outside global range) | Yes (extreme locally too) |
| Sudden spike from 1kW to 8kW in 1 minute | Maybe (8kW is high but possible) | Yes (local deviation is huge) |
| Gradual drift to 0.001kW over hours (dying sensor) | No (still within global range) | Yes (deviation from recent norm) |
| Consistently high readings at 6kW (heavy user) | Yes (above global Q3+3×IQR) | No (stable locally) |

Combined detection gives better coverage than either alone.

**Results by column:**

| Measurement | IQR Anomalies | Rolling Z-Score | Total | % of Data |
|-------------|---------------|-----------------|-------|-----------|
| active_power_kw | 14,306 | 75,698 | 87,289 | 4.21% |
| reactive_power_kvar | 3,886 | 20,440 | 22,982 | 1.11% |
| voltage_v | 810 | 22,838 | 23,565 | 1.14% |
| current_a | 16,523 | 74,582 | 87,796 | 4.23% |
| sub_meter_2_wh | 64,414 | 76,247 | 130,758 | 6.30% |
| sub_meter_3_wh | 0 | 73,246 | 73,246 | 3.53% |

### Step 5: Visualizations (4 plots)

#### 1. Distribution Overview (`distributions.png`)
Histograms + box plots for the 3 primary measurements (active power, reactive power, voltage). Shows the right-skewed nature of power consumption and the tight distribution of voltage around 240V.

#### 2. Time Series with Anomaly Markers (`timeseries_anomalies.png`)
4-year time series for 3 key measurements with red dots marking detected anomalies. Shows seasonal patterns (higher consumption in winter) and where anomalies cluster.

#### 3. Missing Data Heatmap (`missing_data_heatmap.png`)
Daily resampled missing data pattern across all columns. Reveals that missing values cluster on specific days (complete sensor outages) rather than being randomly distributed.

#### 4. Anomaly Detection Comparison (`anomaly_summary.png`)
Side-by-side bar chart comparing IQR vs rolling z-score detections per column. Shows that rolling z-score catches significantly more anomalies — local deviations are more common than global extremes in sensor data.

## Results

| Metric | Value |
|--------|-------|
| Rows processed | 2,075,259 |
| Date range | Dec 2006 -- Nov 2010 (nearly 4 years) |
| Missing values in raw data | 181,853 |
| Interpolation method | Linear (limit=10) |
| Missing values remaining | 180,229 (long outage gaps) |
| Anomalies flagged | 512,925 |
| Voltage standard detected | EU 230V |
| Unit conversions applied | 7 renames + 1 derived column |
| Visualizations generated | 4 plots |

## Project Structure

```
project3_sensor_data_pipeline/
├── pipeline.py              # Main pipeline — run this
├── anomaly_detector.py      # Standalone anomaly detection module
├── visualizations.py        # Standalone visualization functions
├── requirements.txt         # Python dependencies
├── data/                    # Raw Kaggle data (not committed)
├── output/
│   ├── cleaned_power_data.csv          # Full cleaned dataset with anomaly flags
│   ├── sample_cleaned_power_data.csv   # First 200 rows for quick inspection
│   ├── anomaly_report.json            # Detailed anomaly statistics
│   ├── distributions.png              # Histograms and box plots
│   ├── timeseries_anomalies.png       # Time series with anomaly markers
│   ├── missing_data_heatmap.png       # Missing data patterns
│   └── anomaly_summary.png           # Detection method comparison
└── docs/
    └── index.html           # GitHub Pages dashboard (embeds all plots)
```

### File Details

**`pipeline.py`** (main): Orchestrates all 5 steps. Each step is a pure function that takes a DataFrame and returns a modified DataFrame + report dict.

**`anomaly_detector.py`** (reusable module): Standalone anomaly detection functions that can be imported independently:
```python
from anomaly_detector import combined_detector
mask, details = combined_detector(df["voltage_v"], iqr_mult=3.0, z_window=60)
print(f"Anomalies: {details['combined_count']} ({details['anomaly_pct']}%)")
```

**`visualizations.py`** (reusable module): Standalone plotting functions for before/after distributions, anomaly timelines, and correlation heatmaps.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset from Kaggle
kaggle datasets download -d uciml/electric-power-consumption-data-set --unzip -p data/

# 3. Run the pipeline (takes ~2 minutes for 2M rows)
python pipeline.py
```

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, datetime parsing, resampling, rolling windows |
| `numpy` | Numeric operations, quantile computation, boolean masking |
| `matplotlib` | Time series plots, histograms, bar charts |
| `seaborn` | Heatmaps, statistical plot styling |
| `scikit-learn` | Available for additional analysis (StandardScaler, etc.) |

## Configuration

Key parameters that can be tuned for different sensor data:

| Parameter | Default | Where | Purpose |
|-----------|---------|-------|---------|
| IQR multiplier | 3.0 | `detect_anomalies()` | Higher = less sensitive (3x for sensors, 1.5x for transactions) |
| Rolling window | 60 | `detect_anomalies()` | Minutes of context for z-score. Match to data's natural cycle. |
| Z-score threshold | 3.0 | `detect_anomalies()` | Standard deviations. 2=aggressive, 3=balanced, 4=conservative |
| Interpolation limit | 10 | `handle_missing_values()` | Max consecutive NaNs to interpolate across |
| Forward-fill limit | 30 | `handle_missing_values()` | Max minutes to propagate last known value |
| Missing % thresholds | 5%, 20% | `handle_missing_values()` | Boundaries between interpolation/ffill/flag strategies |
