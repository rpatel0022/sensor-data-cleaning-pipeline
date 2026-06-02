# Industrial Sensor Data Cleaning Pipeline

Cleans 2M+ rows of household electric power consumption data: handles missing readings, detects anomalies, standardizes units, and produces visualizations.

**Interview story:** "Since AMETEK builds power systems, I built a pipeline for cleaning industrial sensor data -- handling missing readings, detecting anomalies, and standardizing units. This shows I understand the domain."

## What it does

1. **Loads time-series data** (semicolon-delimited, mixed formats, `?` as NA)
2. **Handles missing values**: linear interpolation for small gaps, forward/backward fill for medium gaps, flagging for large gaps
3. **Standardizes units**: renames columns to include units (kW, V, A, Wh), adds kW-to-W conversion, detects voltage standard (EU 230V)
4. **Anomaly detection**: dual-method approach
   - IQR method (3x multiplier for sensor data)
   - Rolling z-score (window=60, catches temporal spikes)
5. **Generates 4 visualizations**: distributions, time series with anomaly highlights, missing data heatmap, anomaly summary by method

## Results

| Metric | Value |
|--------|-------|
| Rows processed | 2,075,259 |
| Date range | Dec 2006 - Nov 2010 |
| Missing values handled | 181,853 |
| Anomalies flagged | 512,925 |
| Voltage standard detected | EU 230V |
| Plots generated | 4 |

## Run it

```bash
# Download data from Kaggle first:
# kaggle datasets download -d uciml/electric-power-consumption-data-set --unzip -p data/

python pipeline.py
```

## Output

- `output/cleaned_power_data.csv` -- cleaned dataset with anomaly flags
- `output/anomaly_report.json` -- detailed anomaly statistics
- `output/distributions.png` -- histograms and box plots
- `output/timeseries_anomalies.png` -- time series with red anomaly markers
- `output/missing_data_heatmap.png` -- missing data patterns over time
- `output/anomaly_summary.png` -- anomaly counts by detection method

## Tech

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
