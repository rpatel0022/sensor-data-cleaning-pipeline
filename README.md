# Industrial Sensor Data Cleaning Pipeline

Cleans 2M+ rows of household electric power consumption data: handles missing readings, detects anomalies, standardizes units, and produces visualizations.

## What it does

1. **Loads time-series data** (semicolon-delimited, mixed formats, `?` as NA)
2. **Handles missing values**: linear interpolation for small gaps (<5%), forward/backward fill for medium gaps, flagging for large gaps
3. **Standardizes units**: renames columns to include units (kW, V, A, Wh), adds kW-to-W conversion, detects voltage standard (EU 230V)
4. **Anomaly detection**: dual-method approach (IQR + rolling z-score)
5. **Generates 4 visualizations**: distributions, time series with anomaly highlights, missing data heatmap, anomaly summary

## Results

| Metric | Value |
|--------|-------|
| Rows processed | 2,075,259 |
| Date range | Dec 2006 -- Nov 2010 |
| Missing values handled | 181,853 |
| Anomalies flagged | 512,925 |
| Voltage standard detected | EU 230V |
| Plots generated | 4 |

## Key design decisions

- **Linear interpolation for <5% missing**: power consumption changes smoothly minute-to-minute; `limit=10` prevents interpolating across long outages
- **Forward-fill for 5-20% missing**: assumes steady-state operation (reasonable for appliances running continuously)
- **3x IQR instead of 1.5x**: sensor data has natural spikes (oven turns on) that aren't errors -- 1.5x would flag too much normal usage
- **Rolling z-score window=60**: captures hourly patterns without being thrown off by day/night cycles
- **Dual detection**: IQR catches global outliers, rolling z-score catches sudden local deviations -- combined gives better coverage than either alone

## Run it

```bash
pip install -r requirements.txt

# Download data from Kaggle:
# kaggle datasets download -d uciml/electric-power-consumption-data-set --unzip -p data/

python pipeline.py
```

## Output

- `output/cleaned_power_data.csv` -- cleaned dataset with anomaly flags
- `output/sample_cleaned_power_data.csv` -- first 200 rows for quick inspection
- `output/anomaly_report.json` -- detailed anomaly statistics
- `output/distributions.png` -- histograms and box plots
- `output/timeseries_anomalies.png` -- time series with red anomaly markers
- `output/missing_data_heatmap.png` -- missing data patterns over time
- `output/anomaly_summary.png` -- anomaly counts by detection method

## Tech

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
