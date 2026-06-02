"""
Industrial Sensor Data Cleaning Pipeline
==========================================
Cleans time-series power consumption data: handles missing readings,
detects anomalies, standardizes units, and produces visualizations.

Pipeline steps:
1. Load power consumption data, parse dates, handle mixed delimiters
2. Detect & handle missing values (forward-fill, interpolation, flagging)
3. Unit standardization (kW vs W, voltage formats)
4. Anomaly detection: IQR + rolling z-score
5. Generate visualizations and anomaly report
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load & Parse
# ---------------------------------------------------------------------------
def load_power_data(data_dir):
    """Load the UCI Household Electric Power Consumption dataset."""
    # The dataset uses ';' as separator and '?' for missing values
    data_file = None
    for f in os.listdir(data_dir):
        if f.endswith(".txt") or f.endswith(".csv"):
            data_file = os.path.join(data_dir, f)
            break

    if data_file is None:
        raise FileNotFoundError(f"No data file found in {data_dir}")

    print(f"  Loading {os.path.basename(data_file)}...")

    # Try semicolon separator first (UCI format), fall back to comma
    try:
        df = pd.read_csv(data_file, sep=";", low_memory=False, na_values=["?", ""])
    except Exception:
        df = pd.read_csv(data_file, low_memory=False, na_values=["?", ""])

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Parse datetime
    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df = df.drop(columns=["Date", "Time"])
        df = df.set_index("datetime").sort_index()
        df = df[df.index.notna()]
    elif any("date" in c.lower() or "time" in c.lower() for c in df.columns):
        # Try to find and parse date columns generically
        date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df = df[df.index.notna()]

    # Convert all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Step 2: Handle Missing Values
# ---------------------------------------------------------------------------
def handle_missing_values(df):
    """Handle missing sensor readings with appropriate strategies."""
    missing_report = {}

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        pct = n_missing / len(df) * 100
        missing_report[col] = {
            "count": int(n_missing),
            "pct": round(pct, 2),
        }

        if pct < 5:
            # Small gaps: linear interpolation (best for continuous sensor data)
            df[col] = df[col].interpolate(method="linear", limit=10)
            missing_report[col]["method"] = "linear_interpolation"
        elif pct < 20:
            # Medium gaps: forward fill then backward fill
            df[col] = df[col].ffill(limit=30).bfill(limit=30)
            missing_report[col]["method"] = "forward_backward_fill"
        else:
            # Large gaps: flag but don't impute (data may be unreliable)
            df[f"{col}_missing_flag"] = df[col].isna().astype(int)
            missing_report[col]["method"] = "flagged_only"

        missing_report[col]["remaining_nulls"] = int(df[col].isna().sum())

    return df, missing_report


# ---------------------------------------------------------------------------
# Step 3: Unit Standardization
# ---------------------------------------------------------------------------
def standardize_units(df):
    """Standardize power measurements to consistent units."""
    unit_changes = []

    # Map of known column names to standard names and units
    column_standards = {
        "Global_active_power": {"name": "active_power_kw", "unit": "kW", "factor": 1.0},
        "Global_reactive_power": {"name": "reactive_power_kvar", "unit": "kVAR", "factor": 1.0},
        "Voltage": {"name": "voltage_v", "unit": "V", "factor": 1.0},
        "Global_intensity": {"name": "current_a", "unit": "A", "factor": 1.0},
        "Sub_metering_1": {"name": "sub_meter_1_wh", "unit": "Wh", "factor": 1.0},
        "Sub_metering_2": {"name": "sub_meter_2_wh", "unit": "Wh", "factor": 1.0},
        "Sub_metering_3": {"name": "sub_meter_3_wh", "unit": "Wh", "factor": 1.0},
    }

    rename_map = {}
    for old_name, spec in column_standards.items():
        if old_name in df.columns:
            # Apply conversion factor if needed
            if spec["factor"] != 1.0:
                df[old_name] = df[old_name] * spec["factor"]
                unit_changes.append(f"{old_name}: applied factor {spec['factor']}")
            rename_map[old_name] = spec["name"]
            unit_changes.append(f"{old_name} -> {spec['name']} ({spec['unit']})")

    df = df.rename(columns=rename_map)

    # Add derived columns
    if "active_power_kw" in df.columns:
        df["active_power_w"] = df["active_power_kw"] * 1000  # kW to W conversion
        unit_changes.append("Added active_power_w (kW * 1000)")

    # Voltage sanity check: typical household is 220-240V (EU) or 110-120V (US)
    if "voltage_v" in df.columns:
        median_v = df["voltage_v"].median()
        if median_v > 200:
            df["voltage_standard"] = "EU_230V"
        else:
            df["voltage_standard"] = "US_120V"
        unit_changes.append(f"Detected voltage standard: {'EU 230V' if median_v > 200 else 'US 120V'}")

    return df, unit_changes


# ---------------------------------------------------------------------------
# Step 4: Anomaly Detection
# ---------------------------------------------------------------------------
def detect_anomalies(df, numeric_cols=None):
    """Detect anomalies using IQR and rolling z-score methods."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    anomaly_report = {}
    df["anomaly_flag"] = False

    for col in numeric_cols:
        if df[col].isna().all():
            continue

        # Method 1: IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 3 * iqr  # Using 3x IQR for sensor data (less aggressive)
        upper = q3 + 3 * iqr
        iqr_anomalies = (df[col] < lower) | (df[col] > upper)

        # Method 2: Rolling z-score (captures temporal anomalies)
        if hasattr(df.index, 'freq') or isinstance(df.index, pd.DatetimeIndex):
            rolling_mean = df[col].rolling(window=60, min_periods=10).mean()
            rolling_std = df[col].rolling(window=60, min_periods=10).std()
            z_scores = ((df[col] - rolling_mean) / rolling_std).abs()
            rolling_anomalies = z_scores > 3
        else:
            rolling_anomalies = pd.Series(False, index=df.index)

        # Combine: flag if either method detects anomaly
        combined = iqr_anomalies | rolling_anomalies
        df[f"{col}_anomaly"] = combined

        n_anomalies = combined.sum()
        anomaly_report[col] = {
            "iqr_anomalies": int(iqr_anomalies.sum()),
            "rolling_z_anomalies": int(rolling_anomalies.sum()),
            "total_anomalies": int(n_anomalies),
            "pct": round(n_anomalies / len(df) * 100, 3),
            "bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
        }

        # Update global anomaly flag
        df["anomaly_flag"] = df["anomaly_flag"] | combined

    return df, anomaly_report


# ---------------------------------------------------------------------------
# Step 5: Visualizations
# ---------------------------------------------------------------------------
def generate_visualizations(df, anomaly_report):
    """Generate before/after plots and anomaly highlights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    sns.set_style("whitegrid")
    plot_paths = []

    # Get numeric columns (excluding flags)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.endswith("_anomaly") and c != "anomaly_flag"
                    and not c.endswith("_missing_flag")]

    # --- Plot 1: Distribution overview ---
    n_cols = min(len(numeric_cols), 6)
    if n_cols > 0:
        fig, axes = plt.subplots(2, min(n_cols, 3), figsize=(15, 8))
        if n_cols <= 3:
            axes = axes.reshape(2, -1)

        for i, col in enumerate(numeric_cols[:min(n_cols, 3)]):
            ax_top = axes[0, i]
            ax_bot = axes[1, i]

            # Histogram
            data = df[col].dropna()
            ax_top.hist(data, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
            ax_top.set_title(f"{col}\n(distribution)", fontsize=10)
            ax_top.set_ylabel("Count")

            # Box plot
            ax_bot.boxplot(data, vert=False)
            ax_bot.set_title(f"{col}\n(box plot)", fontsize=10)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)
        print(f"  Saved: {os.path.basename(path)}")

    # --- Plot 2: Time series with anomalies ---
    if isinstance(df.index, pd.DatetimeIndex) and len(numeric_cols) > 0:
        # Sample for plotting (full dataset too large)
        sample_size = min(50000, len(df))
        df_sample = df.iloc[::max(1, len(df) // sample_size)]

        fig, axes = plt.subplots(min(len(numeric_cols), 3), 1, figsize=(15, 4 * min(len(numeric_cols), 3)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, col in enumerate(numeric_cols[:3]):
            ax = axes[i]
            ax.plot(df_sample.index, df_sample[col], linewidth=0.5, color="steelblue", alpha=0.7, label=col)

            # Highlight anomalies
            anomaly_col = f"{col}_anomaly"
            if anomaly_col in df_sample.columns:
                anomalies = df_sample[df_sample[anomaly_col] == True]
                if len(anomalies) > 0:
                    ax.scatter(anomalies.index, anomalies[col], color="red", s=5, alpha=0.5, label="anomaly", zorder=5)

            ax.set_ylabel(col, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.xlabel("Date")
        plt.suptitle("Time Series with Anomaly Detection", fontsize=14, y=1.01)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "timeseries_anomalies.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)
        print(f"  Saved: {os.path.basename(path)}")

    # --- Plot 3: Missing data heatmap ---
    fig, ax = plt.subplots(figsize=(12, 4))
    missing_data = df[numeric_cols].isna()

    if isinstance(df.index, pd.DatetimeIndex):
        # Resample to daily for visualization
        daily_missing = missing_data.resample("D").mean() * 100
        sns.heatmap(daily_missing.T, cmap="YlOrRd", ax=ax, cbar_kws={"label": "% Missing"})
        ax.set_title("Daily Missing Data Pattern", fontsize=14)
        ax.set_xlabel("Date")
    else:
        # Show first 500 rows
        sns.heatmap(missing_data.head(500).T, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Missing"})
        ax.set_title("Missing Data Pattern (first 500 rows)", fontsize=14)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "missing_data_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path)
    print(f"  Saved: {os.path.basename(path)}")

    # --- Plot 4: Anomaly summary bar chart ---
    if anomaly_report:
        fig, ax = plt.subplots(figsize=(10, 5))
        cols = list(anomaly_report.keys())
        iqr_vals = [anomaly_report[c]["iqr_anomalies"] for c in cols]
        rolling_vals = [anomaly_report[c]["rolling_z_anomalies"] for c in cols]

        x = np.arange(len(cols))
        width = 0.35
        ax.bar(x - width/2, iqr_vals, width, label="IQR Method", color="steelblue")
        ax.bar(x + width/2, rolling_vals, width, label="Rolling Z-Score", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
        ax.set_ylabel("Anomalies Detected")
        ax.set_title("Anomaly Detection Results by Method", fontsize=14)
        ax.legend()

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "anomaly_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)
        print(f"  Saved: {os.path.basename(path)}")

    return plot_paths


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("INDUSTRIAL SENSOR DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading power consumption data...")
    df = load_power_data(DATA_DIR)
    rows_before = len(df)
    nulls_before = int(df.isna().sum().sum())
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "")
    print(f"  Total null values: {nulls_before:,}")

    # Step 2: Missing values
    print("\n[2/5] Handling missing values...")
    df, missing_report = handle_missing_values(df)
    for col, info in missing_report.items():
        print(f"  {col}: {info['count']:,} missing ({info['pct']}%) -> {info['method']}, {info['remaining_nulls']} remaining")

    # Step 3: Standardize units
    print("\n[3/5] Standardizing units...")
    df, unit_changes = standardize_units(df)
    for change in unit_changes:
        print(f"  {change}")

    # Step 4: Anomaly detection
    print("\n[4/5] Detecting anomalies...")
    # Only run on core measurement columns
    core_cols = [c for c in df.columns if not c.endswith(("_anomaly", "_flag", "_standard", "_missing_flag"))]
    core_numeric = [c for c in core_cols if pd.api.types.is_numeric_dtype(df[c])]
    df, anomaly_report = detect_anomalies(df, core_numeric)
    total_anomalies = sum(v["total_anomalies"] for v in anomaly_report.values())
    print(f"  Total anomalies detected: {total_anomalies:,}")
    for col, info in anomaly_report.items():
        print(f"  {col}: {info['total_anomalies']:,} anomalies ({info['pct']}%)")

    # Step 5: Visualizations
    print("\n[5/5] Generating visualizations...")
    plot_paths = generate_visualizations(df, anomaly_report)

    # Save cleaned data
    output_csv = os.path.join(OUTPUT_DIR, "cleaned_power_data.csv")
    df.to_csv(output_csv)
    print(f"\n  Saved: {output_csv}")

    # Save anomaly report — only count original measurement columns for null comparison
    original_cols = [c for c in df.columns if not c.endswith(("_anomaly", "_flag", "_standard", "_missing_flag", "_w"))]
    nulls_after = int(df[original_cols].isna().sum().sum())
    report = {
        "rows": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())] if isinstance(df.index, pd.DatetimeIndex) else None,
        "missing_values": missing_report,
        "unit_changes": unit_changes,
        "anomaly_detection": anomaly_report,
        "summary": {
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
            "null_reduction_pct": round((1 - nulls_after / max(nulls_before, 1)) * 100, 1),
            "total_anomalies": total_anomalies,
            "plots_generated": len(plot_paths),
        },
    }

    import json
    with open(os.path.join(OUTPUT_DIR, "anomaly_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: output/anomaly_report.json")

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Rows processed: {rows_before:,}")
    print(f"  Null values: {nulls_before:,} -> {nulls_after:,} ({report['summary']['null_reduction_pct']}% reduction)")
    print(f"  Anomalies flagged: {total_anomalies:,}")
    print(f"  Plots generated: {len(plot_paths)}")
    print(f"  Output files in: {OUTPUT_DIR}")

    return df, report


if __name__ == "__main__":
    run_pipeline()
