"""
Anomaly Detector Module
=======================
Statistical anomaly detection methods for time-series sensor data.
Can be used standalone or as part of the pipeline.
"""

import pandas as pd
import numpy as np


def iqr_detector(series, multiplier=1.5):
    """Detect outliers using IQR method.

    Args:
        series: pandas Series of numeric values
        multiplier: IQR multiplier (1.5 = standard, 3.0 = conservative)

    Returns:
        Boolean mask where True = anomaly
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def rolling_zscore_detector(series, window=60, threshold=3.0, min_periods=10):
    """Detect anomalies using rolling z-score.

    Computes z-score relative to a rolling window, catching sudden
    spikes or drops in otherwise stable sensor readings.

    Args:
        series: pandas Series of numeric values
        window: rolling window size
        threshold: z-score threshold for anomaly
        min_periods: minimum observations in window

    Returns:
        Boolean mask where True = anomaly
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z_scores = ((series - rolling_mean) / rolling_std).abs()
    return z_scores > threshold


def combined_detector(series, iqr_mult=3.0, z_window=60, z_threshold=3.0):
    """Combine IQR and rolling z-score detection.

    Flags a point as anomalous if EITHER method detects it.

    Returns:
        Tuple of (anomaly_mask, details_dict)
    """
    iqr_mask = iqr_detector(series, multiplier=iqr_mult)
    z_mask = rolling_zscore_detector(series, window=z_window, threshold=z_threshold)
    combined = iqr_mask | z_mask

    details = {
        "iqr_count": int(iqr_mask.sum()),
        "zscore_count": int(z_mask.sum()),
        "combined_count": int(combined.sum()),
        "total_points": len(series),
        "anomaly_pct": round(combined.sum() / len(series) * 100, 3),
    }

    return combined, details
