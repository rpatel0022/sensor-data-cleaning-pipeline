"""
Visualization Module
====================
Standalone visualization functions for sensor data analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")


def plot_before_after_distributions(df_before, df_after, columns, output_dir):
    """Plot distributions before and after cleaning for comparison."""
    n = min(len(columns), 4)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(columns[:n]):
        if col in df_before.columns:
            axes[i, 0].hist(df_before[col].dropna(), bins=50, color="coral", alpha=0.7, edgecolor="white")
            axes[i, 0].set_title(f"BEFORE: {col}", fontsize=11)
            axes[i, 0].set_ylabel("Count")

        if col in df_after.columns:
            axes[i, 1].hist(df_after[col].dropna(), bins=50, color="steelblue", alpha=0.7, edgecolor="white")
            axes[i, 1].set_title(f"AFTER: {col}", fontsize=11)

    plt.suptitle("Before vs After Cleaning", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "before_after_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_anomaly_timeline(df, col, anomaly_col, output_dir):
    """Plot a time series with anomalies highlighted."""
    fig, ax = plt.subplots(figsize=(15, 5))

    sample = df.iloc[::max(1, len(df) // 20000)]
    ax.plot(sample.index, sample[col], linewidth=0.5, color="steelblue", alpha=0.7)

    if anomaly_col in sample.columns:
        anomalies = sample[sample[anomaly_col] == True]
        ax.scatter(anomalies.index, anomalies[col], color="red", s=8, alpha=0.6, zorder=5, label="Anomaly")

    ax.set_title(f"Anomaly Detection: {col}", fontsize=14)
    ax.set_ylabel(col)
    ax.legend()

    if isinstance(df.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(output_dir, f"anomaly_timeline_{col}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_correlation_heatmap(df, numeric_cols, output_dir):
    """Plot correlation matrix of numeric columns."""
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
