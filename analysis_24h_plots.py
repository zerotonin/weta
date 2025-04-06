#!/usr/bin/env python3
"""
analysis_24h_plots.py

This script performs 24-hour analysis of rock temperature and humidity data.
It loads a wide-format CSV file with columns:
  rock, date_time, elapsed_date_time, temperature_in_C, temperature_out_C, temperature_diff_C, humidity_perc_RH

For each rock:
  - It groups data by hour-of-day (0-23) and computes nan-aware statistics:
      • mean, median, standard error (SEM) and 95% confidence interval (via bootstrapping)
    for temperature_in_C, temperature_out_C, temperature_diff_C and humidity_perc_RH.
  - It then produces three plot variants:
      1. Mean + SEM
      2. Mean + 95% CI
      3. Median + 95% CI
    Each plot is a figure with two subplots:
      • Top subplot: a dual-y-axis plot showing inside temperature (red), outside temperature (blue)
        and temperature difference (black on secondary y-axis). A custom legend is added.
      • Bottom subplot: humidity (orange).
    The x-axis is set to run from 0 to 24.
    Figures are saved in both PNG and SVG formats under the folder:
      /home/geuba03p/weta_project/burrow_temperature/figures/24h_perRock/
      
Next, it collates all hourly per-rock stats into a single CSV file at:
  /home/geuba03p/weta_project/burrow_temperature/data/24h_hourly_averages.csv

Finally, it aggregates the hourly stats across all rocks (by taking the per‐rock means/medians)
and produces corresponding across-rock plots (saved under
  /home/geuba03p/weta_project/burrow_temperature/figures/24h_comparisson/).

Note: All plotting is done without grid lines.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Parameters and Paths ---
N_BOOTSTRAP = 1000        # Number of bootstrap resamples for confidence intervals
CONF_LEVEL = 0.95         # Confidence level for bootstrapped CIs

DATA_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv"
OUTPUT_DATA_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/24h_hourly_averages.csv"
FIG_DIR_PER_ROCK = "/home/geuba03p/weta_project/burrow_temperature/figures/24h_perRock"
FIG_DIR_ACROSS = "/home/geuba03p/weta_project/burrow_temperature/figures/24h_comparisson"

# Create output directories if they do not exist
os.makedirs(FIG_DIR_PER_ROCK, exist_ok=True)
os.makedirs(FIG_DIR_ACROSS, exist_ok=True)

# --- Helper Functions ---

def bootstrap_ci(data, stat_func=np.nanmean, n_resamples=N_BOOTSTRAP, confidence=CONF_LEVEL):
    """
    Computes a bootstrap confidence interval for a given statistic.
    Uses only non-NaN values.
    Returns a tuple (lower_bound, upper_bound).
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return (np.nan, np.nan)
    if data.size == 1:
        return (data[0], data[0])
    reps = np.empty(n_resamples)
    n = data.size
    for i in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        reps[i] = stat_func(sample)
    alpha = 1 - confidence
    lower = np.percentile(reps, 100 * (alpha/2))
    upper = np.percentile(reps, 100 * (1 - alpha/2))
    return (lower, upper)

# --- Load Data ---
df = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
# Extract hour-of-day (0-23)
df["hour"] = df["date_time"].dt.hour

# --- Per-Rock Analysis ---
# Prepare list for per-rock hourly stats
stats_records = []  # Each record is a dict for a specific rock and hour

# Get unique rock IDs
rocks = df["rock"].unique()

for rock in rocks:
    rock_df = df[df["rock"] == rock]
    # Group by hour-of-day
    hour_groups = rock_df.groupby("hour")
    # Prepare arrays of length 24 (one for each hour)
    hours = np.arange(24)
    # Initialize arrays with NaN values
    temp_in_mean   = np.full(24, np.nan)
    temp_in_median = np.full(24, np.nan)
    temp_out_mean   = np.full(24, np.nan)
    temp_out_median = np.full(24, np.nan)
    temp_diff_mean   = np.full(24, np.nan)
    temp_diff_median = np.full(24, np.nan)
    hum_mean       = np.full(24, np.nan)
    hum_median     = np.full(24, np.nan)
    
    temp_in_sem   = np.full(24, np.nan)
    temp_out_sem  = np.full(24, np.nan)
    temp_diff_sem = np.full(24, np.nan)
    hum_sem       = np.full(24, np.nan)
    
    temp_in_mean_ci_low   = np.full(24, np.nan)
    temp_in_mean_ci_up    = np.full(24, np.nan)
    temp_out_mean_ci_low  = np.full(24, np.nan)
    temp_out_mean_ci_up   = np.full(24, np.nan)
    temp_diff_mean_ci_low = np.full(24, np.nan)
    temp_diff_mean_ci_up  = np.full(24, np.nan)
    hum_mean_ci_low       = np.full(24, np.nan)
    hum_mean_ci_up        = np.full(24, np.nan)
    
    temp_in_median_ci_low   = np.full(24, np.nan)
    temp_in_median_ci_up    = np.full(24, np.nan)
    temp_out_median_ci_low  = np.full(24, np.nan)
    temp_out_median_ci_up   = np.full(24, np.nan)
    temp_diff_median_ci_low = np.full(24, np.nan)
    temp_diff_median_ci_up  = np.full(24, np.nan)
    hum_median_ci_low       = np.full(24, np.nan)
    hum_median_ci_up        = np.full(24, np.nan)
    
    # Loop over each hour group for this rock
    for hour, grp in hour_groups:
        # Get non-NaN values for each variable
        vals_in   = grp["temperature_in_C"].values
        vals_out  = grp["temperature_out_C"].values
        vals_diff = grp["temperature_diff_C"].values
        vals_hum  = grp["humidity_perc_RH"].values
        
        # Use nanmean and nanmedian to ignore NaNs
        temp_in_mean[hour]   = np.nanmean(vals_in)
        temp_in_median[hour] = np.nanmedian(vals_in)
        temp_out_mean[hour]  = np.nanmean(vals_out)
        temp_out_median[hour] = np.nanmedian(vals_out)
        temp_diff_mean[hour] = np.nanmean(vals_diff)
        temp_diff_median[hour] = np.nanmedian(vals_diff)
        hum_mean[hour]       = np.nanmean(vals_hum)
        hum_median[hour]     = np.nanmedian(vals_hum)
        
        n = np.sum(~np.isnan(vals_in))  # Using inside as an example; you could handle each separately
        if n > 1:
            temp_in_sem[hour]   = np.nanstd(vals_in, ddof=1) / np.sqrt(n)
            temp_out_sem[hour]  = np.nanstd(vals_out, ddof=1) / np.sqrt(n)
            temp_diff_sem[hour] = np.nanstd(vals_diff, ddof=1) / np.sqrt(n)
            hum_sem[hour]       = np.nanstd(vals_hum, ddof=1) / np.sqrt(n)
        else:
            temp_in_sem[hour] = temp_out_sem[hour] = temp_diff_sem[hour] = hum_sem[hour] = 0.0
        
        # Bootstrap CIs for each statistic, using non-NaN values
        ci_low, ci_up = bootstrap_ci(vals_in, stat_func=np.nanmean)
        temp_in_mean_ci_low[hour] = ci_low; temp_in_mean_ci_up[hour] = ci_up
        ci_low, ci_up = bootstrap_ci(vals_in, stat_func=np.nanmedian)
        temp_in_median_ci_low[hour] = ci_low; temp_in_median_ci_up[hour] = ci_up
        
        ci_low, ci_up = bootstrap_ci(vals_out, stat_func=np.nanmean)
        temp_out_mean_ci_low[hour] = ci_low; temp_out_mean_ci_up[hour] = ci_up
        ci_low, ci_up = bootstrap_ci(vals_out, stat_func=np.nanmedian)
        temp_out_median_ci_low[hour] = ci_low; temp_out_median_ci_up[hour] = ci_up
        
        ci_low, ci_up = bootstrap_ci(vals_diff, stat_func=np.nanmean)
        temp_diff_mean_ci_low[hour] = ci_low; temp_diff_mean_ci_up[hour] = ci_up
        ci_low, ci_up = bootstrap_ci(vals_diff, stat_func=np.nanmedian)
        temp_diff_median_ci_low[hour] = ci_low; temp_diff_median_ci_up[hour] = ci_up
        
        ci_low, ci_up = bootstrap_ci(vals_hum, stat_func=np.nanmean)
        hum_mean_ci_low[hour] = ci_low; hum_mean_ci_up[hour] = ci_up
        ci_low, ci_up = bootstrap_ci(vals_hum, stat_func=np.nanmedian)
        hum_median_ci_low[hour] = ci_low; hum_median_ci_up[hour] = ci_up
        
        # Append stats for this rock-hour to the records list
        stats_records.append({
            "rock": rock,
            "hour": hour,
            "temp_in_mean": temp_in_mean[hour],
            "temp_in_median": temp_in_median[hour],
            "temp_in_sem": temp_in_sem[hour],
            "temp_in_mean_ci_low": temp_in_mean_ci_low[hour],
            "temp_in_mean_ci_up": temp_in_mean_ci_up[hour],
            "temp_in_median_ci_low": temp_in_median_ci_low[hour],
            "temp_in_median_ci_up": temp_in_median_ci_up[hour],
            "temp_out_mean": temp_out_mean[hour],
            "temp_out_median": temp_out_median[hour],
            "temp_out_sem": temp_out_sem[hour],
            "temp_out_mean_ci_low": temp_out_mean_ci_low[hour],
            "temp_out_mean_ci_up": temp_out_mean_ci_up[hour],
            "temp_out_median_ci_low": temp_out_median_ci_low[hour],
            "temp_out_median_ci_up": temp_out_median_ci_up[hour],
            "temp_diff_mean": temp_diff_mean[hour],
            "temp_diff_median": temp_diff_median[hour],
            "temp_diff_sem": temp_diff_sem[hour],
            "temp_diff_mean_ci_low": temp_diff_mean_ci_low[hour],
            "temp_diff_mean_ci_up": temp_diff_mean_ci_up[hour],
            "temp_diff_median_ci_low": temp_diff_median_ci_low[hour],
            "temp_diff_median_ci_up": temp_diff_median_ci_up[hour],
            "hum_mean": hum_mean[hour],
            "hum_median": hum_median[hour],
            "hum_sem": hum_sem[hour],
            "hum_mean_ci_low": hum_mean_ci_low[hour],
            "hum_mean_ci_up": hum_mean_ci_up[hour],
            "hum_median_ci_low": hum_median_ci_low[hour],
            "hum_median_ci_up": hum_median_ci_up[hour]
        })
    
    # --- Per-Rock Plotting ---
    # x-axis: hours 0 to 23
    x = hours

    # Colors for plots
    color_in = 'red'
    color_out = 'blue'
    color_diff = 'black'
    
    # Custom legend for temperature plot: inside (red), outside (blue), difference (black)
    custom_lines = [Line2D([0], [0], color=color_in, lw=2),
                    Line2D([0], [0], color=color_out, lw=2),
                    Line2D([0], [0], color=color_diff, lw=2)]
    
    # 1. Mean + SEM Plot
    fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    # Temperature subplot
    ax_temp.plot(x, temp_in_mean, color=color_in)
    ax_temp.fill_between(x, temp_in_mean - temp_in_sem, temp_in_mean + temp_in_sem, color=color_in, alpha=0.3)
    ax_temp.plot(x, temp_out_mean, color=color_out)
    ax_temp.fill_between(x, temp_out_mean - temp_out_sem, temp_out_mean + temp_out_sem, color=color_out, alpha=0.3)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_xlim(0, 24)
    # Secondary y-axis for temperature difference
    ax_temp_2 = ax_temp.twinx()
    ax_temp_2.plot(x, temp_diff_mean, color=color_diff)
    ax_temp_2.fill_between(x, temp_diff_mean - temp_diff_sem, temp_diff_mean + temp_diff_sem, color='gray', alpha=0.3)
    ax_temp_2.set_ylabel("Temp. Difference (°C)")
    # Add custom legend to top subplot
    ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
    # Humidity subplot
    ax_hum.plot(x, hum_mean, color='darkorange')
    ax_hum.fill_between(x, hum_mean - hum_sem, hum_mean + hum_sem, color='orange', alpha=0.3)
    ax_hum.set_ylabel("Humidity (% RH)")
    ax_hum.set_xlabel("Hour of day")
    ax_hum.set_xticks(range(0,24,3))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_mean_SEM.png"))
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_mean_SEM.svg"))
    plt.close(fig)

    # 2. Mean + 95% CI Plot
    fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    ax_temp.plot(x, temp_in_mean, color=color_in)
    ax_temp.fill_between(x, temp_in_mean_ci_low, temp_in_mean_ci_up, color=color_in, alpha=0.3)
    ax_temp.plot(x, temp_out_mean, color=color_out)
    ax_temp.fill_between(x, temp_out_mean_ci_low, temp_out_mean_ci_up, color=color_out, alpha=0.3)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_xlim(0, 24)
    ax_temp_2 = ax_temp.twinx()
    ax_temp_2.plot(x, temp_diff_mean, color=color_diff)
    ax_temp_2.fill_between(x, temp_diff_mean_ci_low, temp_diff_mean_ci_up, color='gray', alpha=0.3)
    ax_temp_2.set_ylabel("Temp. Difference (°C)")
    ax_hum.plot(x, hum_mean, color='darkorange')
    ax_hum.fill_between(x, hum_mean_ci_low, hum_mean_ci_up, color='orange', alpha=0.3)
    ax_hum.set_ylabel("Humidity (% RH)")
    ax_hum.set_xlabel("Hour of day")
    ax_hum.set_xticks(range(0,24,3))
    # Add legend on top subplot
    ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_mean_95CI.png"))
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_mean_95CI.svg"))
    plt.close(fig)

    # 3. Median + 95% CI Plot
    fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    ax_temp.plot(x, temp_in_median, color=color_in)
    ax_temp.fill_between(x, temp_in_median_ci_low, temp_in_median_ci_up, color=color_in, alpha=0.3)
    ax_temp.plot(x, temp_out_median, color=color_out)
    ax_temp.fill_between(x, temp_out_median_ci_low, temp_out_median_ci_up, color=color_out, alpha=0.3)
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_xlim(0, 24)
    ax_temp_2 = ax_temp.twinx()
    ax_temp_2.plot(x, temp_diff_median, color=color_diff)
    ax_temp_2.fill_between(x, temp_diff_median_ci_low, temp_diff_median_ci_up, color='gray', alpha=0.3)
    ax_temp_2.set_ylabel("Temp. Difference (°C)")
    ax_hum.plot(x, hum_median, color='darkorange')
    ax_hum.fill_between(x, hum_median_ci_low, hum_median_ci_up, color='orange', alpha=0.3)
    ax_hum.set_ylabel("Humidity (% RH)")
    ax_hum.set_xlabel("Hour of day")
    ax_hum.set_xticks(range(0,24,3))
    ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_median_95CI.png"))
    fig.savefig(os.path.join(FIG_DIR_PER_ROCK, f"rock_{rock}_median_95CI.svg"))
    plt.close(fig)

# End of per-rock loop

# --- Save Per-Rock Hourly Stats ---
stats_df = pd.DataFrame(stats_records)
stats_df.sort_values(["rock", "hour"], inplace=True)
stats_df.to_csv(OUTPUT_DATA_FILE, index=False)

# --- Aggregate Across Rocks ---
agg_records = []
for hour in range(24):
    hour_data = stats_df[stats_df["hour"] == hour]
    if hour_data.empty:
        continue
    n_rocks = hour_data["rock"].nunique()
    mean_temp_in_val = np.nanmean(hour_data["temp_in_mean"])
    mean_temp_out_val = np.nanmean(hour_data["temp_out_mean"])
    mean_temp_diff_val = np.nanmean(hour_data["temp_diff_mean"])
    mean_hum_val = np.nanmean(hour_data["hum_mean"])
    
    med_temp_in_val = np.nanmedian(hour_data["temp_in_median"])
    med_temp_out_val = np.nanmedian(hour_data["temp_out_median"])
    med_temp_diff_val = np.nanmedian(hour_data["temp_diff_median"])
    med_hum_val = np.nanmedian(hour_data["hum_median"])
    
    if n_rocks > 1:
        sem_temp_in_val = np.nanstd(hour_data["temp_in_mean"], ddof=1) / np.sqrt(n_rocks)
        sem_temp_out_val = np.nanstd(hour_data["temp_out_mean"], ddof=1) / np.sqrt(n_rocks)
        sem_temp_diff_val = np.nanstd(hour_data["temp_diff_mean"], ddof=1) / np.sqrt(n_rocks)
        sem_hum_val = np.nanstd(hour_data["hum_mean"], ddof=1) / np.sqrt(n_rocks)
    else:
        sem_temp_in_val = sem_temp_out_val = sem_temp_diff_val = sem_hum_val = 0.0
    
    # 95% CI for means using percentile of per-rock means
    ci_mean_temp_in = np.nanpercentile(hour_data["temp_in_mean"], [2.5, 97.5])
    ci_mean_temp_out = np.nanpercentile(hour_data["temp_out_mean"], [2.5, 97.5])
    ci_mean_temp_diff = np.nanpercentile(hour_data["temp_diff_mean"], [2.5, 97.5])
    ci_mean_hum = np.nanpercentile(hour_data["hum_mean"], [2.5, 97.5])
    
    # 95% CI for medians using percentile of per-rock medians
    ci_med_temp_in = np.nanpercentile(hour_data["temp_in_median"], [2.5, 97.5])
    ci_med_temp_out = np.nanpercentile(hour_data["temp_out_median"], [2.5, 97.5])
    ci_med_temp_diff = np.nanpercentile(hour_data["temp_diff_median"], [2.5, 97.5])
    ci_med_hum = np.nanpercentile(hour_data["hum_median"], [2.5, 97.5])
    
    agg_records.append({
        "hour": hour,
        "mean_temp_in": mean_temp_in_val,
        "mean_temp_out": mean_temp_out_val,
        "mean_temp_diff": mean_temp_diff_val,
        "mean_hum": mean_hum_val,
        "sem_temp_in": sem_temp_in_val,
        "sem_temp_out": sem_temp_out_val,
        "sem_temp_diff": sem_temp_diff_val,
        "sem_hum": sem_hum_val,
        "ci_mean_temp_in_low": ci_mean_temp_in[0],
        "ci_mean_temp_in_up": ci_mean_temp_in[1],
        "ci_mean_temp_out_low": ci_mean_temp_out[0],
        "ci_mean_temp_out_up": ci_mean_temp_out[1],
        "ci_mean_temp_diff_low": ci_mean_temp_diff[0],
        "ci_mean_temp_diff_up": ci_mean_temp_diff[1],
        "ci_mean_hum_low": ci_mean_hum[0],
        "ci_mean_hum_up": ci_mean_hum[1],
        "med_temp_in": med_temp_in_val,
        "med_temp_out": med_temp_out_val,
        "med_temp_diff": med_temp_diff_val,
        "med_hum": med_hum_val,
        "ci_med_temp_in_low": ci_med_temp_in[0],
        "ci_med_temp_in_up": ci_med_temp_in[1],
        "ci_med_temp_out_low": ci_med_temp_out[0],
        "ci_med_temp_out_up": ci_med_temp_out[1],
        "ci_med_temp_diff_low": ci_med_temp_diff[0],
        "ci_med_temp_diff_up": ci_med_temp_diff[1],
        "ci_med_hum_low": ci_med_hum[0],
        "ci_med_hum_up": ci_med_hum[1]
    })

agg_df = pd.DataFrame(agg_records)

# --- Plot Aggregated Across Rocks ---

x = agg_df["hour"].values
# Mean of means lines
mean_temp_in_all = agg_df["mean_temp_in"].values
mean_temp_out_all = agg_df["mean_temp_out"].values
mean_temp_diff_all = agg_df["mean_temp_diff"].values
mean_hum_all = agg_df["mean_hum"].values
# Median of medians lines
med_temp_in_all = agg_df["med_temp_in"].values
med_temp_out_all = agg_df["med_temp_out"].values
med_temp_diff_all = agg_df["med_temp_diff"].values
med_hum_all = agg_df["med_hum"].values
# SEM arrays
sem_temp_in_all = agg_df["sem_temp_in"].values
sem_temp_out_all = agg_df["sem_temp_out"].values
sem_temp_diff_all = agg_df["sem_temp_diff"].values
sem_hum_all = agg_df["sem_hum"].values
# CIs for means
ci_mean_temp_in_low_all = agg_df["ci_mean_temp_in_low"].values
ci_mean_temp_in_up_all = agg_df["ci_mean_temp_in_up"].values
ci_mean_temp_out_low_all = agg_df["ci_mean_temp_out_low"].values
ci_mean_temp_out_up_all = agg_df["ci_mean_temp_out_up"].values
ci_mean_temp_diff_low_all = agg_df["ci_mean_temp_diff_low"].values
ci_mean_temp_diff_up_all = agg_df["ci_mean_temp_diff_up"].values
ci_mean_hum_low_all = agg_df["ci_mean_hum_low"].values
ci_mean_hum_up_all = agg_df["ci_mean_hum_up"].values
# CIs for medians
ci_med_temp_in_low_all = agg_df["ci_med_temp_in_low"].values
ci_med_temp_in_up_all = agg_df["ci_med_temp_in_up"].values
ci_med_temp_out_low_all = agg_df["ci_med_temp_out_low"].values
ci_med_temp_out_up_all = agg_df["ci_med_temp_out_up"].values
ci_med_temp_diff_low_all = agg_df["ci_med_temp_diff_low"].values
ci_med_temp_diff_up_all = agg_df["ci_med_temp_diff_up"].values
ci_med_hum_low_all = agg_df["ci_med_hum_low"].values
ci_med_hum_up_all = agg_df["ci_med_hum_up"].values

# 1. Across rocks: Mean + SEM
fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax_temp.plot(x, mean_temp_in_all, color='red', label='Inside (avg of means)')
ax_temp.fill_between(x, mean_temp_in_all - sem_temp_in_all, mean_temp_in_all + sem_temp_in_all, color='red', alpha=0.3)
ax_temp.plot(x, mean_temp_out_all, color='blue', label='Outside (avg of means)')
ax_temp.fill_between(x, mean_temp_out_all - sem_temp_out_all, mean_temp_out_all + sem_temp_out_all, color='blue', alpha=0.3)
ax_temp.set_ylabel("Temperature (°C)")
ax_temp.set_xlim(0, 24)
ax_temp_2 = ax_temp.twinx()
ax_temp_2.plot(x, mean_temp_diff_all, color='black', label='Diff (avg of means)')
ax_temp_2.fill_between(x, mean_temp_diff_all - sem_temp_diff_all, mean_temp_diff_all + sem_temp_diff_all, color='gray', alpha=0.3)
ax_temp_2.set_ylabel("Temp. Difference (°C)")
ax_hum.plot(x, mean_hum_all, color='darkorange', label='Humidity (avg of means)')
ax_hum.fill_between(x, mean_hum_all - sem_hum_all, mean_hum_all + sem_hum_all, color='orange', alpha=0.3)
ax_hum.set_ylabel("Humidity (% RH)")
ax_hum.set_xlabel("Hour of day")
ax_hum.set_xticks(range(0,24,3))
# Custom legend for temperature subplot
ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_mean_SEM.png"))
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_mean_SEM.svg"))
plt.close(fig)

# 2. Across rocks: Mean + 95% CI
fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax_temp.plot(x, mean_temp_in_all, color='red', label='Inside (avg of means)')
ax_temp.fill_between(x, ci_mean_temp_in_low_all, ci_mean_temp_in_up_all, color='red', alpha=0.3)
ax_temp.plot(x, mean_temp_out_all, color='blue', label='Outside (avg of means)')
ax_temp.fill_between(x, ci_mean_temp_out_low_all, ci_mean_temp_out_up_all, color='blue', alpha=0.3)
ax_temp.set_ylabel("Temperature (°C)")
ax_temp.set_xlim(0, 24)
ax_temp_2 = ax_temp.twinx()
ax_temp_2.plot(x, mean_temp_diff_all, color='black', label='Diff (avg of means)')
ax_temp_2.fill_between(x, ci_mean_temp_diff_low_all, ci_mean_temp_diff_up_all, color='gray', alpha=0.3)
ax_temp_2.set_ylabel("Temp. Difference (°C)")
ax_hum.plot(x, mean_hum_all, color='darkorange', label='Humidity (avg of means)')
ax_hum.fill_between(x, ci_mean_hum_low_all, ci_mean_hum_up_all, color='orange', alpha=0.3)
ax_hum.set_ylabel("Humidity (% RH)")
ax_hum.set_xlabel("Hour of day")
ax_hum.set_xticks(range(0,24,3))
ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_mean_95CI.png"))
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_mean_95CI.svg"))
plt.close(fig)

# 3. Across rocks: Median + 95% CI
fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax_temp.plot(x, med_temp_in_all, color='red', label='Inside (median of medians)')
ax_temp.fill_between(x, ci_med_temp_in_low_all, ci_med_temp_in_up_all, color='red', alpha=0.3)
ax_temp.plot(x, med_temp_out_all, color='blue', label='Outside (median of medians)')
ax_temp.fill_between(x, ci_med_temp_out_low_all, ci_med_temp_out_up_all, color='blue', alpha=0.3)
ax_temp.set_ylabel("Temperature (°C)")
ax_temp.set_xlim(0, 24)
ax_temp_2 = ax_temp.twinx()
ax_temp_2.plot(x, med_temp_diff_all, color='black', label='Diff (median of medians)')
ax_temp_2.fill_between(x, ci_med_temp_diff_low_all, ci_med_temp_diff_up_all, color='gray', alpha=0.3)
ax_temp_2.set_ylabel("Temp. Difference (°C)")
ax_hum.plot(x, med_hum_all, color='darkorange', label='Humidity (median of medians)')
ax_hum.fill_between(x, ci_med_hum_low_all, ci_med_hum_up_all, color='orange', alpha=0.3)
ax_hum.set_ylabel("Humidity (% RH)")
ax_hum.set_xlabel("Hour of day")
ax_hum.set_xticks(range(0,24,3))
ax_temp.legend(custom_lines, ['inside', 'outside', 'difference in-out'], loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_median_95CI.png"))
fig.savefig(os.path.join(FIG_DIR_ACROSS, "acrossRocks_median_95CI.svg"))
plt.close(fig)

print("Analysis complete. Figures saved to:")
print(f"  {FIG_DIR_PER_ROCK} (per-rock plots)")
print(f"  {FIG_DIR_ACROSS} (across-rock plots)")
print(f"And hourly averages saved to: {OUTPUT_DATA_FILE}")
