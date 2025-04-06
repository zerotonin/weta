#!/usr/bin/env python3
"""
analysis_full_data_across_rocks.py

This driver script performs full-duration analysis (without hourly/daily averaging)
of rock temperature and humidity data. It loads the wide-format CSV file
located at:

    /home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv

For each rock, it plots the full raw time series using elapsed time (in days)
and saves the figures under:
    /home/geuba03p/weta_project/burrow_temperature/figures/full_perRock/

It also computes overall statistics (mean, median, SEM, 95% CI) for each rock
using the entire observation period, and then aggregates these overall stats
across rocks. A summary bar chart is produced and saved under:
    /home/geuba03p/weta_project/burrow_temperature/figures/full_acrossRocks/

The combined per‑rock overall statistics are saved as:
    /home/geuba03p/weta_project/burrow_temperature/data/full_duration_overall_stats.csv
"""

import os
import pandas as pd
from data_analyzer import DataAnalyzer
from plot_generator import PlotGenerator

# --- Configuration ---
DATA_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv"
OUTPUT_STATS_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/full_duration_overall_stats.csv"
FIG_DIR_PER_ROCK = "/home/geuba03p/weta_project/burrow_temperature/figures/full_perRock"
FIG_DIR_AGG = "/home/geuba03p/weta_project/burrow_temperature/figures/full_acrossRocks_hourly"
OUTPUT_AGG_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/full_duration_hourly_aggregates.csv"

os.makedirs(FIG_DIR_PER_ROCK, exist_ok=True)
os.makedirs(FIG_DIR_AGG, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_FILE, parse_dates=["date_time", "elapsed_date_time"])
# (We assume that the wide CSV already contains the columns specified.)

# Get unique rock IDs
rock_ids = df["rock"].unique()

# Initialize analyzer and plot generator
analyzer = DataAnalyzer()
plotter = PlotGenerator()

# Prepare a list for overall stats for each rock
per_rock_overall_stats = []

# --- Per-Rock Full Data Plots and Statistics ---
for rock in rock_ids:
    rock_df = df[df["rock"] == rock].copy()
    # Plot the full raw time series for this rock (x-axis = elapsed days)
    fig = plotter.plot_full_timeseries(rock_df, rock, x_label="Elapsed Time (Days)")
    plotter.save_figure(fig, FIG_DIR_PER_ROCK, f"rock_{rock}_full_timeseries")
    
    # Compute overall (full-duration) statistics for this rock
    stats = analyzer.compute_overall_stats(rock_df)
    # For convenience, store the rock id along with stats (flattening the dict)
    record = {"rock": rock}
    for var, stat in stats.items():
        for key, value in stat.items():
            record[f"{var}_{key}"] = value
    per_rock_overall_stats.append(record)

# Save per-rock overall stats to CSV
stats_df = pd.DataFrame(per_rock_overall_stats)
stats_df.to_csv(OUTPUT_STATS_FILE, index=False)

# Compute elapsed_hour for each row.
# Since elapsed_date_time has been computed using base January 1,2000,
# we convert it to the number of hours since that base.
base_date = pd.Timestamp("2000-01-01")
df["elapsed_hour"] = ((pd.to_datetime(df["elapsed_date_time"]) - base_date) /
                      pd.Timedelta(hours=1)).astype(int)

# --- Aggregate Hourly Statistics Across Rocks ---
analyzer = DataAnalyzer()
agg_df = analyzer.aggregate_hourly_stats(df)

# Save the aggregated hourly statistics to CSV
agg_df.to_csv(OUTPUT_AGG_FILE, index=False)

# --- Produce Aggregated Plots ---
plotter = PlotGenerator()
fig = plotter.plot_hourly_aggregated_across_rocks(agg_df, x_label="Elapsed Hour")
plotter.save_figure(fig, FIG_DIR_AGG, "acrossRocks_hourly_aggregated")

print("Hourly aggregated analysis complete!")
print(f"Aggregated hourly stats saved to: {OUTPUT_AGG_FILE}")
print(f"Aggregated plot saved to: {FIG_DIR_AGG}")