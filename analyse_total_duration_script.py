#!/usr/bin/env python3
"""
analysis_total_duration_plots.py

This driver script performs analysis over the entire observation time (total duration)
of rock data. It loads the wide-format CSV file located at:
    /home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv

It then computes an "elapsed_day" column (days since the first observation)
and groups data by elapsed day. For each rock, it computes statistics
(mean, median, SEM, and 95% bootstrap confidence intervals) for:
    - temperature_in_C, temperature_out_C, temperature_diff_C, and humidity_perc_RH

For each rock, four plot variants are produced:
    1. Temperature dual-axis plot (Mean ± SEM)
    2. Temperature dual-axis plot (Mean ± 95% CI)
    3. Temperature dual-axis plot (Median ± 95% CI)
    4. Humidity plot (Mean ± 95% CI)

The x-axis is in elapsed days (from 0 to the total duration).
Figures are saved in both PNG and SVG formats under:
    /home/geuba03p/weta_project/burrow_temperature/figures/total_observation_duration/

All per‑rock daily stats are collated and saved as:
    /home/geuba03p/weta_project/burrow_temperature/data/total_duration_averages.csv

Then, an across‑rocks aggregation is performed and corresponding plots are saved in the same folder.

Note: No gridlines are used in any plot.
"""

import os
import pandas as pd
from data_analyzer import DataAnalyzer
from plot_generator import PlotGenerator

# --- Configuration Constants ---
DATA_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv"
OUTPUT_DAILY_STATS = "/home/geuba03p/weta_project/burrow_temperature/data/day_wise_averages.csv"
FIG_DIR_TOTAL = "/home/geuba03p/weta_project/burrow_temperature/figures/day_wise"

# Create output directory if needed
os.makedirs(FIG_DIR_TOTAL, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_FILE, parse_dates=["date_time", "elapsed_date_time"])
# Create elapsed_day column (as integer days from the first observation)
df["elapsed_day"] = (pd.to_datetime(df["elapsed_date_time"]) - pd.to_datetime(df["elapsed_date_time"]).min()).dt.total_seconds() / (3600*24)
df["elapsed_day"] = df["elapsed_day"].astype(int)

# Get unique rock IDs
rock_ids = df["rock"].unique()

# Initialize analyzer and plot generator
analyzer = DataAnalyzer()
plotter = PlotGenerator()

# Prepare list for per-rock daily stats
per_rock_stats_list = []

# --- Per-Rock Analysis and Plotting (Total Duration) ---
for rock in rock_ids:
    rock_df = df[df["rock"] == rock].copy()
    # Compute daily stats by grouping on "elapsed_day"
    stats_df = analyzer.compute_total_duration_stats(rock_df)
    # Save rock identifier with stats and make grouping column a normal column
    stats_df = stats_df.reset_index().rename(columns={"elapsed_day": "day"})
    stats_df["rock"] = rock
    per_rock_stats_list.append(stats_df)
    
    title_prefix = f"Rock {rock}"
    # For total duration, x-axis limits are from the minimum to maximum day.
    x_min = stats_df["day"].min()
    x_max = stats_df["day"].max()
    x_limits = (x_min, x_max)

    # Plot temperature dual-axis: three variants.
    fig1 = plotter.plot_temperature_dual_axis(stats_df.set_index("day"), metric="mean", error="SEM",
                                                title=f"{title_prefix}: Temperature (Mean ± SEM)", x_limits=x_limits)
    plotter.save_figure(fig1, FIG_DIR_TOTAL, f"rock_{rock}_total_mean_SEM")
    
    fig2 = plotter.plot_temperature_dual_axis(stats_df.set_index("day"), metric="mean", error="CI",
                                                title=f"{title_prefix}: Temperature (Mean ± 95% CI)", x_limits=x_limits)
    plotter.save_figure(fig2, FIG_DIR_TOTAL, f"rock_{rock}_total_mean_CI")
    
    fig3 = plotter.plot_temperature_dual_axis(stats_df.set_index("day"), metric="median", error="CI",
                                                title=f"{title_prefix}: Temperature (Median ± 95% CI)", x_limits=x_limits)
    plotter.save_figure(fig3, FIG_DIR_TOTAL, f"rock_{rock}_total_median_CI")
    
    # Plot humidity (mean + 95% CI)
    fig4 = plotter.plot_humidity_series(stats_df.set_index("day"), metric="mean", error="CI",
                                        title=f"{title_prefix}: Humidity (Mean ± 95% CI)", x_limits=x_limits)
    plotter.save_figure(fig4, FIG_DIR_TOTAL, f"rock_{rock}_total_humidity_mean_CI")

# --- Save Combined Per-Rock Daily Stats ---
combined_stats = pd.concat(per_rock_stats_list, ignore_index=True)
combined_stats.to_csv(OUTPUT_DAILY_STATS, index=False)

# --- Across-Rocks Aggregation ---
agg_stats = analyzer.compute_total_duration_stats(df)

# Determine x-axis limits from aggregated stats
x_min = agg_stats.index.min()
x_max = agg_stats.index.max()
x_limits = (x_min, x_max)

# Plot aggregated temperature (across rocks)
fig_a1 = plotter.plot_temperature_dual_axis(agg_stats, metric="mean", error="SEM",
                                             title="Across Rocks: Temperature (Mean ± SEM)", x_limits=x_limits)
plotter.save_figure(fig_a1, FIG_DIR_TOTAL, "acrossRocks_total_mean_SEM")

fig_a2 = plotter.plot_temperature_dual_axis(agg_stats, metric="mean", error="CI",
                                             title="Across Rocks: Temperature (Mean ± 95% CI)", x_limits=x_limits)
plotter.save_figure(fig_a2, FIG_DIR_TOTAL, "acrossRocks_total_mean_CI")

fig_a3 = plotter.plot_temperature_dual_axis(agg_stats, metric="median", error="CI",
                                             title="Across Rocks: Temperature (Median ± 95% CI)", x_limits=x_limits)
plotter.save_figure(fig_a3, FIG_DIR_TOTAL, "acrossRocks_total_median_CI")

# Plot aggregated humidity
fig_a4 = plotter.plot_humidity_series(agg_stats, metric="mean", error="CI",
                                      title="Across Rocks: Humidity (Mean ± 95% CI)", x_limits=x_limits)
plotter.save_figure(fig_a4, FIG_DIR_TOTAL, "acrossRocks_total_humidity_mean_CI")

print("Total observation duration analysis complete!")
print(f"Per-rock daily figures saved to: {FIG_DIR_TOTAL}")
print(f"Combined daily statistics saved to: {OUTPUT_DAILY_STATS}")
