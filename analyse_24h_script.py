#!/usr/bin/env python3
"""
analysis_24h_plots.py

This driver script performs 24‑hour analysis of rock temperature and humidity data.
It uses the wide‑format CSV file located at:
    /home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv

Each rock’s data is grouped by hour-of-day (from the elapsed_date_time column) to compute:
    - Mean, median, SEM, and 95% bootstrap confidence intervals for:
         • temperature_in_C, temperature_out_C, temperature_diff_C, and humidity_perc_RH.
         
For each rock, three types of temperature plots are produced (dual‑axis plots):
    1. Mean ± SEM
    2. Mean ± 95% CI
    3. Median ± 95% CI
    
Additionally, a humidity plot (mean ± 95% CI) is produced.
Figures are saved in both PNG and SVG formats in:
    /home/geuba03p/weta_project/burrow_temperature/figures/24h_perRock/

All per‑rock hourly stats are collated and saved as:
    /home/geuba03p/weta_project/burrow_temperature/data/24h_hourly_averages.csv

Then, an across‑rocks aggregation is performed and corresponding plots are saved in:
    /home/geuba03p/weta_project/burrow_temperature/figures/24h_comparisson/
    
Note: No gridlines are used in any plot.
"""

import os
import pandas as pd
from data_analyzer import DataAnalyzer
from plot_generator import PlotGenerator

# --- Configuration Constants ---
DATA_FILE = "/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv"
OUTPUT_HOURLY_STATS = "/home/geuba03p/weta_project/burrow_temperature/data/24h_hourly_averages.csv"
FIG_DIR_PER_ROCK = "/home/geuba03p/weta_project/burrow_temperature/figures/24h_perRock"
FIG_DIR_ACROSS = "/home/geuba03p/weta_project/burrow_temperature/figures/24h_comparisson"

# Ensure output directories exist
os.makedirs(FIG_DIR_PER_ROCK, exist_ok=True)
os.makedirs(FIG_DIR_ACROSS, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_FILE, parse_dates=["date_time", "elapsed_date_time"])
# Create an Hour column from elapsed_date_time if not present
if "Hour" not in df.columns:
    df["Hour"] = pd.to_datetime(df["elapsed_date_time"]).dt.hour

# Get unique rock IDs
rock_ids = df["rock"].unique()

# Initialize our analysis and plotting objects
analyzer = DataAnalyzer()
plotter = PlotGenerator()

# Prepare list to collect per-rock stats for later aggregation
per_rock_stats_list = []

# --- Per-Rock Analysis and Plotting ---
for rock in rock_ids:
    # Subset data for this rock
    rock_df = df[df["rock"] == rock].copy()
    # Compute hourly statistics for this rock
    stats_df = analyzer.compute_hourly_stats(rock_df, hour_col="Hour")
    # Save rock identifier with the stats
    stats_df["rock"] = rock
    # Save per-rock stats for aggregation (make Hour a column)
    stats_df = stats_df.reset_index()  # Hour becomes a column
    per_rock_stats_list.append(stats_df)
    
    # Plot temperature: three variants
    title_prefix = f"Rock {rock}"
    # 1. Temperature Mean + SEM
    fig1 = plotter.plot_temperature_dual_axis(stats_df.set_index("Hour"), metric="mean", error="SEM",
                                                title=f"{title_prefix}: Temperature (Mean ± SEM)")
    plotter.save_figure(fig1, FIG_DIR_PER_ROCK, f"rock_{rock}_mean_SEM")
    # 2. Temperature Mean + 95% CI
    fig2 = plotter.plot_temperature_dual_axis(stats_df.set_index("Hour"), metric="mean", error="CI",
                                                title=f"{title_prefix}: Temperature (Mean ± 95% CI)")
    plotter.save_figure(fig2, FIG_DIR_PER_ROCK, f"rock_{rock}_mean_CI")
    # 3. Temperature Median + 95% CI
    fig3 = plotter.plot_temperature_dual_axis(stats_df.set_index("Hour"), metric="median", error="CI",
                                                title=f"{title_prefix}: Temperature (Median ± 95% CI)")
    plotter.save_figure(fig3, FIG_DIR_PER_ROCK, f"rock_{rock}_median_CI")
    # 4. Humidity plot (mean + 95% CI)
    fig4 = plotter.plot_humidity_series(stats_df.set_index("Hour"), metric="mean", error="CI",
                                        title=f"{title_prefix}: Humidity (Mean ± 95% CI)")
    plotter.save_figure(fig4, FIG_DIR_PER_ROCK, f"rock_{rock}_humidity_mean_CI")

# --- Save Combined Per-Rock Hourly Stats ---
combined_stats = pd.concat(per_rock_stats_list, ignore_index=True)
combined_stats.to_csv(OUTPUT_HOURLY_STATS, index=False)

# --- Across-Rocks Aggregation ---
# Here we aggregate by pooling all data (ignoring rock identifiers)
agg_stats = analyzer.compute_hourly_stats(df, hour_col="Hour")

# Plot aggregated temperature
fig_a1 = plotter.plot_temperature_dual_axis(agg_stats, metric="mean", error="SEM",
                                             title="Across Rocks: Temperature (Mean ± SEM)")
plotter.save_figure(fig_a1, FIG_DIR_ACROSS, "acrossRocks_mean_SEM")
fig_a2 = plotter.plot_temperature_dual_axis(agg_stats, metric="mean", error="CI",
                                             title="Across Rocks: Temperature (Mean ± 95% CI)")
plotter.save_figure(fig_a2, FIG_DIR_ACROSS, "acrossRocks_mean_CI")
fig_a3 = plotter.plot_temperature_dual_axis(agg_stats, metric="median", error="CI",
                                             title="Across Rocks: Temperature (Median ± 95% CI)")
plotter.save_figure(fig_a3, FIG_DIR_ACROSS, "acrossRocks_median_CI")
# Plot aggregated humidity
fig_a4 = plotter.plot_humidity_series(agg_stats, metric="mean", error="CI",
                                      title="Across Rocks: Humidity (Mean ± 95% CI)")
plotter.save_figure(fig_a4, FIG_DIR_ACROSS, "acrossRocks_humidity_mean_CI")

print("Analysis complete!")
print(f"Per-rock figures saved to: {FIG_DIR_PER_ROCK}")
print(f"Across-rock figures saved to: {FIG_DIR_ACROSS}")
print(f"Combined hourly statistics saved to: {OUTPUT_HOURLY_STATS}")
