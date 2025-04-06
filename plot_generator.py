#!/usr/bin/env python3
"""
plot_generator.py

This module defines the PlotGenerator class that provides reusable plotting functions.
It builds on matplotlib (with seaborn styling) to create:
  - A dual-axis temperature plot showing inside (red) and outside (blue) temperatures on the left,
    and temperature difference (black dashed) on the right.
  - A humidity plot (single-axis) in dark orange.
It also includes a helper function to save figures in both PNG and SVG formats.

Optional x-axis limits can be specified.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

class PlotGenerator:
    """Provides functions to generate and save plots for rock data analysis."""

    def __init__(self):
        # Set a seaborn theme with no gridlines (override grid style)
        sns.set_theme(style="white")
    
    def save_figure(self, fig, output_folder, base_filename):
        """
        Save a matplotlib Figure in both PNG and SVG formats.

        Parameters:
            fig (Figure): The figure to save.
            output_folder (str): Destination folder.
            base_filename (str): Base filename (without extension).

        Returns:
            None.
        """
        os.makedirs(output_folder, exist_ok=True)
        png_path = os.path.join(output_folder, f"{base_filename}.png")
        svg_path = os.path.join(output_folder, f"{base_filename}.svg")
        fig.savefig(png_path, format="png", dpi=300)
        fig.savefig(svg_path, format="svg")
        plt.close(fig)

    def plot_temperature_dual_axis(self, stats_df, metric="mean", error="SEM", title=None, x_limits=None):
        """
        Create a dual-axis temperature plot.

        Parameters:
            stats_df (DataFrame): DataFrame indexed by the grouping variable (e.g., hour or elapsed_day)
                                  with columns:
                                    inside_<metric>, outside_<metric>, diff_<metric>
                                    and error columns:
                                    inside_sem / inside_ci_lower_mean, etc.
            metric (str): "mean" or "median" (default "mean").
            error (str): "SEM" or "CI" (default "SEM").
            title (str): Optional plot title.
            x_limits (tuple): Optional (xmin, xmax) limits for the x-axis.

        Returns:
            Figure: The generated matplotlib figure.
        """
        metric = metric.lower()
        error = error.upper()
        inside_col = f"inside_{metric}"
        outside_col = f"outside_{metric}"
        diff_col = f"diff_{metric}"
        if error == "SEM":
            err_inside_low = stats_df[inside_col] - stats_df["inside_sem"]
            err_inside_high = stats_df[inside_col] + stats_df["inside_sem"]
            err_outside_low = stats_df[outside_col] - stats_df["outside_sem"]
            err_outside_high = stats_df[outside_col] + stats_df["outside_sem"]
            err_diff_low = stats_df[diff_col] - stats_df["diff_sem"]
            err_diff_high = stats_df[diff_col] + stats_df["diff_sem"]
        else:  # CI
            suffix = "mean" if metric == "mean" else "med"
            err_inside_low = stats_df[f"inside_ci_lower_{suffix}"]
            err_inside_high = stats_df[f"inside_ci_upper_{suffix}"]
            err_outside_low = stats_df[f"outside_ci_lower_{suffix}"]
            err_outside_high = stats_df[f"outside_ci_upper_{suffix}"]
            err_diff_low = stats_df[f"diff_ci_lower_{suffix}"]
            err_diff_high = stats_df[f"diff_ci_upper_{suffix}"]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()
        x = stats_df.index.values
        # Plot inside (red) and outside (blue) on primary axis.
        ax1.plot(x, stats_df[inside_col], color="red", label="Inside")
        ax1.fill_between(x, err_inside_low, err_inside_high, color="red", alpha=0.2)
        ax1.plot(x, stats_df[outside_col], color="blue", label="Outside")
        ax1.fill_between(x, err_outside_low, err_outside_high, color="blue", alpha=0.2)
        ax1.set_xlabel("Time")
        if x_limits:
            ax1.set_xlim(x_limits)
        ax1.set_ylabel("Temperature (°C)")
        # Plot temperature difference on secondary axis (black dashed).
        ax2.plot(x, stats_df[diff_col], color="black", linestyle="--", label="Difference")
        ax2.fill_between(x, err_diff_low, err_diff_high, color="black", alpha=0.15)
        ax2.set_ylabel("Temperature Difference (°C)")
        # Build custom legend.
        custom_lines = [Line2D([0], [0], color="red", lw=2),
                        Line2D([0], [0], color="blue", lw=2),
                        Line2D([0], [0], color="black", lw=2)]
        ax1.legend(custom_lines, ["inside", "outside", "difference in-out"], loc="upper left")
        if title:
            ax1.set_title(title)
        ax1.grid(False)
        ax2.grid(False)
        return fig

    def plot_humidity_series(self, stats_df, metric="mean", error="CI", title=None, x_limits=None):
        """
        Create a humidity time-series plot.

        Parameters:
            stats_df (DataFrame): DataFrame indexed by the grouping variable with humidity stats.
            metric (str): "mean" or "median" (default "mean").
            error (str): "SEM" or "CI" (default "CI").
            title (str): Optional title.
            x_limits (tuple): Optional x-axis limits.

        Returns:
            Figure: The generated figure.
        """
        metric = metric.lower()
        error = error.upper()
        hum_col = f"hum_{metric}"
        if error == "SEM":
            err_low = stats_df[hum_col] - stats_df["hum_sem"]
            err_high = stats_df[hum_col] + stats_df["hum_sem"]
        else:
            suffix = "mean" if metric == "mean" else "med"
            err_low = stats_df[f"hum_ci_lower_{suffix}"]
            err_high = stats_df[f"hum_ci_upper_{suffix}"]
        fig, ax = plt.subplots(figsize=(6, 3))
        x = stats_df.index.values
        ax.plot(x, stats_df[hum_col], color="darkorange", label="Humidity")
        ax.fill_between(x, err_low, err_high, color="orange", alpha=0.2)
        ax.set_xlabel("Time")
        if x_limits:
            ax.set_xlim(x_limits)
        ax.set_ylabel("Humidity (% RH)")
        ax.legend(loc="upper right")
        ax.grid(False)
        if title:
            ax.set_title(title)
        return fig
    
    def plot_full_timeseries(self, df: 'pd.DataFrame', rock: str, x_label="Elapsed Time (Days)") -> plt.Figure:
        """
        Plot the full raw time series for one rock.
        The top subplot shows inside (red) and outside (blue) temperatures (if available)
        on a primary y-axis and the temperature difference (black dashed) on a secondary y-axis.
        The bottom subplot shows humidity (orange).
        
        Parameters:
            df (pd.DataFrame): DataFrame for one rock. Must contain:
                              date_time, elapsed_date_time, temperature_in_C,
                              temperature_out_C, temperature_diff_C, humidity_perc_RH.
            rock (str): Rock identifier.
            x_label (str): Label for the x-axis.
        
        Returns:
            Figure: The generated figure.
        """
        # Use elapsed_date_time converted to elapsed days for x-axis.
        df["elapsed_days"] = (pd.to_datetime(df["elapsed_date_time"]) - pd.to_datetime(df["elapsed_date_time"]).min()).dt.total_seconds()/(3600*24)
        fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
        # Top subplot: temperature data
        ax_temp.plot(df["elapsed_days"], df["temperature_in_C"], color="red", label="Inside")
        if "temperature_out_C" in df.columns and not df["temperature_out_C"].isnull().all():
            ax_temp.plot(df["elapsed_days"], df["temperature_out_C"], color="blue", label="Outside")
        ax_temp.set_ylabel("Temperature (°C)")
        # Secondary y-axis: temperature difference
        ax_temp2 = ax_temp.twinx()
        ax_temp2.plot(df["elapsed_days"], df["temperature_diff_C"], color="black", linestyle="--", label="Diff")
        ax_temp2.set_ylabel("Temp. Difference (°C)")
        # Custom legend for temperature subplot
        custom_lines = [Line2D([0], [0], color="red", lw=2),
                        Line2D([0], [0], color="blue", lw=2),
                        Line2D([0], [0], color="black", lw=2)]
        ax_temp.legend(custom_lines, ["inside", "outside", "difference in-out"], loc="upper left")
        ax_temp.set_title(f"Rock {rock} – Full Time Series")
        # Bottom subplot: humidity
        ax_hum.plot(df["elapsed_days"], df["humidity_perc_RH"], color="darkorange", label="Humidity")
        ax_hum.set_ylabel("Humidity (%RH)")
        ax_hum.set_xlabel(x_label)
        # Remove grid lines
        ax_temp.grid(False)
        ax_temp2.grid(False)
        ax_hum.grid(False)
        return fig

    def plot_full_timeseries(self, df: 'pd.DataFrame', rock: str, x_label="Elapsed Time (Days)") -> plt.Figure:
        """
        Plot the full raw time series for one rock.
        The top subplot shows inside (red) and outside (blue) temperature on a primary y-axis,
        and the temperature difference (black dashed) on a secondary y-axis.
        The bottom subplot shows humidity (orange).
        
        Parameters:
            df (pd.DataFrame): DataFrame for one rock. Must contain:
                              date_time, elapsed_date_time, temperature_in_C,
                              temperature_out_C, temperature_diff_C, humidity_perc_RH.
            rock (str): Rock identifier.
            x_label (str): Label for the x-axis.
        
        Returns:
            Figure: The generated figure.
        """
        # Compute elapsed days from elapsed_date_time (which has base date 2000-01-01)
        base_date = pd.Timestamp("2000-01-01")
        df = df.copy()
        df["elapsed_days"] = ((pd.to_datetime(df["elapsed_date_time"]) - base_date)
                              / pd.Timedelta(days=1))
        fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
        # Top subplot: temperature data
        ax_temp.plot(df["elapsed_days"], df["temperature_in_C"], color="red", label="Inside")
        if "temperature_out_C" in df.columns and not df["temperature_out_C"].isnull().all():
            ax_temp.plot(df["elapsed_days"], df["temperature_out_C"], color="blue", label="Outside")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_xlim(df["elapsed_days"].min(), df["elapsed_days"].max())
        ax_temp2 = ax_temp.twinx()
        ax_temp2.plot(df["elapsed_days"], df["temperature_diff_C"], color="black", linestyle="--", label="Diff")
        ax_temp2.set_ylabel("Temp. Difference (°C)")
        # Custom legend for temperature subplot
        custom_lines = [Line2D([0], [0], color="red", lw=2),
                        Line2D([0], [0], color="blue", lw=2),
                        Line2D([0], [0], color="black", lw=2)]
        ax_temp.legend(custom_lines, ["inside", "outside", "difference in-out"], loc="upper left")
        ax_temp.set_title(f"Rock {rock} – Full Time Series")
        # Bottom subplot: humidity
        ax_hum.plot(df["elapsed_days"], df["humidity_perc_RH"], color="darkorange", label="Humidity")
        ax_hum.set_ylabel("Humidity (%RH)")
        ax_hum.set_xlabel(x_label)
        # Remove grid lines
        ax_temp.grid(False)
        ax_temp2.grid(False)
        ax_hum.grid(False)
        return fig

    def plot_hourly_aggregated_across_rocks(self, agg_df: pd.DataFrame, x_label="Elapsed Hour") -> plt.Figure:
        """
        Plot the aggregated hourly statistics (across rocks) on a dual-y axis plot.
        The primary y-axis shows temperature (inside in red and outside in blue) with shaded error
        (either SEM or 95% CI), and the secondary y-axis shows temperature difference (black)
        with shaded error and humidity (darkorange) with shaded error.
        
        Parameters:
            agg_df (pd.DataFrame): Aggregated hourly statistics DataFrame. Expected to have columns:
              elapsed_hour, temperature_in_C_mean, temperature_in_C_sem, temperature_in_C_ci_lower, temperature_in_C_ci_upper,
              temperature_out_C_mean, temperature_out_C_sem, temperature_out_C_ci_lower, temperature_out_C_ci_upper,
              temperature_diff_C_mean, temperature_diff_C_sem, temperature_diff_C_ci_lower, temperature_diff_C_ci_upper,
              humidity_perc_RH_mean, humidity_perc_RH_sem, humidity_perc_RH_ci_lower, humidity_perc_RH_ci_upper.
            x_label (str): Label for the x-axis.
        
        Returns:
            Figure: The generated figure.
        """
        x = agg_df["elapsed_hour"].values
        # Extract variables for temperature (inside, outside, difference) and humidity
        temp_in_mean   = agg_df["temperature_in_C_mean"].values
        temp_out_mean  = agg_df["temperature_out_C_mean"].values
        temp_diff_mean = agg_df["temperature_diff_C_mean"].values
        hum_mean       = agg_df["humidity_perc_RH_mean"].values
        
        # For error bars, here we use SEM (you could also use CI columns by modifying the code)
        temp_in_err   = agg_df["temperature_in_C_sem"].values
        temp_out_err  = agg_df["temperature_out_C_sem"].values
        temp_diff_err = agg_df["temperature_diff_C_sem"].values
        hum_err       = agg_df["humidity_perc_RH_sem"].values
        
        fig, (ax_temp, ax_hum) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
        # Top subplot: temperature data
        color_in, color_out, color_diff = "red", "blue", "black"
        ax_temp.plot(x, temp_in_mean, color=color_in, label="Inside")
        ax_temp.fill_between(x, temp_in_mean - temp_in_err, temp_in_mean + temp_in_err, color=color_in, alpha=0.3)
        ax_temp.plot(x, temp_out_mean, color=color_out, label="Outside")
        ax_temp.fill_between(x, temp_out_mean - temp_out_err, temp_out_mean + temp_out_err, color=color_out, alpha=0.3)
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_xlim(x.min(), x.max())
        ax_temp2 = ax_temp.twinx()
        ax_temp2.plot(x, temp_diff_mean, color=color_diff, label="Difference", linestyle="--")
        ax_temp2.fill_between(x, temp_diff_mean - temp_diff_err, temp_diff_mean + temp_diff_err, color="gray", alpha=0.3)
        ax_temp2.set_ylabel("Temp. Difference (°C)")
        # Custom legend for temperature subplot
        custom_lines = [Line2D([0], [0], color=color_in, lw=2),
                        Line2D([0], [0], color=color_out, lw=2),
                        Line2D([0], [0], color=color_diff, lw=2)]
        ax_temp.legend(custom_lines, ["inside", "outside", "difference in-out"], loc="upper left")
        # Bottom subplot: humidity
        ax_hum.plot(x, hum_mean, color="darkorange", label="Humidity")
        ax_hum.fill_between(x, hum_mean - hum_err, hum_mean + hum_err, color="orange", alpha=0.3)
        ax_hum.set_ylabel("Humidity (% RH)")
        ax_hum.set_xlabel(x_label)
        return fig