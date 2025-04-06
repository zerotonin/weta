import numpy as np
import pandas as pd

class DataAnalyzer:
    """Contains stateless analysis functions for computing derived metrics.
        A class for analyzing 24-hour rock temperature and humidity data.
        Provides methods to compute statistical summaries and confidence intervals.
        All methods are stateless (no persistent internal state).
    """
    
    @staticmethod
    def compute_elapsed_daytime(times: pd.Series) -> pd.Series:
        """
        Given a pandas Series of datetime values, compute a new Series representing
        the elapsed day/time relative to the first reading, preserving the hour-of-day.
        Uses January 1, 2000 as the base date.
        """
        floored = times.dt.floor('h')
        t0 = floored.iloc[0]
        base = t0.replace(year=2000, month=1, day=1)
        elapsed = base + (floored - t0)
        return elapsed

    @staticmethod
    def compute_temp_diff(temp_in: pd.Series, temp_out: pd.Series) -> pd.Series:
        """Compute temperature difference (inside minus outside)."""
        return temp_in.astype(float) - temp_out.astype(float)
    
    @staticmethod
    def initial_analysis(data: dict) -> pd.DataFrame:
        """
        Combine raw DataFrames for a single rock into one wide-format table.
        Expects keys in data: 'in_temp' (required), 'out_temp' (optional), 'humidity' (optional).
        Returns a DataFrame with columns:
            - date_time (from inside sensor's parsed data)
            - elapsed_date_time (computed from date_time)
            - temperature_in_C (float)
            - temperature_out_C (float, if available)
            - temperature_diff_C (float, if out available)
            - humidity_perc_RH (float, if available)
        """
        if 'in_temp' not in data:
            raise ValueError("Inside temperature data (in_temp) is required for analysis")
        
        # Use inside temperature data as base.
        df = data['in_temp'].copy()
        # Ensure correct column names for inside temperature DataFrame
        # (Assuming df already has "date_time" and "temperature_in_C" columns)
        df["elapsed_date_time"] = DataAnalyzer.compute_elapsed_daytime(df['date_time'])
        df = df.rename(columns={"Value": "temperature_in_C"})
        # Merge outside temperature if available.
        if 'out_temp' in data and data['out_temp'] is not None:
            df_out = data['out_temp'].copy()
            # Rename columns to ensure consistency:
            df_out = df_out.rename(columns={df_out.columns[0]: "temperature_out_C",
                                            df_out.columns[-1]: "date_time"})
            df_out["elapsed_date_time"] = DataAnalyzer.compute_elapsed_daytime(df_out['date_time'])
            # Merge using a DataFrame that contains both key and measurement
            df = df.merge(df_out[['elapsed_date_time', "temperature_out_C"]],
                          on="elapsed_date_time", how="left")
            df["temperature_diff_C"] = DataAnalyzer.compute_temp_diff(df["temperature_in_C"],
                                                                      df["temperature_out_C"])
        else:
            df["temperature_out_C"] = np.nan
            df["temperature_diff_C"] = np.nan
        
        # Merge humidity if available.
        if 'humidity' in data and data['humidity'] is not None:
            df_hum = data['humidity'].copy()

            df_hum = df_hum.rename(columns={df_hum.columns[0]: "humidity_perc_RH",
                                            df_hum.columns[-1]: "date_time"})
            df_hum["elapsed_date_time"] = DataAnalyzer.compute_elapsed_daytime(df_hum['date_time'])
            # We assume here that merging on date_time is more reliable for humidity,
            # but if needed, you can merge on elapsed_date_time similarly.
            df = df.merge(df_hum[['elapsed_date_time', "humidity_perc_RH"]],
                          on="elapsed_date_time", how="left")
        else:
            df["humidity_perc_RH"] = pd.NA
        
        # Sort by date_time for consistency.
        df = df.sort_values("date_time").reset_index(drop=True)
        return df[["date_time", "elapsed_date_time", "temperature_in_C", 
                   "temperature_out_C", "temperature_diff_C", "humidity_perc_RH"]]


   
    @staticmethod
    def bootstrap_confidence_interval(data, statistic=np.nanmean, confidence=0.95, n_boot=1000):
        """
        Compute a bootstrap confidence interval for a given statistic.
        
        Parameters:
            data (array-like): Numeric data (NaNs are ignored).
            statistic (callable): Function to compute the statistic (default: np.nanmean).
            confidence (float): Confidence level (default: 0.95).
            n_boot (int): Number of bootstrap samples (default: 1000).
        
        Returns:
            tuple: (lower_bound, upper_bound) of the bootstrap confidence interval.
        """
        vals = np.array(data, dtype=float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        if n == 0:
            return (np.nan, np.nan)
        if n == 1:
            return (vals[0], vals[0])
        boot_samples = np.empty(n_boot)
        for i in range(n_boot):
            sample = np.random.choice(vals, size=n, replace=True)
            boot_samples[i] = statistic(sample)
        lower_pct = ((1 - confidence) / 2) * 100
        upper_pct = (confidence + (1 - confidence) / 2) * 100
        return (np.percentile(boot_samples, lower_pct), np.percentile(boot_samples, upper_pct))

    def compute_hourly_stats(self, df: pd.DataFrame, group_col: str = "Hour") -> pd.DataFrame:
        """
        Group data by the specified column (typically hour-of-day) and compute statistics
        (mean, median, SEM, and 95% bootstrap confidence intervals) for:
          - temperature_in_C (as "inside")
          - temperature_out_C (as "outside")
          - temperature_diff_C (as "diff")
          - humidity_perc_RH (as "hum")

        Parameters:
            df (pd.DataFrame): Wide-format DataFrame.
            group_col (str): Column to group by (default "Hour").

        Returns:
            pd.DataFrame: DataFrame indexed by the group with computed statistics.
                        The index is renamed to the group_col.
        """
        # Define a helper for standard error (nan-aware)
        def sem(x):
            arr = np.array(x, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) <= 1:
                return 0.0
            return np.nanstd(arr, ddof=1) / np.sqrt(len(arr))

        stats_records = []
        groups = df.groupby(group_col)
        for grp_val, grp in groups:
            record = {group_col: grp_val}
            for var, prefix in [("temperature_in_C", "inside"),
                                ("temperature_out_C", "outside"),
                                ("temperature_diff_C", "diff"),
                                ("humidity_perc_RH", "hum")]:
                values = grp[var].dropna().values
                if len(values) > 0:
                    record[f"{prefix}_mean"] = np.nanmean(values)
                    record[f"{prefix}_median"] = np.nanmedian(values)
                    record[f"{prefix}_sem"] = sem(values)
                    ci_mean = self.bootstrap_confidence_interval(values, statistic=np.nanmean)
                    record[f"{prefix}_ci_lower_mean"] = ci_mean[0]
                    record[f"{prefix}_ci_upper_mean"] = ci_mean[1]
                    ci_median = self.bootstrap_confidence_interval(values, statistic=np.nanmedian)
                    record[f"{prefix}_ci_lower_med"] = ci_median[0]
                    record[f"{prefix}_ci_upper_med"] = ci_median[1]
                else:
                    record[f"{prefix}_mean"] = np.nan
                    record[f"{prefix}_median"] = np.nan
                    record[f"{prefix}_sem"] = np.nan
                    record[f"{prefix}_ci_lower_mean"] = np.nan
                    record[f"{prefix}_ci_upper_mean"] = np.nan
                    record[f"{prefix}_ci_lower_med"] = np.nan
                    record[f"{prefix}_ci_upper_med"] = np.nan
            stats_records.append(record)
        stats_df = pd.DataFrame(stats_records)
        stats_df.set_index(group_col, inplace=True)
        return stats_df
    
    def aggregate_hourly_stats(self, df, hour_col="Hour"):
        """
        Aggregate hourly statistics across all rocks.

        Parameters:
            df (DataFrame): Combined wide-format DataFrame for all rocks.
            hour_col (str): Column name for hour-of-day (default "Hour").

        Returns:
            DataFrame: Aggregated hourly statistics computed on the pooled data.
            (The same format as compute_hourly_stats.)
        """
        # Here we simply call compute_hourly_stats on the pooled DataFrame.
        return self.compute_hourly_stats(df, hour_col=hour_col)
    

    @staticmethod
    def compute_elapsed_days(times: pd.Series) -> pd.Series:
        """
        Convert a Series of elapsed_date_time values into elapsed days (as float)
        relative to the first observation.

        Parameters:
            times (pd.Series): Series of datetime values.

        Returns:
            pd.Series: Series of elapsed days.
        """
        t0 = times.min()
        # Calculate total seconds difference and convert to days
        elapsed_days = (times - t0).dt.total_seconds() / (3600 * 24)
        return elapsed_days
    
    def compute_total_duration_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics over the entire observation period by grouping on elapsed days.
        The function creates an 'elapsed_day' column (as integer days since the first observation)
        and then groups by this column.

        Parameters:
            df (pd.DataFrame): Wide-format DataFrame with an 'elapsed_date_time' column.

        Returns:
            pd.DataFrame: Aggregated statistics indexed by elapsed_day.
        """
        df = df.copy()
        # Compute elapsed days (as integer days)
        df["elapsed_day"] = self.compute_elapsed_days(pd.to_datetime(df["elapsed_date_time"])).astype(int)
        return self.compute_hourly_stats(df, group_col="elapsed_day")
    
    def compute_overall_stats(self, df: pd.DataFrame) -> dict:
        """
        Compute overall (unbinned) statistics for a single rock's full dataset.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns:
                  date_time, temperature_in_C, temperature_out_C,
                  temperature_diff_C, humidity_perc_RH.
        
        Returns:
            dict: Dictionary with overall statistics for each variable.
                  Each value is another dict with keys: mean, median, sem,
                  ci_lower, ci_upper.
        """
        stats = {}
        variables = {
            "temperature_in_C": "inside",
            "temperature_out_C": "outside",
            "temperature_diff_C": "diff",
            "humidity_perc_RH": "hum"
        }
        for var in variables.keys():
            values = df[var].dropna().values
            if len(values) == 0:
                stats[var] = {"mean": np.nan, "median": np.nan, "sem": np.nan,
                              "ci_lower": np.nan, "ci_upper": np.nan}
                continue
            mean_val = np.nanmean(values)
            median_val = np.nanmedian(values)
            # Standard error (using sample std)
            sem_val = np.nanstd(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            ci_lower, ci_upper = self.bootstrap_confidence_interval(values, statistic=np.nanmean)
            stats[var] = {"mean": mean_val, "median": median_val, "sem": sem_val,
                          "ci_lower": ci_lower, "ci_upper": ci_upper}
        return stats

    def aggregate_overall_stats(self, df: pd.DataFrame) -> dict:
        """
        Given the wide DataFrame for all rocks, compute the overall (full duration)
        statistics aggregated across rocks. It computes the overall mean of each variable
        for each rock and then computes the average (and SEM, 95% CI) of those rock-level
        means.
        
        Parameters:
            df (pd.DataFrame): Wide-format DataFrame with a "rock" column.
            
        Returns:
            dict: Aggregated statistics with keys for each variable.
        """
        rock_groups = df.groupby("rock")
        rock_stats = {var: [] for var in ["temperature_in_C", "temperature_out_C", "temperature_diff_C", "humidity_perc_RH"]}
        for rock, rock_df in rock_groups:
            overall = self.compute_overall_stats(rock_df)
            for var in rock_stats.keys():
                rock_stats[var].append(overall[var]["mean"])
        agg_stats = {}
        for var, vals in rock_stats.items():
            vals = np.array(vals, dtype=float)
            agg_mean = np.nanmean(vals)
            agg_sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci_lower, ci_upper = self.bootstrap_confidence_interval(vals, statistic=np.nanmean)
            # For median across rocks, take median of rock-level means
            agg_median = np.nanmedian(vals)
            agg_stats[var] = {"mean": agg_mean, "median": agg_median, "sem": agg_sem,
                              "ci_lower": ci_lower, "ci_upper": ci_upper}
        return agg_stats
    @staticmethod
    def aggregate_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the full-duration (raw) data across rocks by elapsed hour.
        For each row in the wide-format table, compute an "elapsed_hour" value as the
        number of hours since the base date (January 1, 2000). Then, group by the integer
        elapsed hour and compute nan-aware statistics (mean, median, SEM, and 95% CI)
        for each variable.
        
        Parameters:
            df (pd.DataFrame): Wide-format DataFrame containing full data for all rocks.
            
        Returns:
            pd.DataFrame: Aggregated hourly statistics. The index is the elapsed hour,
                        and columns include for each variable:
                        mean, median, sem, ci_lower, and ci_upper.
        """
        # Compute elapsed_hour from elapsed_date_time.
        base_date = pd.Timestamp("2000-01-01")
        df = df.copy()
        df["elapsed_hour"] = ((pd.to_datetime(df["elapsed_date_time"]) - base_date)
                              / pd.Timedelta(hours=1)).astype(int)
        
        # Variables to aggregate
        variables = ["temperature_in_C", "temperature_out_C", "temperature_diff_C", "humidity_perc_RH"]
        records = []
        for hour, group in df.groupby("elapsed_hour"):
            rec = {"elapsed_hour": hour}
            for var in variables:
                vals = group[var].values
                mean_val = np.nanmean(vals) if np.sum(~np.isnan(vals)) > 0 else np.nan
                median_val = np.nanmedian(vals) if np.sum(~np.isnan(vals)) > 0 else np.nan
                count = np.sum(~np.isnan(vals))
                sem_val = np.nanstd(vals, ddof=1) / np.sqrt(count) if count > 1 else 0.0
                ci_lower, ci_upper = DataAnalyzer.bootstrap_confidence_interval(vals, statistic=np.nanmean)
                rec[f"{var}_mean"] = mean_val
                rec[f"{var}_median"] = median_val
                rec[f"{var}_sem"] = sem_val
                rec[f"{var}_ci_lower"] = ci_lower
                rec[f"{var}_ci_upper"] = ci_upper
            records.append(rec)
        return pd.DataFrame(records).sort_values("elapsed_hour").reset_index(drop=True)