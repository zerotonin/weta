# data_loader.py
import os
import pandas as pd
from datafile import DataFile
from data_analyzer import DataAnalyzer

class DataLoader:
    """Loads and processes all raw data files from a directory into a consolidated DataFrame."""
    def __init__(self, raw_data_dir: str, output_file: str):
        self.raw_data_dir = raw_data_dir
        self.output_file = output_file
        # Container for per-rock data (dictionary mapping rock_id -> combined DataFrame)
        self.rock_data = {}
    
    def scan_files(self) -> dict:
        """
        Walk through the directory tree starting at raw_data_dir and identify all relevant files.
        Returns a dictionary grouping file paths by rock ID and measurement type.
        e.g., {rock_id: {'in_temp': path, 'out_temp': path, 'humidity': path, ...}, ...}
        """
        file_groups = {}
        # Use os.walk to traverse directories recursively&#8203;:contentReference[oaicite:8]{index=8}
        for dirpath, _, filenames in os.walk(self.raw_data_dir):
            for fname in filenames:
                # Only consider likely data files (e.g., .csv or .txt files)
                if not fname.lower().endswith(('.csv', '.txt')):
                    continue
                file_path = os.path.join(dirpath, fname)
                # Try to parse metadata from filename
                try:
                    rock_id, location, measure = DataFile._parse_filename(fname)
                except ValueError:
                    # Skip files that don't match the pattern (or handle separately if needed)
                    continue
                # Determine the key for this file in the group dict
                if measure == 'temperature':
                    key = 'in_temp' if location == 'in' else 'out_temp'
                elif measure == 'humidity':
                    key = 'humidity'  # humidity (assuming it's an internal sensor by default)
                # Initialize the rock entry if not present
                if rock_id not in file_groups:
                    file_groups[rock_id] = {}
                # If a file for the same key already exists, we may have duplicate or shared sensor
                # We'll handle shared outside sensors next
                file_groups[rock_id][key] = file_path
        # Handle shared outside sensors by checking for rocks missing 'out_temp'
        rock_ids = sorted(file_groups.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for rid in rock_ids:
            # If this rock has no outside temp but a numeric neighbor does, assign neighbor's out_temp
            if 'out_temp' not in file_groups[rid]:
                # Check neighbor with ID one less or one more (as string keys)
                rid_int = int(rid) if rid.isdigit() else None
                if rid_int is not None:
                    neighbor1 = str(rid_int - 1)
                    neighbor2 = str(rid_int + 1)
                    if neighbor1 in file_groups and 'out_temp' in file_groups[neighbor1]:
                        file_groups[rid]['out_temp'] = file_groups[neighbor1]['out_temp']
                    elif neighbor2 in file_groups and 'out_temp' in file_groups[neighbor2]:
                        file_groups[rid]['out_temp'] = file_groups[neighbor2]['out_temp']
        return file_groups
    
    def process_all(self):
        """
        Parse all files, perform analysis for each rock, and aggregate the results.
        Saves the final wide-format DataFrame to the output CSV.
        """
        file_groups = self.scan_files()
        all_rocks_df_list = []
        for rock_id, files in file_groups.items():
            # Parse each relevant file for this rock using DataFile
            data_dict = {}
            if 'in_temp' in files:
                df_in = DataFile(files['in_temp']).read()
                data_dict['in_temp'] = df_in
            else:
                # If no inside temperature, skip this rock (inside temp is essential for elapsed time)
                continue
            if 'out_temp' in files:
                # Avoid reading the same shared file multiple times: cache by path if needed
                df_out = DataFile(files['out_temp']).read()
                data_dict['out_temp'] = df_out
            if 'humidity' in files:
                df_hum = DataFile(files['humidity']).read()
                data_dict['humidity'] = df_hum
            # Perform initial analysis to combine and compute derived metrics
            rock_df = DataAnalyzer.initial_analysis(data_dict)
            # Add rock identifier column
            rock_df.insert(0, 'rock', rock_id)  # insert as first column
            all_rocks_df_list.append(rock_df)
            # Store the per-rock data (could also store in self.rock_data if needed for later)
            self.rock_data[rock_id] = rock_df
        # Concatenate all rocks' DataFrames into one
        if all_rocks_df_list:
            final_df = pd.concat(all_rocks_df_list, ignore_index=True, sort=False)
        else:
            final_df = pd.DataFrame(columns=['rock','date_time','elapsed_date_time',
                                            'temperature_in_C','temperature_out_C',
                                            'temperature_diff_C','humidity_perc_RH'])
        # Optional: sort final data by rock then date_time for neatness
        final_df = final_df.sort_values(['rock', 'date_time']).reset_index(drop=True)
        # Save to CSV (wide-format table)
        final_df.to_csv(self.output_file, index=False)
        return final_df
