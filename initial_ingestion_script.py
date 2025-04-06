#!/usr/bin/env python3
"""
Run the data processing pipeline for the burrow_temperature project.

This script uses the DataLoader, DataFile, and DataAnalyzer classes to:
  - Identify and group raw data files from the raw_data folder.
  - Parse and process the files for each rock (including computing elapsed time and temperature differences).
  - Combine the processed data into one wide-format DataFrame.
  - Save the resulting DataFrame to:
       /home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv
"""

import os
from data_loader import DataLoader

def main():
    # Define the paths based on your project tree
    raw_data_dir = "/home/geuba03p/weta_project/burrow_temperature/raw_data"
    output_csv = "/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv"
    
    # Create a DataLoader instance which will:
    # - Scan raw_data_dir for all data files (using DataFile for parsing)
    # - Group files by rock and perform initial analysis via DataAnalyzer functions.
    loader = DataLoader(raw_data_dir, output_csv)
    
    # Process all files and combine the data into one wide-format DataFrame.
    final_df = loader.process_all()
    
    # Print a summary of the resulting data.
    print("Final combined data shape:", final_df.shape)
    print("First few rows:")
    print(final_df.head())
    
    print(f"\nCombined data successfully saved to: {output_csv}")

if __name__ == '__main__':
    main()
