# Burrow Temperature Analysis Pipeline

This project is a Python-based pipeline for ingesting, analyzing, and visualizing rock temperature and humidity data from iButton sensors deployed in burrow studies. The pipeline is designed to handle messy, real-world data (including inconsistent file naming and missing values) and to provide robust statistical summaries and visualizations using bootstrapping and nan-aware computations.

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Data Ingestion](#data-ingestion)
  - [24‑Hour Analysis](#24-hour-analysis)
  - [Full Duration Analysis](#full-duration-analysis)
  - [Across‑Rocks Analysis (Hourly Aggregation)](#across-rocks-analysis-hourly-aggregation)
- [Module Descriptions](#module-descriptions)
  - [DataFile](#datafile)
  - [DataLoader](#dataloader)
  - [DataAnalyzer](#dataanalyzer)
  - [PlotGenerator](#plotgenerator)
- [Extending and Customizing the Pipeline](#extending-and-customizing-the-pipeline)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The pipeline processes raw sensor data files, computes derived metrics such as elapsed time (using a base date of January 1, 2000 to preserve hour-of-day), temperature differences (inside minus outside), and statistical summaries (mean, median, SEM, and 95% bootstrap confidence intervals). It then generates standardized plots for each rock and across rocks—both on an hourly basis (24‑hour analysis) and over the full observation period.

The project is built using a modular design:
- **Data Ingestion** is handled by the `DataLoader` and `DataFile` classes.
- **Data Analysis** is performed by the `DataAnalyzer` class.
- **Plot Generation** is managed by the `PlotGenerator` class.
- **Driver Scripts** coordinate the pipeline steps.

## File Structure

```bash
/home/geuba03p/weta_project/
└── burrow_temperature/
    ├── raw_data/
    │   ├── Rock 10 in temp.csv
    │   ├── Rock 10 out temp.csv
    │   ├── Rock 11 in temp.csv
    │   ├── Rock 15 in temp.csv
    │   ├── Rock 15 out hum.csv
    │   ├── Rock 15 out temp.csv
    │   ├── ... (other sensor files)
    ├── data/
    │   ├── initial_wide_format.csv         # Combined wide-format data after ingestion
    │   ├── 24h_hourly_averages.csv           # Hourly aggregated stats for 24h analysis
    │   ├── full_duration_overall_stats.csv    # Overall full-duration stats per rock
    │   └── full_duration_hourly_aggregates.csv  # Aggregated hourly stats across rocks (full duration)
    ├── figures/
    │   ├── 24h_perRock/                     # 24-hour analysis figures for each rock
    │   ├── 24h_comparisson/                 # Across-rock 24-hour aggregated figures
    │   ├── full_perRock/                    # Full-duration time series plots for each rock
    │   └── full_acrossRocks_hourly/          # Aggregated hourly plots for full-duration analysis
    ├── initial_ingestion_script.py         # Driver script for data ingestion
    ├── datafile.py                         # Contains DataFile class for parsing raw files
    ├── data_loader.py                      # Contains DataLoader class for grouping and processing files
    ├── data_analyzer.py                    # Contains DataAnalyzer class for computing derived metrics  
    ├── plot_generator.py                   # Contains PlotGenerator class for creating standardized plots
    ├── analysis_24h_plots.py               # Driver script for 24-hour (hourly) analysis and plotting
    ├── analysis_full_data_across_rocks.py    # Driver script for full-duration analysis across rocks
    └── analyse_total_duration_script.py      # Driver script for full observation duration analysis
```


## Requirements

- Python 3.6+
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

## Setup Instructions

1. **Clone the Repository or Copy the Project Folder**

   Ensure that the project folder is organized as shown above.

2. **Install Dependencies**

   Use pip to install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```

## Usage
### Data Ingestion

Run the `initial_ingestion_script.py` to scan the raw data files, parse them using the `DataFile` and `DataLoader` classes, and produce a combined wide-format CSV file:

`python initial_ingestion_script.py`

The output file is saved to:

`/home/geuba03p/weta_project/burrow_temperature/data/initial_wide_format.csv`

### 24‑Hour Analysis

Run the `analysis_24h_plots.py` script to perform hourly analysis. This script:

- Loads the combined CSV.
    
- Computes hourly statistics (mean, median, SEM, 95% CI) per rock.
    
- Generates dual-axis temperature plots and humidity plots for each rock.
    
- Aggregates and plots hourly statistics across rocks.
    

`python analysis_24h_plots.py`

### Full Duration Analysis

Run the `analysis_full_data_across_rocks.py` script to analyze the full observation duration (without hourly averaging) for each rock and across rocks.


`python analysis_full_data_across_rocks.py`

### Across‑Rocks Analysis (Hourly Aggregation)

Run the `analyse_total_duration_script.py` script to analyze the entire observation period on a per-day basis, compute statistics, and generate corresponding plots.


`python analyse_total_duration_script.py`

## Module Descriptions

### DataFile

- **Purpose**: Parses raw ASCII log files (temperature or humidity) into pandas DataFrames.
    
- **Key Functions**:
    
    - `_parse_filename`: Extracts metadata (rock ID, location, measure) from file names.
        
    - `read`: Reads the file after scanning for the header marker `"Date/Time,Unit,Value"` and returns a DataFrame with parsed datetime and measurement columns.
        

### DataLoader

- **Purpose**: Loads and processes all raw data files from the designated directory, groups them by rock, and combines them into a wide-format DataFrame.
    
- **Key Functions**:
    
    - `scan_files`: Recursively scans the raw data directory and groups files based on parsed metadata.
        
    - `process_all`: Uses `DataFile` to read files, applies initial analysis via `DataAnalyzer`, and produces the combined CSV.
        

### DataAnalyzer

- **Purpose**: Provides stateless functions for computing derived metrics and statistical summaries.
    
- **Key Functions**:
    
    - `compute_elapsed_daytime`: Computes elapsed time (with a base date of January 1, 2000) to preserve the hour-of-day.
        
    - `compute_temp_diff`: Computes the temperature difference (inside minus outside).
        
    - `initial_analysis`: Combines raw DataFrames for a single rock into one wide-format table, merging temperature and humidity data.
        
    - `bootstrap_confidence_interval`: Computes bootstrap confidence intervals for a given statistic.
        
    - `compute_hourly_stats`: Groups data by hour (or elapsed day) and computes nan-aware statistics (mean, median, SEM, CI).
        
    - `aggregate_hourly_stats`: Aggregates hourly statistics across all rocks.
        
    - `compute_elapsed_days`: Converts elapsed datetime values to elapsed days.
        
    - `compute_total_duration_stats`: Computes statistics over the full observation period by grouping on elapsed days.
        
    - `compute_overall_stats`: Computes overall (unbinned) statistics for a rock's dataset.
        
    - `aggregate_overall_stats`: Aggregates overall statistics across rocks.
        

### PlotGenerator

- **Purpose**: Contains reusable plotting functions for generating standardized plots.
    
- **Key Functions**:
    
    - `save_figure`: Saves a matplotlib figure in both PNG and SVG formats.
        
    - `plot_temperature_dual_axis`: Creates a dual-axis plot for temperature data (inside, outside, and difference).
        
    - `plot_humidity_series`: Creates a time-series plot for humidity data.
        
    - `plot_full_timeseries`: Plots the full raw time series for a rock (with temperature and humidity).
        
    - `plot_hourly_aggregated_across_rocks`: Creates an aggregated dual-axis plot from hourly statistics across rocks.
        

## Extending and Customizing the Pipeline

The modular design of this pipeline allows you to easily extend and customize each stage:

- **Data Ingestion**: Modify `DataFile` or `DataLoader` if file naming conventions change.
    
- **Data Analysis**: Add new functions to `DataAnalyzer` for additional metrics or alternative statistical methods.
    
- **Plot Generation**: Reuse and extend functions in `PlotGenerator` to produce new visualization styles.
    
- **Driver Scripts**: Use or modify the provided driver scripts to run specific analyses or to integrate new modules.
    

## Troubleshooting

- **Date Parsing Issues**: Ensure that the raw data files follow the expected date/time format (`'%d/%m/%y %I:%M:%S %p'`). Adjust the format in `DataFile.read()` if necessary.
    
- **Missing Data**: The pipeline is designed to be nan-aware. However, if too many values are missing, statistical computations (e.g., SEM, CI) may not be reliable.
    
- **File Naming**: Files that do not match the expected naming pattern are skipped. Check file names if data seems to be missing from the final output.
    

## License