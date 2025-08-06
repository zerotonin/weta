# datafile.py
import re
import pandas as pd
import io
import os

class DataFile:
    """Parses a raw ASCII log file (temperature or humidity) into a pandas DataFrame."""
    
    def __init__(self, file_path: str):
        """
        Initialize DataFile by extracting metadata from the filename.
        Metadata includes rock ID, location (in/out), and measure type (temperature/humidity).
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.rock_id, self.location, self.measure = self._parse_filename(self.file_name)
    
    @staticmethod
    def _parse_filename(file_name: str):
        """
        Extract rock ID, location, and measure from a filename.
        Returns: (rock_id as str, location as str, measure as str).
        """
        name = file_name.lower()
        name = re.sub(r'\.[^.]+$', '', name)
        name = name.replace('_', ' ').replace('-', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
        pattern = re.compile(r"(?:rock\s*)?(\d+)\s*(in|inside|internal|out|outside|external)?\s*(temp|temperature|hum|humidity)", re.IGNORECASE)
        match = pattern.search(name)
        if not match:
            raise ValueError(f"Filename '{file_name}' does not match expected pattern")
        rock_num, location, meas = match.groups()
        location = location.lower() if location else 'in'
        meas = meas.lower()
        if meas.startswith('temp'):
            meas = 'temperature'
        elif meas.startswith('hum'):
            meas = 'humidity'
        return rock_num, location, meas
    
    def _read_file_text(self) -> str:
        """Read the file contents, trying multiple encodings for robustness."""
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
        last_error = None
        for enc in encodings:
            try:
                with open(self.file_path, 'r', encoding=enc, errors='strict') as f:
                    return f.read()
            except UnicodeDecodeError as e:
                last_error = e
                continue
        if last_error:
            raise last_error
        raise IOError(f"Failed to read file: {self.file_path}")
    
    def read(self) -> pd.DataFrame:
        """
        Read the file contents by scanning for the header marker "Date/Time,Unit,Value"
        and then reading all lines below that as CSV data. If the first column cannot be
        parsed as a datetime but the second column can, swap the columns.
        
        Returns a DataFrame with:
            - 'date_time' (as datetime64)
            - A measurement column, renamed to either "temperature_in_C" (or out) or "humidity_perc_RH"
        """
        text = self._read_file_text()
        lines = text.splitlines()
        
        # Look for the header marker exactly (ignoring case)
        header_marker = "Date/Time,Unit,Value"
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower() == header_marker.lower():
                start_idx = i + 1
                break
        if start_idx is None:
            raise ValueError(f"Header marker '{header_marker}' not found in file: {self.file_name}")
        
        # Use header marker for column names
        columns = [col.strip() for col in header_marker.split(',')]
        data_str = "\n".join(lines[start_idx:]).strip()
        data_str = data_str.replace("\u202f", " ")
        if not data_str:
            raise ValueError(f"No data found in file: {self.file_name}")
        
        try:
            df = pd.read_csv(io.StringIO(data_str), header=None, names=columns)
        except Exception as e:
            raise ValueError(f"Error parsing CSV data from file {self.file_name}: {e}")
        
        # Try to parse the first column ("Date/Time") as datetime using our expected format
        df["parsed_dt"] = pd.to_datetime(df["Date/Time"], format='%d/%m/%y %I:%M:%S %p', errors='coerce')
        
        # If most of the first column fails to parse, try a four digit year format
        if df["parsed_dt"].isna().sum() > len(df)/2:
            df["parsed_dt"] = pd.to_datetime(df["Date/Time"], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
        # If most of the first column fails to parse, trysecond column
        if df["parsed_dt"].isna().sum() > len(df)/2:
            df["parsed_dt"] = pd.to_datetime(df.iloc[:, 1], format='%d/%m/%y %I:%M:%S %p', errors='coerce')
            measurement_col = df.columns[0]
        else:
            measurement_col = df.columns[-1]
        # If still NaT, raise an error
        if df["parsed_dt"].isna().all():
            raise ValueError(f"Could not parse any datetime values in file: {self.file_name}")
        
        # Drop the original "Date/Time" and "Unit" columns if they exist
        df = df.drop(columns=["Date/Time", "Unit"], errors='ignore')
        # Remove any rows where parsed_dt is still NaT
        df = df.dropna(subset=["parsed_dt"]).reset_index(drop=True)
        
        # Rename the parsed datetime column to "date_time"
        df = df.rename(columns={"parsed_dt": "date_time"})
        
        # Rename the measurement column according to sensor type and location
        if self.measure == "temperature":
            new_val_name = f"temperature_{self.location}_C"
        else:
            new_val_name = "humidity_perc_RH"
        df = df.rename(columns={measurement_col: new_val_name})
        
        self.data = df.copy()
        return self.data
