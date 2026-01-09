
import sys
import os
import csv
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), 'data')

def robust_read_processed_data():
    candidates = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not candidates:
        raise FileNotFoundError("No CSV found in data/")
    
    filename = 'processed_data.csv' if 'processed_data.csv' in candidates else candidates[0]
    path = os.path.join(DATA_DIR, filename)

    print(f"Reading {path}...")

    # Try common encodings
    for encoding in ['mac_turkish', 'cp1254', 'utf-8', 'iso-8859-9', 'latin1', 'cp1252']:
        try:
            print(f"Testing encoding: {encoding}")
            # Check separator
            with open(path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    sep = dialect.delimiter
                    print(f"Sniffer detected sep: '{sep}'")
                except csv.Error:
                    print("Sniffer failed.")
                    if ';' in sample.splitlines()[0]:
                         sep = ';'
                         print("Fallback to sep: ';'")

            df = pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines='skip')
            print(f"Success with {encoding}! Shape: {df.shape}")
            return df, filename
        except Exception as e:
            print(f"Failed {encoding}: {e}")
            continue
            
    raise ValueError(f"Could not read {filename} with any standard encoding")

try:
    df, filename = robust_read_processed_data()
    print("Columns:", df.columns.tolist())
    print(df.head(2))
except Exception as e:
    print(e)
