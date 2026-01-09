"""preprocessing.py
Functions to load and clean the processed CSV used by notebooks.
"""
from pathlib import Path
import pandas as pd


def detect_sep(path: Path) -> str:
    from csv import Sniffer
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2048)
        dialect = Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter
    except Exception:
        return ','


def _clean_header_line(header_line: str) -> list:
    # Strip extraneous quotes and whitespace, then split by common separators
    h = header_line.strip().strip('"').strip()
    # Sometimes header might be quoted repeatedly e.g., '"""ORDERID""";...'
    h = h.replace('"""', '"')
    # Choose separator by checking common delimiters
    for sep in [';', ',', '\t', '|']:
        if sep in h:
            parts = [c.strip().strip('"') for c in h.split(sep)]
            return parts, sep
    # fallback: whitespace split
    return [c.strip().strip('"') for c in h.split()], ','


def robust_read_processed(path1: str = 'data/processed_data.csv', path2: str = 'processed_data.csv') -> tuple:
    import os
    import csv as _csv
    
    candidates = [path1, path2]
    # If using from webapp, we might need absolute path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    for pth in candidates:
        path = pth
        if not os.path.exists(path):
            path = os.path.join(base_dir, pth)
        if not os.path.exists(path):
            continue
            
        # Try common encodings
        for enc in ['mac_turkish', 'cp1254', 'utf-8', 'iso-8859-9', 'latin1', 'cp1252']:
            try:
                # Detect separator
                sep = ','
                with open(path, 'r', encoding=enc) as f:
                    sample = f.read(1024)
                    # Robust check: if first line contains ';', it is likely the separator
                    first_line = sample.splitlines()[0] if sample else ''
                    if ';' in first_line:
                        sep = ';'
                    else:
                        try:
                            dialect = _csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
                            sep = dialect.delimiter
                        except Exception:
                            pass
                
                df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines='skip')
                return df, pth
            except Exception:
                continue
    raise FileNotFoundError('Processed data file not found or unreadable')
    raise FileNotFoundError('Processed data file not found or unreadable')


def clean_numeric_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    import numpy as np
    if cols is None:
        cols = ['TOTALBASKET','UNITPRICE','TOTALPRICE','AMOUNT']
    for col in cols:
        if col in df.columns:
            s = df[col].astype(str)
            # Handle formats like '1.234,56' (dot thousands separator, comma decimal)
            mask = s.str.contains(r'\.') & s.str.contains(',')
            s.loc[mask] = s.loc[mask].str.replace('.', '', regex=False)
            # Remove whitespace
            s = s.str.replace(r'\s', '', regex=True)
            # Replace comma decimal separator with dot
            s = s.str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce')
    return df


def parse_dates(df: pd.DataFrame, col='DATE_') -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df


import csv

def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names and string columns to avoid embedded newlines/quotes that break CSV output."""
    df = df.copy()
    # Clean column names
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        s = df[col].astype(str)
        s = s.str.replace('\r|\n|\t', ' ', regex=True)
        s = s.str.replace('\x00', '', regex=False)
        s = s.str.replace('"{2,}', '"', regex=True)
        s = s.str.strip()
        df[col] = s
    return df


def save_processed(df: pd.DataFrame, out='data/processed_data.csv', sep=';') -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_clean = sanitize_for_csv(df)
    # Use explicit separator and encoding and safe quoting to avoid malformed files
    df_clean.to_csv(out, index=False, sep=sep, encoding='utf-8', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
