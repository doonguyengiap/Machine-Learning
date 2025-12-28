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


def robust_read_processed(path1: str = 'data/processed_data.csv', path2: str = 'processed_data.csv') -> tuple:
    """Try reading processed data from common locations and return (df, sep).

    Returns:
        (pd.DataFrame, str): the dataframe and the detected separator
    """
    for p in [path1, path2]:
        path = Path(p)
        if not path.exists():
            continue
        sep = detect_sep(path)
        try:
            df = pd.read_csv(path, sep=sep, encoding='utf-8', engine='python', on_bad_lines='skip')
            return df, sep
        except Exception:
            df = pd.read_csv(path, sep=';', encoding='latin1', engine='python', on_bad_lines='skip')
            return df, ';'
    raise FileNotFoundError('Processed data file not found')


def clean_numeric_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    import numpy as np
    if cols is None:
        cols = ['TOTALBASKET','UNITPRICE','TOTALPRICE','AMOUNT']
    for col in cols:
        if col in df.columns:
            s = df[col].astype(str)
            # Handle formats like '1.234,56' (dot thousands separator, comma decimal)
            mask = s.str.contains('\.') & s.str.contains(',')
            s.loc[mask] = s.loc[mask].str.replace('.', '', regex=False)
            # Remove whitespace
            s = s.str.replace('\s', '', regex=True)
            # Replace comma decimal separator with dot
            s = s.str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce')
    return df


def parse_dates(df: pd.DataFrame, col='DATE_') -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df


def save_processed(df: pd.DataFrame, out='data/processed_data.csv') -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
