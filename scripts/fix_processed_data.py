"""Validate and fix data/processed_data.csv

Actions:
- Load file robustly (try src.preprocessing.robust_read_processed if available)
- Normalize numeric columns (clean_numeric_columns)
- Convert likely ID columns to integers (round)
- Parse date column if present
- Save cleaned file (overwrite original)
- Produce a JSON report in reports/data_fix_report.json with summary statistics
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Helpers
try:
    from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
except Exception:
    robust_read_processed = None
    clean_numeric_columns = None
    parse_dates = None

INPUT = 'data/processed_data.csv'
REPORT = 'reports/data_fix_report.json'

def try_load(path):
    if robust_read_processed is not None:
        try:
            df, sep = robust_read_processed(path)
            return df, sep
        except Exception:
            pass
    # fallback attempts
    encodings = ['utf-8', 'latin1', 'cp1252']
    seps = [';', ',', '\t']
    last_exc = None
    for enc in encodings:
        for s in seps:
            try:
                df = pd.read_csv(path, sep=s, encoding=enc)
                return df, s
            except Exception as e:
                last_exc = e
    raise RuntimeError(f"Failed to read {path}: {last_exc}")


def summarize_df(df):
    summary = {
        'shape': df.shape,
        'columns': {c: str(dtype) for c, dtype in df.dtypes.items()},
        'null_counts': df.isna().sum().to_dict()
    }
    return summary


def main():
    df, sep = try_load(INPUT)
    before = summarize_df(df)

    report = {'loaded_sep': sep, 'before': before, 'fixes': {}, 'timestamp': datetime.utcnow().isoformat()}

    # Normalize numeric columns using helper if available
    if clean_numeric_columns is not None:
        # attempt to detect numeric-ish columns by name
        numeric_hint = ['TOTALBASKET','UNITPRICE','TOTALPRICE','AMOUNT','AGE']
        cols_to_try = [c for c in numeric_hint if c in df.columns]
        try:
            df = clean_numeric_columns(df, cols=cols_to_try)
            report['fixes']['clean_numeric_columns'] = {'cols': cols_to_try}
        except Exception as e:
            report['fixes']['clean_numeric_columns_error'] = str(e)

    # ID-like columns: round and convert to integer if present
    id_candidates = ['ORDERID','ORDERDETAILID','ITEMID','USERID']
    id_present = [c for c in id_candidates if c in df.columns]
    id_report = {}
    for c in id_present:
        # coerce to numeric
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # round to nearest integer
        df[c] = df[c].round()
        # convert to Int64 to allow NA
        try:
            df[c] = df[c].astype('Int64')
        except Exception:
            # fallback to float
            df[c] = df[c]
        after_na = df[c].isna().sum()
        id_report[c] = {'before_na': int(before_na), 'after_na': int(after_na)}
    if id_report:
        report['fixes']['id_columns'] = id_report

    # Parse date column if exists
    date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'DATE_' == c]
    date_report = {}
    for c in date_cols:
        before_na = df[c].isna().sum()
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=False, infer_datetime_format=True)
            after_na = df[c].isna().sum()
            date_report[c] = {'before_na': int(before_na), 'after_na': int(after_na)}
        except Exception as e:
            date_report[c] = {'error': str(e)}
    if date_report:
        report['fixes']['date_parsing'] = date_report

    # Ensure categorical columns are strings
    cat_hints = ['ITEMCODE','BRANCH_ID','USERNAME_','REGION','CITY','DISTRICT','STATUS_','USERGENDER']
    cat_present = [c for c in cat_hints if c in df.columns]
    for c in cat_present:
        df[c] = df[c].astype(str).str.strip()

    # Final summary
    after = summarize_df(df)
    report['after'] = after

    # Save cleaned file (overwrite original) using sep=';'
    os.makedirs(os.path.dirname(INPUT), exist_ok=True)
    df.to_csv(INPUT, sep=';', index=False)

    # write report
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print('Fix complete. Report written to', REPORT)


if __name__ == '__main__':
    main()
