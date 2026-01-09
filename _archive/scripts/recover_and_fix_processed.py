"""Recover and fix data/processed_data.csv when separator/encoding caused single-column read.

Steps:
- Read raw file bytes and decode latin1
- Split lines, parse header by semicolon
- Split each row by semicolon (pad/trim) to ensure consistent columns
- Build DataFrame, clean numeric columns, cast IDs, parse dates
- Overwrite the original CSV with sep=';'
- Write a JSON report to reports/data_fix_report.json
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.preprocessing import clean_numeric_columns
except Exception:
    clean_numeric_columns = None

INPUT = 'data/processed_data.csv'
REPORT = 'reports/data_fix_report.json'


def parse_semicolon_text(text):
    lines = text.splitlines()
    if not lines:
        raise RuntimeError('Empty file')
    header = lines[0]
    cols = header.split(';')
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        # split but allow extra separators in last field: we will split into at most len(cols)-1 parts and keep last as remainder
        parts = line.split(';')
        if len(parts) < len(cols):
            # pad
            parts += [''] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            # join extras into last column
            parts = parts[:len(cols)-1] + [';'.join(parts[len(cols)-1:])]
        rows.append(parts)
    df = pd.DataFrame(rows, columns=[c.strip() for c in cols])
    return df


def main():
    # read raw bytes and decode latin1 to avoid decode errors
    with open(INPUT, 'rb') as fh:
        raw = fh.read().decode('latin1')

    # quick check
    total_lines = raw.count('\n') + 1
    # parse by semicolon
    df = parse_semicolon_text(raw)

    report = {
        'detected_lines': int(total_lines),
        'parsed_shape': list(df.shape),
        'timestamp': datetime.utcnow().isoformat(),
        'notes': []
    }

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns are numeric using helper
    numeric_hint = ['TOTALBASKET','UNITPRICE','TOTALPRICE','AMOUNT','Age']
    cols_to_try = [c for c in numeric_hint if c in df.columns]
    if clean_numeric_columns is not None and cols_to_try:
        try:
            df = clean_numeric_columns(df, cols=cols_to_try)
            report['notes'].append(f'Applied clean_numeric_columns to {cols_to_try}')
        except Exception as e:
            report['notes'].append(f'clean_numeric_columns failed: {e}')

    # Fix ID-like columns
    id_candidates = ['ORDERID','ORDERDETAILID','ITEMID','USERID']
    id_present = [c for c in id_candidates if c in df.columns]
    id_report = {}
    for c in id_present:
        before_na = int(df[c].isna().sum())
        df[c] = pd.to_numeric(df[c].str.replace(',', '').str.replace('.', ''), errors='coerce')
        # rounding
        df[c] = df[c].round()
        # convert to int (using Int64 to allow NA)
        try:
            df[c] = df[c].astype('Int64')
        except Exception:
            pass
        after_na = int(df[c].isna().sum())
        id_report[c] = {'before_na': before_na, 'after_na': after_na}
    if id_report:
        report['id_report'] = id_report

    # Parse dates
    date_cols = [c for c in df.columns if 'DATE' in c.upper() or c.lower().endswith('date')]
    date_report = {}
    for c in date_cols:
        before_na = int(df[c].isna().sum())
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=False, infer_datetime_format=True)
            after_na = int(df[c].isna().sum())
            date_report[c] = {'before_na': before_na, 'after_na': after_na}
        except Exception as e:
            date_report[c] = {'error': str(e)}
    if date_report:
        report['date_report'] = date_report

    # Final shape and write
    report['final_shape'] = list(df.shape)

    # Overwrite file with semicolon sep and standard decimal (dot)
    df.to_csv(INPUT, sep=';', index=False)

    # write report
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print('Recovery complete, report written to', REPORT)


if __name__ == '__main__':
    main()
