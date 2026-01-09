"""Repair a malformed data/processed_data.csv without overwriting original.

Usage:
    python scripts/repair_processed.py --input data/processed_data.csv --backup data/cleandata_backup/processed_data.csv --output data/processed_data_fixed.csv

The script will try to read the backup header to get canonical column names and then
reconstruct rows from the malformed file using tolerant splitting.
"""
import argparse
from pathlib import Path
import csv
import pandas as pd
import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path to import src
proj_root = str(_Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed_data.csv')
    parser.add_argument('--backup', default='data/cleandata_backup/processed_data.csv')
    parser.add_argument('--output', default='data/processed_data_fixed.csv')
    args = parser.parse_args()

    inp = Path(args.input)
    backup = Path(args.backup)
    out = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(inp)
    if not backup.exists():
        raise FileNotFoundError(backup)

    # Read header from backup
    with backup.open('r', encoding='latin1', errors='ignore') as f:
        first = f.readline().strip()
    # Use the backup header split by ';' as canonical columns
    cols = [c.strip() for c in first.split(';')]

    # Read raw lines from input and tolerant-split by ';'
    rows = []
    with inp.open('r', encoding='latin1', errors='ignore') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    for ln in lines[1:]:
        parts = ln.split(';')
        if len(parts) == len(cols):
            rows.append(parts)
            continue
        # try csv reader with semicolon
        try:
            parsed = next(csv.reader([ln], delimiter=';'))
            if len(parsed) == len(cols):
                rows.append(parsed)
                continue
        except Exception:
            pass
        # pad/truncate
        parts = (parts + [''] * len(cols))[:len(cols)]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=cols)
    # minimal cleaning
    from src.preprocessing import clean_numeric_columns
    df = clean_numeric_columns(df)
    df.to_csv(out, index=False, sep=';', encoding='utf-8')
    print(f'Wrote repaired dataset to {out} with shape {df.shape}')


if __name__ == '__main__':
    main()
