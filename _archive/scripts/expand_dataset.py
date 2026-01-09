"""Expand an existing processed dataset to a larger size by sampling with replacement.

Usage:
    python scripts/expand_dataset.py --input data/processed_data.csv --output data/processed_data_150k.csv --n 150000 --seed 42 --jitter 0.01 --force

Notes:
- This script samples rows with replacement and applies small gaussian jitter to numeric columns to avoid exact duplicates.
- By default the separator used when writing will match the input separator detected by src.preprocessing.robust_read_processed if available.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# robust reader import will be attempted inside main() to avoid scoping issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed_data.csv')
    parser.add_argument('--output', default='data/processed_data_150k.csv')
    parser.add_argument('--n', type=int, default=150000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jitter', type=float, default=0.01, help='Relative jitter (fraction of std) to add to numeric cols')
    parser.add_argument('--force', action='store_true', help='Overwrite output if exists')
    args = parser.parse_args()

    inp = args.input
    out = args.output

    if os.path.exists(out) and not args.force:
        print(f"Output file {out} exists. Use --force to overwrite.")
        sys.exit(1)

    if not os.path.exists(inp):
        # fallback to sample
        fallback = 'data/sample/processed_sample.csv'
        if os.path.exists(fallback):
            print(f"Input {inp} not found, falling back to {fallback}")
            inp = fallback
        else:
            raise FileNotFoundError(f"Neither {args.input} nor {fallback} exist")

    # Ensure project root on sys.path for src import (if script run directly)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Try robust reader first (import inside function)
    try:
        from src.preprocessing import robust_read_processed as local_robust
    except Exception:
        local_robust = None

    sep = ','
    if local_robust is not None:
        try:
            df, sep = local_robust(inp)
            print(f"Loaded {inp} with detected sep='{sep}' shape={df.shape}")
        except Exception as e:
            print(f"local robust_read_processed failed: {e} — will try fallback readers")
            local_robust = None

    # Fallback: try combinations of encodings and separators
    if local_robust is None:
        encodings = ['utf-8', 'latin1', 'cp1252']
        seps = [',', ';', '\t']
        df = None
        last_exc = None
        for enc in encodings:
            for s in seps:
                try:
                    df = pd.read_csv(inp, sep=s, encoding=enc)
                    sep = s
                    print(f"Loaded {inp} with sep='{sep}' encoding='{enc}' shape={df.shape}")
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
            if last_exc is None:
                break
        if df is None:
            # Try python engine with sep=None to let pandas guess
            try:
                df = pd.read_csv(inp, sep=None, engine='python', encoding='latin1')
                sep = ','
                print(f"Loaded {inp} using python engine fallback with encoding=latin1 shape={df.shape}")
            except Exception as e:
                raise RuntimeError(f"Failed to read {inp}: {last_exc or e}")

    n = args.n
    rng = np.random.RandomState(args.seed)

    # Sample with replacement
    df_exp = df.sample(n=n, replace=True, random_state=args.seed).reset_index(drop=True)

    # Apply jitter to numeric columns
    numeric_cols = df_exp.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        print(f"Applying jitter to numeric columns ({len(numeric_cols)}): {numeric_cols}")
        for col in numeric_cols:
            col_std = df_exp[col].std()
            if np.isfinite(col_std) and col_std > 0:
                noise = rng.normal(loc=0.0, scale=col_std * args.jitter, size=n)
                df_exp[col] = df_exp[col] + noise
                # If column was originally non-negative, clip
                if df[col].min() >= 0:
                    df_exp[col] = df_exp[col].clip(lower=0)

    # Save with detected sep if possible
    write_sep = sep if sep else ','
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_exp.to_csv(out, sep=write_sep, index=False)
    print(f"Wrote expanded dataset to {out} (n={n}) using sep='{write_sep}'")


if __name__ == '__main__':
    main()
