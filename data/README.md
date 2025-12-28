# Data directory

Canonical raw directory: `data/rawdata/`

Required raw files (place originals here, do not commit):
- Orders.csv
- Order_Details.csv
- Customers.csv

Notes on formats:
- Raw CSVs may use `;` as delimiter and `,` as decimal. Use `src.preprocessing.detect_sep` and `clean_numeric_columns` to read and normalize.
- Processed dataset (can be committed): `data/processed_data.csv` — this file is created by running `notebooks/data_preprocessing.ipynb`.
- For fast CI/testing, a small sample is available at `data/sample/processed_sample.csv`.
