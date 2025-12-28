<<<<<<< HEAD
# 3A Superstore Analysis

This repository contains notebooks and helper code to analyze and model the Superstore dataset.

## Setup

Recommended: create a conda environment from `environment.yml` or use `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate superstore
# or
python -m pip install -r requirements.txt
```

## Key files
- `data_preprocessing.ipynb` — data cleaning and saving `data/processed_data.csv`.
- `3A_Superstore.ipynb` — analysis, classification, time-series, market-basket, churn.
- `src/` — helper modules: `preprocessing.py`, `features.py`, `models.py`.
- `scripts/group_cv_tune.py` — run group-aware RandomizedSearchCV and save models.
- `scripts/generate_reports.py` — generate report and confusion matrix plots.
- `tests/` — pytest unit tests for core functions.

## Reproduce training and reports

Run hyperparameter tuning (group-aware):

```bash
python scripts/group_cv_tune.py
```

Generate human-readable report (confusion matrices + markdown):

```bash
python scripts/generate_reports.py
```

Verify saved models quickly:

```bash
PYTHONPATH=. python scripts/verify_models.py
```

## Tests & CI

Run tests locally with:

```bash
PYTHONPATH=. pytest -q
```

A GitHub Actions workflow is present at `.github/workflows/ci.yml` to run tests and lint.

## Notes
- The data uses semicolon delimiters and comma decimals; `src.preprocessing.detect_sep` and `clean_numeric_columns` handle these formats.
- Models are saved to `models/` and CV reports are under `reports/`.

## Project layout (reorganized)
```
project-name/                # Thư mục gốc của dự án
│
├── README.md                # Giới thiệu dự án, cách chạy code, mục tiêu, etc.
├── requirements.txt         # Liệt kê các thư viện Python cần thiết
│
├── data/                    # Chứa dữ liệu (processed + raw backups)
│   ├── processed_data.csv   # File dữ liệu đã xử lý
│   └── rawdata/             # Canonical raw files (required): Orders.csv, Order_Details.csv, Customers.csv
│
├── notebooks/               # Jupyter Notebooks (exploratory + preprocessing)
│   └── exploratory_analysis.ipynb
│
├── src/                     # Thư mục chứa code
│   ├── main.py              # File chạy chính (CLI)
│   ├── lr_model.py          # Logistic Regression wrapper
│   ├── knn_model.py         # KNN wrapper
│   ├── rf_model.py          # Random Forest wrapper
│   ├── xgb_model.py         # XGBoost wrapper
│   ├── gb_model.py          # Gradient Boosting wrapper
│   └── adaboost_model.py    # AdaBoost wrapper
│
├── reports/                 # Thư mục báo cáo và phân công
│   ├── report.md
│   └── task_assignment.md
│
└── .gitignore               # Để loại bỏ các file không cần thiết khi push
```

=======
# Machine-Learning
Machine Learning
>>>>>>> d2c21b0d61a78b834fec12945c7652e16c79c361
