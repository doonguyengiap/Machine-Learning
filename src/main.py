"""Main CLI to train and save models on processed dataset.
Usage: PYTHONPATH=. python -m src.main --model rf
"""
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.preprocessing import robust_read_processed, clean_numeric_columns
import pandas as pd
import os

MODEL_MAP = {
    'lr': 'src.lr_model',
    'knn': 'src.knn_model',
    'rf': 'src.rf_model',
    'xgb': 'src.xgb_model',
    'gb': 'src.gb_model',
    'adaboost': 'src.adaboost_model'
}

def load_data(path=None):
    df, sep = robust_read_processed(path or 'data/processed_data.csv')
    # Normalize numeric columns (handle comma decimals) to ensure numeric ops succeed
    df = clean_numeric_columns(df, cols=['TOTALBASKET','UNITPRICE','TOTALPRICE','AMOUNT'])

    expected_cols = ['TOTALBASKET', 'Age', 'USERGENDER', 'REGION', 'AMOUNT']
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df.dropna(subset=available_cols)
    threshold = pd.to_numeric(df['TOTALBASKET'], errors='coerce').median()
    df['HighValueCustomer'] = (df['TOTALBASKET'] > threshold).astype(int)
    feature_cols = [c for c in available_cols if c != 'TOTALBASKET']
    if 'Month' in df.columns:
        feature_cols.append('Month')
    X = df[feature_cols].copy()
    y = df['HighValueCustomer']
    # encode categorical
    for col in ['USERGENDER','REGION']:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_MAP.keys(), required=True)
    parser.add_argument('--out', default='models')
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    module_name = MODEL_MAP[args.model]
    module = __import__(module_name, fromlist=['train','save_model'])
    model = module.train(X_train, y_train)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'best_{args.model}.pkl')
    module.save_model(model, out_path)
    print('Saved model to', out_path)

if __name__ == '__main__':
    main()
