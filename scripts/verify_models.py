"""Verify saved models by loading and predicting on a sample test split."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.models import load_model

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'order_level_data.csv')
MODEL_DIR = os.path.join(ROOT, 'models')

print('Loading data...')
df = pd.read_csv(DATA_PATH)
cols_to_drop = ['ORDERID', 'USERID'] if 'ORDERID' in df.columns else ['USERID']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + ['profitable'])
y = df['profitable'].astype(int)
X = X.select_dtypes(include=[np.number]).fillna(0)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, df['USERID']))
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

for model_file in ['best_rf.pkl','best_xgb.pkl']:
    path = os.path.join(MODEL_DIR, model_file)
    print('\nLoading', path)
    m = load_model(path)
    preds = m.predict(X_test)
    print('Sample predictions:', preds[:10])
    print('Unique preds:', set(preds))
