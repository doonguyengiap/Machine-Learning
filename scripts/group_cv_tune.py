"""
Group-aware hyperparameter tuning for RandomForest and XGBoost on order-level dataset.
Generates reports in `reports/` and saves best models in `models/`.
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

RND = 42

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'order_level_data.csv')
REPORT_DIR = os.path.join(ROOT, 'reports')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print('Loading', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Prepare features and target
cols_to_drop = ['ORDERID', 'USERID'] if 'ORDERID' in df.columns else ['USERID']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + ['profitable'])
y = df['profitable'].astype(int)
groups = df['USERID'] if 'USERID' in df.columns else None

# Simple feature selection: numeric columns only
X = X.select_dtypes(include=[np.number]).fillna(0)

print('X shape:', X.shape, 'y shape:', y.shape)

# Create a held-out test set using GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RND)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print('Train/Test sizes:', X_train.shape, X_test.shape)

# Utility to run RandomizedSearchCV with GroupKFold
def run_random_search(estimator, param_dist, n_iter=40, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    rs = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=n_iter, scoring='accuracy', cv=cv, n_jobs=-1, random_state=RND, verbose=1)
    rs.fit(X_train, y_train, groups=groups_train)
    return rs

# RandomForest
rf = RandomForestClassifier(random_state=RND)
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
print('Tuning RandomForest...')
rf_rs = run_random_search(rf, rf_params, n_iter=30)
print('Best RF params:', rf_rs.best_params_, 'Best score:', rf_rs.best_score_)

# Evaluate on test
rf_best = rf_rs.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_metrics = {
    'accuracy': accuracy_score(y_test, rf_preds),
    'precision': precision_score(y_test, rf_preds, zero_division=0),
    'recall': recall_score(y_test, rf_preds, zero_division=0),
    'f1': f1_score(y_test, rf_preds, zero_division=0),
}
print('RF Test metrics:', rf_metrics)

# Save RF model and cv results
joblib.dump(rf_best, os.path.join(MODEL_DIR, 'best_rf.pkl'))
rf_cv_df = pd.DataFrame(rf_rs.cv_results_)
rf_cv_df.to_csv(os.path.join(REPORT_DIR, 'rf_cv_results.csv'), index=False)

with open(os.path.join(REPORT_DIR, 'rf_test_metrics.json'), 'w') as f:
    json.dump({'cv_best_score': rf_rs.best_score_, 'test_metrics': rf_metrics, 'best_params': rf_rs.best_params_}, f, indent=2)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RND, verbosity=0)
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
}
print('Tuning XGBoost...')
xgb_rs = run_random_search(xgb, xgb_params, n_iter=30)
print('Best XGB params:', xgb_rs.best_params_, 'Best score:', xgb_rs.best_score_)

xgb_best = xgb_rs.best_estimator_
xgb_preds = xgb_best.predict(X_test)
xgb_metrics = {
    'accuracy': accuracy_score(y_test, xgb_preds),
    'precision': precision_score(y_test, xgb_preds, zero_division=0),
    'recall': recall_score(y_test, xgb_preds, zero_division=0),
    'f1': f1_score(y_test, xgb_preds, zero_division=0),
}
print('XGB Test metrics:', xgb_metrics)

joblib.dump(xgb_best, os.path.join(MODEL_DIR, 'best_xgb.pkl'))
xgb_cv_df = pd.DataFrame(xgb_rs.cv_results_)
xgb_cv_df.to_csv(os.path.join(REPORT_DIR, 'xgb_cv_results.csv'), index=False)

with open(os.path.join(REPORT_DIR, 'xgb_test_metrics.json'), 'w') as f:
    json.dump({'cv_best_score': xgb_rs.best_score_, 'test_metrics': xgb_metrics, 'best_params': xgb_rs.best_params_}, f, indent=2)

# Save a combined report
combined = {
    'rf': {'best_score_cv': rf_rs.best_score_, 'test_metrics': rf_metrics, 'best_params': rf_rs.best_params_},
    'xgb': {'best_score_cv': xgb_rs.best_score_, 'test_metrics': xgb_metrics, 'best_params': xgb_rs.best_params_},
}
with open(os.path.join(REPORT_DIR, 'combined_model_report.json'), 'w') as f:
    json.dump(combined, f, indent=2)

print('Done. Reports and models saved to', REPORT_DIR, MODEL_DIR)
