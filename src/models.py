"""models.py
Helpers for training & evaluating models (classification & regression).
"""
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict
import pandas as pd


def evaluate_classification(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred)
    }


def train_models(models: Dict, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = evaluate_classification(y_test, y_pred)
        m['Model'] = name
        results.append(m)
    return pd.DataFrame(results).set_index('Model')


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def predict_with_model(path: str, X):
    model = load_model(path)
    return model.predict(X)

