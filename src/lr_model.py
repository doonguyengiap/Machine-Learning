"""Logistic Regression model wrapper
Simple train/predict API used by scripts and notebooks.
"""
from sklearn.linear_model import LogisticRegression
import joblib

def train(X, y, **kwargs):
    model = LogisticRegression(max_iter=1000, **kwargs)
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
