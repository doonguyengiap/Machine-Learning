"""Gradient Boosting wrapper"""
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train(X, y, random_state=42, **kwargs):
    model = GradientBoostingClassifier(random_state=random_state, **kwargs)
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
