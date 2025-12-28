"""XGBoost wrapper"""
import xgboost as xgb
import joblib

def train(X, y, random_state=42, **kwargs):
    model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, **kwargs)
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
