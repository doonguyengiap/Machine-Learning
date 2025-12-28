"""KNN model wrapper"""
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train(X, y, n_neighbors=5, **kwargs):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X, y)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
