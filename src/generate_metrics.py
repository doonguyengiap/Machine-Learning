
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.main import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

def generate():
    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'lr': LogisticRegression(max_iter=1000),
        'knn': KNeighborsClassifier(),
        'rf': RandomForestClassifier(n_estimators=100),
        'dt': DecisionTreeClassifier(),
        'gb': GradientBoostingClassifier()
    }

    report = {}

    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Some models don't have predict_proba easily or we just want AUC
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = accuracy_score(y_test, y_pred) # fallback

        report[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'auc': auc
        }
        
        # Save the model while we are at it
        joblib.dump(model, f'models/best_{name}.pkl')

    with open('reports/model_comparison.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Done! Metrics saved to reports/model_comparison.json")

if __name__ == "__main__":
    generate()
