"""evaluation.py
Functions for model evaluation, including metrics and confusion matrices.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Calculate and print performance metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_prob)
    
    print(f"--- {model_name} Evaluation ---")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return metrics

def plot_confusion_matrix(y_test, y_pred_encoded, model_name="Model", output_path=None):
    """Plot and save confusion matrix."""
    if output_path is None:
        output_path = f'reports/figures/cm_{model_name.lower().replace(" ", "_")}.png'
    
    cm = confusion_matrix(y_test, y_pred_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_model_comparison(results_df, output_path='reports/model_comparison.json'):
    """Save model comparison results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_json(output_path, orient='records', indent=4)
    print(f"Comparison saved to {output_path}")
