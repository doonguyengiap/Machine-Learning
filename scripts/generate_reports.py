"""Generate human-readable model report and plots from saved results and models."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.models import load_model

ROOT = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(ROOT, 'reports')
MODEL_DIR = os.path.join(ROOT, 'models')
DATA_PATH = os.path.join(ROOT, 'data', 'order_level_data.csv')

os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
print('Loading data...')
df = pd.read_csv(DATA_PATH)
cols_to_drop = ['ORDERID', 'USERID'] if 'ORDERID' in df.columns else ['USERID']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + ['profitable'])
y = df['profitable'].astype(int)
X = X.select_dtypes(include=[np.number]).fillna(0)

# load holdout via GroupShuffleSplit serialization approach
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, df['USERID']))
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

report_lines = ["# Model Report\n"]

for model_name in ['best_rf.pkl', 'best_xgb.pkl']:
    path = os.path.join(MODEL_DIR, model_name)
    print('Processing', model_name)
    model = load_model(path)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(model_name)
    plot_path = os.path.join(REPORT_DIR, f'cm_{model_name.replace(".pkl","")}.png')
    fig.savefig(plot_path)
    plt.close(fig)

    # Load metric json
    metric_file = os.path.join(REPORT_DIR, model_name.replace('.pkl','_test_metrics.json'))
    metrics = None
    if os.path.exists(metric_file):
        with open(metric_file) as f:
            metrics = json.load(f)

    report_lines.append(f"## {model_name}\n")
    if metrics:
        report_lines.append('**CV best score:** ' + str(metrics.get('cv_best_score', 'N/A')) + '\n')
        report_lines.append('**Test metrics:**\n')
        for k,v in metrics.get('test_metrics', {}).items():
            report_lines.append(f'- {k}: {v}')
        report_lines.append('\n')
    report_lines.append(f'![confusion matrix]({os.path.basename(plot_path)})\n')

# Save markdown report
md_path = os.path.join(REPORT_DIR, 'model_report.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(report_lines))

print('Report saved to', md_path)
