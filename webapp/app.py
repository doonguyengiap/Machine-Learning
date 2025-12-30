from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import json
import pandas as pd
import joblib
from werkzeug.utils import secure_filename

# try to import preprocessing helpers
try:
    from src.preprocessing import clean_numeric_columns, robust_read_processed
except Exception:
    clean_numeric_columns = None
    robust_read_processed = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'model_comparison.json')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as fh:
                metrics = json.load(fh)
        except Exception:
            metrics = {}

    figures = []
    if os.path.isdir(FIGURES_DIR):
        for f in os.listdir(FIGURES_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                figures.append(f)
    return render_template('index.html', metrics=metrics, figures=figures)


@app.route('/figures/<path:filename>')
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(target_path)

            # Attempt to read file
            try:
                # try robust reader if available
                if robust_read_processed is not None:
                    df, sep = robust_read_processed(target_path)
                else:
                    df = pd.read_csv(target_path)
            except Exception as e:
                flash(f'Error reading file: {e}')
                return redirect(url_for('index'))

            # Clean numeric columns if helper available
            try:
                if clean_numeric_columns is not None:
                    df = clean_numeric_columns(df)
            except Exception:
                pass

            # Try to load a model
            model_files = []
            if os.path.isdir(MODELS_DIR):
                for nm in os.listdir(MODELS_DIR):
                    if nm.endswith('.pkl'):
                        model_files.append(nm)
            if not model_files:
                flash('No models found in models/ to run predictions. Train and save a model first.')
                return redirect(url_for('index'))

            # Load the first model available
            model_path = os.path.join(MODELS_DIR, model_files[0])
            try:
                model = joblib.load(model_path)
            except Exception as e:
                flash(f'Failed to load model {model_path}: {e}')
                return redirect(url_for('index'))

            # Attempt to predict (model is expected to be a pipeline)
            try:
                preds = model.predict(df)
                proba = None
                try:
                    proba = model.predict_proba(df)[:, 1]
                except Exception:
                    proba = None
                result = pd.DataFrame({'prediction': preds})
                if proba is not None:
                    result['probability'] = proba
                preview = result.head(20).to_html(index=False)
                return render_template('predict.html', filename=filename, preview=preview)
            except Exception as e:
                flash(f'Prediction failed: {e}')
                return redirect(url_for('index'))
        else:
            flash('Invalid file type — only CSV allowed')
            return redirect(url_for('index'))
    # GET
    return render_template('predict.html', filename=None, preview=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
