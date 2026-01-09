"""app.py
Flask application for the ML Dashboard.
Supports file upload, EDA visualization, and real-time prediction.
"""
import os
import sys
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
from src.feature_engineering import create_target_variable, build_rfm_features, scale_features
from src.eda import run_full_eda
from src.model_student import train_random_forest

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = "superstore_secret_key"
UPLOAD_FOLDER = 'data/uploads'
REPORT_FOLDER = 'reports/figures'
MODEL_PATH = 'models/best_rf.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global state to keep track of loaded data and model
current_df = None
model = None
scaler = None

def get_trained_model(df):
    global model, scaler
    df = create_target_variable(df)
    rfm = build_rfm_features(df)
    user_target = df.groupby('USERID')['IS_PROFIT'].max().reset_index()
    rfm = rfm.merge(user_target, on='USERID')
    
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features]
    y = rfm['IS_PROFIT']
    
    X_scaled, scaler_obj = scale_features(X, features)
    scaler = scaler_obj
    model = train_random_forest(X_scaled, y)
    joblib.dump((model, scaler), MODEL_PATH)
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html', has_data=(current_df is not None))

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        # Load and process
        try:
            df, _ = robust_read_processed(path)
            df = clean_numeric_columns(df)
            df = parse_dates(df)
            current_df = df
            
            # Generate EDA plots
            eda_df = create_target_variable(df)
            run_full_eda(eda_df)
            
            # Train/Update model
            get_trained_model(df)
            
            flash('File uploaded and processed successfully!')
        except Exception as e:
            flash(f'Error processing file: {e}')
            
        return redirect(url_for('index'))

@app.route('/sample')
def load_sample():
    global current_df
    try:
        df, _ = robust_read_processed('data/processed_data.csv')
        df = clean_numeric_columns(df)
        df = parse_dates(df)
        current_df = df
        
        # Generate EDA plots
        eda_df = create_target_variable(df)
        run_full_eda(eda_df)
        
        # Train/Update model
        get_trained_model(df)
        
        flash('Sample data loaded successfully!')
    except Exception as e:
        flash(f'Error loading sample: {e}')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        flash('Model not trained yet. Please upload data.')
        return redirect(url_for('index'))
    
    try:
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        
        input_data = pd.DataFrame([[recency, frequency, monetary]], 
                                 columns=['Recency', 'Frequency', 'Monetary'])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = "Profitable" if prediction == 1 else "Non-Profitable"
        return render_template('index.html', has_data=True, 
                               prediction_text=f'Result: {result} (Probability: {probability:.2%})')
    except Exception as e:
        flash(f'Prediction error: {e}')
        return redirect(url_for('index'))

@app.route('/reports/figures/<filename>')
def get_report(filename):
    return send_from_directory(os.path.join('../', REPORT_FOLDER), filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
