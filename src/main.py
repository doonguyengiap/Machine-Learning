"""main.py
Main entry point for the ML pipeline.
Orchestrates preprocessing, EDA, feature engineering, modeling, and evaluation.
"""
import sys
import os
from sklearn.model_selection import train_test_split

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
from src.eda import run_full_eda
from src.feature_engineering import create_target_variable, build_rfm_features, scale_features, get_model_features
from src.model_student import train_random_forest, train_logistic_regression
from src.evaluation import evaluate_model, plot_confusion_matrix, save_model_comparison
import pandas as pd

def run_pipeline():
    print("--- Starting ML Pipeline ---")
    
    # 1. Preprocessing
    print("Loading and cleaning data...")
    df, _ = robust_read_processed('data/processed_data.csv')
    df = clean_numeric_columns(df)
    df = parse_dates(df)
    
    # 2. EDA
    print("Running EDA...")
    df = create_target_variable(df) # Need target for EDA plots
    run_full_eda(df)
    
    # 3. Feature Engineering
    print("Calculating RFM features...")
    rfm = build_rfm_features(df)
    
    # Prepare features and target (using a simplified strategy for demo)
    # We'll merge the target variable from df back to rfm if needed, 
    # but the notebook usually aggregates IS_PROFIT per user.
    # For simplicity, let's assume we want to predict if a user is profitable overall.
    user_target = df.groupby('USERID')['IS_PROFIT'].max().reset_index()
    rfm = rfm.merge(user_target, on='USERID')
    
    features = get_model_features()
    X = rfm[features]
    y = rfm['IS_PROFIT']
    
    X_scaled, scaler = scale_features(X, features)
    
    # 4. Split and Train
    print("Training models...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    
    # 5. Evaluation
    print("Evaluating models...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    plot_confusion_matrix(y_test, rf_model.predict(X_test), "Random Forest")
    
    results_df = pd.DataFrame([rf_metrics, lr_metrics])
    save_model_comparison(results_df)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == '__main__':
    run_pipeline()
