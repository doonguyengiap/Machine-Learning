"""feature_engineering.py
Functions for RFM analysis, target variable (IS_PROFIT) creation, and feature scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_target_variable(df, alpha=0.7):
    """Create ESTIMATED_COST, PROFIT, and IS_PROFIT target variable."""
    df = df.copy()
    df['ESTIMATED_COST'] = df['UNITPRICE'] * df['AMOUNT'] * alpha
    df['PROFIT'] = df['TOTALPRICE'] - df['ESTIMATED_COST']
    df['IS_PROFIT'] = (df['PROFIT'] > 0).astype(int)
    return df

def build_rfm_features(df, date_col='DATE_'):
    """Calculate Recency, Frequency, and Monetary features for each user."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    today = df[date_col].max()
    
    rfm = df.groupby('USERID').agg({
        date_col: lambda x: (today - x.max()).days, # Recency
        'ORDERID': 'count',                        # Frequency
        'TOTALPRICE': 'sum'                        # Monetary
    }).reset_index()
    
    rfm.columns = ['USERID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def scale_features(df, columns):
    """Scale specified features using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler

def get_model_features():
    """Returns the list of features used in the final model."""
    return ['Recency', 'Frequency', 'Monetary']
