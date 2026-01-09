"""eda.py
Functions for Exploratory Data Analysis and visualization.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_profit_distribution(df, output_path='reports/figures/profit_dist.png'):
    """Plot the distribution of profitable vs non-profitable orders."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='IS_PROFIT', data=df)
    plt.title("Phân bố đơn hàng sinh lời")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_feature_vs_profit(df, feature, output_path=None):
    """Plot a feature vs profitability using boxplots."""
    if output_path is None:
        output_path = f'reports/figures/{feature}_vs_profit.png'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='IS_PROFIT', y=feature, data=df)
    plt.title(f"{feature} và khả năng sinh lời")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_time_trends(df, date_col='DATE_', period='Month', output_path='reports/figures/time_trends.png'):
    """Plot profit trends over time."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if period == 'Month':
        trend = df.groupby(df[date_col].dt.month)['PROFIT'].sum()
    else:
        trend = df.groupby(df[date_col].dt.year)['PROFIT'].sum()
    
    plt.figure(figsize=(12, 6))
    trend.plot(kind='bar')
    plt.title(f"Xu hướng lợi nhuận theo {period}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def run_full_eda(df):
    """Run all EDA plots."""
    plot_profit_distribution(df)
    plot_feature_vs_profit(df, 'AMOUNT')
    plot_feature_vs_profit(df, 'UNITPRICE')
    plot_feature_vs_profit(df, 'Age')
    plot_time_trends(df, period='Month', output_path='reports/figures/profit_by_month.png')
    plot_time_trends(df, period='Year', output_path='reports/figures/profit_by_year.png')
    print("EDA reports generated in reports/figures/")
