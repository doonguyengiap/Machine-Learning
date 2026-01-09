
from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
import pandas as pd

try:
    df, _ = robust_read_processed('data/processed_data.csv')
    df = clean_numeric_columns(df)
    df = parse_dates(df)
    
    print("--- STATUS Value Counts ---")
    print(df['STATUS_'].value_counts())
    
    print("\n--- CITY Value Counts (Top 10) ---")
    print(df['CITY'].value_counts().head(10))
    
    print("\n--- Sales by Day of Week ---")
    df['DayOfWeek'] = df['DATE_'].dt.day_name()
    print(df['DayOfWeek'].value_counts().sort_index())
    
    print("\n--- Summary Stats for TOTALBASKET ---")
    print(df['TOTALBASKET'].describe())

except Exception as e:
    print(f"Error: {e}")
