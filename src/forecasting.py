
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.preprocessing import robust_read_processed as robust_read_processed_data

def get_revenue_forecast():
    try:
        df, _ = robust_read_processed_data()
        
        # Ensure correct types
        df['TOTALPRICE'] = pd.to_numeric(df['TOTALPRICE'], errors='coerce').fillna(0)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        
        # Aggregate monthly revenue by region
        monthly_revenue = df.groupby(['REGION', 'Year', 'Month'])['TOTALPRICE'].sum().reset_index()
        
        # Create a time index (months passed since start)
        min_year = monthly_revenue['Year'].min()
        monthly_revenue['TimeIndex'] = (monthly_revenue['Year'] - min_year) * 12 + monthly_revenue['Month']
        
        regions = monthly_revenue['REGION'].unique()
        forecast_results = {}
        
        latest_time = monthly_revenue['TimeIndex'].max()
        
        for region in regions:
            region_data = monthly_revenue[monthly_revenue['REGION'] == region].sort_values('TimeIndex')
            
            if len(region_data) < 2:
                continue
                
            X = region_data[['TimeIndex']].values
            y = region_data['TOTALPRICE'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for next 3 and 6 months
            # We want total revenue for next 3 months and next 6 months
            
            # Predict 3 months (latest_time + 1, +2, +3)
            next_3 = np.array([[latest_time + i] for i in range(1, 4)])
            pred_3 = model.predict(next_3)
            total_3m = float(np.sum(pred_3))
            
            # Predict 6 months (latest_time + 1 to +6)
            next_6 = np.array([[latest_time + i] for i in range(1, 7)])
            pred_6 = model.predict(next_6)
            total_6m = float(np.sum(pred_6))
            
            forecast_results[region] = {
                'forecast_3m': max(0, total_3m),
                'forecast_6m': max(0, total_6m),
                'trend': float(model.coef_[0])
            }
            
        return forecast_results
    except Exception as e:
        print(f"Forecasting error: {e}")
        return {}

if __name__ == "__main__":
    results = get_revenue_forecast()
    print(results)
