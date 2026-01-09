import os
import sys
import json
import csv
from flask import Flask, jsonify, render_template, redirect, url_for, send_from_directory, request, flash
from werkzeug.utils import secure_filename
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'model_comparison.json')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')

# Add project root to Python path so src module can be imported
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {'csv'}

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def list_data_files():
    files = []
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(('.csv',)):
                files.append(f)
    return sorted(files)


def list_model_files():
    files = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.lower().endswith(('.pkl', '.joblib')):
                files.append(f)
    return sorted(files)


def list_figures():
    figs = []
    if os.path.isdir(FIGURES_DIR):
        for f in os.listdir(FIGURES_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                figs.append(f)
    return sorted(figs)


@app.route('/')
def index():
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as fh:
                metrics = json.load(fh)
        except Exception:
            metrics = {}
    return render_template(
        'index.html',
        metrics=metrics,
        figures=list_figures(),
        data_files=list_data_files(),
        models=list_model_files(),
    )


@app.route('/data/<path:filename>')
def serve_data(filename):
    # Secure: only serve files that exist in data dir
    if filename not in list_data_files():
        return "Not found", 404
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route('/models/<path:filename>')
def serve_model(filename):
    if filename not in list_model_files():
        return "Not found", 404
    return send_from_directory(MODELS_DIR, filename, as_attachment=True)


@app.route('/figures/<path:filename>')
def serve_figure(filename):
    if filename not in list_figures():
        return "Not found", 404
    return send_from_directory(FIGURES_DIR, filename)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    models = list_model_files()
    if request.method == 'POST':
        file = request.files.get('file')
        model_name = request.form.get('model_name')
        if not file:
            flash('No file uploaded')
            return redirect(url_for('index'))
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXT:
            flash('Invalid file type, only CSV allowed')
            return redirect(url_for('index'))
        filename = secure_filename(file.filename)
        target = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(target)

        # Read CSV (robust options could be added)
        try:
            df = pd.read_csv(target)
        except Exception as e:
            flash(f'Failed to read CSV: {e}')
            return redirect(url_for('index'))

        if not models:
            flash('No models available in models/')
            return redirect(url_for('index'))

        # Choose model
        model_to_load = model_name if model_name in models else models[0]
        try:
            model = joblib.load(os.path.join(MODELS_DIR, model_to_load))
        except Exception as e:
            flash(f'Failed to load model: {e}')
            return redirect(url_for('index'))

        try:
            preds = model.predict(df)
            proba = None
            try:
                proba = model.predict_proba(df)[:, 1]
            except Exception:
                proba = None
            res = pd.DataFrame({'prediction': preds})
            if proba is not None:
                res['prob'] = proba
            preview = res.head(20).to_html(index=False)
            return render_template('predict.html', filename=filename, preview=preview, models=models, selected_model=model_to_load)
        except Exception as e:
            flash(f'Prediction failed: {e}')
            return redirect(url_for('index'))

    # GET
    return render_template('predict.html', filename=None, preview=None, models=models, selected_model=None)


@app.route('/api/models')
def api_models():
    return jsonify(list_model_files())


@app.route('/api/data')
def api_data():
    """Return list of data files available in `data/` directory."""
    return jsonify(list_data_files())


@app.route('/api/data/<path:filename>/preview')
def api_data_preview(filename):
    """Return a small JSON preview (rows) of a data CSV file in `data/`.

    Query string params:
    - rows: number of rows to return (default 200)
    """
    files = list_data_files()
    if filename not in files:
        return jsonify({'error': 'file not found'}), 404
    rows = int(request.args.get('rows', 200))
    path = os.path.join(DATA_DIR, filename)
    
    from src.preprocessing import robust_read_processed
    try:
        df, _ = robust_read_processed(path)
        df = df.head(rows)
    except Exception as e:
         return jsonify({'error': str(e)}), 500

    records = df.where(pd.notnull(df), None).to_dict(orient='records')
    return jsonify(records)

@app.route('/api/sales_over_time')
def api_sales_over_time():
    """Return monthly sales (revenue) over time."""
    try:
        from src.preprocessing import robust_read_processed
        df, _ = robust_read_processed()
        # Group by Year, Month and sum TOTALPRICE
        if 'Year' not in df.columns or 'Month' not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns (Year, Month, TOTALPRICE) missing'}), 400
        
        # Ensure numeric
        from src.preprocessing import clean_numeric_columns
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        
        daily = df.groupby(['Year', 'Month'])['TOTALPRICE'].sum().reset_index()
        # Sort chronologically
        daily['Day'] = 1
        daily['Date'] = pd.to_datetime(daily[['Year', 'Month', 'Day']])
        daily = daily.sort_values('Date')
        
        data = {
            'labels': daily['Date'].dt.strftime('%Y-%m').tolist(),
            'values': daily['TOTALPRICE'].tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/revenue_by_region')
def api_revenue_by_region():
    """Return total revenue by region."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        if 'REGION' not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400
            
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby('REGION')['TOTALPRICE'].sum().sort_values(ascending=False)
        
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profitability_by_city')
def api_profitability_by_city():
    """Return profitability rate (IS_PROFIT mean) by top cities."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        
        # Clean necessary columns
        df = clean_numeric_columns(df, cols=['UNITPRICE', 'TOTALPRICE', 'AMOUNT'])
        
        # Calculate IS_PROFIT if missing
        if 'IS_PROFIT' not in df.columns:
            ALPHA = 0.7
            df['ESTIMATED_COST'] = df['UNITPRICE'] * df['AMOUNT'] * ALPHA
            df['PROFIT'] = df['TOTALPRICE'] - df['ESTIMATED_COST']
            df['IS_PROFIT'] = (df['PROFIT'] > 0).astype(int)
            
        if 'CITY' not in df.columns:
            return jsonify({'error': 'CITY column missing'}), 400
        
        # Get top 10 cities by order count
        top_cities = df['CITY'].value_counts().head(10).index
        
        # Calculate profitability rate (mean of IS_PROFIT) for top cities
        city_profit = df[df['CITY'].isin(top_cities)].groupby('CITY')['IS_PROFIT'].mean().sort_values(ascending=False)
        
        return jsonify({
            'labels': city_profit.index.tolist(),
            'values': city_profit.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top_products')
def api_top_products():
    """Return top products by revenue (TOTALPRICE)."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        # Use ITEMCODE or ITEMID
        item_col = 'ITEMCODE' if 'ITEMCODE' in df.columns else 'ITEMID'
        if item_col not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400
            
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby(item_col)['TOTALPRICE'].sum().sort_values(ascending=False).head(10)
        
        return jsonify({
            # Prefix to make labels readable on the chart instead of raw IDs
            'labels': [f"Product {code}" for code in grouped.index.astype(str)],
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demographics')
def api_demographics():
    """Return gender and age group statistics."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        
        # Gender
        gender_col = next((c for c in df.columns if c.upper() in ['USERGENDER', 'GENDER']), None)
        gender_data = {}
        if gender_col:
            raw_gender = df[gender_col].fillna('Unknown').value_counts().to_dict()
            # Map K/E to Female/Male
            mapping = {'K': 'Female', 'E': 'Male'}
            gender_data = {mapping.get(k, k): v for k, v in raw_gender.items()}

        # Age Groups
        age_col = next((c for c in df.columns if c.upper() == 'AGE'), None)
        age_data = {}
        if age_col:
            df = clean_numeric_columns(df, cols=[age_col])
            bins = [0, 18, 25, 35, 45, 55, 65, 100]
            labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            age_groups = pd.cut(df[age_col], bins=bins, labels=labels)
            age_data = age_groups.value_counts().sort_index().to_dict()

        return jsonify({
            'gender': {'labels': list(gender_data.keys()), 'values': list(gender_data.values())},
            'age': {'labels': list(age_data.keys()), 'values': list(age_data.values())}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/branch_performance')
def api_branch_performance():
    """Return top branches by revenue."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        branch_col = next((c for c in df.columns if c.upper() in ['BRANCH_ID', 'BRANCH']), None)
        
        if not branch_col or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400

        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby(branch_col)['TOTALPRICE'].sum().sort_values(ascending=False).head(10)
        
        return jsonify({
            'labels': grouped.index.astype(str).tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/overall_stats')
def api_overall_stats():
    """Return key summary metrics for the store."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        
        total_revenue = float(df['TOTALPRICE'].sum())
        total_orders = int(df['ORDERID'].nunique()) if 'ORDERID' in df.columns else 0
        total_customers = int(df['USERID'].nunique()) if 'USERID' in df.columns else 0
        
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        return jsonify({
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'total_customers': total_customers,
            'avg_order_value': avg_order_value
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def api_predictions():
    """Return model comparison metrics normalized for the dashboard."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as fh:
                data = json.load(fh)
            
            # Normalize list format vs dict format
            normalized = []
            if isinstance(data, dict):
                for model_name, metrics in data.items():
                    # Map common variations to standard keys
                    # Expected by frontend: Accuracy, F1-score, ROC-AUC
                    normalized.append({
                        'Model': model_name.upper(),
                        'Accuracy': metrics.get('accuracy', metrics.get('Accuracy', 0)),
                        'F1-score': metrics.get('f1', metrics.get('F1-score', metrics.get('precision', 0))), # fallback
                        'ROC-AUC': metrics.get('auc', metrics.get('ROC-AUC', 0))
                    })
            elif isinstance(data, list):
                for m in data:
                    normalized.append({
                        'Model': m.get('Model', 'Unknown'),
                        'Accuracy': m.get('Accuracy', m.get('accuracy', 0)),
                        'F1-score': m.get('F1-score', m.get('f1', 0)),
                        'ROC-AUC': m.get('ROC-AUC', m.get('auc', 0))
                    })
            
            return jsonify(normalized)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify([])

@app.route('/api/revenue_by_city')
def api_revenue_by_city():
    """Return top 10 cities by revenue."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        if 'CITY' not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby('CITY')['TOTALPRICE'].sum().sort_values(ascending=False).head(10)
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales_by_day')
def api_sales_by_day():
    """Return revenue grouped by day of week."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
        df, _ = robust_read_processed()
        df = parse_dates(df)
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        df['DayOfWeek'] = df['DATE_'].dt.day_name()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grouped = df.groupby('DayOfWeek')['TOTALPRICE'].sum().reindex(order).fillna(0)
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/basket_distribution')
def api_basket_distribution():
    """Return binned distribution of TOTALBASKET."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        import numpy as np
        df, _ = robust_read_processed()
        df = clean_numeric_columns(df, cols=['TOTALBASKET'])
        
        # Fixed bins and labels to match count
        bins = [0, 50000, 100000, 150000, 200000, 300000, 500000, float('inf')]
        labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-300k', '300k-500k', '500k+']
        df['BasketRange'] = pd.cut(df['TOTALBASKET'], bins=bins, labels=labels)
        counts = df['BasketRange'].value_counts().sort_index()
        return jsonify({
            'labels': counts.index.tolist(),
            'values': counts.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monthly_seasonality')
def api_monthly_seasonality():
    """Return revenue by month (aggregated)."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
        df, _ = robust_read_processed()
        df = parse_dates(df)
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        df['MonthName'] = df['DATE_'].dt.month_name()
        order = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        grouped = df.groupby('MonthName')['TOTALPRICE'].sum().reindex(order).fillna(0)
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/revenue_by_town')
def api_revenue_by_town():
    """Return top 10 towns by revenue."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        if 'TOWN' not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby('TOWN')['TOTALPRICE'].sum().sort_values(ascending=False).head(10)
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/revenue_by_district')
def api_revenue_by_district():
    """Return top 10 districts by revenue."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        if 'DISTRICT' not in df.columns or 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'Required columns missing'}), 400
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        grouped = df.groupby('DISTRICT')['TOTALPRICE'].sum().sort_values(ascending=False).head(10)
        return jsonify({
            'labels': grouped.index.tolist(),
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/revenue_by_year')
def api_revenue_by_year():
    """Return revenue by year."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns, parse_dates
        df, _ = robust_read_processed()
        df = parse_dates(df, col='DATE_')
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        df['Year'] = df['DATE_'].dt.year
        grouped = df.groupby('Year')['TOTALPRICE'].sum().sort_index()
        return jsonify({
            'labels': [str(int(y)) for y in grouped.index.tolist()],
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profit_analysis')
def api_profit_analysis():
    """Return profit analysis (IS_PROFIT distribution)."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        import numpy as np
        df, _ = robust_read_processed()
        df = clean_numeric_columns(df, cols=['UNITPRICE', 'TOTALPRICE', 'AMOUNT'])
        
        # Calculate profit if not exists
        ALPHA = 0.7
        if 'IS_PROFIT' not in df.columns:
            df['ESTIMATED_COST'] = df['UNITPRICE'] * df['AMOUNT'] * ALPHA
            df['PROFIT'] = df['TOTALPRICE'] - df['ESTIMATED_COST']
            df['IS_PROFIT'] = (df['PROFIT'] > 0).astype(int)
        
        profit_counts = df['IS_PROFIT'].value_counts().sort_index()
        labels = ['No Profit' if idx == 0 else 'Profit' for idx in profit_counts.index]
        
        return jsonify({
            'labels': labels,
            'values': profit_counts.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/age_revenue')
def api_age_revenue():
    """Return revenue by age groups."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        import pandas as pd
        df, _ = robust_read_processed()
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        
        if 'Age' not in df.columns:
            return jsonify({'error': 'Age column missing'}), 400
        
        # Create age groups
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        grouped = df.groupby('AgeGroup')['TOTALPRICE'].sum()
        
        return jsonify({
            'labels': [str(x) for x in grouped.index.tolist()],
            'values': grouped.values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast')
def api_forecast():
    """Return revenue forecast for next 3 and 6 months by region."""
    from src.forecasting import get_revenue_forecast
    results = get_revenue_forecast()
    return jsonify(results)

def robust_read_processed_data():
    """Legacy helper (replaced by src.preprocessing)"""
    from src.preprocessing import robust_read_processed
    return robust_read_processed()
            
    # Fallback
    raise ValueError(f"Could not read {filename} with any standard encoding")


# @app.route('/goto/<page>')
# def goto(page):
#     # simple safe mapping used by templates
#     mapping = {
#         'home': '/',
#         'store': '/store',
#         'analytics': '/analytics'
#     }
#     return redirect(mapping.get(page, '/'))


@app.route('/store')
def store():
    return render_template('store.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/api/age_pyramid')
def api_age_pyramid():
    """Return age distribution split by gender (Age Pyramid)."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        import numpy as np
        
        df, _ = robust_read_processed()
        
        # Check for required columns
        age_col = next((c for c in df.columns if c.upper() == 'AGE'), None)
        gender_col = next((c for c in df.columns if c.upper() in ['USERGENDER', 'GENDER']), None)
        
        if not age_col or not gender_col:
            return jsonify({'error': 'Required columns (Age, Gender) missing'}), 400
            
        df = clean_numeric_columns(df, cols=[age_col])
        
        # Create age bins
        bins = [0, 18, 25, 35, 45, 55, 65, 100]
        labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df['AgeGroup'] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        
        # Standardize gender
        # Map K/E to Female/Male or ensure Male/Female
        df['GenderNorm'] = df[gender_col].fillna('Unknown').apply(
            lambda x: 'Female' if str(x).upper() in ['K', 'FEMALE', 'F'] else ('Male' if str(x).upper() in ['E', 'MALE', 'M'] else 'Other')
        )
        
        # Calculate counts
        grouped = df.groupby(['AgeGroup', 'GenderNorm']).size().unstack(fill_value=0)
        
        # Ensure Male and Female columns exist
        if 'Male' not in grouped.columns:
            grouped['Male'] = 0
        if 'Female' not in grouped.columns:
            grouped['Female'] = 0
            
        return jsonify({
            'labels': labels,
            'male': grouped['Male'].reindex(labels, fill_value=0).tolist(),
            'female': grouped['Female'].reindex(labels, fill_value=0).tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pareto_customers')
def api_pareto_customers():
    """Return Pareto analysis (80/20 rule) for customers."""
    try:
        from src.preprocessing import robust_read_processed, clean_numeric_columns
        df, _ = robust_read_processed()
        
        if 'TOTALPRICE' not in df.columns:
            return jsonify({'error': 'TOTALPRICE column missing'}), 400
            
        user_col = next((c for c in df.columns if c.upper() in ['USERID', 'CUSTOMERID', 'USER_ID']), None)
        if not user_col:
             return jsonify({'error': 'User ID column missing'}), 400
             
        df = clean_numeric_columns(df, cols=['TOTALPRICE'])
        
        # Group by customer and sum revenue
        customer_revenue = df.groupby(user_col)['TOTALPRICE'].sum().sort_values(ascending=False)
        
        # Calculate cumulative percentage
        total_revenue = customer_revenue.sum()
        cumulative_revenue = customer_revenue.cumsum()
        cumulative_pct = (cumulative_revenue / total_revenue) * 100
        
        # Limit to top N for visualization (e.g., top 50 or top 20% of customers if list is huge)
        # For the chart, we might want to show the "head" clearly. 
        # Let's take top 50 customers to keep chart readable, but ensure we return metrics about the tail
        
        top_n = 50
        top_customers = customer_revenue.head(top_n)
        top_cumulative_pct = cumulative_pct.head(top_n)
        
        return jsonify({
            'labels': top_customers.index.astype(str).tolist(),
            'values': [float(x) for x in top_customers.values],
            'cumulative_pct': [float(x) for x in top_cumulative_pct.values],
            'total_customers': int(len(customer_revenue)),
            'total_revenue': float(total_revenue),
            'top_20_pct_count': int(len(customer_revenue) * 0.2),
            'top_20_pct_revenue_share': float(cumulative_pct.iloc[int(len(customer_revenue) * 0.2)]) if len(customer_revenue) > 5 else 0.0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Enable debug and auto-reload when running in development (FLASK_ENV=development or FLASK_DEBUG=1)
    debug_mode = os.environ.get('FLASK_ENV', '').lower() == 'development' or os.environ.get('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=debug_mode, use_reloader=debug_mode)
