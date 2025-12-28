"""features.py
Functions to build dataset-level features: order-level aggregation, RFM, time-series features.
"""
import pandas as pd


def aggregate_order_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detail-level data (one row per ORDERID).
    Returns: DataFrame with ORDERID, total_revenue, total_items, unique_items, avg_unit_price
    """
    agg = df.groupby('ORDERID').agg(
        total_revenue=('TOTALPRICE','sum'),
        total_items=('AMOUNT','sum'),
        unique_items=('ITEMCODE', pd.Series.nunique),
        avg_unit_price=('UNITPRICE','mean')
    ).reset_index()
    return agg


def build_rfm(df: pd.DataFrame, date_col='DATE_') -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    ref_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby('USERID').agg(
        recency_days = (date_col, lambda x: (ref_date - x.max()).days),
        frequency = ('ORDERID', 'nunique'),
        monetary = ('TOTALPRICE', 'sum')
    ).reset_index()
    return rfm
