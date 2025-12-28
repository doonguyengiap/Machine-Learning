import pandas as pd
from src.features import aggregate_order_level, build_rfm


def test_aggregate_order_level():
    df = pd.DataFrame({
        'ORDERID': [1,1,2,2,2],
        'TOTALPRICE': [10, 15, 5, 5, 5],
        'AMOUNT': [1,2,1,1,1],
        'ITEMCODE': [100, 100, 101, 102, 101],
        'UNITPRICE': [10, 7.5, 5, 5, 5]
    })
    agg = aggregate_order_level(df)
    assert agg.loc[agg['ORDERID']==1,'total_revenue'].values[0] == 25
    assert agg.loc[agg['ORDERID']==2,'total_items'].values[0] == 3
    assert agg.shape[0] == 2


def test_build_rfm():
    df = pd.DataFrame({
        'USERID': [1,1,2],
        'ORDERID': [10,11,12],
        'DATE_': ['01/01/2020','05/01/2020','01/02/2020'],
        'TOTALPRICE': [100, 50, 20]
    })
    rfm = build_rfm(df, date_col='DATE_')
    assert 'monetary' in rfm.columns
    assert 'frequency' in rfm.columns
    assert rfm.loc[rfm['USERID']==1,'frequency'].values[0] == 2
