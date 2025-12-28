import pandas as pd
from src.preprocessing import robust_read_processed, clean_numeric_columns
from src import main


def test_robust_read_semicolon(tmp_path):
    # create a small semicolon-delimited CSV with comma decimal
    p = tmp_path / "test.csv"
    p.write_text("col1;AMOUNT;UNITPRICE;TOTALPRICE\n1;2;10,5;21,0\n2;3;5,0;15,0\n")
    df, sep = robust_read_processed(str(p),)
    assert sep in [";",","]
    df2 = clean_numeric_columns(df, cols=['UNITPRICE','TOTALPRICE','AMOUNT'])
    assert df2['UNITPRICE'].dtype.kind in 'fi'
    assert df2['TOTALPRICE'].dtype.kind in 'fi'


def test_main_load_from_sample(tmp_path):
    # ensure main.load_data can load from sample processed file
    sample = tmp_path / 'sample.csv'
    sample.write_text("TOTALBASKET,Age,USERGENDER,REGION,AMOUNT\n100,30,M,RegionA,2\n200,40,F,RegionB,3\n300,50,M,RegionA,1\n")
    X, y = main.load_data(path=str(sample))
    assert len(X) == len(y)
    assert 'Age' in X.columns or 'AGE' in X.columns
