import os
from pathlib import Path
import pandas as pd
import tempfile
from src.preprocessing import detect_sep, clean_numeric_columns, parse_dates


def test_detect_sep_semicolon():
    txt = 'a;b\nc;d\n'
    tmp = Path(tempfile.gettempdir()) / 'tmp_semicolon.csv'
    tmp.write_text(txt)
    assert detect_sep(tmp) == ';'


def test_clean_numeric_columns_commas():
    df = pd.DataFrame({'TOTALBASKET': ['1.234,56', '78,9', '1.000'], 'UNITPRICE': ['4,5','2','3,14']})
    out = clean_numeric_columns(df.copy(), cols=['TOTALBASKET','UNITPRICE'])
    assert out['TOTALBASKET'].dtype.kind in 'fi'
    assert not out['TOTALBASKET'].isna().any()
    assert out['UNITPRICE'].iloc[0] == 4.5


def test_parse_dates():
    df = pd.DataFrame({'DATE_': ['17/5/23', '01/01/2020']})
    out = parse_dates(df.copy())
    assert out['DATE_'].dtype == 'datetime64[ns]'
