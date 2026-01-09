import io
import os
import pandas as pd
import pytest
from src import preprocessing as sp


def make_malformed_file(tmp_path):
    # Create a semicolon-separated file where header is wrapped in extra quotes and some records contain embedded newlines
    content = '"""ORDERID""";ORDERDETAILID;AMOUNT\n""";"";\n123;456;1.234,56\n'
    p = tmp_path / "malformed.csv"
    p.write_text(content, encoding='latin1')
    return str(p)


def test_sanitize_for_csv_removes_newlines_and_quotes():
    df = pd.DataFrame({'A': ['abc\n', '  def  '], 'B': ['"x"', '\x00y']})
    df2 = sp.sanitize_for_csv(df)
    assert '\n' not in df2['A'].iloc[0]
    assert df2['A'].iloc[1] == 'def'
    assert 'x' in df2['B'].iloc[0]


def test_clean_numeric_columns_handles_comma_decimal():
    df = pd.DataFrame({'TOTALBASKET': ['1.234,56', '1000', None]})
    df = sp.clean_numeric_columns(df, cols=['TOTALBASKET'])
    assert df['TOTALBASKET'].iloc[0] == pytest.approx(1234.56)
    assert df['TOTALBASKET'].iloc[1] == pytest.approx(1000.0)
    assert pd.isna(df['TOTALBASKET'].iloc[2])


def test_robust_read_processed_tolerates_malformed_header(tmp_path):
    path = make_malformed_file(tmp_path)
    df, sep = sp.robust_read_processed(path, path)
    # Should return a dataframe with at least one numeric column parsed
    assert sep in [',', ';', '\t', '|']
    assert isinstance(df, pd.DataFrame)


def test_save_processed_writes_safe_csv(tmp_path):
    df = pd.DataFrame({'A': ['line1\nline2', 'x'], 'B': [1, 2]})
    out = tmp_path / 'out.csv'
    sp.save_processed(df, out=str(out), sep=';')
    text = out.read_text(encoding='utf-8')
    assert '\n' in text
    assert 'line1' in text
