import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.filters import apply_eligibility


def _universe():
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "market_cap": [1000, 500, 50, 2000, 800],
        "adtv": [10, 1, 5, 20, None],
        "listing_date": pd.to_datetime(
            ["2015-01-01", "2015-01-01", "2015-01-01", "2024-06-01", "2015-01-01"]),
        "sector": ["Tech", "Tech", "Energy", "Tech", "Health"],
    })


def test_market_cap_filter():
    passed, funnel, _ = apply_eligibility(_universe(), min_market_cap=600)
    assert set(passed["ticker"]) == {"A", "D", "E"}
    assert funnel.iloc[-1]["종목 수"] == 3


def test_adtv_filter_requires_column():
    df = _universe().drop(columns=["adtv"])
    with pytest.raises(ValueError):
        apply_eligibility(df, min_adtv=1)


def test_adtv_filter_excludes_missing_and_low():
    passed, _, _ = apply_eligibility(_universe(), min_adtv=5)
    # B(1) 미달, E(None) 미달 -> A, C, D 남음
    assert set(passed["ticker"]) == {"A", "C", "D"}


def test_listing_days_filter_needs_as_of_date():
    with pytest.raises(ValueError):
        apply_eligibility(_universe(), min_listing_days=365)


def test_listing_days_filter_excludes_recent_ipo():
    passed, _, _ = apply_eligibility(
        _universe(), min_listing_days=365, as_of_date="2024-12-31")
    assert "D" not in set(passed["ticker"])  # 2024-06 상장, 1년 미만
    assert "A" in set(passed["ticker"])


def test_sector_exclusion():
    passed, _, _ = apply_eligibility(_universe(), exclude_sectors=["Energy"])
    assert "C" not in set(passed["ticker"])
    assert len(passed) == 4


def test_funnel_step_order_and_counts():
    passed, funnel, _ = apply_eligibility(
        _universe(), min_market_cap=600, min_adtv=5)
    # 최초 5 -> 시가총액 필터 후 3(A,D,E) -> 유동성 필터 후 2(A,D; E는 adtv 결측)
    assert list(funnel["종목 수"]) == [5, 3, 2]
    assert set(passed["ticker"]) == {"A", "D"}


def test_reason_log_records_exclusion_reason():
    _, _, reasons = apply_eligibility(_universe(), min_market_cap=600)
    row = reasons.loc[reasons["티커"] == "C"].iloc[0]
    assert row["결과"] == "제외"
    assert "시가총액" in row["사유"]
