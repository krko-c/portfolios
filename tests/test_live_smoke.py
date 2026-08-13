"""
Live Data Smoke Test (docs/PLAN.md 0-4).

실제 Yahoo Finance를 호출해 "연결이 되고 기본적으로 동작하는가"만 확인한다.
골든 숫자와는 비교하지 않는다 — 그건 tools/golden.py(고정 fixture, 네트워크
없음)의 역할이다. 이 테스트는 네트워크 실패 시 실패가 아니라 skip 처리해서
CI에 빨간불이 나지 않게 한다 (야후 쪽 일시 장애로 무관한 PR이 막히는 것 방지).

    python3 -m pytest tests/test_live_smoke.py -v
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _app_extract import extract  # noqa: E402

NEED_FUNCS = {
    "build_price_frame", "probe_ticker", "load_ticker", "load_fx",
    "_suffix_currency", "_is_symbol_list", "_clean_name", "_fi_get",
}
PRELUDE = """
import re
import pandas as pd
import streamlit as st
import yfinance as yf
"""

# 골든 fixture의 고정 입력 세트와는 무관한, 스모크 테스트 전용의 짧은 최근 구간이다.
# 여기서는 "요즘도 데이터가 내려오는가"만 보면 되므로 5년치를 받을 필요가 없다.
TICKERS = ["SPY", "005930.KS"]  # 미국(USD) + 한국(KRW) — 통화변환 경로까지 확인
START = (date.today() - timedelta(days=30)).isoformat()
END = date.today().isoformat()


@pytest.fixture(scope="module")
def live_frame():
    try:
        mod = extract(NEED_FUNCS, prelude=PRELUDE)
        raw_spy = mod.load_ticker("SPY", START, END, currency="USD", name="SPY")
        prices, meta, fx_used = mod.build_price_frame(
            TICKERS, START, END, "KRW", True, fx_hedge=False)
    except Exception as ex:
        pytest.skip(f"네트워크/데이터 실패로 스킵: {type(ex).__name__}: {ex}")
    if prices.empty:
        pytest.skip("빈 데이터로 스킵 (야후 쪽 일시적 문제로 추정)")
    return prices, meta, fx_used, raw_spy


def test_price_data_not_empty(live_frame):
    prices, _, _, _ = live_frame
    assert not prices.empty


def test_required_columns_exist(live_frame):
    prices, _, _, _ = live_frame
    for t in TICKERS:
        assert t in prices.columns


def test_currency_conversion_changes_value(live_frame):
    """KRW 환산 SPY 값이 원시(USD) 값과 실제로 다른지 — 변환 로직이 동작했는지 직접 확인."""
    prices, meta, _, raw_spy = live_frame
    assert meta["SPY"]["currency"] == "USD"
    common = prices.index.intersection(raw_spy["close"].dropna().index)
    assert len(common) > 0
    converted = float(prices.loc[common[-1], "SPY"])
    raw = float(raw_spy["close"].loc[common[-1]])
    assert converted != pytest.approx(raw, rel=1e-6)


def test_data_not_stale(live_frame):
    prices, _, _, _ = live_frame
    last = prices.index.max()
    staleness_days = (pd.Timestamp.now().normalize() - last).days
    assert staleness_days <= 10, f"최근 데이터가 {staleness_days}일 전 — 너무 오래됨"
