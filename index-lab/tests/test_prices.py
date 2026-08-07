import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.prices import fetch_close_prices, kr_candidates

IDX = pd.date_range("2024-01-02", periods=3, freq="B")


def _multiindex_download(prices: dict):
    """prices: {symbol: [close,...]} -> yf.download(group_by='column') 형태의
    MultiIndex(level0=field, level1=symbol) DataFrame을 흉내낸다."""
    cols = pd.MultiIndex.from_product([["Close"], list(prices.keys())])
    data = {("Close", sym): vals for sym, vals in prices.items()}
    return pd.DataFrame(data, index=IDX, columns=cols)


def _flat_download(symbol: str, closes):
    """단일 심볼 요청 시 컬럼이 평평하게(OHLCV) 오는 yfinance 버전을 흉내낸다."""
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": [1000] * len(closes),
    }, index=IDX)


def test_kr_candidates_bare_six_digit_tries_ks_then_kq():
    assert kr_candidates("005930") == ["005930.KS", "005930.KQ"]


def test_kr_candidates_already_suffixed_or_foreign_unchanged():
    assert kr_candidates("005930.KS") == ["005930.KS"]
    assert kr_candidates("AAPL") == ["AAPL"]
    assert kr_candidates("^GSPC") == ["^GSPC"]


def test_fetch_close_prices_bare_ticker_resolves_via_ks_first_try():
    def downloader(syms):
        assert syms == ["005930.KS"]
        return _multiindex_download({"005930.KS": [100, 101, 102]})

    px, failed = fetch_close_prices(["005930"], downloader)
    assert failed == []
    assert list(px.columns) == ["005930"]
    assert px["005930"].tolist() == [100, 101, 102]


def test_fetch_close_prices_falls_back_to_kq_when_ks_has_no_data():
    calls = []

    def downloader(syms):
        calls.append(list(syms))
        if syms == ["247540.KS"]:
            return _multiindex_download({"247540.KS": [None, None, None]})
        assert syms == ["247540.KQ"]
        return _multiindex_download({"247540.KQ": [50, 51, 52]})

    px, failed = fetch_close_prices(["247540"], downloader)
    assert failed == []
    assert px["247540"].tolist() == [50, 51, 52]
    assert calls == [["247540.KS"], ["247540.KQ"]]


def test_fetch_close_prices_reports_failure_when_both_ks_and_kq_missing():
    def downloader(syms):
        if syms == ["999999.KS"]:
            return _multiindex_download({"999999.KS": [None, None, None]})
        return _multiindex_download({"999999.KQ": [None, None, None]})

    px, failed = fetch_close_prices(["999999"], downloader)
    assert failed == ["999999"]
    assert "999999" not in px.columns


def test_fetch_close_prices_non_kr_ticker_not_retried():
    calls = []

    def downloader(syms):
        calls.append(list(syms))
        return _multiindex_download({"AAPL": [200, 201, 202]})

    px, failed = fetch_close_prices(["AAPL"], downloader)
    assert failed == []
    assert px["AAPL"].tolist() == [200, 201, 202]
    assert len(calls) == 1  # 재시도 없이 1회만 호출


def test_fetch_close_prices_mixed_batch_preserves_input_order():
    def downloader(syms):
        if set(syms) == {"AAPL", "005930.KS"}:
            return _multiindex_download({
                "AAPL": [10, 11, 12], "005930.KS": [100, 101, 102],
            })
        raise AssertionError(f"unexpected retry call with {syms}")

    px, failed = fetch_close_prices(["005930", "AAPL"], downloader)
    assert failed == []
    assert list(px.columns) == ["005930", "AAPL"]


def test_fetch_close_prices_handles_flat_single_symbol_response():
    def downloader(syms):
        assert syms == ["005930.KS"]
        return _flat_download("005930.KS", [100, 101, 102])

    px, failed = fetch_close_prices(["005930"], downloader)
    assert failed == []
    assert px["005930"].tolist() == [100, 101, 102]


def test_fetch_close_prices_empty_input():
    px, failed = fetch_close_prices([], lambda syms: pd.DataFrame())
    assert px.empty and failed == []
