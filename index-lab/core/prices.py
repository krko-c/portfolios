"""야후파이낸스 종가 조회 — 한국 종목 맨숫자 티커의 거래소 접미사 자동보정 포함.

야후파이낸스는 한국 종목을 `005930.KS`(코스피) / `005930.KQ`(코스닥) 형태로만
인식한다. 사용자가 `005930`처럼 접미사 없이 입력하면 전량 "가격 데이터 없음"
오류가 난다. 이 모듈은 맨숫자 6자리 티커에 한해 `.KS`를 먼저 시도하고, 실패한
종목만 `.KQ`로 재시도한다.
"""
import re

import pandas as pd

_KR_BARE = re.compile(r"^\d{6}$")


def kr_candidates(ticker: str) -> list:
    """맨숫자 6자리 한국 종목코드는 [코스피, 코스닥] 순으로 후보를 준다."""
    t = (ticker or "").strip()
    if _KR_BARE.match(t):
        return [t + ".KS", t + ".KQ"]
    return [t]


def _extract_close(raw, requested):
    """yf.download() 결과에서 종가만 뽑아 컬럼명을 요청 심볼로 맞춘다."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        return raw["Close"]
    if "Close" not in raw.columns:
        return pd.DataFrame()
    # 심볼 1개만 요청하면 일부 yfinance 버전은 컬럼을 평평하게 돌려준다.
    return raw[["Close"]].rename(columns={"Close": requested[0]})


def fetch_close_prices(tickers, downloader):
    """
    tickers 각각의 종가 시계열을 받아온다.

    downloader(symbol_list) -> yfinance 스타일 원본 DataFrame 을 받는 콜러블.
    (`lambda syms: yf.download(syms, start=start, end=end, auto_adjust=True,
    progress=False)` 형태로 호출부에서 주입 — 네트워크 접근을 격리해 테스트
    가능하게 한다.)

    반환 DataFrame 의 컬럼명은 입력한 원래 티커 문자열 그대로다. 예:
    '005930' 입력 시 내부적으로 '005930.KS'를 먼저 시도하고, 데이터가 없으면
    '005930.KQ'만 재시도하되, 결과 컬럼명은 '005930'.

    반환: (price_df, failed_tickers)
    """
    tickers = list(dict.fromkeys(t.strip() for t in tickers if t and t.strip()))
    if not tickers:
        return pd.DataFrame(), []

    candidates = {t: kr_candidates(t) for t in tickers}
    first_try = [candidates[t][0] for t in tickers]
    px1 = _extract_close(downloader(first_try), first_try)

    ok = {}
    retry = {}
    for t in tickers:
        c0 = candidates[t][0]
        if c0 in px1.columns and not px1[c0].dropna().empty:
            ok[t] = px1[c0]
        elif len(candidates[t]) > 1:
            retry[t] = candidates[t][1]

    if retry:
        retry_list = list(retry.values())
        px2 = _extract_close(downloader(retry_list), retry_list)
        for t, c1 in retry.items():
            if c1 in px2.columns and not px2[c1].dropna().empty:
                ok[t] = px2[c1]

    failed = [t for t in tickers if t not in ok]
    if not ok:
        return pd.DataFrame(), failed

    result = pd.concat(ok, axis=1)
    result = result[[t for t in tickers if t in ok]]
    return result, failed
