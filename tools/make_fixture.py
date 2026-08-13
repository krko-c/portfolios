#!/usr/bin/env python3
"""
골든 회귀 테스트용 원시 데이터를 1회 생성한다 (docs/PLAN.md 0-2).

이 스크립트는 1회성입니다. tests/fixtures/ 에 parquet을 만든 뒤
커밋하고, 다시 실행하지 마십시오. yfinance는 소급 수정이 잦아서
재실행하면 같은 입력으로도 다른 값이 나올 수 있고, 그러면 골든
기준선이 바뀌어 과거 골든과 비교할 수 없게 됩니다.

이미 fixture가 존재하면 실행을 중단합니다 (실수로 덮어쓰는 사고 방지).

실제 yfinance 네트워크 호출이 필요합니다. 이 저장소를 체크아웃한 뒤
네트워크가 되는 환경에서 한 번 실행하십시오.

    python3 tools/make_fixture.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _app_extract import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
PRICES_PATH = FIXTURE_DIR / "golden_prices_raw.parquet"
FX_PATH = FIXTURE_DIR / "golden_fx_raw.parquet"
README_PATH = FIXTURE_DIR / "README.md"

# 고정 입력 세트 (docs/PLAN.md 0-2 — 변경 금지. 바꾸면 과거 골든과 비교 불가)
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "005930.KS"]
TICKER_CCY = {"SPY": "USD", "QQQ": "USD", "TLT": "USD", "GLD": "USD", "005930.KS": "KRW"}
START, END = "2018-01-01", "2025-12-31"
BASE_CCY = "KRW"

NEED_FUNCS = {"probe_ticker", "load_ticker", "load_fx",
              "_suffix_currency", "_is_symbol_list", "_clean_name", "_fi_get"}
PRELUDE = """
import re
import pandas as pd
import streamlit as st
import yfinance as yf
"""


def _load_real_loaders():
    return extract(NEED_FUNCS, prelude=PRELUDE)


def main():
    if PRICES_PATH.exists() or FX_PATH.exists():
        existing = PRICES_PATH if PRICES_PATH.exists() else FX_PATH
        print(f"❌ 이미 fixture가 있습니다: {existing}")
        print("   재실행하면 골든 기준선이 바뀝니다. 다시 만들어야 한다면")
        print("   왜 필요한지 먼저 기록한 뒤 파일을 직접 지우고 실행하십시오.")
        return 1

    mod = _load_real_loaders()

    print(f"가격 데이터 수집: {START} ~ {END}")
    price_cols = {}
    for t in TICKERS:
        pr = mod.probe_ticker((t,))
        ccy = pr.get("currency") or None
        if ccy and ccy != TICKER_CCY[t]:
            print(f"❌ {t}: 예상 통화 {TICKER_CCY[t]} 인데 probe_ticker는 {ccy} 를 반환했습니다.")
            print("   고정 입력 세트(TICKER_CCY)가 더 이상 맞지 않을 수 있습니다. 중단합니다.")
            return 1
        d = mod.load_ticker(t, START, END, currency=TICKER_CCY[t], name=pr.get("name"))
        price_cols[f"{t}__close"] = d["close"]
        price_cols[f"{t}__adjclose"] = d["adjclose"]
        print(f"  {t} ({TICKER_CCY[t]}, {d['name']}): "
              f"close {len(d['close'])}건 · adjclose {len(d['adjclose'])}건, "
              f"{d['close'].index.min().date()} ~ {d['close'].index.max().date()}")

    prices = pd.DataFrame(price_cols)
    prices.index.name = "date"

    fx_needed = sorted({ccy for ccy in TICKER_CCY.values() if ccy != BASE_CCY})
    print(f"\n환율 데이터 수집 (기준통화 {BASE_CCY}):")
    fx_cols = {}
    for ccy in fx_needed:
        s = mod.load_fx(ccy, BASE_CCY, START, END)
        fx_cols[f"{ccy}{BASE_CCY}"] = s
        print(f"  {ccy}->{BASE_CCY}: {len(s)}건, "
              f"{s.index.min().date()} ~ {s.index.max().date()}")

    fx = pd.DataFrame(fx_cols)
    fx.index.name = "date"

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PATH)
    fx.to_parquet(FX_PATH)
    print(f"\n저장: {PRICES_PATH.relative_to(ROOT)}  ({prices.shape[0]}행 × {prices.shape[1]}열)")
    print(f"저장: {FX_PATH.relative_to(ROOT)}  ({fx.shape[0]}행 × {fx.shape[1]}열)")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    README_PATH.write_text(_readme_text(prices, fx, generated_at), encoding="utf-8")
    print(f"저장: {README_PATH.relative_to(ROOT)}")
    print("\n이 fixture는 골든 회귀 테스트의 기준선입니다.")
    print("커밋한 뒤 이 스크립트를 다시 실행하지 마십시오.")
    return 0


def _readme_text(prices: pd.DataFrame, fx: pd.DataFrame, generated_at: str) -> str:
    per_ticker = "\n".join(
        f"- `{t}` ({TICKER_CCY[t]}): "
        f"{prices[f'{t}__close'].dropna().index.min().date()} ~ "
        f"{prices[f'{t}__close'].dropna().index.max().date()}, "
        f"{prices[f'{t}__close'].dropna().shape[0]}행"
        for t in TICKERS
    )
    fx_lines = "\n".join(
        f"- `{c}`: {fx[c].dropna().index.min().date()} ~ {fx[c].dropna().index.max().date()}, "
        f"{fx[c].dropna().shape[0]}행"
        for c in fx.columns
    )
    return f"""# 골든 회귀 테스트 fixture

`tools/make_fixture.py` 로 생성했습니다. **이 디렉터리의 parquet은 갱신 금지입니다.**

## 생성 시점

{generated_at} (UTC)

## 출처

Yahoo Finance (`yfinance`), `app.py` 의 `probe_ticker`/`load_ticker`/`load_fx` 를
그대로 통해 받았습니다 (골든 테스트가 실제 앱과 같은 경로로 데이터를 받았는지
보장하기 위함). `tools/make_fixture.py` 실행 시점에 1회만 호출했습니다.

## 고정 입력 세트

```
tickers:   {TICKERS}
start:     {START}
end:       {END}
base_ccy:  {BASE_CCY}
```

이 값들은 `tools/make_fixture.py` 와 `tools/golden.py` 양쪽에 하드코딩되어
있습니다 (docs/PLAN.md 0-2). **바꾸지 마십시오** — 바꾸면 이 fixture로 찍은
과거 골든과 비교할 수 없게 됩니다.

## 파일

- `golden_prices_raw.parquet` — 종목별 **원시** 가격 (각 현지통화, 환율 미적용).
  컬럼은 `{{티커}}__close`, `{{티커}}__adjclose`.
- `golden_fx_raw.parquet` — 원시 환율. 컬럼은 `{{통화쌍}}` (예: `USDKRW`).

원시 가격과 원시 환율을 따로 얼린 이유: 최종(환산 완료) 가격 프레임만 얼리면
`build_price_frame()` 의 FX 변환 로직 자체가 검증 대상에서 빠집니다.
과거 환율 `bfill` 사고가 났던 바로 그 경로라, 원시 데이터부터 통과시켜야
FX 변환 → dropna → 수익률 계산 경로 전체가 회귀 테스트에 들어갑니다.

## 수집 결과

종목별 가격:
{per_ticker}

환율:
{fx_lines}

## 갱신 금지

이 fixture를 다시 만들고 싶다면 (예: 고정 입력 세트 자체를 바꾸기로
결정한 경우) `tools/make_fixture.py` 를 재실행하기 전에:

1. 왜 바꿔야 하는지 `docs/PLAN.md` 0순위 항목 아래에 먼저 기록하고
2. 기존 골든 JSON들이 전부 무효가 된다는 것을 인지한 뒤
3. 이 파일들을 직접 지우고 실행하십시오. `tools/make_fixture.py` 는
   fixture가 이미 있으면 스스로 중단합니다.
"""


if __name__ == "__main__":
    sys.exit(main())
