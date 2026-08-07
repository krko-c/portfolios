"""
유니버스 데이터 품질 점검.

문제가 있는 데이터를 조용히 제외하지 않고, 어떤 종목이 왜 걸렸는지 표로
보여준다. 필터·점수·선정 단계로 넘어가기 전에 반드시 이 점검부터 거친다.
"""
import pandas as pd

REQUIRED_COLS = ["ticker", "name", "market_cap"]
ISSUE_COLS = ["구분", "심각도", "종목", "사유"]


def check_universe(df: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    """
    유니버스 원본을 점검해 문제 목록을 돌려준다.
    반환: DataFrame[구분, 심각도, 종목, 사유]. 문제가 없으면 빈 표.
    """
    issues = []

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append({"구분": "스키마", "심각도": "오류", "종목": "-",
                       "사유": f"필수 컬럼 누락: {', '.join(missing_cols)}"})
        return pd.DataFrame(issues, columns=ISSUE_COLS)

    tickers = df["ticker"].astype(str).str.strip()

    empty_n = int((tickers == "").sum())
    if empty_n:
        issues.append({"구분": "식별자", "심각도": "오류", "종목": "(빈 값)",
                       "사유": f"티커가 빈 행 {empty_n}개"})

    dup = tickers[tickers.duplicated(keep=False) & (tickers != "")]
    for t in sorted(dup.unique()):
        issues.append({"구분": "중복", "심각도": "오류", "종목": t,
                       "사유": "같은 티커가 여러 행에 존재 (정규화 전 원본 기준)"})

    mc = pd.to_numeric(df["market_cap"], errors="coerce")
    for t in df.loc[mc.isna() | (mc <= 0), "ticker"]:
        issues.append({"구분": "시가총액", "심각도": "오류", "종목": str(t),
                       "사유": "시가총액 누락 또는 0 이하"})

    if "adtv" in df.columns:
        adtv = pd.to_numeric(df["adtv"], errors="coerce")
        for t in df.loc[adtv.isna(), "ticker"]:
            issues.append({"구분": "유동성", "심각도": "경고", "종목": str(t),
                           "사유": "일평균 거래대금(ADTV) 누락 — 유동성 필터를 걸면 제외됨"})

    if "price" in df.columns:
        px = pd.to_numeric(df["price"], errors="coerce")
        for t in df.loc[px.isna() | (px <= 0), "ticker"]:
            issues.append({"구분": "가격", "심각도": "경고", "종목": str(t),
                           "사유": "가격 누락 또는 0 이하"})

    if "currency" in df.columns:
        ccy = df["currency"].dropna().astype(str).unique()
        if len(ccy) > 1:
            issues.append({"구분": "통화", "심각도": "경고", "종목": "-",
                           "사유": f"통화가 섞여 있음: {', '.join(sorted(ccy))} — "
                                  f"환산 없이 시가총액을 비교하면 왜곡됨"})

    if "listing_date" in df.columns and as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        ld = pd.to_datetime(df["listing_date"], errors="coerce")
        for t in df.loc[ld.notna() & (ld > as_of), "ticker"]:
            issues.append({"구분": "미래 데이터", "심각도": "오류", "종목": str(t),
                           "사유": f"기준일({as_of.date()}) 이후 상장일 — "
                                  f"이 기준일 시점에는 존재하지 않았던 종목"})

    if "delisting_date" in df.columns and as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        dd = pd.to_datetime(df["delisting_date"], errors="coerce")
        for t in df.loc[dd.notna() & (dd <= as_of), "ticker"]:
            issues.append({"구분": "상장폐지", "심각도": "경고", "종목": str(t),
                           "사유": f"기준일({as_of.date()}) 이전 상장폐지 — "
                                  f"편입 대상에서 빠져야 함"})

    return pd.DataFrame(issues, columns=ISSUE_COLS)


def has_blocking_errors(issues: pd.DataFrame) -> bool:
    """다음 단계로 진행 가능한지. '오류'가 하나라도 있으면 막는다."""
    return bool(len(issues)) and (issues["심각도"] == "오류").any()
