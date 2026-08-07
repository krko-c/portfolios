"""
적격성 필터. 단계마다 몇 종목이 왜 빠졌는지 남긴다 — 결과만 보여주면
"왜 이 종목이 없지?"에 답할 수 없다.
"""
import pandas as pd

FunnelStep = tuple  # (단계명, 남은 종목수, 제외 종목수, 주요 제외 사유)


def apply_eligibility(df: pd.DataFrame, *, min_market_cap=0.0, min_adtv=0.0,
                      min_listing_days=0, as_of_date=None,
                      exclude_sectors=None):
    """
    적격성 필터를 순서대로 적용한다.
    반환: (통과한 종목 DataFrame, 깔때기 표 DataFrame, 종목별 편입/제외 사유 DataFrame)
    """
    cur = df.copy()
    cur["ticker"] = cur["ticker"].astype(str).str.strip()
    reasons = {t: [] for t in cur["ticker"]}
    funnel = [("최초 유니버스", len(cur), 0, "-")]

    def _step(name, mask, reason):
        nonlocal cur
        excluded = cur.loc[~mask, "ticker"]
        for t in excluded:
            reasons[t].append(reason)
        cur = cur.loc[mask]
        funnel.append((name, len(cur), int((~mask).sum()), reason))

    if min_market_cap > 0:
        mc = pd.to_numeric(cur["market_cap"], errors="coerce").fillna(0)
        _step("시가총액", mc >= min_market_cap, f"시가총액 < {min_market_cap:,.0f}")

    if min_adtv > 0:
        if "adtv" not in cur.columns:
            raise ValueError("유동성 필터를 쓰려면 'adtv' 컬럼이 필요합니다.")
        adtv = pd.to_numeric(cur["adtv"], errors="coerce").fillna(0)
        _step("유동성", adtv >= min_adtv, f"일평균 거래대금 < {min_adtv:,.0f}")

    if min_listing_days > 0:
        if "listing_date" not in cur.columns or as_of_date is None:
            raise ValueError("상장기간 필터를 쓰려면 'listing_date' 컬럼과 "
                             "기준일이 필요합니다.")
        as_of = pd.Timestamp(as_of_date)
        ld = pd.to_datetime(cur["listing_date"], errors="coerce")
        days = (as_of - ld).dt.days
        _step("상장기간", days >= min_listing_days,
              f"상장 후 {min_listing_days}일 미만")

    if "delisting_date" in cur.columns and as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        dd = pd.to_datetime(cur["delisting_date"], errors="coerce")
        _step("상장폐지 제외", dd.isna() | (dd > as_of), "기준일 이전 상장폐지")

    if exclude_sectors:
        if "sector" not in cur.columns:
            raise ValueError("산업 제외를 쓰려면 'sector' 컬럼이 필요합니다.")
        excl = set(exclude_sectors)
        _step("산업 제외", ~cur["sector"].isin(excl),
              f"제외 산업: {', '.join(sorted(excl))}")

    funnel_df = pd.DataFrame(funnel, columns=["단계", "종목 수", "제외 종목 수", "주요 제외 사유"])
    reason_df = pd.DataFrame(
        [{"티커": t, "결과": "통과" if t in set(cur["ticker"]) else "제외",
          "사유": "; ".join(reasons[t]) if reasons[t] else "-"}
         for t in df["ticker"].astype(str).str.strip()])
    return cur.reset_index(drop=True), funnel_df, reason_df
