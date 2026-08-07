"""1️⃣ 데이터·유니버스 업로드 및 품질 점검."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.validation import check_universe, has_blocking_errors

st.set_page_config(page_title="Index Lab · 데이터·유니버스", page_icon="📥", layout="wide")
st.title("📥 1️⃣ 데이터·유니버스")
st.caption("투자 유니버스를 만듭니다. 파일을 올리거나, 표에 직접 티커를 입력해 "
          "야후파이낸스로 종목명·시가총액을 자동조회할 수도 있습니다.")

with st.expander("필수/선택 컬럼", expanded=False):
    st.markdown("""
| 컬럼 | 필수 | 설명 |
|---|---|---|
| `ticker` | ✅ | 종목 식별자 (야후파이낸스 형식, 예: `005930.KS`, `AAPL`) |
| `name` | ✅ | 종목명 |
| `market_cap` | ✅ | 시가총액 |
| `adtv` | 유동성 필터 시 | 일평균 거래대금 |
| `listing_date` | 상장기간 필터 시 | 상장일 |
| `delisting_date` | 선택 | 상장폐지일 |
| `sector` | 산업 제외 필터 시 | 산업분류 |
| `currency` | 선택 | 거래통화 |
| 그 외 숫자 컬럼 | 점수 설계용 | 예: theme_score, growth_score 등 자유롭게 |
""")

as_of = st.date_input("기준일 (as-of date)", value=pd.Timestamp.today(),
                      help="이 유니버스 데이터가 어느 시점 기준인지입니다. "
                           "상장일·상장폐지일 점검에 쓰입니다.")

MANUAL_COLS = ["ticker", "name", "market_cap", "adtv", "sector", "listing_date"]

mode = st.radio("입력 방식", ["⌨️ 직접 입력", "📤 파일 업로드", "🧪 샘플 데이터"],
                horizontal=True)

df = None

if mode == "⌨️ 직접 입력":
    if "il_manual_df" not in st.session_state:
        st.session_state["il_manual_df"] = pd.DataFrame(
            [{"ticker": "", "name": "", "market_cap": None, "adtv": None,
              "sector": "", "listing_date": None} for _ in range(8)])[MANUAL_COLS]

    st.caption("`ticker`만 채우고 **🔍 자동조회**를 누르면 종목명·시가총액·업종을 "
              "야후파이낸스에서 가져와 빈 칸만 채웁니다(이미 입력한 값은 안 건드림). "
              "이 기능은 이 개발 환경에서 실제 호출을 검증하지 못했습니다.")
    edited = st.data_editor(
        st.session_state["il_manual_df"], num_rows="dynamic", width="stretch",
        key="il_manual_editor", column_config={
            "ticker": st.column_config.TextColumn("티커", width="small"),
            "name": st.column_config.TextColumn("종목명", width="medium"),
            "market_cap": st.column_config.NumberColumn("시가총액", width="small"),
            "adtv": st.column_config.NumberColumn("ADTV", width="small"),
            "sector": st.column_config.TextColumn("업종", width="small"),
            "listing_date": st.column_config.DateColumn("상장일", width="small"),
        })
    st.session_state["il_manual_df"] = edited

    if st.button("🔍 야후파이낸스로 자동조회 (빈 칸만 채움)"):
        import yfinance as yf
        filled = edited.copy()
        n_ok, n_fail = 0, 0
        with st.spinner("종목 정보 조회 중..."):
            for i, row in filled.iterrows():
                t = str(row.get("ticker") or "").strip()
                if not t:
                    continue
                try:
                    info = yf.Ticker(t).info
                    if not str(row.get("name") or "").strip():
                        filled.at[i, "name"] = info.get("shortName") or info.get("longName") or t
                    if pd.isna(row.get("market_cap")):
                        mc = info.get("marketCap")
                        if mc:
                            filled.at[i, "market_cap"] = mc
                    if not str(row.get("sector") or "").strip():
                        filled.at[i, "sector"] = info.get("sector") or ""
                    n_ok += 1
                except Exception:
                    n_fail += 1
        st.session_state["il_manual_df"] = filled
        st.success(f"조회 완료 — 성공 {n_ok}건, 실패 {n_fail}건")
        st.rerun()

    live = edited.copy()
    live["ticker"] = live["ticker"].astype(str).str.strip()
    df = live[live["ticker"] != ""].reset_index(drop=True)
    if df.empty:
        st.info("👆 표에 티커를 입력해주세요.")

elif mode == "📤 파일 업로드":
    up = st.file_uploader("유니버스 파일 (Excel 또는 CSV)", type=["xlsx", "csv"])
    if up is not None:
        try:
            df = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
        except Exception as ex:
            st.error(f"파일을 읽지 못했습니다: {ex}")

else:  # 샘플 데이터
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "ticker": [f"SIM{i:03d}" for i in range(n)],
        "name": [f"Sample Co {i}" for i in range(n)],
        "market_cap": rng.lognormal(9, 1.2, n).round(0),
        "adtv": rng.lognormal(5, 1.0, n).round(0),
        "listing_date": pd.Timestamp("2010-01-01") +
                        pd.to_timedelta(rng.integers(0, 5000, n), unit="D"),
        "sector": rng.choice(["Tech", "Health", "Energy", "Consumer", "Industrial"], n),
        "theme_score": rng.normal(0, 1, n).round(3),
        "growth_score": rng.normal(0, 1, n).round(3),
        "currency": "KRW",
    })

if df is None or df.empty:
    st.stop()

st.subheader("2️⃣ 데이터 품질 점검")
issues = check_universe(df, as_of_date=as_of)
if len(issues):
    st.dataframe(issues, width="stretch", hide_index=True)
    n_err = int((issues["심각도"] == "오류").sum())
    n_warn = int((issues["심각도"] == "경고").sum())
    if n_err:
        st.error(f"🚫 오류 {n_err}건 — 다음 단계로 진행할 수 없습니다.")
    if n_warn:
        st.warning(f"⚠️ 경고 {n_warn}건 — 진행은 가능하지만 확인해주세요.")
else:
    st.success("✅ 문제 없음")

if has_blocking_errors(issues):
    st.stop()

st.subheader("3️⃣ 미리보기")
st.dataframe(df, width="stretch", height=300)
st.caption(f"총 {len(df):,}개 종목")

with st.expander("🧩 4️⃣ DART로 업종·설립일 보강 (선택, 실험적)", expanded=False):
    st.warning("⚠️ 이 기능은 개발 환경 네트워크 제약으로 실제 API 호출을 "
              "검증하지 못했습니다. 응답 스키마가 문서와 다르면 실패할 수 "
              "있습니다 — 소량으로 먼저 시도해보세요.")
    api_key = st.text_input("DART API 키", type="password",
                            help="https://opendart.fss.or.kr 에서 무료 발급")
    if st.button("DART 조회 실행", disabled=not api_key):
        try:
            from core.dart_client import enrich_universe
            with st.spinner("DART 조회 중... (종목 수가 많으면 오래 걸립니다)"):
                df = enrich_universe(df, api_key)
            n_fail = int(df["dart_오류"].notna().sum())
            if n_fail:
                st.warning(f"{n_fail}개 종목은 DART에서 매칭/조회하지 못했습니다.")
                st.dataframe(df.loc[df["dart_오류"].notna(), ["ticker", "dart_오류"]],
                            width="stretch", hide_index=True)
            st.success("완료 — 아래 미리보기와 이후 단계에 반영됩니다.")
            st.dataframe(df, width="stretch", height=300)
        except Exception as ex:
            st.error(f"🚫 DART 조회 실패: {ex}")

st.session_state["il_universe_raw"] = df
st.session_state["il_as_of_date"] = pd.Timestamp(as_of)
st.session_state.setdefault("il_methodology", {})
st.session_state["il_methodology"]["기준일"] = str(as_of)
st.session_state["il_methodology"]["유니버스 종목수"] = len(df)

st.success("✅ 다음 단계(2️⃣ 적격성 필터)로 진행할 수 있습니다.")
