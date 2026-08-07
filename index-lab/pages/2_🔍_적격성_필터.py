"""2️⃣ 적격성 필터."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.filters import apply_eligibility

st.set_page_config(page_title="Index Lab · 적격성 필터", page_icon="🔍", layout="wide")
st.title("🔍 2️⃣ 적격성 필터")
st.caption("필수 조건입니다. 여기서 걸러진 종목은 이후 점수·선정 대상에서 완전히 빠집니다 "
          "(순위점수와는 별개).")

if "il_universe_raw" not in st.session_state:
    st.warning("먼저 1️⃣ 데이터·유니버스 단계를 완료해주세요.")
    st.stop()

df = st.session_state["il_universe_raw"]
as_of = st.session_state["il_as_of_date"]

c1, c2, c3 = st.columns(3)
min_mc = c1.number_input("최소 시가총액", min_value=0.0, value=0.0, step=1000.0)
min_adtv = c2.number_input("최소 일평균 거래대금(ADTV)", min_value=0.0, value=0.0, step=100.0,
                           help="0이면 필터를 걸지 않습니다.")
min_listing = c3.number_input("최소 상장기간(일)", min_value=0, value=0, step=30)

sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
excl_sectors = st.multiselect("제외할 산업", sectors)

try:
    passed, funnel, reasons = apply_eligibility(
        df, min_market_cap=min_mc, min_adtv=min_adtv,
        min_listing_days=min_listing, as_of_date=as_of,
        exclude_sectors=excl_sectors or None)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

st.subheader("3️⃣ 필터 깔때기")
st.dataframe(funnel, width="stretch", hide_index=True)

with st.expander("종목별 편입·제외 사유", expanded=False):
    st.dataframe(reasons, width="stretch", hide_index=True, height=300)

if passed.empty:
    st.error("🚫 필터를 통과한 종목이 없습니다. 조건을 완화해주세요.")
    st.stop()

st.success(f"✅ {len(passed):,}개 종목이 적격성 필터를 통과했습니다.")

st.session_state["il_universe_filtered"] = passed
st.session_state["il_funnel"] = funnel
st.session_state["il_reason_log"] = reasons
st.session_state["il_methodology"]["적격성 필터"] = {
    "최소 시가총액": min_mc, "최소 ADTV": min_adtv,
    "최소 상장기간(일)": min_listing, "제외 산업": excl_sectors,
}
