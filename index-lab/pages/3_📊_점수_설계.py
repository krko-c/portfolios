"""3️⃣ 테마 관련성 및 점수 설계."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.scoring import compute_score

st.set_page_config(page_title="Index Lab · 점수 설계", page_icon="📊", layout="wide")
st.title("📊 3️⃣ 종합점수 설계")
st.caption("필터를 통과한 종목을 순위화합니다. 필수 조건(적격성)과 순위점수는 분리돼 있으니, "
          "여기서 점수가 낮다고 탈락하지 않습니다 — 순서만 매깁니다.")

if "il_universe_filtered" not in st.session_state:
    st.warning("먼저 2️⃣ 적격성 필터 단계를 완료해주세요.")
    st.stop()

df = st.session_state["il_universe_filtered"]
non_factor_cols = {"ticker", "name", "sector", "currency", "listing_date",
                   "delisting_date", "adtv", "market_cap"}
numeric_cols = [c for c in df.columns if c not in non_factor_cols
               and pd.api.types.is_numeric_dtype(df[c])]
if "market_cap" in df.columns:
    numeric_cols = ["market_cap"] + numeric_cols

if not numeric_cols:
    st.error("점수화할 수 있는 숫자 컬럼이 없습니다. 1️⃣ 단계에서 요인 컬럼을 추가해주세요.")
    st.stop()

st.subheader("요인 선택")
chosen = st.multiselect("종합점수에 쓸 요인", numeric_cols,
                        default=numeric_cols[:min(2, len(numeric_cols))])
if not chosen:
    st.info("요인을 하나 이상 선택해주세요.")
    st.stop()

st.subheader("요인별 설정")
factors = {}
for col in chosen:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.markdown(f"**{col}**")
    w = c2.number_input("가중치", min_value=0.0, value=1.0, step=0.1, key=f"w_{col}")
    d = c3.selectbox("방향", ["클수록 유리(+1)", "작을수록 유리(-1)"], key=f"d_{col}")
    m = c4.selectbox("방식", ["zscore", "percentile", "raw"], key=f"m_{col}")
    factors[col] = {"weight": w, "direction": 1 if d.startswith("클수록") else -1,
                    "method": m}

if st.button("🔬 종합점수 계산", type="primary", width="stretch"):
    st.session_state["il_run_score"] = True
if not st.session_state.get("il_run_score"):
    st.info("👆 종합점수 계산을 눌러주세요.")
    st.stop()

try:
    scored = compute_score(df, factors)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

st.subheader("결과")
show_cols = ["ticker", "name", "순위", "종합점수"] + \
    [c for c in scored.columns if c.endswith("_점수")]
show_cols = [c for c in show_cols if c in scored.columns]
st.dataframe(scored[show_cols], width="stretch", height=400, hide_index=True)

st.session_state["il_scored"] = scored
st.session_state["il_methodology"]["점수 설계"] = {
    k: {"가중치": v["weight"], "방향": v["direction"], "방식": v["method"]}
    for k, v in factors.items()}
st.success(f"✅ {len(scored):,}개 종목에 종합점수를 매겼습니다. "
          f"4️⃣ 구성종목 선정으로 진행하세요.")
