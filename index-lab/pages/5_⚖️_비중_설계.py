"""5️⃣ 비중 설계 + 상한 반복 재배분."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.weighting import compute_weights, apply_caps

st.set_page_config(page_title="Index Lab · 비중 설계", page_icon="⚖️", layout="wide")
st.title("⚖️ 5️⃣ 비중 설계")
st.caption("사전에 정한 결정론적 규칙만 씁니다 — 최적화 solver는 쓰지 않습니다.")

if "il_selection" not in st.session_state:
    st.warning("먼저 4️⃣ 구성종목 선정 단계를 완료해주세요.")
    st.stop()

universe = st.session_state["il_universe_filtered"]
scored = st.session_state["il_scored"]
tickers = st.session_state["il_selected_tickers"]

merged = universe.merge(scored[["ticker", "종합점수"]], on="ticker", how="left")
merged = merged[merged["ticker"].isin(tickers)].reset_index(drop=True)
if "market_cap" in merged.columns:
    vol_col = None  # 변동성 컬럼은 유니버스 파일에 없을 수 있어 선택 사항으로 둔다
else:
    vol_col = None

scheme = st.selectbox("비중 방식", ["동일가중", "시가총액가중", "점수가중",
                                  "역변동성가중", "시총점수혼합"])
blend_alpha = 0.5
if scheme == "시총점수혼합":
    blend_alpha = st.slider("시가총액 비중 (나머지는 점수)", 0.0, 1.0, 0.5, 0.05)

try:
    w = compute_weights(merged, ticker_col="ticker", scheme=scheme,
                        score_col="종합점수", mktcap_col="market_cap",
                        vol_col=vol_col, blend_alpha=blend_alpha)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

st.subheader("상한 제약")
c1, c2 = st.columns(2)
cap_stock = c1.slider("종목별 최대 비중", 0.01, 1.0, 1.0, 0.01)
use_group_cap = c2.checkbox("산업별 상한 사용", value=False)

group = cap_group = None
if use_group_cap and "sector" in merged.columns:
    cap_group = st.slider("산업별 최대 비중", 0.05, 1.0, 0.5, 0.05)
    group = merged.set_index("ticker")["sector"].reindex(w.index)

try:
    w_final = apply_caps(w, cap_per_stock=cap_stock, group=group, cap_per_group=cap_group)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

st.subheader("결과")
res = pd.DataFrame({"티커": w_final.index, "원래비중(%)": (w * 100).reindex(w_final.index).values,
                    "최종비중(%)": (w_final * 100).values}).sort_values(
    "최종비중(%)", ascending=False)
st.dataframe(res.style.format({"원래비중(%)": "{:.2f}", "최종비중(%)": "{:.2f}"}),
            width="stretch", height=400, hide_index=True)

k1, k2, k3 = st.columns(3)
top5 = float(w_final.sort_values(ascending=False).head(5).sum() * 100)
k1.metric("상위 5개 비중", f"{top5:.1f}%")
k2.metric("실효 종목 수", f"{1/float((w_final**2).sum()):.1f}")
k3.metric("종목 수", f"{len(w_final)}")

if group is not None:
    gsum = (w_final.groupby(group).sum() * 100).sort_values(ascending=False)
    st.markdown("**산업별 비중**")
    st.dataframe(gsum.rename("비중(%)").to_frame().style.format("{:.2f}"), width="stretch")

st.session_state["il_weights"] = w_final
st.session_state["il_methodology"]["비중 설계"] = {
    "방식": scheme, "종목상한": cap_stock, "산업상한": cap_group,
}
st.success("✅ 비중 설계 완료. 6️⃣ 정기변경 백테스트로 진행하세요.")
