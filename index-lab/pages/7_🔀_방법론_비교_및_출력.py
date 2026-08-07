"""7️⃣ 방법론 대안 비교 + 결과 출력."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import analytics as an
from core.export import build_excel, methodology_to_json
from core.index_simulator import simulate_index
from core.weighting import apply_caps, compute_weights

st.set_page_config(page_title="Index Lab · 방법론 비교·출력", page_icon="🔀", layout="wide")
st.title("🔀 7️⃣ 방법론 대안 비교 및 결과 출력")

if "il_bt_prices" not in st.session_state:
    st.warning("먼저 6️⃣ 정기변경 백테스트 단계를 완료해주세요.")
    st.stop()

st.subheader("방법론 대안 비교")
st.caption("같은 유니버스·같은 구성종목·같은 기간·같은 거래비용으로, 비중 방식만 바꿔 비교합니다. "
          "1등만 보지 말고 상위 대안의 안정성도 함께 보세요.")

universe = st.session_state["il_universe_filtered"]
scored = st.session_state["il_scored"]
tickers = st.session_state["il_selected_tickers"]
merged = universe.merge(scored[["ticker", "종합점수"]], on="ticker", how="left")
merged = merged[merged["ticker"].isin(tickers)].reset_index(drop=True)

px = st.session_state["il_bt_prices"]
dates = st.session_state["il_bt_dates"]
cost_bp = st.session_state["il_bt_cost_bp"]
bench_level = st.session_state.get("il_bt_bench_level")
cap_stock = st.session_state["il_methodology"].get("비중 설계", {}).get("종목상한", 1.0)

alt_schemes = st.multiselect(
    "비교할 비중 방식", ["동일가중", "시가총액가중", "점수가중", "역변동성가중"],
    default=["동일가중", "시가총액가중", "점수가중"])

rows = []
for scheme in alt_schemes:
    try:
        w = compute_weights(merged, ticker_col="ticker", scheme=scheme,
                            score_col="종합점수", mktcap_col="market_cap")
        w = apply_caps(w, cap_per_stock=cap_stock)
        tw = {d: w.to_dict() for d in dates}
        res = simulate_index(px[list(w.index)], tw, cost_bp=cost_bp)
        summ = an.summary_table(res["level"], bench_level=bench_level)
        summ["방식"] = scheme
        summ["연 회전율"] = float(res["turnover"]["단방향회전율"].sum() /
                               max(1, (dates[-1] - dates[0]).days / 365.25))
        rows.append(summ)
    except ValueError as ex:
        st.warning(f"{scheme}: {ex}")

if rows:
    cmp_df = pd.DataFrame(rows).set_index("방식")
    fmt = {c: "{:.2%}" for c in cmp_df.columns if c != "샤프" and c != "소르티노"
          and c != "정보비율"}
    fmt.update({"샤프": "{:.2f}", "소르티노": "{:.2f}", "정보비율": "{:.2f}"})
    st.dataframe(cmp_df.style.format(fmt, na_rep="-"), width="stretch")

st.divider()
st.subheader("결과 출력")

methodology = dict(st.session_state.get("il_methodology", {}))
json_str = methodology_to_json(methodology)
c1, c2 = st.columns(2)
c1.download_button("📄 방법론 JSON", json_str.encode("utf-8"),
                   "methodology.json", "application/json", width="stretch")

try:
    xlsx = build_excel(
        constituents=st.session_state["il_selection"],
        selection_log=st.session_state["il_selection"],
        funnel=st.session_state.get("il_funnel", pd.DataFrame()),
        backtest_summary=an.summary_table(
            st.session_state["il_backtest_level"], bench_level=bench_level),
        constituent_history=st.session_state.get("il_backtest_hist", pd.DataFrame()),
        methodology=methodology,
    )
    c2.download_button(
        "📊 결과 Excel", xlsx,
        f"index_lab_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")
except Exception as ex:
    st.error(f"Excel 생성 실패: {ex}")

with st.expander("📋 방법론 설정 미리보기", expanded=False):
    st.json(methodology)
