"""6️⃣ 정기변경을 반영한 지수 백테스트 (탐색형)."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import analytics as an
from core.index_simulator import simulate_index
from core.prices import fetch_close_prices

st.set_page_config(page_title="Index Lab · 백테스트", page_icon="📅", layout="wide")
st.title("📅 6️⃣ 정기변경 백테스트")
st.error("⚠️ **탐색형 백테스트입니다.** 지금 선정된 구성종목·비중을 과거 가격에 "
        "그대로 적용합니다 — 과거 그 시점에 실제로 이 종목들이 선정 기준을 "
        "만족했는지는 검증하지 않습니다(생존편향). 방법론 최종 검증에는 쓰지 마세요.")

if "il_weights" not in st.session_state:
    st.warning("먼저 5️⃣ 비중 설계 단계를 완료해주세요.")
    st.stop()

w = st.session_state["il_weights"]
tickers = list(w.index)

c1, c2, c3 = st.columns(3)
years = c1.selectbox("백테스트 기간", [1, 2, 3, 5], index=1)
rebal_freq = c2.selectbox("정기변경 주기", ["분기별", "반기별", "연별"], index=1)
cost_bp = c3.number_input("정기변경 거래비용 (편도, bp)", min_value=0.0,
                          value=0.0, step=5.0)
bench_tk = st.text_input("벤치마크 티커 (선택, Yahoo Finance 기준)", value="")

FREQ_CODE = {"분기별": "QS", "반기별": "2QS", "연별": "AS"}

if st.button("🚀 백테스트 실행", type="primary", width="stretch"):
    st.session_state["il_run_bt"] = True
if not st.session_state.get("il_run_bt"):
    st.info("👆 위 설정을 확인하고 백테스트 실행을 눌러주세요.")
    st.stop()

end = pd.Timestamp.today()
start = end - pd.DateOffset(years=years)

fetch_list = list(tickers) + ([bench_tk.strip()] if bench_tk.strip() else [])
try:
    with st.spinner("가격 데이터 조회 중..."):
        downloader = lambda syms: yf.download(  # noqa: E731
            syms, start=start, end=end, auto_adjust=True, progress=False)
        px, missing = fetch_close_prices(fetch_list, downloader)
except Exception as ex:
    st.error(f"🚫 가격 데이터를 가져오지 못했습니다: {ex}")
    st.stop()

missing = [t for t in missing if t in tickers]
if missing:
    st.error(f"🚫 가격 데이터가 없는 종목: {', '.join(missing)}. "
            f"티커 형식을 확인해주세요 — 한국 종목은 코스피 `.KS`(예: "
            f"`005930.KS`) / 코스닥 `.KQ`(예: `247540.KQ`) 접미사가 필요합니다 "
            f"(맨숫자만 있으면 `.KS`를 먼저, 안 되면 `.KQ`를 자동으로 시도했습니다). "
            f"샘플 티커를 쓰셨다면 실제 Yahoo Finance 티커로 유니버스를 다시 "
            f"올려주세요.")
    st.stop()

dates = pd.date_range(start, end, freq=FREQ_CODE[rebal_freq])
if len(dates) == 0 or dates[0] > px.index[0]:
    dates = pd.DatetimeIndex([px.index[0]]).append(dates)
target_weights = {d: w.to_dict() for d in dates}

try:
    result = simulate_index(px[tickers], target_weights, cost_bp=cost_bp,
                            base_value=1000.0)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

level = result["level"]
bench_level = None
if bench_tk.strip() and bench_tk.strip() in px.columns:
    b = px[bench_tk.strip()].dropna()
    b = b.loc[level.index[0]:level.index[-1]]
    bench_level = 1000.0 * b / b.iloc[0]

st.subheader("성과")
summary = an.summary_table(level, bench_level=bench_level)
cols = st.columns(len(summary))
for (k, v), col in zip(summary.items(), cols):
    pct_metrics = {"CAGR", "연변동성", "MDD", "초과수익(연)", "추적오차"}
    col.metric(k, f"{v*100:.2f}%" if k in pct_metrics else f"{v:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=level.index, y=level.values, name="지수"))
if bench_level is not None:
    fig.add_trace(go.Scatter(x=bench_level.index, y=bench_level.values,
                             name=f"벤치마크 ({bench_tk})"))
fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0),
                  yaxis=dict(title="지수 레벨 (시작=1,000)"))
st.plotly_chart(fig, width="stretch")

st.subheader("정기변경·회전율")
st.dataframe(result["turnover"].style.format(
    {"회전율": "{:.2%}", "단방향회전율": "{:.2%}"}), width="stretch", hide_index=True)

st.session_state["il_backtest_level"] = level
st.session_state["il_backtest_bench"] = bench_level
st.session_state["il_backtest_turnover"] = result["turnover"]
st.session_state["il_backtest_hist"] = result["constituent_history"]
st.session_state["il_bt_prices"] = px[tickers]
st.session_state["il_bt_dates"] = dates
st.session_state["il_bt_cost_bp"] = cost_bp
st.session_state["il_bt_bench_level"] = bench_level
st.session_state["il_methodology"]["백테스트"] = {
    "기간(년)": years, "정기변경 주기": rebal_freq, "거래비용(bp)": cost_bp,
    "벤치마크": bench_tk or "-",
}
