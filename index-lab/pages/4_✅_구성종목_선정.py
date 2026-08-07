"""4️⃣ 구성종목 선정 + 버퍼룰."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.selection import select_constituents, current_constituents

st.set_page_config(page_title="Index Lab · 구성종목 선정", page_icon="✅", layout="wide")
st.title("✅ 4️⃣ 구성종목 선정")
st.caption("종합점수 상위 N개를 선정합니다. 버퍼룰을 쓰면 순위가 살짝 흔들릴 때마다 "
          "종목을 교체하지 않아도 됩니다.")

if "il_scored" not in st.session_state:
    st.warning("먼저 3️⃣ 점수 설계 단계를 완료해주세요.")
    st.stop()

scored = st.session_state["il_scored"]
n_max = len(scored)

c1, c2 = st.columns(2)
target_n = c1.number_input("목표 구성종목 수", min_value=1, max_value=n_max,
                           value=min(30, n_max), step=1)
use_buffer = c2.checkbox("버퍼룰 사용", value=False)

buffer_in = buffer_out = None
if use_buffer:
    b1, b2 = st.columns(2)
    buffer_in = b1.number_input("신규 편입 기준 (순위 이내)", min_value=1,
                                max_value=n_max, value=min(int(target_n * 0.85), n_max))
    buffer_out = b2.number_input("기존 유지 기준 (순위 이내)", min_value=1,
                                 max_value=n_max, value=min(int(target_n * 1.15), n_max))
    if buffer_in > buffer_out:
        st.error("🚫 신규 편입 기준이 유지 기준보다 느슨할 수 없습니다.")
        st.stop()

st.caption("최초 구성이면 '직전 구성종목'을 비워두세요. 이후 정기변경을 이어서 "
          "시뮬레이션하려면 직전 구성종목을 입력하세요.")
prev_input = st.text_input("직전 구성종목 (콤마 구분, 선택)", value="")
prev = [t.strip() for t in prev_input.split(",") if t.strip()] or None

try:
    sel = select_constituents(scored, target_n=int(target_n),
                              buffer_in=buffer_in, buffer_out=buffer_out,
                              prev_constituents=prev)
except ValueError as ex:
    st.error(f"🚫 {ex}")
    st.stop()

st.subheader("선정 결과")
st.dataframe(sel, width="stretch", height=400, hide_index=True)

picked = current_constituents(sel)
st.success(f"✅ 최종 {len(picked):,}개 종목 선정 — "
          f"{(sel['상태']=='신규편입').sum()}개 신규편입, "
          f"{(sel['상태']=='유지').sum()}개 유지, "
          f"{(sel['상태']=='편출').sum()}개 편출, "
          f"{(sel['상태']=='조건미달편출').sum()}개 조건미달편출")

st.session_state["il_selection"] = sel
st.session_state["il_selected_tickers"] = picked
st.session_state["il_methodology"]["선정"] = {
    "목표 종목수": int(target_n), "버퍼룰 사용": use_buffer,
    "신규편입 기준": buffer_in, "유지 기준": buffer_out,
}
