"""Index Lab — 규칙 기반 ETF·인덱스 설계·검증 도구. 진입 화면."""
import streamlit as st

st.set_page_config(page_title="Index Lab", page_icon="📐", layout="wide")

st.title("📐 Index Lab")
st.caption("규칙 기반 ETF·인덱스 설계·검증 도구 — 1단계(MVP)")

st.markdown("""
이 도구는 **투자 아이디어를 규칙으로 바꿔 구성종목·비중을 결정론적으로
산출하고 검증**합니다. Portfolio Analyzer처럼 이미 정해진 포트폴리오를
분석하는 게 아니라, "어떤 종목을 어떤 규칙으로 편입할 것인가"부터 다룹니다.

### 이 단계에서 하는 것과 안 하는 것

**합니다**
- 유니버스 데이터 품질 점검 → 적격성 필터 → 종합점수 → 구성종목 선정(버퍼룰
  포함) → 비중 설계(상한 반복 재배분 포함) → 정기변경 백테스트 → 방법론 대안
  비교 → 방법론 JSON·Excel 출력

**안 합니다** (2단계 이후)
- 최적화 solver, 블랙-리터만 같은 자산배분 도구 — 인덱스는 사전에 정한
  규칙의 결과여야지, 최적화로 찾은 결과여선 안 됩니다
- Point-in-Time 데이터 검증, 감사로그, 방법론 버전관리
- 실시간 데이터 자동화 (DART·ECOS API 연동)
""")

st.warning("⚠️ **탐색형 백테스트입니다.** 현재 조회 가능한 종목군과 데이터를 "
          "과거로 소급해 계산하므로 생존편향이 있습니다. 실제 방법론 검증에는 "
          "쓰지 말고, 아이디어를 빠르게 점검하는 용도로만 보세요.")

st.divider()
st.markdown("""
### 시작하는 법

왼쪽 사이드바에서 **1️⃣ 데이터·유니버스**부터 순서대로 진행하세요. 각
단계는 이전 단계의 결과를 세션에 보관합니다(서버에 저장하지 않음 — 브라우저
탭을 닫으면 사라지니, 마지막 단계에서 JSON/Excel로 받아두세요).
""")

if "il_methodology" not in st.session_state:
    st.session_state["il_methodology"] = {}

with st.expander("📋 프로젝트 개요 (선택 입력)", expanded=False):
    c1, c2 = st.columns(2)
    proj_name = c1.text_input("프로젝트명", key="il_proj_name")
    index_name = c2.text_input("지수명", key="il_index_name")
    theme = st.text_area("투자 주제·목적", key="il_theme",
                         help="예: AI 인프라 관련 국내 상장기업 중 유동성 상위군")
    st.session_state["il_methodology"].update({
        "프로젝트명": proj_name, "지수명": index_name, "투자 주제": theme,
    })
