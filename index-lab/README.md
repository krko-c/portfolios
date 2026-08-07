# Index Lab

규칙 기반 ETF·인덱스 설계·검증 도구. Portfolio Analyzer(`../app.py`)와는
목적이 다르다 — 이미 정해진 포트폴리오를 분석하는 게 아니라, 투자 아이디어를
규칙으로 바꿔 구성종목·비중을 결정론적으로 산출하고 검증한다.

## 현재 단계

1단계(규칙 기반 MVP) 진행 중. 최적화 solver·블랙-리터만·Point-in-Time
데이터·감사로그는 아직 없다 — 사전에 정한 결정론적 규칙(필터→점수→선정→
비중→정기변경)만 쓴다. 데이터는 yfinance 수준만 가정하므로 백테스트는
"탐색형"이다(현재 종목군을 과거로 소급, 생존편향 있음).

## 구조

```
core/               계산 로직 (Streamlit 의존 없음)
  validation.py     유니버스 데이터 품질 점검
  filters.py        적격성 필터 (깔때기 표 포함)
  scoring.py        종합점수 계산
  selection.py      구성종목 선정 + 버퍼룰
  weighting.py       비중 설계 + 상한 반복 재배분
  index_simulator.py 정기변경을 반영한 동적 지수 시뮬레이터
  analytics.py       성과지표 (Portfolio Analyzer에서 검증된 공식 재사용)
  export.py          방법론 JSON / Excel 출력
tests/              core/ 각 모듈의 단위테스트 (pytest)
pages/              Streamlit 화면 (예정)
```

## 실행

```bash
pip install -r requirements.txt
python -m pytest tests/ -v      # 계산 로직 검증
streamlit run app.py            # 화면 (예정)
```

## 저장

세션 상태 + JSON 다운로드/업로드만 쓴다. 서버 파일에 저장하지 않는다 —
Portfolio Analyzer에서 겪은 것처럼, 여러 사람이 같은 배포 인스턴스를 쓰면
서버 파일이 세션 간에 섞일 수 있기 때문이다. 실제로 여러 명이 협업하며
쓰기 시작하면 그때 진짜 저장소(SQLite 등)로 바꾼다.
