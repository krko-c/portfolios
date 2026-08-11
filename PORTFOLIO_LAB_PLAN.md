# Portfolio Lab 개선 계획

작성일: 2026-08-11
대상: `github.com/krko-c/portfolios` — 루트 `app.py` (Portfolio Analyzer / Portfolio Lab)
기준 커밋: `63f96af` (app.py 10,135줄, 최상위 함수 163개)

---

## 시작하기 전에 (최초 1회)

### 이 문서를 저장소에 커밋한다

`docs/PLAN.md` 로 커밋합니다. 그래야 이후 Claude Code 세션에서 파일을 다시 붙여넣지 않고 "PLAN.md 0순위부터 시작해줘"로 바로 시작할 수 있습니다.

### HANDOVER.md에 포인터를 추가한다

이 프로젝트는 `HANDOVER.md` 가 진입 문서이므로 Claude Code가 그것을 먼저 읽습니다. 계획서가 그 밖에 있으면 못 볼 수 있습니다. `HANDOVER.md` 에 다음 섹션을 추가하십시오.

```markdown
## 진행 중인 개선 계획

개선 작업은 `docs/PLAN.md` 를 따른다.

- 0순위(골든 스냅샷)부터 순서대로 진행한다. 순서를 건너뛰지 않는다.
- 각 항목이 끝나면 PLAN.md에 완료 표시를 하고,
  계획과 달라진 점(추가로 필요했던 제약, 재사용이 안 된 함수,
  겪은 사고)을 해당 항목 아래에 기록한다.
- PLAN.md Part 3(보류·기각 기록)에 있는 항목은 다시 제안하지 않는다.
  이유가 이미 기록되어 있다.
- PLAN.md Part 5(전역 원칙)는 모든 작업에 적용된다.
```

### 줄 번호는 참고용이다

이 문서의 `(app.py:1369)` 같은 표기는 **커밋 `63f96af` 기준**입니다. 코드를 수정하면 그 아래 모든 줄 번호가 밀리므로, 작업이 진행될수록 실제 위치와 어긋납니다.

**항상 함수명으로 검색해서 찾으십시오.** 줄 번호로 바로 이동하지 마십시오.

```bash
grep -n "^def build_price_frame" app.py
```

`build_price_frame`, `feasibility`, `factor_betas`, `black_litterman`, `walk_forward`, `turnover_series`, `growth_contribution`, `perf_row`, `risk_cvar`, `guess_asset_kind`, `assumptions_panel`, `snapshot`, `_git_commit` 같은 함수명은 바뀌지 않습니다. 이 문서에서 줄 번호와 함수명이 함께 표기된 곳은 **함수명이 정본**입니다.

---

## 이 문서의 사용법

**Part 1 (0~2순위)** 은 Claude Code가 그대로 작업할 수 있는 상세 스펙입니다. 파일 위치, 함수 시그니처, 로직, 검증 항목, 주의사항이 들어 있습니다.

**Part 2 (3순위 이후)** 는 상세 스펙이 아니라 **결정 기록**입니다. Asset Master의 실제 형태가 확정되기 전에 상세 스펙을 쓰면 대부분 다시 쓰게 되므로, 대신 **무엇을 / 왜 / 어떤 함정을 피해야 하는지**를 기록합니다. 각 항목에 착수할 때 이 기록을 근거로 상세 스펙을 씁니다.

**Part 3 (보류·기각)** 은 검토했으나 지금 하지 않기로 한 것들과 그 이유입니다. 나중에 같은 제안이 다시 올라올 때 재논의를 막기 위한 기록입니다.

**Part 4** 는 순위와 무관하게 지금 병행 시작하는 작업입니다.

**Part 5** 는 개별 항목이 아니라 앞으로 모든 작업에 적용되는 전역 원칙입니다.

---

## 전체 순서

| 순위 | 작업 | 성격 |
|---|---|---|
| **0** | 회귀검증 기반 + Run 식별정보 | 안전망 |
| **1** | Data Coverage 공통계층 + 데이터 출처 메타 + CVaR 라벨 | 결과 무결성 |
| **2** | Asset Master (`core/`) + PortfolioCandidate 계약 | 이후 전부의 기반 |
| 3 | Exposure Dashboard | 현업 활용도 |
| 4 | Portfolio Mandate + **μ 의존도 표시** | 기관형 의사결정 |
| 5 | Mandate 적합성 + Feasibility 진단 | 실제 배분 |
| 6 | Transition + AUM + 유동성 + **no-trade zone** | 실제 실행 |
| 7 | Robustness 체크리스트 | 판단 신뢰 |
| 8 | Adopted Portfolio (JSON) | 반복 사용 |
| 9 | **성과귀속 (Brinson-Fachler)** | 사후 평가 |
| 10 | **블랙-리터만 2단계** | 11번의 전제조건 |
| 11 | Index Lab → Candidate / MarketView → BL | 워크플로 연결 |
| 12 | Investment Memo (템플릿 조립 한정) | 업무 산출물 |

굵게 표시한 4개는 이번에 추가 채택한 항목입니다.

**전역 원칙**: 기준안 대비 Δ (3순위부터 모든 비교 화면에 적용)
**병행 작업**: Index Lab PIT 스냅샷 적재 (지금 시작)

**Index Lab 본 작업(정기변경 강화, 방법론 비교 심화, 테마 키워드 필터)은 Portfolio Lab을 마친 뒤 별도 트랙으로 진행합니다.** PIT 스냅샷 적재만 예외적으로 지금 시작합니다 (Part 4 참조).

---

# Part 1 — 상세 스펙

## 0순위. 회귀검증 기반 + Run 식별정보

### 왜 이게 0순위인가

1순위 작업은 `build_price_frame()` 의 반환 구조를 바꾸는 일입니다. 이 함수는 **모든 화면이 사용**하고, 앱은 10,135줄 단일 파일이며, 기존 self-test 12종은 개별 계산 함수만 검증하고 **화면 단위 동작은 검증하지 않습니다.**

즉 계획의 첫걸음이 동시에 가장 위험한 걸음입니다. 변경 전후를 비교할 수단 없이 시작하면, 이후 모든 작업이 "고쳤는데 다른 게 깨졌는지 모르는" 상태로 진행됩니다.

### 0-1. 골든 스냅샷 스크립트

**파일**: `tools/golden.py` (신규)

고정 입력으로 앱의 핵심 계산 결과를 JSON으로 덤프하고, 이전 덤프와 비교하는 스크립트입니다.

```
고정 입력 세트 (변경 금지 — 변경하면 과거 골든과 비교 불가):
  tickers: ["SPY", "QQQ", "TLT", "GLD", "005930.KS"]
  weights: [30, 20, 25, 15, 10]
  start: "2018-01-01"
  end:   "2025-12-31"
  base_ccy: "KRW"
  use_dividends: True
  fx_hedge: False
  cost_bp: 10
  rebalance: "분기"
  rf_rate: 0.03
```

한국 종목을 하나 넣는 이유는 환율 처리 경로를 반드시 타게 하기 위함입니다. 이 경로는 과거에 `bfill` 사고가 있었던 곳입니다.

```python
def collect_golden(out_path: str) -> dict:
    """
    고정 입력으로 핵심 계산을 실행하고 결과를 dict로 반환 + JSON 저장.

    수집 항목:
      meta:
        run_id, git_commit, collected_at, python/pandas/numpy 버전
      data:
        price_frame_shape        # (행, 열)
        price_frame_start/end    # 실제 공통기간
        per_ticker_first_last    # {ticker: [첫날, 마지막날]}
        na_ratio                 # 결측 비율
      perf:                      # perf_row() 결과 전체 (app.py:8542)
        cagr, vol, sharpe, sortino, calmar, mdd, ulcer, martin, ...
      risk:
        risk_cvar, risk_semisd, var95, 각 위험측도 함수 결과
      turnover:
        turnover_series() 요약 (평균, 합계, 리밸런싱 횟수)
      contribution:
        growth_contribution() 종목별 기여도
      optimize:
        min_vol / max_sharpe / risk_parity / hrp 각각의 비중 벡터
      factor:
        factor_betas() 베타 및 R²

    주의:
      - float은 소수점 10자리로 반올림해서 저장 (부동소수 노이즈 제거)
      - dict/Series는 키 정렬 후 저장 (순서 차이로 인한 오탐 방지)
      - 네트워크 호출이 있으므로 가격 데이터를 별도 캐시 파일에 저장하고
        재실행 시 캐시를 우선 사용 (yfinance 소급 수정으로 인한 오탐 방지)
        캐시 경로: tools/_golden_cache.parquet
    """

def compare_golden(before_path: str, after_path: str, tol: float = 1e-9) -> list:
    """
    두 골든 JSON을 비교하고 차이 목록을 반환.

    반환: [{"key": "perf.sharpe", "before": 0.684, "after": 0.691,
            "diff": 0.007, "status": "CHANGED"}, ...]

    - 수치는 tol 이내면 동일 취급
    - meta 블록은 비교 대상에서 제외 (run_id, commit은 당연히 달라짐)
    - 키가 추가/삭제된 경우도 리포트 (ADDED / REMOVED)
    """
```

CLI 사용:
```bash
python tools/golden.py collect --out tools/golden_before.json
# ... 코드 수정 ...
python tools/golden.py collect --out tools/golden_after.json
python tools/golden.py compare tools/golden_before.json tools/golden_after.json
```

**검증**
- 코드를 전혀 바꾸지 않고 두 번 collect → compare 결과가 빈 목록이어야 함
- 캐시를 지우고 재실행 → 가격 데이터 재조회 후에도 결과가 동일해야 함 (yfinance 소급 수정이 있으면 여기서 드러남, 이건 정상 동작)
- `sortino_ratio` 의 상수 하나를 일부러 바꿔보고 → compare가 그걸 잡아내는지 확인 후 원복

**주의**
- 이 스크립트는 `app.py` 를 import 합니다. `app.py` 최상단에 Streamlit 초기화 코드가 실행되면 CLI에서 에러가 납니다. `if __name__` 가드나 지연 import로 회피하되, **`app.py` 구조를 크게 바꾸지 말 것** — 0순위에서 앱을 건드리면 골든의 의미가 없어집니다.
- 캐시 파일은 `.gitignore` 에 추가. 골든 JSON은 커밋합니다.

### 0-2. Run 식별정보 (의존성 없는 부분만)

**파일**: `app.py` — `assumptions_panel()` (app.py:836) 확장

이미 `_git_commit()` (app.py:824) 이 있고 `assumptions_panel` 이 코드 버전을 표시합니다. 여기에 다음을 추가합니다.

```
추가할 필드 (이번 단계):
  Run ID        : PF-YYYYMMDD-HHMMSS-XXX  (XXX는 3자리 랜덤 hex)
  분석 실행시각  : 로컬 타임존 포함 ISO 8601
  기준통화, 배당처리, 환헤지, 거래비용, 리밸런싱 주기, 무위험수익률
  코드 버전     : (기존)
```

Run ID는 세션 시작 시 한 번 생성해 `st.session_state["run_id"]` 에 보관하고, 설정이 바뀌면 재생성합니다.

```python
def make_run_id() -> str:
    """PF-20260811-143022-A71 형식."""

def get_or_make_run_id(settings_fingerprint: str) -> str:
    """
    settings_fingerprint: 주요 설정값을 정렬 후 해시한 문자열.
    session_state에 저장된 fingerprint와 다르면 새 Run ID 발급.
    """
```

**다음 단계로 미루는 필드**: 실제 분석기간, 종목별 데이터 최종일, 데이터 사용률. 이건 1순위 Data Coverage 작업에 딸려옵니다.

**검증**
- 같은 설정으로 화면을 여러 번 전환해도 Run ID가 유지되는지
- 거래비용을 바꾸면 Run ID가 새로 발급되는지
- Excel 내보내기에 Run ID가 포함되는지

**주의**
- Run ID는 결과값이 아니므로 **골든 스냅샷 비교 대상에서 제외**해야 합니다 (`meta` 블록에 넣습니다).
- 0-1과 0-2 중 **0-1을 먼저** 완료하고, 0-2 작업 후 골든 비교를 돌려 아무것도 안 바뀌었는지 확인합니다. 이게 이 도구의 첫 실전 사용입니다.

---

## 1순위. Data Coverage 공통계층 + 데이터 출처 메타 + CVaR 라벨

### 왜 공통계층인가

`dropna()` 로 공통기간이 잘리는 문제는 후보탐색 화면만의 문제가 아닙니다. `build_price_frame()` (app.py:1369) 에서 일어나고, 이 함수는 최적화, 상관관계, 스트레스, 최종 대안 비교 등 **모든 화면**이 사용합니다. 화면별로 구현하면 또 흩어집니다.

### 1-1. build_price_frame 반환 구조 변경

**파일**: `app.py:1369`

```python
@dataclass
class DataCoverage:
    requested_start: pd.Timestamp
    requested_end:   pd.Timestamp
    actual_start:    pd.Timestamp      # dropna 후 실제 공통기간 시작
    actual_end:      pd.Timestamp
    usage_ratio:     float             # 실제 영업일수 / 요청 영업일수
    per_ticker:      dict              # {ticker: {"first": ts, "last": ts, "n_obs": int}}
    limiting:        list              # 공통기간을 제약한 종목 (actual_start를 결정한 티커들)
    na_ratio:        float             # dropna 이전 결측 비율
    retrieved_at:    pd.Timestamp      # 데이터 조회 시각

def build_price_frame(tickers, start, end, base_ccy, use_dividends,
                      ..., return_coverage: bool = False):
    """
    기존 동작은 그대로 유지한다. return_coverage=True 일 때만
    (prices, coverage) 튜플을 반환한다.

    이 기본값 설계가 중요하다 — 기존 호출부를 한꺼번에 바꾸지 않아도
    동작하므로, 화면을 하나씩 옮기면서 골든 비교를 돌릴 수 있다.
    """
```

`limiting` 산출 로직:
```
각 티커의 첫 유효 관측일을 구한다.
actual_start와 같거나 그보다 늦은 시작일을 가진 티커가 제약 종목이다.
동률이면 전부 포함한다.
```

### 1-2. Coverage 표시 컴포넌트

**파일**: `app.py` — 신규 함수, `assumptions_panel` 근처에 배치

```python
def coverage_panel(cov: DataCoverage, *, warn_below: float = 0.7):
    """
    화면 상단에 공통 표시. 형태:

      요청 분석기간   2015-01-01 ~ 2026-08-07
      실제 분석기간   2022-04-15 ~ 2026-08-07
      사용률          38%
      제한 종목       ABC (2022-04-15 상장)

    usage_ratio < warn_below 이면 st.warning, 아니면 st.caption.
    결측률이 5%를 넘으면 별도 경고 한 줄 추가.
    """
```

적용 대상 화면(전부):
- 포트폴리오 분석
- 최적화
- 자산 상관관계
- 신규자산 추가
- 스트레스 테스트
- 거시 국면
- 최종 대안 비교

### 1-3. 최소 이력 필터 (선택 기능)

**파일**: `app.py` — 신규자산 추가 화면

```python
def filter_by_min_history(tickers: list, prices_raw: pd.DataFrame,
                          min_years: float) -> tuple:
    """
    반환: (통과 티커, 제외 로그 DataFrame)
    로그: 티커 | 이력(년) | 기준(년) | 사유
    """
```

UI: `st.selectbox("후보 최소 이력", ["제한 없음", "1년", "3년", "5년"], index=0)`

**기본값은 "제한 없음"** 입니다. Coverage 표시가 기본이고 필터는 선택입니다. 필터를 기본으로 켜면 사용자가 모르는 사이에 후보가 사라집니다 — 지금 고치려는 문제와 같은 종류의 문제입니다.

### 1-4. Run 메타에 데이터 출처 추가

0-2에서 만든 `assumptions_panel` 확장에 다음을 추가합니다.

```
데이터 조회시각  : cov.retrieved_at
실제 분석기간    : cov.actual_start ~ cov.actual_end
데이터 사용률    : cov.usage_ratio
종목별 최종일    : cov.per_ticker의 last를 표로 (expander 안에)
```

**종목별 최종일이 가장 중요합니다.** yfinance는 소급 수정이 잦아서, 이게 없으면 "지난번이랑 숫자가 왜 달라졌나"에 답할 수 없습니다.

### 1-5. CVaR 라벨 정리

**파일**: `app.py:1808` `risk_cvar()` 및 위험측도 표시부

현재 일간 CVaR에 `√252` 를 곱한 값을 "CVaR"로 표시합니다. 계산을 바꾸지 말고 **라벨과 도움말만** 수정합니다.

```
표시명: "CVaR 95% (연율 환산 근사)"
도움말: "일간 수익률 하위 5%의 평균을 √252로 연율 환산한 근사치입니다.
        수익률 분포가 정규분포가 아닐 경우 실제 연간 손실 규모를
        과소평가할 수 있습니다."
```

**Sortino는 건드리지 마십시오.** 이전 검토에서 "정의가 잘못됐다"는 지적이 있었으나, 최신 코드(app.py:1603)는 이미 전체 표본 기준 RMS 하방편차를 사용하며 `risk_semisd` 와 정의가 일치합니다. 오래된 스냅샷을 기준으로 한 지적이었습니다.

**검증 (1순위 전체)**
- 골든 비교: 1-1 ~ 1-4 작업 후 **숫자가 하나도 바뀌지 않아야 함** (반환 구조 추가와 표시 변경뿐이므로)
- 1-5 후에도 숫자 불변 (라벨만 변경)
- 신규 상장 ETF를 후보에 넣었을 때 `usage_ratio` 가 실제로 낮게 나오고 `limiting` 에 그 종목이 잡히는지
- 전 종목이 같은 기간이면 `usage_ratio` 가 1.0에 가깝고 `limiting` 이 비는지

**주의**
- `return_coverage=False` 기본값을 지키십시오. 화면을 하나씩 옮기면서 매번 골든 비교를 돌릴 수 있어야 합니다.
- 화면 7개를 한 커밋에 다 바꾸지 말고, 화면 단위로 나누어 커밋하십시오.

---

## 2순위. Asset Master + PortfolioCandidate 계약

### 왜 필요한가

지금 자산 정보가 세 곳에 흩어져 있습니다. `guess_asset_kind()` (app.py:3294) 의 티커 기반 추정, 스트레스 화면의 사용자 입력(듀레이션, 신용 민감도), 최적화 화면의 제약 설정입니다. 화면을 옮기면 다시 입력해야 합니다.

Exposure Dashboard(3순위), Mandate(4순위), Feasibility(5순위)는 **셋 다 자산별 속성이 있어야 성립**합니다. 이걸 먼저 만들지 않으면 3순위에서 "자산군 비중을 보여줘야 하는데 자산군이 어디에도 정의되어 있지 않은" 상황을 만납니다.

### 2-1. core/ 디렉터리 신설

**이것이 코드 분리의 시작점입니다.** 별도 리팩토링 프로젝트로 잡지 말고, **새로 만드는 공통 객체를 처음부터 `core/` 에 두는 방식**으로 진행합니다. 나중에 13,000줄에서 뜯어내는 것보다 압도적으로 저렴합니다.

```
core/
  __init__.py
  asset_master.py      # 2순위
  settings.py          # AnalysisSettings — 2순위에 껍데기만, 4순위에 완성
  mandate.py           # PortfolioMandate — 4순위
  contracts.py         # PortfolioCandidate, MarketView — 2순위
```

`app.py` 는 `from core.asset_master import AssetMaster` 형태로 import합니다. Streamlit 상태에 의존하는 코드는 `core/` 에 넣지 마십시오 — 순수 데이터·로직만입니다.

### 2-2. AssetMaster

**파일**: `core/asset_master.py` (신규)

```python
@dataclass
class AssetAttrs:
    ticker:        str
    name:          str = ""
    asset_class:   str = ""      # 주식 / 채권 / 대체 / 현금
    sub_class:     str = ""      # 미국 국채, 국내 대형주, 금 ...
    region:        str = ""      # 미국 / 한국 / 유럽 / 일본 / 신흥 / 글로벌
    currency:      str = ""      # USD / KRW / EUR ...
    duration:      float = None  # 채권만
    credit:        str = ""      # Sovereign / IG / HY / None
    fx_exposure:   str = ""      # 실질 환노출 통화 (환헤지 ETF는 다를 수 있음)
    beta_bench:    str = ""      # 스트레스에서 쓸 기준지수 티커
    source:        str = "auto"  # auto(추정) / user(사용자 수정)

class AssetMaster:
    """티커 → AssetAttrs 매핑을 관리한다."""

    def infer(self, ticker: str, name: str = "",
              quote_type: str = "") -> AssetAttrs:
        """
        guess_asset_kind() (app.py:3294) 를 재사용해 초기 추정값을 만든다.
        추정 못 한 필드는 빈 값으로 두고 source="auto" 로 표시한다.
        추정을 지어내지 말 것 — 모르면 비워둔다.
        """

    def upsert(self, attrs: AssetAttrs) -> None:
        """사용자 수정 반영. source를 "user"로 바꾼다."""

    def get(self, ticker: str) -> AssetAttrs: ...

    def group_weights(self, weights: dict, by: str) -> dict:
        """
        by: "asset_class" | "region" | "currency" | "sub_class"
        반환: {그룹명: 합산 비중}
        속성이 비어 있는 자산은 "미분류"로 묶고, 호출부에서 경고할 수 있도록
        미분류 비중을 함께 반환한다.
        """

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "AssetMaster": ...
```

**UI** (`app.py` — 공통 설정 영역 또는 별도 expander):
- `st.data_editor` 로 티커별 속성 표 편집
- `source="auto"` 인 셀은 배경색으로 구분 (추정값임을 표시)
- 사용자가 수정하면 `source="user"` 로 전환
- session_state에 보관 + JSON 내보내기/불러오기

### 2-3. PortfolioCandidate / MarketView 계약

**파일**: `core/contracts.py` (신규)

```python
@dataclass
class PortfolioCandidate:
    """Index Lab, 최적화, 수동 구성 등 모든 포트폴리오 안의 공통 형식."""
    name:           str
    source:         str            # "index-lab" / "optimizer" / "manual" / "bl"
    holdings:       dict           # {ticker: weight(0~1)}
    asset_metadata: dict           # AssetMaster.to_dict() 부분집합
    created_at:     str            # ISO 8601
    notes:          str = ""

    def validate(self) -> list:
        """비중 합계 1.0 확인(±1e-6), 음수 비중 확인, 티커 중복 확인.
           문제 목록을 반환. 빈 목록이면 정상."""

@dataclass
class MarketView:
    """시장동향 화면 → BL 전달용."""
    factor:     str     # "US10Y" / "USD" / "Growth" / "Inflation" ...
    direction:  str     # "up" / "down" / "neutral"
    magnitude:  float   # 표준화 단위 또는 %p — 단위를 factor별로 문서화할 것
    horizon:    str     # "3M" / "6M" / "12M"
    confidence: float   # 0~1
```

**MarketView는 지금 정의합니다** — 소비자(BL 화면)가 이미 존재하고 필드가 작고 확정적입니다.
**PortfolioCandidate는 Asset Master와 함께 정의합니다** — `asset_metadata` 가 곧 Asset Master이므로 그 전에 확정하면 다시 맞춰야 합니다.

**검증 (2순위)**
- `core/` 를 import한 뒤 골든 비교: **숫자 불변** (아직 계산에 쓰지 않으므로)
- `AssetMaster.infer()` 가 SPY / TLT / GLD / 005930.KS 를 각각 합리적으로 분류하는지
- 모르는 티커에 대해 **빈 값을 반환하고 지어내지 않는지**
- `group_weights()` 미분류 비중이 정확히 집계되는지
- `PortfolioCandidate.validate()` 가 비중 합 0.99, 음수 비중, 중복 티커를 각각 잡는지
- `to_dict` → `from_dict` 왕복 후 동일한지

**주의**
- `guess_asset_kind()` 를 재사용하되 **수정하지 마십시오.** 기존 호출부가 있습니다. 새 로직이 필요하면 `core/` 안에서 감싸십시오.
- Asset Master를 만들었다고 기존 화면을 바로 옮기지 마십시오. 2순위는 **객체를 만들고 편집 UI를 붙이는 것까지**입니다. 실제 사용은 3순위부터입니다.
- `core/` 에 `import streamlit` 을 넣지 마십시오. 테스트 가능성이 사라집니다.

---

# Part 2 — 결정 기록 (3순위 이후)

각 항목에 착수할 때, 아래 기록을 근거로 상세 스펙을 작성합니다.
**"왜"와 "함정"이 핵심입니다. 이게 안 적히면 나중에 같은 논쟁을 반복합니다.**

---

## 3순위. Exposure Dashboard

**무엇** — 현재 포트폴리오의 성격을 한 화면에 요약. 자산군/지역/통화 비중, Equity Beta, Duration, 위험기여 상위 자산, 거시 민감도.

**왜** — 새 계산이 거의 없이 기존 계산을 조립하는 것뿐인데 사용 경험이 바뀝니다. 값/비용 비율이 이 계획 전체에서 가장 좋습니다. 현재는 "내 포트폴리오가 대체 어떤 성격인가"를 알려면 여러 화면을 돌아야 합니다.

**전제** — Asset Master(2순위). 자산군·지역·통화 없이는 성립하지 않습니다.

**함정**
- Duration, Credit, 실질 FX 노출은 **자동으로 안 나옵니다.** 현재 이 값들은 스트레스 화면에서 사용자가 자산별로 입력합니다. Asset Master에 넣되, **비어 있으면 그 항목을 표시하지 말고 "미입력"으로 두십시오.** 0으로 채우면 채권 듀레이션 0인 포트폴리오처럼 보입니다.
- 미분류 자산 비중이 10%를 넘으면 화면 상단에 경고. 분류가 엉성한 상태의 요약은 오도합니다.
- "핵심 취약성" 같은 서술형 문구를 넣을 때, **규칙 기반으로만** 생성하십시오 (예: 특정 그룹 비중 > 임계값 → 해당 문구). 자유 서술은 금지입니다.

**재사용 대상** — `growth_contribution()` (app.py:1770), `factor_betas()` (app.py:4575), 기존 위험기여 계산, `AssetMaster.group_weights()`

---

## 4순위. Portfolio Mandate + μ 의존도 표시

### 4-A. Portfolio Mandate

**무엇** — 투자 목적과 제약을 정의하는 객체 및 화면. 목표 변동성, MDD 허용, 자산군/지역/통화 한도, 단일자산 한도, 회전율 한도.

**왜** — "Sharpe 0.8이니 좋다"가 "목표 변동성 8% 이하를 충족하면서 예상수익이 가장 높은 안"으로 바뀝니다. 개인 도구와 기관 도구의 차이입니다.

**중요한 구조 결정 — AnalysisSettings와 분리합니다.**

```
core/settings.py   AnalysisSettings  — 계산 방법에 관한 것
                   분석기간, 배당 처리, 환헤지, 거래비용, 결측처리, 무위험금리

core/mandate.py    PortfolioMandate  — 투자 목적과 제약에 관한 것
                   투자목적, 기준통화, 목표 변동성, MDD 허용,
                   자산군/지역/통화 한도, 단일자산 한도, 회전율 한도
```

**경계가 모호한 필드의 판별 기준**: *"이걸 바꾸면 정답이 달라지는가, 계산 방식만 달라지는가."*
정답이 달라지면 Mandate, 계산만 달라지면 Settings입니다.
이 기준으로 **`base_ccy` 는 Mandate**에 둡니다 (원화 펀드냐 달러 펀드냐는 만데이트 사항). AnalysisSettings가 Mandate의 `base_ccy` 를 참조하는 구조입니다.

**함정**
- **측정기준 라벨을 반드시 붙이십시오.** "목표 변동성 ≤10% / 현재 11.8% ⚠️" 에서 11.8%는 **과거 실현 변동성**이고 목표는 **미래 기대치**입니다. 성격이 다른 둘을 직접 비교해 판정을 찍으면 없는 정밀도를 만드는 것입니다. 판정 옆에 "최근 3년 실현 기준"처럼 항상 표기하십시오.
- 비중 제약(주식 57%)은 현재 상태이므로 이 문제가 없습니다. **변동성·MDD 같은 통계량만** 해당됩니다.
- Mandate를 만들면서 기존 전역 설정을 한꺼번에 정리하려 하지 마십시오. Mandate에 해당하는 필드만 옮기고, 나머지는 그대로 둡니다.

### 4-B. μ 의존도 표시 (채택 항목 A)

**무엇** — 최적화 목표별로 과거 평균수익률 의존도를 표시하고, 의존도 높은 목표 선택 시 경고 한 줄.

**왜** — MVO 비중 불안정의 주범은 공분산이 아니라 기대수익(μ)입니다. 그런데 이 앱은 μ를 안 쓰는 대안(Min Vol, Risk Parity, HRP)과 μ를 뷰로 대체하는 대안(BL)을 **이미 전부 갖고 있습니다.** 새 추정 모델을 만들 게 아니라, 사용자가 목표를 고를 때 이 차이를 알게 하는 것으로 충분합니다. 공분산 shrinkage를 넣는 것보다 훨씬 저렴하고 효과가 큽니다.

**형태**
```
최적화 목표 선택 옆에 의존도 표시:
  Max Sharpe       과거 평균수익률 의존: 높음
  Max Return       과거 평균수익률 의존: 매우 높음
  Min Vol          의존 없음
  Risk Parity      의존 없음
  HRP              의존 없음
  Black-Litterman  사용자 뷰 기반

의존도 높음/매우 높음 선택 시:
  "과거 평균수익률을 미래 기대수익률의 대용치로 사용합니다.
   추정오차에 민감하므로 Min Vol / Risk Parity / BL 결과와
   함께 비교하는 것을 권장합니다."
```

**작업량** — 반나절. 계산 변경 없음, 표시만.

**함께 정리할 것** — 최적화에서 쓰는 기대수익은 CAGR이 아니라 **연율 산술평균**입니다. 라벨을 "연율 기대수익(산술평균)"으로 명확히 하십시오.

---

## 5순위. Mandate 적합성 + Feasibility 진단

**무엇** — 현재/제안 포트폴리오가 Mandate를 충족하는지 판정하고, 최적화에 제약을 걸었을 때 해가 없으면 **어느 제약이 충돌하는지 진단**.

**왜** — 제약을 10개 넣는 것 자체는 어렵지 않습니다. 진짜 문제는 동시에 걸면 해가 없어진다는 것이고, 이건 이 프로젝트에서 이미 겪은 일입니다(제약 7개 동시 적용 시 해 없음). "최적화 실패"로 끝내면 사용자는 제약을 하나씩 껐다 켜며 헤맵니다.

**핵심은 제약 추가가 아니라 진단입니다.**
```
원하는 출력:
  현재 제약으로 가능한 포트폴리오가 없습니다.

  가장 큰 충돌:
    미국 비중 ≤40% 와 USD 노출 ≤30% 를 동시에 충족하기 어렵습니다.

  USD 한도를 36% 이상으로 완화하면 해가 생성됩니다.
```

**전제** — Asset Master(2), Mandate(4). 자산군·지역·통화 분류 없이는 해당 제약을 걸 수 없습니다.

**재사용 대상** — `feasibility()` (app.py:2161) 확장

**함정**
- 완화 제안값("36% 이상")을 계산하려면 제약을 하나씩 이완시키며 재탐색해야 합니다. 제약 수만큼 solver를 돌리므로 느려집니다. 진단은 **해가 없을 때만** 실행하십시오.
- solver 실패 시 동일가중으로 fallback하면 안 됩니다. 최신 코드는 `strict=True` 로 `None` 을 반환하도록 이미 수정되어 있습니다. 이 동작을 유지하십시오.

---

## 6순위. Transition Analysis + AUM + 유동성 + no-trade zone

### 6-A. Transition Analysis

**무엇** — 현재안에서 제안안으로 실제로 바꿀 때 무엇을 얼마나 사고팔아야 하는지.

**왜** — 운용사에서 배분안을 올릴 때 위원회가 가장 먼저 묻는 질문입니다. 지금 이 도구는 "어떤 안이 좋은가"까지만 답하고 "그 안으로 어떻게 가는가"에는 답하지 못합니다.

**결정적 설계 원칙 — 화면을 물리적으로 두 영역으로 나눕니다.**

```
[영역 1] Historical Validation
  현재안과 제안안을 과거 동일 조건으로 각각 백테스트.
  CAGR / Vol / Sharpe / MDD / 위기구간 / OOS

[영역 2] Implementation Today
  오늘 기준 전환 분석.
  종목별 매도/매수 비중, AUM 기준 거래금액, 총 교체비중,
  명시적 거래비용, ADV 대비 주문규모, 유동성 경고,
  이후 예상 리밸런싱 회전율
```

**두 영역의 숫자를 절대 합산하지 마십시오.** 영역 1은 과거 백테스트 기반이고 영역 2는 현재 시점 계산입니다. 합쳐서 "제안안의 순효과"처럼 만들면 오도합니다. 초기 검토안에서 실제로 이 실수가 있었습니다.

### 6-B. AUM 입력과 유동성

**AUM 입력이 필수입니다.** 비중만으로는 거래금액이 나오지 않습니다.

```
Portfolio AUM: 500억원  ← 사용자 입력

종목      매매금액   20D ADV    ADV 대비   판단
SPY        30억      매우 큼     0.01%     여유
ABC ETF    20억      35억        57%       ⚠️ 실행 부담
```

**함정 — ADV 비율은 시장충격비용이 아닙니다.**
"ADV 대비 35%" 에서 "시장충격비용 42bp" 를 도출하지 마십시오. 무료 가격·거래량 데이터로는 근거가 없습니다. **비용 모델이 아니라 실행가능성 경고**로 표현합니다. 이건 이 프로젝트의 기존 원칙(없는 정밀도를 만들지 않기)과 같은 계열입니다.

### 6-C. no-trade zone (채택 항목 C)

**무엇** — 최적 비중과 현재 비중의 차이가 임계값 미만이면 거래하지 않는 밴드.

**왜** — 실제 운용 비용을 크게 줄입니다. 리밸런싱 때마다 전 종목을 소폭씩 조정하는 것은 비용만 발생시키고 효과가 거의 없습니다.

**형태**
```python
def apply_no_trade_zone(current_w, target_w, threshold_bp: float = 100):
    """
    |target - current| < threshold 인 자산은 current 유지.
    나머지만 target으로 변경 후 합계 100% 정규화.
    threshold_bp=0 이면 기존 동작(항상 리밸런싱)과 동일해야 한다.
    """
```

`walk_forward()` (app.py:2090) 의 각 리밸런싱 시점에 적용하고, `prev_w` 를 `current_w` 로 넘깁니다.

**출력** — 적용 전/후 평균 회전율 비교, 건너뛴 리밸런싱 횟수

**함정**
- 정규화 과정에서 no-trade 자산의 비중도 미세하게 변합니다. 이건 허용합니다(실무도 동일).
- 임계값이 너무 높으면 사실상 리밸런싱을 안 하게 됩니다. "최근 4회 중 3회 건너뜀 — 임계값이 높을 수 있습니다" 경고를 넣으십시오.

**재사용 대상** — `turnover_series()` (app.py:1466) 는 이미 `prev_w` 를 받아 실제 교체량을 계산할 수 있는 구조입니다.

---

## 7순위. Robustness 체크리스트

**무엇** — 분석 결과를 얼마나 믿을 수 있는지 체크 항목으로 표시.

```
Data
  공통기간 9.2년 ✅
  요청기간 사용률 91% ✅

Optimization
  학습기간 변경 시 최대 비중변화 12%p ✅
  OOS Sharpe +0.08 ✅
  Walk-forward 5회 중 3회 개선 ⚠️

Implementation
  비용 5→20bp에서도 순위 유지 ✅
  최대 ADV 사용 42% ⚠️

주요 주의점: ABC ETF의 데이터 이력이 짧고 실행 유동성이 낮습니다.
```

**절대 하지 말 것 — 종합 점수나 등급을 만들지 마십시오.**
"종합 82점", "신뢰도: 높음" 같은 압축은 두 가지 문제가 있습니다. 서로 다른 성격의 체크를 합산할 때 가중치에 근거가 없고, 등급이 뜨면 사용자가 개별 항목을 안 봅니다. 이 프로젝트의 원칙(없는 정밀도를 만들지 않기)에 정면으로 어긋납니다.

**체크리스트를 그대로 보여주고, 빨간불 항목만 강조합니다.** 구현도 훨씬 간단해집니다.

**전제** — 1순위(Data Coverage), 6순위(ADV) 의 계산 결과를 가져다 씁니다. 새 계산은 거의 없습니다.

---

## 8순위. Adopted Portfolio

**무엇** — 최종 대안 비교에서 "채택" 버튼 → 목표 비중과 채택 시점 저장 → 이후 drift 확인, 목표 대비 괴리, 리밸런싱 필요 여부.

**왜** — 일회성 분석기에서 지속 사용 프로그램으로 바뀝니다.

**결정 — 저장 방식은 JSON 내보내기/불러오기로 시작합니다.**

현재 Streamlit Cloud 배포는 세션 상태 + JSON만 쓰기로 되어 있고, 이건 여러 명이 같은 인스턴스를 쓸 때 데이터가 섞이는 문제를 겪고 내린 결정입니다. `ENABLE_LOCAL_PERSISTENCE` (app.py:41) 가 이미 있으므로 로컬 실행에서는 파일 저장이 가능합니다.

**단계적으로 갑니다.** ① JSON 내보내기/불러오기 → ② 실제로 자주 쓰이는 것이 확인되면 로컬 SQLite 도입.

**함정**
- "한 달 뒤 앱을 켜면 현재 상태가 나온다"는 **JSON 방식에서는 안 됩니다.** 사용자가 JSON을 다시 올려야 합니다. 화면 문구를 이 동작에 맞게 쓰십시오.
- 저장할 것은 목표 비중, 채택일, 채택 당시 Mandate, 채택 근거 메모입니다. 가격 시계열은 저장하지 마십시오(용량).

**재사용 대상** — `snapshot()` (app.py:1001)

---

## 9순위. 성과귀속 Brinson-Fachler (채택 항목 D)

**무엇** — 벤치마크 대비 초과수익을 배분효과와 선택효과로 분해.

**왜** — "이번 분기에 주식을 많이 담은 게 좋았나, 종목을 잘 골랐나"에 정량적으로 답합니다. **사후 평가라는 새 업무단계**를 프로그램 안으로 가져옵니다. 지금 이 도구에 그 단계가 없습니다.

**공식**
```
배분효과   = (wp - wb) × (Rb_group - Rb_total)
선택효과   = wb × (Rp_group - Rb_group)
상호작용   = (wp - wb) × (Rp_group - Rb_group)
세 효과의 합 = 초과수익 (정확히 일치해야 함 — 이게 핵심 검증)
```

**전제** — Asset Master(2순위)의 자산군 분류, Adopted Portfolio(8순위)의 채택 시점. 8순위 뒤에 두는 이유는 "언제부터의 성과인가"의 기준점이 채택일이기 때문입니다.

**함정**
- 자산군이 1개면 배분효과가 무의미합니다. "자산군 2개 이상 필요" 메시지.
- 벤치마크가 없으면 기능 비활성화.
- 일별 Brinson은 계산은 되지만 해석이 어렵습니다. 월별 이상을 기본으로 하십시오.

**재사용 대상** — `growth_contribution()` (app.py:1770) 이 종목별 기여도를 계산하므로, 자산군 레벨로 묶어 벤치마크 대비 분해합니다.

---

## 10순위. 블랙-리터만 2단계 (채택 항목 B)

**무엇** — 뷰를 반영한 실제 비중 조정안 출력. 현재 BL은 뷰 입력과 영향 분석(1단계)까지만 되어 있고 비중 조정이 미완성입니다.

**왜 — 11순위의 전제조건입니다.** 11순위에서 MarketView를 BL로 전달하는데, 받는 쪽이 끝을 못 내면 연결의 의미가 없습니다. 이 의존성 때문에 11순위 바로 앞에 둡니다.

**로직**
```
1. 틸트 대상 필터링
   factor_betas() (app.py:4575) 로 각 자산의 R² 산출.
   R² < 0.10 인 자산은 틸트 대상에서 제외 — 현재 비중 유지.
   화면에 "R² 0.10 미만으로 틸트 제외" 표시.

2. 틸트 강도 = √R² 가중 (1단계 영향분석과 동일 적용)

3. 제약 — 최대변경폭 하나만
   st.slider("종목당 최대 비중 변경폭(%p)", 1, 20, 5)
   ±max_delta 로 clip 후 합계 100% 비례 조정.
   다른 제약은 이 단계에서 넣지 말 것.

4. 적용 범위
   모드 A: 영향 분석만 (현재 상태)
   모드 B: 보유자산 내 조정
   모드 C(후보자산 포함)는 나중 과제.
```

**출력** — 자산 | 현재비중 | 조정비중 | 변경폭 | R² | 틸트근거, 그리고 조정 전후 성과 비교

**함정**
- **제약을 여러 개 걸면 해가 없어집니다.** 이 단계에서는 최대변경폭 하나만입니다.
- 전 종목 R² < 0.10 이면 "조정 대상 없음" 메시지를 표시하고 끝냅니다. 억지로 조정하지 마십시오. 한국 개별종목은 R²가 0.05~0.07 수준으로 실측된 바 있습니다.

**재사용 대상** — `black_litterman()` (app.py:4689), `bl_weights()` (app.py:4711)

---

## 11순위. Index Lab → Candidate / MarketView → BL

**무엇** — Index Lab에서 설계한 인덱스를 `PortfolioCandidate` 로 내보내 Portfolio Lab에서 신규 자산처럼 평가. 시장동향 화면의 판단을 `MarketView` 로 BL에 전달.

**왜** — 두 도구를 각각 95점으로 만드는 것보다 연결 하나가 업무가치를 더 올립니다. 상품개발 관점에서 "새 ETF를 만들면 기존 라인업에 어떤 영향인가"를 한 흐름으로 볼 수 있습니다.

**전제** — 2순위(계약 정의), 10순위(BL 2단계 완성)

**구현 부담이 작습니다** — 계약이 2순위에서 이미 정의되어 있으므로 변환 함수와 버튼만 붙이면 됩니다.

**함정**
- 한국 종목은 `.KS` / `.KQ` 접미사가 붙어 있어야 yfinance 조회가 됩니다. Index Lab에서 정규화된 티커를 그대로 넘기십시오.
- 비중 합계 검증(`PortfolioCandidate.validate()`) 후 내보내기.
- Index Lab의 백테스트는 탐색형(생존편향 있음)입니다. Portfolio Lab으로 넘어온 뒤에도 이 사실이 표시되어야 합니다.

---

## 12순위. Investment Memo

**무엇** — 분석 결과를 업무 문서용 서술로 정리.

**왜** — 사용자가 업무에서 원하는 최종 산출물은 결국 보고서입니다. "Sharpe 0.83 / MDD -16.2%"에서 끝나는 것과, 검토 의견 문단까지 나오는 것은 업무효율이 다릅니다.

**결정적 제약 — 규칙 기반 템플릿 조립만 허용합니다. 자유 생성 금지.**

```
허용:  숫자 → 조건 판정 → 사전 정의된 문장 조각 조립
       "변동성 개선 + MDD 개선 + 교체비중 28% + ADV 경고"
       → 해당 조합에 대응하는 템플릿 문장 출력

금지:  LLM이 숫자를 보고 자유롭게 서술
```

자유 생성은 같은 입력에 다른 문장이 나오고 근거 없는 해석이 섞입니다. 이 프로젝트의 결정론적 원칙과 정면으로 충돌하며, 그 문서가 실명으로 보고서에 들어간다면 위험합니다.

**전제** — 앞의 항목들이 대부분 있어야 쓸 재료가 생깁니다. 마지막에 두는 이유입니다.

---

# Part 3 — 보류·기각 기록

나중에 같은 제안이 올라올 때 재논의를 막기 위한 기록입니다.

| 항목 | 판정 | 이유 |
|---|---|---|
| 환헤지 비용 시점별 반영 | 보류 | FRED 한국 금리 시리즈가 불안정해 사용자 입력 fallback 필수. 4순위 AnalysisSettings 정리와 겹치므로 그때 재검토. |
| 시나리오 이력 (버린 안 포함) | 보류 | 8순위 Adopted Portfolio가 채택한 안을 저장하므로 대부분 커버. 실제로 버린 안까지 필요해지면 그때. |
| 롤링 팩터 익스포저 | 보류 | 3순위 Exposure Dashboard가 현재 시점 노출을 커버. 시간 변화 추적이 실제로 필요해지면 그때. |
| Proxy 지정 (짧은 이력 보완) | 보류 | Data Coverage(1순위)를 먼저 넣고 실제 불편이 반복되는지 확인. 도입 시 Proxy 구간과 실제 구간을 반드시 구분 표시하고, 전환비용·회전율 분석에서는 제외할 것. yfinance로 국내 ETF 기초지수는 거의 조회 불가. |
| 공분산 Shrinkage | 보류 | MVO 불안정의 주범은 공분산이 아니라 μ. 4-B(μ 의존도 표시)가 훨씬 저렴하고 효과가 큼. 그 뒤에도 필요하면 도입. |
| 시장별/글로벌 Stress transmission | 보류 | 글로벌 충격 → 시장별 충격 → 개별자산의 2단계 구조가 필요해 작업량이 큼. 당분간 현재 방식에 **"단일 시장 proxy 기반 근사"** 라고 명시할 것. |
| Robustness 종합점수 | **기각** | 서로 다른 성격의 체크를 합산할 때 가중치에 근거가 없고, 등급이 뜨면 개별 항목을 안 봄. 체크리스트만 유지. |
| Investment Memo 자유 생성 | **기각** | 결정론적 원칙과 충돌. 템플릿 조립만 허용. |
| ADV 기반 시장충격비용(bp) 산출 | **기각** | 무료 데이터로는 근거 없음. 실행가능성 경고로만 표현. |
| 위험지표 추가, 최적화 알고리즘 추가 | **기각** | 새로운 업무단계를 가져오지 않음. Part 5의 판단 기준 참조. |
| Monte Carlo / GARCH / Copula | **기각** | 동일. |
| ECOS 자유검색, 한국 경기×물가 국면 | **기각** | Portfolio Lab의 정체성은 데이터 탐색기가 아니라 의사결정 도구. |
| 다중 사용자 협업 | **기각** | 현 단계에서 불필요. |
| Index Lab 룰북 자동생성(docx) | 보류 | Index Lab 트랙에서 나중 과제로. |

---

# Part 4 — 병행 작업: Index Lab PIT 스냅샷 적재

**Index Lab 본 작업은 Portfolio Lab 완료 후로 미뤘지만, 이 항목만 지금 시작합니다.**

**왜 지금인가** — PIT 스냅샷은 **시작해야만 쌓이는** 데이터입니다. 지금 시작하면 6개월 뒤 6개월치 실제 PIT 백테스트가 가능하고, 6개월 뒤에 시작하면 그때부터 다시 0에서 시작합니다. Portfolio Lab 작업 기간이 통째로 날아갑니다.

**화면 개발이 필요 없습니다.** 스크립트 하나로 충분합니다.

**파일**: `index-lab/core/pit.py`, `index-lab/tools/snapshot.py` (신규)

```
저장 항목 (날짜별):
  유니버스 (그 시점 조회 가능한 종목 목록)
  종목 기본정보 (시총, 상장일, 업종)
  재무 데이터 (DART rcept_dt 포함 — 발표 시점 필터링용)
  팩터 입력값
  선정 결과

저장 형식: JSON 또는 Parquet, 날짜별 파일
경로: index-lab/data/snapshots/YYYY-MM-DD/

한국: DART API — corpCode 전체 목록 + 상장일/상폐일, 재무제표 rcept_dt
미국: yfinance — ETF 구성종목(SPY/IVV 등)을 유니버스 프록시로 사용,
      가격이 끊긴 종목을 상폐/합병으로 감지
```

**PIT 등급 표시** — 이후 Index Lab 백테스트 화면에 3등급 배지를 붙입니다.
```
🔴 탐색형     현재 유니버스 고정 (생존편향 있음)
🟡 부분 PIT   상폐 종목 반영, 재무 발표 지연 반영
🟢 스냅샷 기반 실제 시점 데이터
```
현재는 항상 🔴. 스냅샷이 쌓인 구간부터 자동으로 🟢으로 전환됩니다.

**지금 할 것은 적재 스크립트와 저장 구조뿐입니다.** 등급 표시와 백테스트 연동은 Index Lab 트랙에서 합니다.

---

# Part 5 — 전역 원칙

개별 항목이 아니라, 앞으로 모든 작업에 적용되는 규칙입니다.

## 5-1. 기준안 대비 Δ

현재 포트폴리오가 있으면 **어디서든** 절대값과 함께 변화량을 보여줍니다. 별도 프로젝트가 아니라 UX 원칙입니다. 3순위부터 새 화면을 만들 때마다 자동으로 따라와야 합니다.

```
Sharpe  0.72 → 0.84  (+0.12)
MDD    -21.4 → -17.8 (+3.6%p 개선)
Vol     13.2 → 12.1  (-1.1%p)
```

## 5-2. 새 공통 객체는 core/ 에

코드 분리를 별도 프로젝트로 잡지 않습니다. **새로 만드는 공통 객체(순수 데이터·로직)는 처음부터 `core/` 에 둡니다.** `core/` 안에는 `import streamlit` 을 넣지 않습니다. 기존 코드를 뜯어내는 작업은 하지 않습니다.

## 5-3. 측정기준 라벨

과거 실현치와 미래 목표치를 비교해 판정을 찍을 때는 측정 기준을 항상 표기합니다. ("최근 3년 실현 기준")

## 5-4. 골든 비교를 매 작업마다

각 순위 작업이 끝날 때마다 골든 비교를 돌립니다. **숫자가 바뀌면 안 되는 작업에서 바뀌었다면 그 자리에서 원인을 찾습니다.** 다음 단계로 넘어가지 마십시오.

숫자가 바뀌어야 정상인 작업(계산 로직 변경)에서는, 바뀐 항목이 **예상한 것뿐인지** 확인하고 골든을 갱신합니다.

## 5-5. 기능 추가 판단 기준

> **"이 기능이 새로운 현업 업무단계를 프로그램 안으로 가져오는가?"**

가져오면 추가할 가치가 있습니다.

| 기능 | 가져오는 업무단계 |
|---|---|
| Exposure Dashboard | 포트폴리오를 이해하는 업무 |
| Mandate | 운용목표를 정의하는 업무 |
| Feasibility | 현실적인 배분안을 만드는 업무 |
| Transition | 실행계획을 만드는 업무 |
| Adopted Portfolio | 결정을 기록하고 관리하는 업무 |
| 성과귀속 | 사후 평가하는 업무 |
| Index/Market 연결 | 아이디어를 검증으로 전달하는 업무 |
| Investment Memo | 결론을 업무문서로 만드는 업무 |

반대로 위험지표 하나 더, 최적화 알고리즘 하나 더는 새로운 업무단계를 만들지 않으므로 추가하지 않습니다.

## 5-6. 없는 정밀도를 만들지 않기

기존 원칙의 재확인입니다. 이 계획에서 특히 적용되는 곳:
- ADV 비율에서 시장충격비용(bp)을 도출하지 않음
- Robustness 종합점수를 만들지 않음
- Investment Memo를 자유 생성하지 않음
- Asset Master에서 모르는 속성을 추정으로 채우지 않음
- R² 0.10 미만 자산을 BL 틸트 대상에 넣지 않음
