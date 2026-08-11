# Portfolio Lab 개선 계획

작성일: 2026-08-11 (v2)
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

- 0순위(회귀검증 기반)부터 순서대로 진행한다. 순서를 건너뛰지 않는다.
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

`build_price_frame`, `feasibility`, `factor_betas`, `black_litterman`, `bl_weights`, `walk_forward`, `turnover_series`, `growth_contribution`, `perf_row`, `risk_cvar`, `risk_semisd`, `guess_asset_kind`, `assumptions_panel`, `snapshot`, `_git_commit` 같은 함수명은 바뀌지 않습니다. 이 문서에서 줄 번호와 함수명이 함께 표기된 곳은 **함수명이 정본**입니다.

---

## 이 문서의 사용법

**Part 1 (0~2순위)** 은 Claude Code가 그대로 작업할 수 있는 상세 스펙입니다. 파일 위치, 함수 시그니처, 로직, 검증 항목, 주의사항이 들어 있습니다.

**Part 2 (3순위 이후)** 는 상세 스펙이 아니라 **결정 기록**입니다. Asset Master의 실제 형태가 확정되기 전에 상세 스펙을 쓰면 대부분 다시 쓰게 되므로, 대신 **무엇을 / 왜 / 어떤 함정을 피해야 하는지**를 기록합니다. 각 항목에 착수할 때 이 기록을 근거로 상세 스펙을 씁니다.

**Part 3 (보류·기각)** 은 검토했으나 지금 하지 않기로 한 것들과 그 이유입니다. 나중에 같은 제안이 다시 올라올 때 재논의를 막기 위한 기록입니다.

**Part 4** 는 Index Lab 트랙입니다. Portfolio Lab을 마친 뒤 착수합니다.

**Part 5** 는 개별 항목이 아니라 앞으로 모든 작업에 적용되는 전역 원칙입니다.

---

## 전체 순서

| 순위 | 작업 | 성격 |
|---|---|---|
| **0** | 회귀검증 기반 + Run/Input 식별정보 | 안전망 |
| **1** | Data Coverage 공통계층 + 데이터 출처 메타 + CVaR 라벨 | 결과 무결성 |
| **2** | Asset Master (`core/`) + 계약 정의 | 이후 전부의 기반 |
| 3 | Exposure Dashboard | 현업 활용도 |
| 4 | Portfolio Mandate + μ 의존도 표시 | 기관형 의사결정 |
| 5 | Mandate 적합성 + Feasibility 진단 | 실제 배분 |
| 6 | Transition + AUM + 유동성 + no-trade zone | 실제 실행 |
| 7 | Robustness 체크리스트 | 판단 신뢰 |
| 8 | Adopted Portfolio (JSON) | 반복 사용 |
| 9 | 블랙-리터만 2단계 | 10번의 전제조건 |
| 10 | Index Lab → Candidate / MarketView → BL | 워크플로 연결 |
| 11 | Policy Benchmark | 12번의 전제조건 |
| 12 | 성과귀속 (Brinson-Fachler) | 사후 평가 |
| 13 | Investment Memo (템플릿 조립 한정) | 업무 산출물 |

**전역 원칙**: 기준안 대비 Δ (3순위부터 모든 비교 화면에 적용)

**Index Lab 작업은 Portfolio Lab을 마친 뒤 별도 트랙으로 진행합니다** (Part 4).

---

# Part 1 — 상세 스펙

## 0순위. 회귀검증 기반 + Run/Input 식별정보

### 왜 이게 0순위인가

1순위 작업은 `build_price_frame()` 의 반환 구조를 바꾸는 일입니다. 이 함수는 **모든 화면이 사용**하고, 앱은 10,135줄 단일 파일이며, 기존 self-test 12종은 개별 계산 함수만 검증하고 **화면 단위 동작은 검증하지 않습니다.**

즉 계획의 첫걸음이 동시에 가장 위험한 걸음입니다. 변경 전후를 비교할 수단 없이 시작하면, 이후 모든 작업이 "고쳤는데 다른 게 깨졌는지 모르는" 상태로 진행됩니다.

### 0-1. 테스트를 두 종류로 분리한다

**이 분리가 핵심입니다.** 하나로 합치면 회귀검증이 재현 불가능해집니다.

```
A. Deterministic Regression Test  (회귀검증)
   저장소에 커밋된 고정 가격 데이터 사용
   → 코드 수정 전후 계산 결과가 동일한지 확인
   → 네트워크 호출 없음, 항상 재현 가능

B. Live Data Smoke Test  (연결 확인)
   Yahoo Finance 실제 호출
   → 데이터가 정상적으로 내려오는지만 확인
   → Golden 숫자와 일치할 필요 없음
```

**왜 분리하는가** — 가격 데이터를 저장소 밖(캐시)에 두고 결과 JSON만 커밋하면, 몇 달 뒤 다른 환경에서 골든을 재생산할 수 없습니다. yfinance는 소급 수정이 잦아서 같은 입력을 다시 얻을 수 없기 때문입니다. **회귀검증 도구가 재현 불가능하면 도구의 의미가 없습니다.**

### 0-2. 고정 fixture 생성

**파일**: `tests/fixtures/` (신규)

```
tests/fixtures/golden_prices_raw.parquet   # 종목별 원시 가격 (각 현지통화)
tests/fixtures/golden_fx_raw.parquet       # USDKRW 원시 환율
tests/fixtures/README.md                   # 생성 시점, 출처, 갱신 금지 명시
```

**중요 — 원시 데이터를 얼립니다. 환율 변환이 끝난 최종 프레임을 얼리지 마십시오.**

최종 프레임을 얼리면 `build_price_frame()` 의 **FX 변환 로직 자체가 검증 대상에서 빠집니다.** 과거에 환율 `bfill` 사고가 났던 바로 그 경로입니다. 원시 가격 + 원시 환율을 따로 얼려야 골든 테스트가 FX 변환 → dropna → 수익률 계산 경로를 전부 통과합니다.

**고정 입력 세트 (변경 금지 — 변경하면 과거 골든과 비교 불가)**
```
tickers:   ["SPY", "QQQ", "TLT", "GLD", "005930.KS"]
weights:   [30, 20, 25, 15, 10]
start:     "2018-01-01"
end:       "2025-12-31"
base_ccy:  "KRW"
use_dividends: True
fx_hedge:  False
cost_bp:   10
rebalance: "분기"
rf_rate:   0.03
```

기준통화가 KRW이므로 **SPY / QQQ / TLT / GLD가 FX 변환 경로를 탑니다.** `005930.KS` 는 원화 자산과 외화 자산이 동시에 존재하는 혼합 포트폴리오와 **국가별 거래일 차이(휴장일 불일치)** 를 검증하는 역할입니다.

fixture 생성 스크립트는 1회성입니다. 생성 후 parquet을 커밋하고, 스크립트는 `tools/make_fixture.py` 로 남겨두되 **재실행하지 마십시오.**

### 0-3. 골든 수집·비교

**파일**: `tools/golden.py` (신규)

```python
def collect_golden(out_path: str, fixture_dir: str) -> dict:
    """
    고정 fixture로 핵심 계산을 실행하고 결과를 JSON 저장.
    네트워크 호출 없음.

    수집 항목:
      meta:                      # 비교 대상에서 제외
        run_id, input_hash, git_commit, collected_at,
        python/pandas/numpy/scipy 버전
      data:
        price_frame_shape, actual_start, actual_end
        per_ticker_first_last, na_ratio
      perf:                      # perf_row() 결과 전체
        cagr, vol, sharpe, sortino, calmar, mdd, ulcer, martin, ...
      risk:
        risk_cvar, risk_semisd, var95, 각 위험측도 함수 결과
      turnover:
        turnover_series() 요약 (평균, 합계, 리밸런싱 횟수)
      contribution:
        growth_contribution() 종목별 기여도
      optimize:                  # 목표별로 아래 3개를 함께 저장
        {목표: {"weights": [...], "objective": float,
                "constraints_ok": bool}}
        목표: min_vol / max_sharpe / risk_parity / hrp
      factor:
        factor_betas() 베타 및 R²

    주의:
      - float은 소수점 10자리로 반올림해서 저장
      - dict/Series는 키 정렬 후 저장 (순서 차이로 인한 오탐 방지)
    """

def compare_golden(before_path, after_path,
                   tol_default: float = 1e-9,
                   tol_weights: float = 1e-4) -> list:
    """
    두 골든 JSON을 비교하고 차이 목록을 반환.

    tolerance를 분리한다:
      - 일반 지표: tol_default (1e-9)
      - optimize.*.weights: tol_weights (1e-4)
        SLSQP는 scipy 버전에 따라 비중이 극미세하게 달라질 수 있다.
      - optimize.*.objective: tol_default
      - optimize.*.constraints_ok: 정확히 일치해야 함 (bool)

    비중이 tol_weights 이내로 달라도 목적함수 값이 tol_default를
    넘게 달라졌다면 CHANGED로 보고한다. 진짜 계산 변화이기 때문이다.

    meta 블록은 비교 대상에서 제외.
    키 추가/삭제도 리포트 (ADDED / REMOVED).
    """
```

CLI:
```bash
python tools/golden.py collect --out tools/golden_before.json
# ... 코드 수정 ...
python tools/golden.py collect --out tools/golden_after.json
python tools/golden.py compare tools/golden_before.json tools/golden_after.json
```

### 0-4. Live smoke test

**파일**: `tests/test_live_smoke.py` (신규)

```python
def test_live_data_smoke():
    """
    Yahoo 실제 호출. 골든 숫자와 비교하지 않는다.
    확인 항목:
      - 가격 데이터가 비어 있지 않은가
      - 필요한 컬럼이 존재하는가
      - 통화 변환이 동작하는가 (KRW 기준 값이 원시값과 다른가)
      - 최근 영업일 기준 데이터가 지나치게 오래되지 않았는가
    네트워크 실패 시 skip 처리 (CI에서 빨간불이 나지 않도록).
    """
```

### 0-5. Run ID / Input Hash 분리

**파일**: `app.py` — `assumptions_panel()` (app.py:836) 확장

`_git_commit()` (app.py:824) 이 이미 있고 `assumptions_panel` 이 코드 버전을 표시합니다. 여기에 추가합니다.

**세 개를 구분합니다.**

```
Session ID   앱을 연 세션 단위. 선택 사항.
Run ID       분석 실행 시마다 새로 발급. PF-20260811-143022-A71
Input Hash   입력조건을 정렬·직렬화해 해시. IH-3f9c2a71
```

```python
def make_run_id() -> str:
    """PF-YYYYMMDD-HHMMSS-XXX (XXX는 3자리 hex)."""

def compute_input_hash(*, tickers, weights, start, end, base_ccy,
                       use_dividends, fx_hedge, cost_bp, rebalance,
                       rf_rate, screen_params: dict) -> str:
    """
    모든 입력을 정렬 후 JSON 직렬화하고 sha256의 앞 8자리를 반환.
    screen_params는 화면별 추가 설정 (최적화 목표, 학습기간 등).
    float은 소수점 10자리로 반올림 후 해시 (부동소수 노이즈 제거).
    """
```

**왜 분리하는가**
```
같은 Input Hash + 다른 결과  → 데이터 소급수정 또는 코드 변경 의심
다른 Input Hash              → 사용자가 입력을 바꾼 것
```
"설정이 같으면 Run ID 유지"로 만들면 그건 Run ID가 아니라 Scenario ID입니다. 그리고 티커·비중·화면별 파라미터까지 fingerprint에 넣어야 하는데, 그건 Input Hash의 역할입니다.

**표시할 필드 (이번 단계)**
```
Run ID, Input Hash, 분석 실행시각(타임존 포함 ISO 8601),
기준통화, 배당처리, 환헤지, 거래비용, 리밸런싱 주기, 무위험수익률,
코드 버전(기존)
```

**다음 단계로 미루는 필드** — 실제 분석기간, 종목별 데이터 최종일, 커버리지 지표. 1순위에 딸려옵니다.

### 검증 (0순위)

- 코드를 전혀 바꾸지 않고 두 번 collect → compare 결과가 빈 목록
- fixture를 지우고 재생성하지 말 것. 지웠다면 커밋에서 복원
- `sortino_ratio` 상수를 일부러 바꿔보고 compare가 잡아내는지 확인 후 원복
- optimizer 비중을 1e-5만큼 흔들어도 통과하고, 1e-3 흔들면 잡히는지
- 같은 입력을 두 번 실행 → Run ID는 다르고 Input Hash는 같은지
- 거래비용을 바꾸면 Input Hash가 달라지는지
- 화면을 전환만 하면 Input Hash가 유지되는지

### 주의 (0순위)

- `tools/golden.py` 는 `app.py` 를 import합니다. Streamlit 초기화가 CLI에서 실행되면 에러가 납니다. 지연 import로 회피하되 **`app.py` 구조를 크게 바꾸지 마십시오** — 0순위에서 앱을 건드리면 골든의 의미가 없어집니다.
- Run ID / Input Hash는 결과값이 아니므로 **골든 비교 대상에서 제외** (`meta` 블록).
- **0-1~0-4를 먼저 완료**하고, 0-5 작업 후 골든 비교를 돌려 아무것도 안 바뀌었는지 확인하십시오. 이게 이 도구의 첫 실전 사용입니다.
- fixture parquet은 `.gitignore` 에 넣지 마십시오. **반드시 커밋합니다.**

---

## 1순위. Data Coverage 공통계층 + 데이터 출처 메타 + CVaR 라벨

### 왜 공통계층인가

`dropna()` 로 공통기간이 잘리는 문제는 후보탐색 화면만의 문제가 아닙니다. `build_price_frame()` (app.py:1369) 에서 일어나고, 이 함수는 최적화, 상관관계, 스트레스, 최종 대안 비교 등 **모든 화면**이 사용합니다.

### 1-1. 커버리지 지표를 두 종류로 구분한다

**하나의 비율로 합치지 마십시오.** 성격이 다른 두 문제입니다.

```
History Span Coverage  — 이력이 얼마나 잘렸는가
  요청 2015~2026, 실제 2022~2026 → history_span_ratio = 38%
  신규 ETF 때문에 공통기간이 잘리는 문제

Observation Coverage   — 그 기간 안에서 데이터가 얼마나 빠졌는가
  실제기간 중 관측치 존재율 → observation_ratio = 97%
  휴장일 불일치·결측 문제
```

사용자에게 **크게 보여줄 것은 `history_span_ratio`** 입니다.

### 1-2. build_price_frame 반환 구조 변경

**파일**: `app.py:1369`

```python
@dataclass
class DataCoverage:
    requested_start:    pd.Timestamp
    requested_end:      pd.Timestamp
    actual_start:       pd.Timestamp   # dropna 후 실제 공통기간 시작
    actual_end:         pd.Timestamp
    history_span_ratio: float          # 실제기간 영업일 / 요청기간 영업일
    observation_ratio:  float          # 실제기간 중 관측치 존재율
    na_ratio:           float          # dropna 이전 결측 비율
    per_ticker:         dict           # {ticker: {"first", "last", "n_obs"}}
    limiting:           list           # actual_start를 결정한 티커들
    retrieved_at:       pd.Timestamp
    source:             str = "yahoo"  # 데이터 원천

def build_price_frame(tickers, start, end, base_ccy, use_dividends,
                      ..., return_coverage: bool = False):
    """
    기존 동작은 그대로 유지한다. return_coverage=True 일 때만
    (prices, coverage) 튜플을 반환한다.

    이 기본값 설계가 중요하다 — 기존 호출부를 한꺼번에 바꾸지 않아도
    동작하므로, 화면을 하나씩 옮기면서 골든 비교를 돌릴 수 있다.
    """
```

`limiting` 산출:
```
각 티커의 첫 유효 관측일을 구한다.
actual_start와 같거나 그보다 늦은 시작일을 가진 티커가 제약 종목이다.
동률이면 전부 포함한다.
```

### 1-3. Coverage 표시 컴포넌트

**파일**: `app.py` — 신규 함수, `assumptions_panel` 근처

```python
def coverage_panel(cov: DataCoverage, *, warn_below: float = 0.7):
    """
      요청 분석기간   2015-01-01 ~ 2026-08-07
      실제 분석기간   2022-04-15 ~ 2026-08-07
      이력 확보율     38%
      관측치 확보율   97%
      제한 종목       ABC (2022-04-15 상장)

    history_span_ratio < warn_below 이면 st.warning, 아니면 st.caption.
    observation_ratio < 0.95 이면 별도 경고 한 줄 추가.
    """
```

**적용 대상 화면** — 포트폴리오 분석 / 최적화 / 자산 상관관계 / 신규자산 추가 / 스트레스 테스트 / 최종 대안 비교

### 1-4. 거시 국면 화면은 예외 처리

**같은 `coverage_panel` 을 그대로 붙이지 마십시오.**

거시 국면은 FRED 데이터와 가격 기반 시장신호가 **서로 다른 데이터원**을 씁니다. 하나의 숫자로 합치면 의미가 없어집니다.

```
FRED Coverage / As-of         (발표 지연 2개월 반영 상태 포함)
Price Signal Coverage / As-of
```
두 개를 각각 표시합니다. `DataCoverage.source` 필드를 활용해 출처별로 분리하십시오.

**원칙**: Data Coverage 철학은 공통으로 쓰되, **출처별 provenance는 분리**합니다.

### 1-5. 최소 이력 필터 (선택 기능)

**파일**: `app.py` — 신규자산 추가 화면

```python
def filter_by_min_history(tickers, prices_raw, min_years: float) -> tuple:
    """반환: (통과 티커, 제외 로그 DataFrame)
       로그: 티커 | 이력(년) | 기준(년) | 사유"""
```

UI: `st.selectbox("후보 최소 이력", ["제한 없음", "1년", "3년", "5년"], index=0)`

**기본값은 "제한 없음"** 입니다. Coverage 표시가 기본이고 필터는 선택입니다. 필터를 기본으로 켜면 사용자가 모르는 사이에 후보가 사라집니다 — 지금 고치려는 문제와 같은 종류입니다.

### 1-6. Run 메타에 데이터 출처 추가

0-5에서 만든 확장에 추가합니다.
```
데이터 조회시각  : cov.retrieved_at
데이터 원천      : cov.source
실제 분석기간    : cov.actual_start ~ cov.actual_end
이력 확보율      : cov.history_span_ratio
관측치 확보율    : cov.observation_ratio
종목별 최종일    : cov.per_ticker의 last를 표로 (expander 안에)
```

**종목별 최종일이 가장 중요합니다.** yfinance는 소급 수정이 잦아서, 이게 없으면 "지난번이랑 숫자가 왜 달라졌나"에 답할 수 없습니다. Input Hash가 같은데 결과가 다르면 여기를 봅니다.

### 1-7. CVaR 라벨 정리

**파일**: `app.py:1808` `risk_cvar()` 및 위험측도 표시부

계산을 바꾸지 말고 **라벨과 도움말만** 수정합니다.
```
표시명: "CVaR 95% (연율 환산 근사)"
도움말: "일간 수익률 하위 5%의 평균을 √252로 연율 환산한 근사치입니다.
        수익률 분포가 정규분포가 아닐 경우 실제 연간 손실 규모를
        과소평가할 수 있습니다."
```

**Sortino는 건드리지 마십시오.** 이전 검토에서 "정의가 잘못됐다"는 지적이 있었으나, 최신 코드(`sortino_ratio`)는 이미 전체 표본 기준 RMS 하방편차를 사용하며 `risk_semisd` 와 정의가 일치합니다. 오래된 스냅샷을 기준으로 한 지적이었습니다.

### 검증 (1순위)

- 골든 비교: 1-2 ~ 1-7 작업 후 **숫자가 하나도 바뀌지 않아야 함** (반환 구조 추가와 표시 변경뿐)
- 신규 상장 ETF를 후보에 넣었을 때 `history_span_ratio` 가 낮게 나오고 `limiting` 에 그 종목이 잡히는지
- 전 종목이 같은 기간이면 `history_span_ratio` 가 1.0에 가깝고 `limiting` 이 비는지
- 한국·미국 종목 혼합 시 휴장일 차이가 `observation_ratio` 에 반영되는지
- 거시 국면 화면에서 FRED와 가격 커버리지가 **따로** 표시되는지

### 주의 (1순위)

- `return_coverage=False` 기본값을 지키십시오. 화면을 하나씩 옮기면서 매번 골든 비교를 돌릴 수 있어야 합니다.
- 화면을 한 커밋에 다 바꾸지 말고 화면 단위로 나누어 커밋하십시오.

---

## 2순위. Asset Master + 계약 정의

### 왜 필요한가

지금 자산 정보가 세 곳에 흩어져 있습니다. `guess_asset_kind()` (app.py:3294) 의 티커 기반 추정, 스트레스 화면의 사용자 입력(듀레이션, 신용 민감도), 최적화 화면의 제약 설정입니다. 화면을 옮기면 다시 입력해야 합니다.

Exposure Dashboard(3), Mandate(4), Feasibility(5)는 **셋 다 자산별 속성이 있어야 성립**합니다. 먼저 만들지 않으면 3순위에서 "자산군 비중을 보여줘야 하는데 자산군이 어디에도 정의되어 있지 않은" 상황을 만납니다.

### 2-1. core/ 디렉터리 신설

**이것이 코드 분리의 시작점입니다.** 별도 리팩토링 프로젝트로 잡지 말고, **새로 만드는 공통 객체를 처음부터 `core/` 에 두는 방식**으로 진행합니다. 나중에 13,000줄에서 뜯어내는 것보다 압도적으로 저렴합니다.

```
core/
  __init__.py
  asset_master.py      # 2순위
  settings.py          # AnalysisSettings — 2순위에 껍데기, 4순위에 완성
  mandate.py           # PortfolioMandate — 4순위
  contracts.py         # PortfolioCandidate, MarketView — 2순위
```

`core/` 안에 `import streamlit` 을 넣지 마십시오. 테스트 가능성이 사라집니다.

### 2-2. AssetMaster — 필드별 provenance

**파일**: `core/asset_master.py` (신규)

```python
@dataclass
class AssetAttrs:
    ticker:      str
    name:        str = ""
    asset_class: str = ""      # 주식 / 채권 / 대체 / 현금
    sub_class:   str = ""      # 미국 국채, 국내 대형주, 금 ...
    region:      str = ""      # 미국 / 한국 / 유럽 / 일본 / 신흥 / 글로벌
    currency:    str = ""      # 표시통화
    duration:    float = None  # 채권만
    credit:      str = ""      # Sovereign / IG / HY / None
    fx_exposure: str = ""      # 실질 환노출 통화 (환헤지 ETF는 다를 수 있음)
    beta_bench:  str = ""      # 스트레스에서 쓸 기준지수 티커

    field_source: dict = field(default_factory=dict)
    # 예: {"asset_class": "auto", "region": "auto",
    #      "currency": "yahoo", "duration": "user", "credit": "user"}
    # 값: "auto"(추정) / "yahoo"(조회) / "user"(사용자 입력) / ""(미입력)
```

**전체 객체에 `source` 하나를 두지 마십시오.** 사용자가 duration 하나만 수정했는데 객체 전체가 `source="user"` 가 되면, 나중에 어떤 값이 검증된 값인지 알 수 없습니다. Exposure Dashboard에서 "USD 노출 41%"가 자동추정인지 사용자 확인값인지 구분할 수 있어야 합니다.

```python
class AssetMaster:
    def infer(self, ticker, name="", quote_type="") -> AssetAttrs:
        """
        guess_asset_kind() (app.py:3294) 를 재사용해 초기 추정값을 만든다.
        추정한 필드는 field_source에 "auto" 또는 "yahoo"로 기록한다.
        추정 못 한 필드는 빈 값으로 두고 field_source도 빈 문자열.
        추정을 지어내지 말 것 — 모르면 비워둔다.
        """

    def upsert_field(self, ticker: str, field_name: str, value) -> None:
        """해당 필드만 갱신하고 field_source[field_name]="user"로 설정."""

    def get(self, ticker: str) -> AssetAttrs: ...

    def group_weights(self, weights: dict, by: str) -> tuple:
        """
        by: "asset_class" | "region" | "currency" | "fx_exposure" | "sub_class"
        반환: ({그룹명: 합산 비중}, 미분류_비중, {그룹명: 자동추정_비중})
        세 번째 값으로 각 그룹에서 자동추정 속성에 기반한 비중을 함께 반환한다.
        Exposure Dashboard가 신뢰도를 표시할 수 있어야 하기 때문이다.
        """

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "AssetMaster": ...
```

**UI** (`app.py` — 공통 설정 영역 또는 별도 expander)
- `st.data_editor` 로 티커별 속성 표 편집
- `field_source` 가 `"auto"` 인 셀은 배경색으로 구분
- 수정하면 해당 필드만 `"user"` 로 전환
- session_state 보관 + JSON 내보내기/불러오기

### 2-3. 계약 정의

**파일**: `core/contracts.py` (신규)

```python
@dataclass
class PortfolioCandidate:
    """Index Lab, 최적화, 수동 구성, 벤치마크 등 모든 holdings의 공통 형식."""
    schema_version: str = "1.0"    # 포맷 변경 시 호환성 관리
    name:           str = ""
    source:         str = ""       # "index-lab" / "optimizer" / "manual"
                                   # / "bl" / "benchmark"
    holdings:       dict = None    # {ticker: weight(0~1)}
    asset_metadata: dict = None    # AssetMaster.to_dict() 부분집합
    created_at:     str = ""       # ISO 8601
    notes:          str = ""

    def validate(self) -> list:
        """비중 합계 1.0(±1e-6), 음수 비중, 티커 중복을 확인.
           문제 목록 반환. 빈 목록이면 정상."""

@dataclass
class MarketView:
    """시장동향 화면 → BL 전달용."""
    schema_version: str = "1.0"
    factor:     str = ""     # "US10Y" / "USD" / "Growth" / "Inflation" ...
    direction:  str = ""     # "up" / "down" / "neutral"
    magnitude:  float = 0.0
    unit:       str = ""     # "bp" / "pct_point" / "sigma" — 필수
    horizon:    str = ""     # "3M" / "6M" / "12M"
    confidence: float = 0.0  # 0~1
```

**`unit` 은 필수입니다.** `magnitude=50` 이 50bp인지 50%p인지 JSON만 보고 알 수 있어야 합니다. "factor별로 문서화"는 앱 간 계약에서 너무 위험합니다.

**`schema_version` 도 필수입니다.** Index Lab과 Portfolio Lab이 별도로 진화하므로 포맷이 달라질 수 있습니다.

**MarketView는 지금 정의합니다** — 소비자(BL 화면)가 이미 존재하고 필드가 확정적입니다.
**PortfolioCandidate는 Asset Master와 함께 정의합니다** — `asset_metadata` 가 곧 Asset Master이므로 그 전에 확정하면 다시 맞춰야 합니다.

### 검증 (2순위)

- `core/` import 후 골든 비교: **숫자 불변** (아직 계산에 쓰지 않으므로)
- `infer()` 가 SPY / TLT / GLD / 005930.KS 를 합리적으로 분류하는지
- 모르는 티커에 대해 **빈 값을 반환하고 지어내지 않는지**
- `upsert_field` 로 duration만 수정했을 때 **다른 필드의 field_source가 유지되는지**
- `group_weights()` 의 미분류 비중과 자동추정 비중이 정확히 집계되는지
- `validate()` 가 비중 합 0.99, 음수 비중, 중복 티커를 각각 잡는지
- `to_dict` → `from_dict` 왕복 후 `field_source` 까지 동일한지

### 주의 (2순위)

- `guess_asset_kind()` 를 재사용하되 **수정하지 마십시오.** 기존 호출부가 있습니다. 새 로직이 필요하면 `core/` 안에서 감싸십시오.
- Asset Master를 만들었다고 기존 화면을 바로 옮기지 마십시오. 2순위는 **객체 생성 + 편집 UI까지**입니다. 실제 사용은 3순위부터입니다.

---

# Part 2 — 결정 기록 (3순위 이후)

각 항목 착수 시 아래 기록을 근거로 상세 스펙을 작성합니다.
**"왜"와 "함정"이 핵심입니다. 이게 안 적히면 나중에 같은 논쟁을 반복합니다.**

---

## 3순위. Exposure Dashboard

**무엇** — 현재 포트폴리오의 성격을 한 화면에 요약. 자산군/지역/통화 비중, Equity Beta, Duration, 위험기여 상위 자산, 거시 민감도.

**왜** — 새 계산이 거의 없이 기존 계산을 조립하는 것뿐인데 사용 경험이 바뀝니다. 값/비용 비율이 이 계획 전체에서 가장 좋습니다. 현재는 "내 포트폴리오가 어떤 성격인가"를 알려면 여러 화면을 돌아야 합니다.

**전제** — Asset Master(2순위)

**함정**
- Duration, Credit, 실질 FX 노출은 **자동으로 안 나옵니다.** 현재 스트레스 화면에서 사용자가 자산별로 입력하는 값입니다. Asset Master에 넣되, **비어 있으면 그 항목을 표시하지 말고 "미입력"으로 두십시오.** 0으로 채우면 채권 듀레이션 0인 포트폴리오처럼 보입니다.
- `group_weights()` 가 반환하는 **자동추정 비중을 표시**하십시오. "USD 노출 41% (이 중 28%p는 자동 추정)" 형태입니다.
- 미분류 자산 비중이 10%를 넘으면 화면 상단에 경고.
- "핵심 취약성" 같은 서술형 문구는 **규칙 기반으로만** 생성하십시오. 자유 서술 금지.

**재사용** — `growth_contribution()`, `factor_betas()`, 기존 위험기여 계산, `AssetMaster.group_weights()`

---

## 4순위. Portfolio Mandate + μ 의존도 표시

### 4-A. Portfolio Mandate

**무엇** — 투자 목적과 제약을 정의하는 객체 및 화면.

**왜** — "Sharpe 0.8이니 좋다"가 "목표 변동성 8% 이하를 충족하면서 예상수익이 가장 높은 안"으로 바뀝니다. 개인 도구와 기관 도구의 차이입니다.

**구조 — AnalysisSettings와 분리합니다.**

```
core/mandate.py    PortfolioMandate — 투자 목적과 제약
    투자목적, nav_currency, 목표 변동성, MDD 허용,
    자산군/지역/통화 한도, 단일자산 한도, 회전율 한도

core/settings.py   AnalysisSettings — 계산 방법
    reporting_currency, 분석기간, 배당 처리, 환헤지,
    거래비용, 결측처리, 무위험금리
```

**통화는 두 개념으로 나눕니다.**
```
mandate.nav_currency          실제 포트폴리오/펀드의 기준통화 (예: KRW)
settings.reporting_currency   분석 결과를 표시할 통화
                              기본값 = mandate.nav_currency
```
원화 펀드를 USD 기준으로 분석한다고 해서 만데이트가 바뀌는 것은 아닙니다. 지금 분리해두는 게 나중에 가장 덜 꼬입니다.

**경계가 모호한 다른 필드의 판별 기준**: *"이걸 바꾸면 정답이 달라지는가, 계산 방식만 달라지는가."* 정답이 달라지면 Mandate, 계산만 달라지면 Settings입니다.

**함정**
- **측정기준 라벨을 반드시 붙이십시오.** "목표 변동성 ≤10% / 현재 11.8% ⚠️" 에서 11.8%는 **과거 실현 변동성**이고 목표는 **미래 기대치**입니다. 성격이 다른 둘을 직접 비교해 판정을 찍으면 없는 정밀도를 만드는 것입니다. 판정 옆에 "최근 3년 실현 기준"처럼 항상 표기하십시오.
- 비중 제약(주식 57%)은 현재 상태이므로 이 문제가 없습니다. **변동성·MDD 같은 통계량만** 해당됩니다.
- Mandate를 만들면서 기존 전역 설정을 한꺼번에 정리하려 하지 마십시오. 해당 필드만 옮깁니다.

### 4-B. μ 의존도 표시

**무엇** — 최적화 목표별로 과거 평균수익률 의존도를 표시하고, 의존도 높은 목표 선택 시 경고 한 줄.

**왜** — MVO 비중 불안정의 주범은 공분산이 아니라 기대수익(μ)입니다. 그런데 이 앱은 μ를 안 쓰는 대안(Min Vol, Risk Parity, HRP)과 μ를 뷰로 대체하는 대안(BL)을 **이미 전부 갖고 있습니다.** 새 추정 모델을 만들 게 아니라 사용자가 목표를 고를 때 이 차이를 알게 하는 것으로 충분합니다. 공분산 shrinkage보다 훨씬 저렴하고 효과가 큽니다.

**형태**
```
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

**함께 정리** — 최적화가 쓰는 기대수익은 CAGR이 아니라 **연율 산술평균**입니다. 라벨을 "연율 기대수익(산술평균)"으로 명확히 하십시오.

---

## 5순위. Mandate 적합성 + Feasibility 진단

**무엇** — 포트폴리오가 Mandate를 충족하는지 판정하고, 제약을 걸었을 때 해가 없으면 **어느 제약이 충돌하는지 진단**.

**왜** — 제약을 10개 넣는 것 자체는 어렵지 않습니다. 문제는 동시에 걸면 해가 없어진다는 것이고, 이건 이 프로젝트에서 이미 겪은 일입니다(제약 7개 동시 적용 시 해 없음). "최적화 실패"로 끝내면 사용자는 제약을 하나씩 껐다 켜며 헤맵니다.

**핵심은 제약 추가가 아니라 진단입니다.**
```
현재 제약으로 가능한 포트폴리오가 없습니다.

가장 큰 충돌:
  미국 비중 ≤40% 와 USD 노출 ≤30% 를 동시에 충족하기 어렵습니다.

USD 한도를 36% 이상으로 완화하면 해가 생성됩니다.
```

**전제** — Asset Master(2), Mandate(4)

**재사용** — `feasibility()` (app.py:2161) 확장

**함정**
- 완화 제안값("36% 이상")을 계산하려면 제약을 하나씩 이완시키며 재탐색해야 합니다. 느려지므로 **해가 없을 때만** 실행하십시오.
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

**두 영역의 숫자를 절대 합산하지 마십시오.** 영역 1은 과거 백테스트 기반이고 영역 2는 현재 시점 계산입니다. 합쳐서 "제안안의 순효과"처럼 만들면 오도합니다.

### 6-B. AUM 입력과 유동성

**AUM 입력이 필수입니다.** 비중만으로는 거래금액이 나오지 않습니다.

```
Portfolio AUM: 500억원  ← 사용자 입력

종목      매매금액   20D ADV    ADV 대비   판단
SPY        30억      매우 큼     0.01%     여유
ABC ETF    20억      35억        57%       ⚠️ 실행 부담
```

**함정 — ADV 비율은 시장충격비용이 아닙니다.** "ADV 대비 35%"에서 "시장충격비용 42bp"를 도출하지 마십시오. 무료 데이터로는 근거가 없습니다. **비용 모델이 아니라 실행가능성 경고**로 표현합니다.

### 6-C. no-trade zone

**무엇** — 최적 비중과 현재 비중의 차이가 임계값 미만이면 거래하지 않는 밴드.

**왜** — 리밸런싱 때마다 전 종목을 소폭씩 조정하는 것은 비용만 발생시키고 효과가 거의 없습니다.

**계산 방식 — 전체 재정규화를 하면 안 됩니다.**

```
잘못된 방식 (전체 정규화):
  A 40 → 40.5 (no trade, 40으로 고정)
  B 30 → 25
  C 30 → 34.5
  → 전체를 100%로 재정규화하면 A도 40이 아니게 된다.
     고정한 게 고정이 안 된다.

올바른 방식:
  1) |target - current| < threshold 인 종목을 완전히 고정
  2) 고정 종목의 비중 합계를 계산 (예: 40%)
  3) 남은 60%만 거래 대상 종목의 target 비율대로 배분
  4) 고정 종목은 재배분 과정에서 건드리지 않는다
```

```python
def apply_no_trade_zone(current_w, target_w, threshold_bp: float = 100):
    """
    threshold_bp=0 이면 기존 동작(항상 리밸런싱)과 동일해야 한다.
    threshold가 매우 크면 전 종목 고정 = 리밸런싱 없음이어야 한다.
    반환 비중 합계는 정확히 1.0.
    """
```

**적용 후 반드시 재검사** — 종목별 최대비중, 자산군/지역/통화 한도, Mandate 중 하나라도 위반하면 경고하거나 Feasibility 재검사를 수행합니다. 거래 대상만 재배분하므로 특정 종목이 상한을 넘을 수 있습니다.

**`walk_forward()` 적용 시 주의** — no-trade 판단 기준은 **직전 목표비중이 아니라 가격 변동으로 drift된 실제 현재비중**입니다. `turnover_series()` (app.py:1466) 가 이미 `prev_w` 로 실제 교체량을 계산하는 구조이므로 그 값을 사용하십시오.

**출력** — 적용 전/후 평균 회전율 비교, 건너뛴 리밸런싱 횟수. 임계값이 너무 높으면 "최근 4회 중 3회 건너뜀 — 임계값이 높을 수 있습니다" 경고.

---

## 7순위. Robustness 체크리스트

**무엇** — 분석 결과를 얼마나 믿을 수 있는지 체크 항목으로 표시.

```
Data
  공통기간 9.2년 ✅
  이력 확보율 91% ✅
  관측치 확보율 98% ✅

Optimization
  학습기간 변경 시 최대 비중변화 12%p ✅
  OOS Sharpe +0.08 ✅
  Walk-forward 5회 중 3회 개선 ⚠️

Implementation
  비용 5→20bp에서도 순위 유지 ✅
  최대 ADV 사용 42% ⚠️

주요 주의점: ABC ETF의 데이터 이력이 짧고 실행 유동성이 낮습니다.
```

**절대 하지 말 것 — 종합 점수나 등급을 만들지 마십시오.** "종합 82점", "신뢰도: 높음" 같은 압축은 두 가지 문제가 있습니다. 서로 다른 성격의 체크를 합산할 때 가중치에 근거가 없고, 등급이 뜨면 사용자가 개별 항목을 안 봅니다. 이 프로젝트의 원칙에 정면으로 어긋납니다.

**체크리스트를 그대로 보여주고 빨간불 항목만 강조합니다.**

**전제** — 1순위(Coverage), 6순위(ADV)의 결과를 가져다 씁니다. 새 계산은 거의 없습니다.

---

## 8순위. Adopted Portfolio

**무엇** — 최종 대안 비교에서 "채택" → 목표 비중과 채택 시점 저장 → 이후 drift 확인, 목표 대비 괴리, 리밸런싱 필요 여부.

**왜** — 일회성 분석기에서 지속 사용 프로그램으로 바뀝니다.

**저장 방식 — JSON 내보내기/불러오기로 시작합니다.** 현재 Streamlit Cloud 배포는 세션 상태 + JSON만 쓰기로 되어 있고, 여러 명이 같은 인스턴스를 쓸 때 데이터가 섞이는 문제를 겪고 내린 결정입니다. `ENABLE_LOCAL_PERSISTENCE` (app.py:41) 가 이미 있으므로 로컬 실행에서는 파일 저장이 가능합니다. **① JSON → ② 자주 쓰이면 로컬 SQLite** 순서입니다.

**Drift 표시 — 가정을 명시해야 합니다.**
```
사용자가 현재 실제비중을 입력하지 않은 경우:
  "무거래·무입출금 가정의 추정 Drift"

사용자가 현재 실제비중을 입력한 경우:
  "실제 현재비중 기준 Drift"
```
목표비중과 현재 가격만으로 계산한 drift는 **채택 이후 매매도 입출금도 없었다는 가정**이 들어갑니다. 실제 펀드는 자금 유출입이 있으므로 이 표시가 없으면 오도합니다.

**함정**
- "한 달 뒤 앱을 켜면 현재 상태가 나온다"는 **JSON 방식에서는 안 됩니다.** 사용자가 JSON을 다시 올려야 합니다. 화면 문구를 이 동작에 맞게 쓰십시오.
- 저장 대상은 목표 비중, 채택일, 채택 당시 Mandate, 채택 근거 메모, Input Hash입니다. 가격 시계열은 저장하지 마십시오.
- `PortfolioCandidate` 형식을 사용하십시오. 별도 포맷을 만들지 마십시오.

**재사용** — `snapshot()` (app.py:1001)

---

## 9순위. 블랙-리터만 2단계

**무엇** — 뷰를 반영한 실제 비중 조정안 출력. 현재 BL은 뷰 입력과 영향 분석(1단계)까지만 되어 있고 비중 조정이 미완성입니다.

**왜 — 10순위의 전제조건입니다.** 10순위에서 MarketView를 BL로 전달하는데 받는 쪽이 끝을 못 내면 연결의 의미가 없습니다.

**구조 — BL은 기대수익까지만 담당하고, 비중은 기존 최적화 엔진이 산출합니다.**

```
Market View
  ↓
Black-Litterman          ← black_litterman() (app.py:4689)
  ↓
Posterior Expected Return
  ↓
기존 제약 최적화 엔진      ← bl_weights() (app.py:4711) 또는 공용 solver
  현재비중 ± max_delta bounds
  합계 100%
  ↓
최종 비중
```

**BL만을 위한 별도 비중조정 휴리스틱을 새로 만들지 마십시오.**

**금지 — clip 후 전체 비례정규화.** ±5%p로 clip한 뒤 전체를 비례조정하면 **비례조정 과정에서 다시 5%p를 넘습니다.** 6-C의 no-trade와 정확히 같은 종류의 버그입니다. 제약은 solver의 bounds로 넘기십시오.

**R² 처리 — 0.10을 절대 기준으로 쓰지 마십시오.**
```
기본 threshold 0.10은 허용하되:
  - "내부 heuristic"임을 화면에 명시
  - 사용자가 threshold 변경 가능
  - R²를 증거품질 표시로도 활용
      R² 높음 → 요인 연결 강함
      R² 중간 → 참고
      R² 낮음 → 근거 약함
```
0.10이라는 값 자체에 근거가 없으므로 하드코딩된 절대 기준으로 만들면 "없는 정밀도를 만들지 않기" 원칙과 충돌합니다.

**전 종목이 threshold 미만이면** 조정하지 않고 "조정 대상 없음"으로 종료합니다. 억지로 조정하지 마십시오. 한국 개별종목은 R²가 0.05~0.07 수준으로 실측된 바 있습니다.

**적용 범위** — 모드 A(영향 분석만, 현재 상태) / 모드 B(보유자산 내 조정). 모드 C(후보자산 포함)는 나중 과제.

**출력** — 자산 | 현재비중 | 조정비중 | 변경폭 | R² | 증거품질, 그리고 조정 전후 성과 비교

---

## 10순위. Index Lab → Candidate / MarketView → BL

**무엇** — Index Lab에서 설계한 인덱스를 `PortfolioCandidate` 로 내보내 Portfolio Lab에서 신규 자산처럼 평가. 시장동향 화면의 판단을 `MarketView` 로 BL에 전달.

**왜** — 두 도구를 각각 95점으로 만드는 것보다 연결 하나가 업무가치를 더 올립니다. 상품개발 관점에서 "새 ETF를 만들면 기존 라인업에 어떤 영향인가"를 한 흐름으로 볼 수 있습니다.

**전제** — 2순위(계약), 9순위(BL 2단계 완성)

**구현 부담이 작습니다** — 계약이 2순위에서 이미 정의되어 있으므로 변환 함수와 버튼만 붙이면 됩니다.

**함정**
- 한국 종목은 `.KS` / `.KQ` 접미사가 있어야 yfinance 조회가 됩니다. Index Lab에서 정규화된 티커를 그대로 넘기십시오.
- `validate()` 후 내보내기.
- Index Lab의 백테스트는 탐색형(생존편향 있음)입니다. Portfolio Lab으로 넘어온 뒤에도 이 사실이 표시되어야 합니다.
- `schema_version` 불일치 시 명시적으로 거부하고 안내하십시오. 조용히 진행하지 마십시오.

---

## 11순위. Policy Benchmark

**무엇** — 벤치마크를 단일 티커가 아니라 **비중을 가진 포트폴리오**로 설정.

```
Policy Benchmark
  SPY  60%
  AGG  30%
  GLD  10%
```

**왜 — 12순위 Brinson의 전제조건입니다.** Brinson-Fachler는 벤치마크의 **자산군별 비중과 수익률**이 있어야 계산됩니다. `^GSPC` 가격 하나로는 "주식 배분효과"와 "채권 선택효과"를 계산할 수 없습니다. 이건 기능적 전제이지 선택 사항이 아닙니다.

**구현** — 벤치마크도 `PortfolioCandidate` (source="benchmark") 를 사용합니다. 별도 구조를 만들지 마십시오. Asset Master로 포트폴리오와 벤치마크를 같은 기준으로 그룹화합니다.

**기존 단일 티커 벤치마크는 유지합니다.** Policy Benchmark는 추가 옵션입니다. 단일 티커만 설정된 경우 Brinson을 비활성화하고 안내합니다.

---

## 12순위. 성과귀속 Brinson-Fachler

**무엇** — 벤치마크 대비 초과수익을 배분효과와 선택효과로 분해.

**왜** — "이번 분기에 주식을 많이 담은 게 좋았나, 종목을 잘 골랐나"에 정량적으로 답합니다. **사후 평가라는 새 업무단계**를 프로그램 안으로 가져옵니다.

**공식**
```
배분효과 = (wp - wb) × (Rb_group - Rb_total)
선택효과 = wb × (Rp_group - Rb_group)
상호작용 = (wp - wb) × (Rp_group - Rb_group)
세 효과의 합 = 초과수익 (정확히 일치해야 함 — 핵심 검증)
```

**전제** — Asset Master(2), Adopted Portfolio(8), **Policy Benchmark(11)**

**함정**
- **Policy Benchmark가 없으면 기능을 비활성화**하고 "자산군별 비중이 있는 Policy Benchmark가 필요합니다"라고 안내하십시오.
- 자산군이 1개면 배분효과가 무의미합니다. "자산군 2개 이상 필요" 메시지.
- 일별 Brinson은 계산은 되지만 해석이 어렵습니다. 월별 이상을 기본으로.

**재사용** — `growth_contribution()` 을 자산군 레벨로 묶어 벤치마크 대비 분해

---

## 13순위. Investment Memo

**무엇** — 분석 결과를 업무 문서용 서술로 정리.

**왜** — 사용자가 업무에서 원하는 최종 산출물은 결국 보고서입니다. "Sharpe 0.83 / MDD -16.2%"에서 끝나는 것과 검토 의견 문단까지 나오는 것은 업무효율이 다릅니다.

**결정적 제약 — 규칙 기반 템플릿 조립만 허용합니다. 자유 생성 금지.**
```
허용:  숫자 → 조건 판정 → 사전 정의된 문장 조각 조립
금지:  LLM이 숫자를 보고 자유롭게 서술
```
자유 생성은 같은 입력에 다른 문장이 나오고 근거 없는 해석이 섞입니다. 결정론적 원칙과 충돌하며, 그 문서가 실명으로 보고서에 들어간다면 위험합니다.

---

# Part 3 — 보류·기각 기록

나중에 같은 제안이 올라올 때 재논의를 막기 위한 기록입니다.

| 항목 | 판정 | 이유 |
|---|---|---|
| Index Lab PIT 스냅샷 적재 | **기각** | 상품개발 목적상 실익이 작음. ETF 유니버스는 시총·유동성 필터를 거치므로 상장폐지가 드물고, 백테스트의 목적도 절대수익이 아니라 회전율·집중도·업종분포 확인. 6개월을 기다릴 가치가 없음. 대신 재무 발표지연·상장일 필터를 Index Lab 트랙에서 처리 (Part 4). |
| 환헤지 비용 시점별 반영 | 보류 | FRED 한국 금리 시리즈가 불안정해 사용자 입력 fallback 필수. 4순위 AnalysisSettings 정리와 겹치므로 그때 재검토. |
| 시나리오 이력 (버린 안 포함) | 보류 | 8순위 Adopted Portfolio가 채택한 안을 저장하므로 대부분 커버. |
| 롤링 팩터 익스포저 | 보류 | 3순위가 현재 시점 노출을 커버. 시간 변화 추적이 실제로 필요해지면 그때. |
| Proxy 지정 (짧은 이력 보완) | 보류 | Data Coverage(1순위)를 먼저 넣고 실제 불편이 반복되는지 확인. 도입 시 Proxy 구간과 실제 구간을 반드시 구분 표시하고 전환비용·회전율 분석에서 제외할 것. yfinance로 국내 ETF 기초지수는 거의 조회 불가. |
| 공분산 Shrinkage | 보류 | MVO 불안정의 주범은 공분산이 아니라 μ. 4-B가 훨씬 저렴하고 효과가 큼. |
| 시장별/글로벌 Stress transmission | 보류 | 글로벌 충격 → 시장별 충격 → 개별자산의 2단계 구조가 필요해 작업량이 큼. 당분간 현재 방식에 **"단일 시장 proxy 기반 근사"** 라고 명시할 것. |
| Robustness 종합점수 | **기각** | 가중치에 근거가 없고, 등급이 뜨면 개별 항목을 안 봄. 체크리스트만 유지. |
| Investment Memo 자유 생성 | **기각** | 결정론적 원칙과 충돌. 템플릿 조립만. |
| ADV 기반 시장충격비용(bp) 산출 | **기각** | 무료 데이터로는 근거 없음. 실행가능성 경고로만. |
| BL 전용 비중조정 휴리스틱 | **기각** | 기존 제약 최적화 엔진을 재사용. clip 후 정규화는 제약을 다시 위반함. |
| 위험지표 추가, 최적화 알고리즘 추가 | **기각** | 새로운 업무단계를 가져오지 않음. Part 5-5 참조. |
| Monte Carlo / GARCH / Copula | **기각** | 동일. |
| ECOS 자유검색, 한국 경기×물가 국면 | **기각** | Portfolio Lab의 정체성은 데이터 탐색기가 아니라 의사결정 도구. |
| 다중 사용자 협업 | **기각** | 현 단계에서 불필요. |
| Index Lab 룰북 자동생성(docx) | 보류 | Index Lab 트랙의 나중 과제. |

---

# Part 4 — Index Lab 트랙 (Portfolio Lab 완료 후)

**Portfolio Lab 13순위까지 마친 뒤 착수합니다.** 지금 병행하는 작업은 없습니다.

### I-1. 재무 발표지연 + 상장일 필터

**무엇** — 정기변경 시점 기준으로 그 시점에 공시되어 있던 재무제표만 사용.

**왜** — 이게 상품개발에서 PIT의 실제 쟁점입니다. 2024년 1월 정기변경에서 ROE로 점수를 매길 때, 2023년 12월 결산 재무제표는 **2024년 3월에야 공시됩니다.** 아직 나오지 않은 숫자로 종목을 고르면 **편입종목·회전율·업종분포가 실제와 달라집니다.** 이것들이 상품 검토에서 실제로 보는 숫자입니다.

**스냅샷 적재 없이 지금 고칠 수 있습니다.** DART 재무제표 API에 접수일(`rcept_dt`)이 있으므로, 정기변경 시점보다 늦게 접수된 재무제표를 제외하면 됩니다. 상장일 필터도 DART에서 소급 조회됩니다.

**함정** — DART가 과거 접수일을 어디까지 제공하는지 실제 키로 검증이 필요합니다. 추측하지 말고 먼저 확인하십시오.

### I-2. 정기변경 백테스트 강화

구성종목 변경 이력 시각화(편입/편출 타임라인), 거래비용 영향 분리(cost=0 vs 실제 cost 2회 실행), 생존편향 경고 유지.

### I-3. 방법론 비교 심화

비중 방식만이 아니라 필터 기준·목표 종목수·점수 가중치까지 변경한 대안을 최대 4개 비교. 파이프라인 전체 재실행 필요(점수가 바뀌면 선정이 바뀌고 비중이 바뀜). session_state 원본을 deepcopy 후 변경할 것.

### I-4. 테마 키워드 필터

DART 사업보고서 텍스트에서 키워드 포함 여부로 1차 선별. 단순 문자열 포함만 사용(형태소 분석 금지, 결정론적 규칙 원칙). 사업내용이 없는 종목(미국 주식 등)은 필터를 건너뛰고 로그에 기록. "1차 선별 도구이며 최종 편입은 직접 확인" 안내 필수.

### I-5. 룰북 자동생성 (나중 과제)

---

# Part 5 — 전역 원칙

## 5-1. 기준안 대비 Δ

현재 포트폴리오가 있으면 **어디서든** 절대값과 함께 변화량을 보여줍니다. 별도 프로젝트가 아니라 UX 원칙입니다. 3순위부터 새 화면을 만들 때마다 자동으로 따라와야 합니다.
```
Sharpe  0.72 → 0.84  (+0.12)
MDD    -21.4 → -17.8 (+3.6%p 개선)
Vol     13.2 → 12.1  (-1.1%p)
```

## 5-2. 새 공통 객체는 core/ 에

코드 분리를 별도 프로젝트로 잡지 않습니다. **새로 만드는 공통 객체(순수 데이터·로직)는 처음부터 `core/` 에 둡니다.** `core/` 안에 `import streamlit` 을 넣지 않습니다. 기존 코드를 뜯어내는 작업은 하지 않습니다.

## 5-3. 재정규화가 제약을 깨뜨리지 않는지 확인

고정한 비중은 재정규화 과정에서도 고정이어야 하고, clip한 상한은 재정규화 후에도 상한이어야 합니다. 6-C(no-trade)와 9순위(BL)에서 같은 종류의 버그가 각각 발견됐습니다. **비중을 조정하는 코드를 쓸 때마다 이걸 확인하십시오.** 제약이 있으면 직접 조정하지 말고 solver의 bounds로 넘기는 것이 안전합니다.

## 5-4. 측정기준 라벨

과거 실현치와 미래 목표치를 비교해 판정을 찍을 때는 측정 기준을 항상 표기합니다. ("최근 3년 실현 기준")

## 5-5. 출처별 provenance 분리

데이터 출처가 다르면 커버리지·as-of를 합치지 않습니다(1-4의 거시 국면). 자산 속성의 출처가 다르면 필드별로 기록합니다(2-2의 `field_source`). **"어디서 온 값인가"를 잃지 마십시오.**

## 5-6. 골든 비교를 매 작업마다

각 순위 작업이 끝날 때마다 골든 비교를 돌립니다. **숫자가 바뀌면 안 되는 작업에서 바뀌었다면 그 자리에서 원인을 찾습니다.** 다음 단계로 넘어가지 마십시오.

숫자가 바뀌어야 정상인 작업에서는 바뀐 항목이 **예상한 것뿐인지** 확인하고 골든을 갱신합니다. fixture는 갱신하지 않습니다.

## 5-7. 기능 추가 판단 기준

> **"이 기능이 새로운 현업 업무단계를 프로그램 안으로 가져오는가?"**

| 기능 | 가져오는 업무단계 |
|---|---|
| Exposure Dashboard | 포트폴리오를 이해하는 업무 |
| Mandate | 운용목표를 정의하는 업무 |
| Feasibility | 현실적인 배분안을 만드는 업무 |
| Transition | 실행계획을 만드는 업무 |
| Adopted Portfolio | 결정을 기록하고 관리하는 업무 |
| Policy Benchmark + Brinson | 사후 평가하는 업무 |
| Index/Market 연결 | 아이디어를 검증으로 전달하는 업무 |
| Investment Memo | 결론을 업무문서로 만드는 업무 |

반대로 위험지표 하나 더, 최적화 알고리즘 하나 더는 새로운 업무단계를 만들지 않으므로 추가하지 않습니다.

## 5-8. 없는 정밀도를 만들지 않기

이 계획에서 특히 적용되는 곳:
- ADV 비율에서 시장충격비용(bp)을 도출하지 않음
- Robustness 종합점수를 만들지 않음
- Investment Memo를 자유 생성하지 않음
- Asset Master에서 모르는 속성을 추정으로 채우지 않음
- R² 0.10을 절대 기준으로 하드코딩하지 않음 (heuristic임을 명시)
- 자동추정 속성에 기반한 노출은 그 사실을 함께 표시
