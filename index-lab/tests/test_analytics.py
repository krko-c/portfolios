import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import analytics as an


def test_cagr_known_case():
    # 1.0 -> 1.21, 정확히 2년(365.25일*2), CAGR = 10%
    start = pd.Timestamp("2020-01-01")
    idx = pd.DatetimeIndex([start, start + pd.Timedelta(days=365.25 * 2)])
    level = pd.Series([1.0, 1.21], index=idx)
    assert an.cagr(level) == pytest.approx(0.10, abs=1e-3)


def test_max_drawdown_known_case():
    level = pd.Series([100, 120, 90, 110],
                      index=pd.date_range("2020-01-01", periods=4))
    # 고점 120 -> 90 = -25%
    assert an.max_drawdown(level) == pytest.approx(-0.25, abs=1e-9)


def test_sortino_downside_deviation_is_sensitive_to_sample_size():
    # Portfolio Analyzer 세션에서 발견된 버그의 재발 방지 테스트.
    # 버그였던 e[e<0].std()는 '나쁜 날들의 부분집합 표준편차'라서, 나쁜 날 2개가
    # 전체 5일 중이든 50일 중이든 분모(d)가 완전히 똑같이 나온다(표본 크기에
    # 둔감). 올바른 공식(전체 표본 기준 RMS 하방편차)은 표본이 늘면 d가
    # 작아져야 한다 — 나쁜 날의 '비중'이 줄었으니 하방위험도 줄어야 하므로.
    bad_days = np.array([-0.01, -0.02])
    r_small = pd.Series(np.concatenate([bad_days, [0.02, 0.03, 0.01]]))
    r_large = pd.Series(np.concatenate([bad_days, np.full(48, 0.001)]))

    def _buggy_subset_std(r):
        e = r - 0.0
        return e[e < 0].std()

    def _correct_downside_dev(r):
        e = r - 0.0
        return np.sqrt(np.mean(np.minimum(e.values, 0.0) ** 2))

    assert _buggy_subset_std(r_small) == pytest.approx(
        _buggy_subset_std(r_large), rel=1e-9), "버그 재현 실패 — 비교 기준이 잘못됨"

    d_small = _correct_downside_dev(r_small)
    d_large = _correct_downside_dev(r_large)
    assert d_large < d_small * 0.5, (
        "하방편차가 표본 크기에 둔감합니다 — "
        "e[e<0].std() 버그가 재발했을 가능성이 있습니다.")


def test_sortino_matches_hand_computed_value():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    d_correct = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2))
    expected = r.mean() / d_correct * np.sqrt(an.TRADING_DAYS)
    assert an.sortino_ratio(r) == pytest.approx(expected, rel=1e-9)


def test_relative_wealth_matches_ratio_of_levels_not_naive_difference():
    idx = pd.date_range("2020-01-01", periods=5)
    port = pd.Series([100, 102, 105, 103, 108], index=idx, dtype=float)
    bench = pd.Series([100, 101, 100, 102, 104], index=idx, dtype=float)
    rw = an.relative_wealth(port, bench)
    expected_last = (108 / 100) / (104 / 100) - 1
    assert rw.iloc[-1] == pytest.approx(expected_last, abs=1e-9)

    # 일별 차이를 그냥 복리로 굴리는(틀린) 방식과는 달라야 한다
    pr, br = an.to_returns(port), an.to_returns(bench)
    naive = (1 + (pr - br)).cumprod() - 1
    assert rw.iloc[-1] != pytest.approx(naive.iloc[-1], abs=1e-6)


def test_excess_return_is_geometric_not_arithmetic_difference():
    # 영업일(주 5일) 빈도로 만들어야 cagr()의 달력일 기준 연율화와
    # excess_return() 의 거래일수(252) 기준 연율화가 서로 맞아떨어진다 —
    # 실제 가격 데이터(영업일 간격)를 쓸 때가 정확히 이 경우다.
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    port = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 300))), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, 300))), index=idx)
    ex = an.excess_return(port, bench)
    ann_p = an.cagr(port)
    ann_b = an.cagr(bench)
    # 두 함수의 연율화 기준(달력일 vs 거래일수)이 정확히 같은 공식은 아니라서
    # 완전히 일치하진 않지만, 영업일 데이터에서는 근사적으로 맞아야 한다.
    assert ex == pytest.approx(ann_p - ann_b, rel=0.06)
