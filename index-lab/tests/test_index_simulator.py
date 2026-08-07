import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.index_simulator import simulate_index


def _prices_ab():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    return pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0]},
                        index=idx)


def test_single_period_buy_and_hold_matches_hand_calc():
    px = _prices_ab()
    out = simulate_index(px, {px.index[0]: {"A": 0.5, "B": 0.5}},
                         cost_bp=0.0, base_value=1000.0)
    level = out["level"]
    # A: +10%/일, B: 0% -> 포트폴리오 가치 = 0.5*growth_A + 0.5*growth_B
    growth = px[["A", "B"]] / px[["A", "B"]].iloc[0]
    expected = 1000.0 * (0.5 * growth["A"] + 0.5 * growth["B"])
    expected = expected.iloc[1:]  # 첫날(기준일 자체)은 level에 없음
    assert list(level.index) == list(expected.index)
    assert np.allclose(level.values, expected.values, rtol=1e-9)


def test_first_rebalance_turnover_equals_full_buy():
    px = _prices_ab()
    out = simulate_index(px, {px.index[0]: {"A": 0.5, "B": 0.5}}, cost_bp=0.0)
    assert out["turnover"].iloc[0]["회전율"] == pytest.approx(1.0)
    assert out["turnover"].iloc[0]["단방향회전율"] == pytest.approx(0.5)


def test_cost_reduces_level_relative_to_zero_cost():
    px = _prices_ab()
    free = simulate_index(px, {px.index[0]: {"A": 0.5, "B": 0.5}}, cost_bp=0.0)
    costly = simulate_index(px, {px.index[0]: {"A": 0.5, "B": 0.5}}, cost_bp=100.0)
    assert costly["level"].iloc[-1] < free["level"].iloc[-1]
    # 회전율 100%에 100bp면 첫날 비용은 정확히 1%
    day1_free = free["level"].iloc[0]
    day1_costly = costly["level"].iloc[0]
    assert day1_costly / day1_free == pytest.approx(0.99, abs=1e-9)


def test_missing_ticker_raises():
    px = _prices_ab()
    with pytest.raises(ValueError):
        simulate_index(px, {px.index[0]: {"A": 0.5, "Z": 0.5}})


def test_rebalance_turnover_reflects_actual_drift_not_full_rebuild():
    # 2구간: 첫 구간 A/B 50:50, 둘째 구간에서 다시 정확히 같은 목표비중으로
    # '재조정'하면, 이미 그 근처로 드리프트돼 있으니 회전율이 100%가 아니라
    # 드리프트만큼만 나와야 한다 (Portfolio Analyzer의 워크포워드와 같은 원칙).
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    a = [100, 110, 121, 121, 133.1, 146.41]
    b = [100, 100, 100, 100, 100, 100]
    px = pd.DataFrame({"A": a, "B": b}, index=idx)
    d0, d1 = idx[0], idx[3]
    out = simulate_index(px, {d0: {"A": 0.5, "B": 0.5}, d1: {"A": 0.5, "B": 0.5}},
                         cost_bp=0.0)
    tno_row1 = out["turnover"].iloc[1]["회전율"]
    assert 0 < tno_row1 < 1.0, (
        "두 번째 정기변경의 회전율이 100%로 나오면 안 됩니다 — "
        "직전 구간 종료 시점의 드리프트된 비중을 넘겨받지 못하고 있을 가능성이 큽니다."
    )


def test_target_weights_do_not_need_exact_keys_order():
    px = _prices_ab()
    out1 = simulate_index(px, {px.index[0]: {"A": 0.5, "B": 0.5}})
    out2 = simulate_index(px, {px.index[0]: {"B": 0.5, "A": 0.5}})
    pd.testing.assert_series_equal(out1["level"], out2["level"], check_names=False)
