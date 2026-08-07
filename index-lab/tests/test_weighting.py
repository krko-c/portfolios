import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.weighting import compute_weights, apply_caps, _iterative_cap


def _df():
    return pd.DataFrame({
        "티커": ["A", "B", "C", "D"],
        "시가총액": [1000.0, 300.0, 200.0, 500.0],
        "점수": [2.0, -1.0, 0.5, 1.0],
        "변동성": [0.10, 0.40, 0.20, 0.05],
        "섹터": ["Tech", "Tech", "Health", "Energy"],
    })


def test_equal_weight():
    w = compute_weights(_df(), scheme="동일가중")
    assert np.allclose(w.values, 0.25)
    assert abs(w.sum() - 1.0) < 1e-12


def test_market_cap_weight_proportional():
    w = compute_weights(_df(), scheme="시가총액가중", mktcap_col="시가총액")
    assert abs(w["A"] - 1000 / 2000) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-9


def test_score_weight_handles_negative_scores():
    w = compute_weights(_df(), scheme="점수가중", score_col="점수")
    assert (w > 0).all()  # 원점수가 음수여도 가중치는 전부 양수여야 함
    assert w["A"] > w["B"]  # 점수 2.0 > -1.0 이므로 비중도 더 커야 함
    assert abs(w.sum() - 1.0) < 1e-9


def test_inverse_vol_weight_favors_low_vol():
    w = compute_weights(_df(), scheme="역변동성가중", vol_col="변동성")
    assert w["D"] > w["B"]  # D 변동성 0.05 < B 변동성 0.40


def test_market_cap_weight_rejects_missing():
    df = _df()
    df.loc[0, "시가총액"] = np.nan
    with pytest.raises(ValueError):
        compute_weights(df, scheme="시가총액가중", mktcap_col="시가총액")


def test_iterative_cap_basic():
    w = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})
    capped = _iterative_cap(w, cap=0.4)
    assert capped["A"] == pytest.approx(0.4)
    assert abs(capped.sum() - 1.0) < 1e-9
    assert (capped <= 0.4 + 1e-9).all()


def test_iterative_cap_cascading_overflow():
    # A를 40%로 자르면 B가 그 초과분을 받아 다시 40%를 넘는 경우
    w = pd.Series({"A": 0.5, "B": 0.35, "C": 0.15})
    capped = _iterative_cap(w, cap=0.4)
    assert (capped <= 0.4 + 1e-9).all()
    assert abs(capped.sum() - 1.0) < 1e-9
    # C가 A, B 초과분을 다 받아서 나머지를 채워야 함
    assert capped["C"] == pytest.approx(0.2, abs=1e-6)


def test_iterative_cap_infeasible_raises():
    w = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
    with pytest.raises(ValueError):
        _iterative_cap(w, cap=0.2)  # 0.2 * 3 = 0.6 < 1.0, 불가능


def test_apply_caps_no_op_when_within_cap():
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    out = apply_caps(w, cap_per_stock=0.5)
    assert np.allclose(out.values, w.values)


def test_apply_caps_group_cap():
    w = pd.Series({"A": 0.4, "B": 0.35, "C": 0.15, "D": 0.10})
    group = pd.Series({"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"})
    out = apply_caps(w, cap_per_stock=1.0, group=group, cap_per_group=0.5)
    tech_sum = out[["A", "B"]].sum()
    assert tech_sum <= 0.5 + 1e-6
    assert abs(out.sum() - 1.0) < 1e-9


def test_blend_scheme_between_mktcap_and_score():
    df = _df()
    w_mc = compute_weights(df, scheme="시가총액가중", mktcap_col="시가총액")
    w_blend = compute_weights(df, scheme="시총점수혼합", mktcap_col="시가총액",
                              score_col="점수", blend_alpha=1.0)
    assert np.allclose(w_mc.values, w_blend.values, atol=1e-9)
