import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.scoring import compute_score


def _df():
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "growth": [10.0, 20.0, 30.0, 40.0],
        "vol": [0.30, 0.10, 0.20, 0.40],  # 낮을수록 좋음 -> direction=-1
    })


def test_single_factor_percentile_rank_order():
    out = compute_score(_df(), {"growth": {"weight": 1.0, "direction": 1,
                                            "method": "percentile"}})
    assert list(out["ticker"]) == ["D", "C", "B", "A"]  # growth 높은 순
    assert out["순위"].tolist() == [1, 2, 3, 4]


def test_direction_minus_one_flips_order():
    out = compute_score(_df(), {"vol": {"weight": 1.0, "direction": -1,
                                        "method": "percentile"}})
    assert out.iloc[0]["ticker"] == "B"  # vol 가장 낮은 게 1등이어야 함


def test_weights_normalize_even_if_not_summing_to_one():
    f = {"growth": {"weight": 2.0, "direction": 1, "method": "zscore"},
        "vol": {"weight": 2.0, "direction": -1, "method": "zscore"}}
    out = compute_score(_df(), f)
    # 가중치가 2+2=4로 정규화 안 된 상태로 들어와도 최종 종합점수 스케일은
    # 정상 범위(대략 -1~1 zscore 합)여야 한다
    assert out["종합점수"].abs().max() < 3


def test_zero_total_weight_raises():
    with pytest.raises(ValueError):
        compute_score(_df(), {"growth": {"weight": 0.0, "direction": 1}})


def test_missing_column_raises():
    with pytest.raises(ValueError):
        compute_score(_df(), {"nope": {"weight": 1.0, "direction": 1}})


def test_missing_value_filled_with_default():
    df = _df()
    df.loc[0, "growth"] = np.nan
    out = compute_score(df, {"growth": {"weight": 1.0, "direction": 1,
                                        "method": "zscore", "fill": 0.0}})
    a_row = out.loc[out["ticker"] == "A"].iloc[0]
    assert a_row["growth_점수"] == 0.0
    assert a_row["growth_결측"] == 1
