import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.selection import select_constituents, current_constituents


def _scored(n=10):
    # 티커 T1..Tn, 순위 1..n (점수는 순위와 반비례)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(1, n + 1)],
        "종합점수": [float(n - i) for i in range(1, n + 1)],
        "순위": list(range(1, n + 1)),
    })


def test_first_time_selection_is_top_n_and_all_new():
    sel = select_constituents(_scored(10), target_n=5)
    assert set(sel["티커"]) == {"T1", "T2", "T3", "T4", "T5"}
    assert (sel["상태"] == "신규편입").all()


def test_buffer_keeps_previously_held_name_that_slipped_slightly():
    # T6가 이전에 편입돼 있었고, 지금 순위 6위(target_n=5, buffer_out=7)
    prev = ["T1", "T2", "T3", "T4", "T6"]
    sel = select_constituents(_scored(10), target_n=5, buffer_in=5,
                              buffer_out=7, prev_constituents=prev)
    row = sel.loc[sel["티커"] == "T6"].iloc[0]
    assert row["상태"] == "유지"
    assert "버퍼" in row["선정근거"]


def test_new_name_needs_stricter_buffer_in_even_if_within_buffer_out():
    # T6은 이전에 없었고 순위 6위 -> buffer_in=5 를 못 넘으면 신규 편입 불가
    # 대신 다른 자리가 비어 충원될 수는 있음 (여기선 자리가 안 빔)
    prev = ["T1", "T2", "T3", "T4", "T5"]
    sel = select_constituents(_scored(10), target_n=5, buffer_in=5,
                              buffer_out=7, prev_constituents=prev)
    assert set(sel.loc[sel["상태"].isin(["신규편입", "유지"]), "티커"]) == \
        {"T1", "T2", "T3", "T4", "T5"}


def test_dropped_out_name_is_marked_exit():
    prev = ["T1", "T2", "T3", "T4", "T9"]  # T9는 9위, buffer_out(7) 밖
    sel = select_constituents(_scored(10), target_n=5, buffer_in=5,
                              buffer_out=7, prev_constituents=prev)
    row = sel.loc[sel["티커"] == "T9"].iloc[0]
    assert row["상태"] == "편출"


def test_name_failing_filter_is_condition_exit():
    scored = _scored(10)
    scored = scored[scored["ticker"] != "T3"]  # T3 가 적격성 필터에서 탈락했다고 가정
    prev = ["T1", "T2", "T3", "T4", "T5"]
    sel = select_constituents(scored, target_n=5, prev_constituents=prev)
    row = sel.loc[sel["티커"] == "T3"].iloc[0]
    assert row["상태"] == "조건미달편출"


def test_pool_shortfall_is_filled_by_rank():
    # target_n=5, buffer_in=2 라 신규편입 후보가 2개뿐 -> 나머지는 순위순 충원
    sel = select_constituents(_scored(10), target_n=5, buffer_in=2, buffer_out=2)
    picked = current_constituents(sel)
    assert len(picked) == 5
    assert set(picked) == {"T1", "T2", "T3", "T4", "T5"}


def test_invalid_buffer_order_raises():
    import pytest
    with pytest.raises(ValueError):
        select_constituents(_scored(10), target_n=5, buffer_in=10, buffer_out=5)
