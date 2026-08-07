"""
구성종목 선정 + 버퍼룰.

버퍼룰은 순위가 살짝 흔들릴 때마다 종목을 교체하는 걸 막기 위한 규칙이다.
신규 편입은 더 엄격한 기준(buffer_in)을, 기존 보유 종목은 더 느슨한 기준
(buffer_out)을 적용해 경계선 근처의 불필요한 회전율을 줄인다.

인원이 넘칠 때는 '유지(보유 종목)' 쪽을 먼저 보호하고 '신규편입' 쪽부터
자른다 — 버퍼룰의 목적 자체가 보유 종목을 함부로 안 바꾸는 것이므로,
반대로 하면 버퍼룰이 무의미해진다.
"""
import pandas as pd


def select_constituents(scored_df: pd.DataFrame, *, target_n: int,
                        ticker_col="ticker", rank_col="순위",
                        buffer_in: int = None, buffer_out: int = None,
                        prev_constituents=None):
    """
    scored_df: scoring.compute_score() 결과 (순위 열 포함, 낮을수록 좋음).
    buffer_in: 신규 편입 커트라인 (예: 25위 이내만 신규 편입 가능).
    buffer_out: 기존 보유 유지 커트라인 (예: 35위 이내면 보유 유지 가능).
               둘 다 None 이면 buffer_in=buffer_out=target_n (버퍼 없음).
    prev_constituents: 직전 구성종목 티커 집합/리스트. None 이면 최초 구성.

    반환: DataFrame[티커, 순위, 종합점수, 상태, 선정근거] (target_n 행 이하)
    """
    buffer_in = target_n if buffer_in is None else buffer_in
    buffer_out = target_n if buffer_out is None else buffer_out
    if buffer_in > buffer_out:
        raise ValueError("buffer_in(신규 편입 기준)은 buffer_out(유지 기준)보다 "
                         "작거나 같아야 합니다 — 신규 편입 문턱이 유지 문턱보다 "
                         "느슨하면 규칙이 모순됩니다.")

    prev = set(prev_constituents or [])
    universe_tk = set(scored_df[ticker_col].astype(str))
    ranked = scored_df.set_index(scored_df[ticker_col].astype(str))[rank_col]

    retained_pool = sorted(
        (prev & universe_tk) &
        set(scored_df.loc[scored_df[rank_col] <= buffer_out, ticker_col].astype(str)),
        key=lambda t: ranked[t])
    new_pool = sorted(
        set(scored_df.loc[scored_df[rank_col] <= buffer_in, ticker_col].astype(str)) - prev,
        key=lambda t: ranked[t])

    # 유지 후보를 먼저 채우고(넘치면 유지 쪽도 순위 나쁜 것부터 자른다),
    # 남는 자리만 신규편입 후보로 채운다.
    if len(retained_pool) >= target_n:
        keep = set(retained_pool[:target_n])
        new_add = set()
    else:
        keep = set(retained_pool)
        slots = target_n - len(keep)
        new_add = set(new_pool[:slots])

    pool = keep | new_add
    if len(pool) < target_n:
        remaining = sorted(universe_tk - pool, key=lambda t: ranked[t])
        pool |= set(remaining[:target_n - len(pool)])

    rows = []
    for t in universe_tk | prev:
        rank = int(ranked[t]) if t in ranked.index else None
        in_pool = t in pool
        was_held = t in prev
        if in_pool and not was_held:
            status = "신규편입"
            reason = (f"순위 {rank}위 (신규 편입 기준 {buffer_in}위 이내)"
                      if rank is not None and rank <= buffer_in
                      else f"순위 {rank}위 (충원)")
        elif in_pool and was_held:
            status = "유지"
            buffered = rank is not None and buffer_in < rank <= buffer_out
            reason = f"순위 {rank}위" + (" (버퍼룰로 유지)" if buffered else "")
        elif was_held and t not in universe_tk:
            status = "조건미달편출"
            reason = "적격성 필터를 통과하지 못함"
        elif was_held:
            status = "편출"
            reason = f"순위 {rank}위 — 유지 기준({buffer_out}위) 밖"
        else:
            continue
        score = float(scored_df.loc[scored_df[ticker_col].astype(str) == t,
                                    "종합점수"].iloc[0]) if t in universe_tk else None
        rows.append({"티커": t, "순위": rank, "종합점수": score,
                    "상태": status, "선정근거": reason})

    out = pd.DataFrame(rows)
    order = {"신규편입": 0, "유지": 1, "편출": 2, "조건미달편출": 3}
    out["_ord"] = out["상태"].map(order)
    out = out.sort_values(["_ord", "순위"], na_position="last").drop(columns="_ord")
    return out.reset_index(drop=True)


def current_constituents(selection_df: pd.DataFrame) -> list:
    """select_constituents() 결과에서 실제 편입 상태(신규편입/유지)인 티커만."""
    return selection_df.loc[selection_df["상태"].isin(["신규편입", "유지"]),
                            "티커"].tolist()
