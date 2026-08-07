"""
동적 지수 시뮬레이터.

Portfolio Analyzer의 정적 비중 계산(portfolio_returns)을 그대로 쓰지 않는
이유는, 인덱스는 정기변경마다 구성종목·비중이 바뀌기 때문이다. 대신 각
정기변경 구간을 '그 구간 동안은 매수 후 보유(드리프트)'로 계산하고,
구간 경계마다 실제 교체비중(직전 구간 종료 시점의 드리프트된 비중 대비
신규 목표비중의 차이)만 거래비용으로 반영한다 — Portfolio Analyzer의
워크포워드에서 검증된 것과 같은 원칙이다: 매 구간을 100% 재구축으로
계산하면 비용이 부풀려진다.
"""
import numpy as np
import pandas as pd


def _segment_returns(seg_px: pd.DataFrame, w: dict):
    """구간 안에서 목표비중으로 매수 후 보유했을 때의 일별 수익률과, 구간 끝
    시점의 드리프트된(실제) 비중을 함께 돌려준다. 데이터가 부족하면 빈 결과."""
    tickers = list(w)
    px = seg_px[tickers].dropna()
    if len(px) < 2:
        return pd.Series(dtype=float), {}
    w_arr = np.array([w[t] for t in tickers])
    daily = px.pct_change().dropna()
    if daily.empty:
        return pd.Series(dtype=float), {}
    growth = (1 + daily).cumprod()
    first = pd.DataFrame([np.ones(len(tickers))], columns=tickers,
                         index=[daily.index[0] - pd.Timedelta(days=1)])
    growth = pd.concat([first, growth])
    port_val = (growth * w_arr).sum(axis=1)
    port_ret = port_val.pct_change().dropna()
    end_w = (growth.iloc[-1] * w_arr) / port_val.iloc[-1]
    return port_ret, end_w.to_dict()


def simulate_index(prices: pd.DataFrame, target_weights_by_date: dict, *,
                   cost_bp: float = 0.0, base_value: float = 1000.0) -> dict:
    """
    prices: 날짜 × 티커 가격 DataFrame.
    target_weights_by_date: {정기변경일: {티커: 목표비중}}. 비중 합은 대략 1이어야
        한다(weighting.apply_caps 결과를 그대로 넣으면 됨).
    반환: {"level": 지수레벨 Series, "turnover": 정기변경일별 회전율 DataFrame,
           "constituent_history": 정기변경일별 목표비중 이력 DataFrame}
    """
    twd = {pd.Timestamp(d): {t: float(w) for t, w in wts.items() if w}
           for d, wts in target_weights_by_date.items()}
    dates = sorted(twd)
    if not dates:
        raise ValueError("정기변경 목표비중이 없습니다.")

    all_tk = sorted({t for wts in twd.values() for t in wts})
    missing_tk = [t for t in all_tk if t not in prices.columns]
    if missing_tk:
        raise ValueError(f"가격 데이터가 없는 종목: {', '.join(missing_tk)}")
    px = prices[all_tk].sort_index()

    turnover_rows, hist_rows = [], []
    level_index, level_vals = [], []
    prev_end_w = {}
    level_val = float(base_value)

    for i, d in enumerate(dates):
        target = twd[d]
        seg_end = dates[i + 1] if i + 1 < len(dates) else px.index[-1]
        seg = px.loc[d:seg_end]

        seg_ret, end_w = _segment_returns(seg, target)
        if seg_ret.empty:
            continue

        turnover = sum(abs(target.get(t, 0.0) - prev_end_w.get(t, 0.0))
                       for t in set(target) | set(prev_end_w))
        turnover_rows.append({"정기변경일": d, "회전율": turnover,
                              "단방향회전율": turnover / 2})
        for t, w in target.items():
            hist_rows.append({"정기변경일": d, "티커": t, "목표비중": w})

        cost = turnover * (cost_bp / 10000.0)
        seg_ret = seg_ret.copy()
        seg_ret.iloc[0] = (1 + seg_ret.iloc[0]) * (1 - cost) - 1

        for dt, r in seg_ret.items():
            level_val = level_val * (1 + r)
            level_index.append(dt)
            level_vals.append(level_val)

        prev_end_w = end_w

    if not level_vals:
        raise ValueError("시뮬레이션 결과가 비어 있습니다 — 가격 데이터 기간을 확인해주세요.")

    level = pd.Series(level_vals, index=pd.DatetimeIndex(level_index)).sort_index()
    level = level[~level.index.duplicated(keep="first")]
    return {"level": level,
           "turnover": pd.DataFrame(turnover_rows),
           "constituent_history": pd.DataFrame(hist_rows)}
