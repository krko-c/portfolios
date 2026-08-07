"""
지수 성과지표. Portfolio Analyzer 세션에서 검증·수정된 공식을 그대로
가져온다 — 특히 소르티노는 하방편차를 '0(목표) 대비 편차를 전체 표본
수로 나눈 값'으로 계산한다(부분집합끼리의 표준편차 아님. 그 실수가
소르티노를 최대 41% 과대평가했던 걸 이 세션에서 발견·수정했다).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
_EPS_STD = 1e-12


def to_returns(level: pd.Series) -> pd.Series:
    return level.pct_change().dropna()


def cagr(level: pd.Series) -> float:
    if len(level) < 2:
        return np.nan
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    return float(level.iloc[-1] / level.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def drawdown_series(level: pd.Series) -> pd.Series:
    return level / level.cummax() - 1.0


def max_drawdown(level: pd.Series) -> float:
    return float(drawdown_series(level).min())


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    e = returns - rf / TRADING_DAYS
    s = e.std()
    if not np.isfinite(s) or s < _EPS_STD:
        return np.nan
    return float(e.mean() / s * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    e = returns - rf / TRADING_DAYS
    d = np.sqrt(np.mean(np.minimum(e.values, 0.0) ** 2))
    if not np.isfinite(d) or d < _EPS_STD:
        return np.nan
    return float(e.mean() / d * np.sqrt(TRADING_DAYS))


def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    act = df.iloc[:, 0] - df.iloc[:, 1]
    return float(act.std() * np.sqrt(TRADING_DAYS))


def excess_return(port_level: pd.Series, bench_level: pd.Series) -> float:
    """연율화 초과수익 = 각자 기하 연율화한 뒤의 차이(일별 차이를 단순 합산하지 않음)."""
    df = pd.concat([to_returns(port_level), to_returns(bench_level)], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    ann_a = (1 + a).prod() ** (TRADING_DAYS / len(a)) - 1
    ann_b = (1 + b).prod() ** (TRADING_DAYS / len(b)) - 1
    return float(ann_a - ann_b)


def information_ratio(port_level: pd.Series, bench_level: pd.Series) -> float:
    te = tracking_error(to_returns(port_level), to_returns(bench_level))
    ex = excess_return(port_level, bench_level)
    return float(ex / te) if te and np.isfinite(te) and te > 1e-12 else np.nan


def relative_wealth(port_level: pd.Series, bench_level: pd.Series) -> pd.Series:
    """누적 초과수익 — 상대 부(富) 기준. 일별 차이를 그냥 복리로 굴리는 방식과
    다르다(변동성이 있으면 둘은 다른 값이 된다)."""
    df = pd.concat([port_level / port_level.iloc[0],
                    bench_level / bench_level.iloc[0]], axis=1).dropna()
    return df.iloc[:, 0] / df.iloc[:, 1] - 1.0


def summary_table(level: pd.Series, rf: float = 0.0, bench_level: pd.Series = None) -> dict:
    r = to_returns(level)
    out = {
        "CAGR": cagr(level), "연변동성": annual_vol(r), "MDD": max_drawdown(level),
        "샤프": sharpe_ratio(r, rf), "소르티노": sortino_ratio(r, rf),
    }
    if bench_level is not None:
        out["초과수익(연)"] = excess_return(level, bench_level)
        out["추적오차"] = tracking_error(r, to_returns(bench_level))
        out["정보비율"] = information_ratio(level, bench_level)
    return out
