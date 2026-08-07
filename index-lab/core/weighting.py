"""
비중 설계. 방법론은 사전에 정한 결정론적 규칙만 쓴다(최적화 solver 없음) —
같은 입력에는 항상 같은 비중이 나와야 인덱스 방법론으로 쓸 수 있다.
"""
import pandas as pd


def compute_weights(df: pd.DataFrame, *, ticker_col="티커", scheme: str,
                    score_col=None, mktcap_col=None, vol_col=None,
                    blend_alpha: float = 0.5) -> pd.Series:
    """
    scheme: "동일가중" | "시가총액가중" | "유동시가총액가중" | "점수가중" |
            "역변동성가중" | "시총점수혼합"
    반환: 티커를 인덱스로 하는 비중 Series (합계 1.0).
    """
    tickers = df[ticker_col].astype(str)
    n = len(df)
    if n == 0:
        raise ValueError("종목이 없습니다.")

    if scheme == "동일가중":
        w = pd.Series(1.0 / n, index=tickers)

    elif scheme in ("시가총액가중", "유동시가총액가중"):
        if not mktcap_col or mktcap_col not in df.columns:
            raise ValueError(f"'{scheme}'에는 시가총액 컬럼이 필요합니다.")
        mc = pd.to_numeric(df[mktcap_col], errors="coerce")
        if mc.isna().any() or (mc <= 0).any():
            raise ValueError("시가총액에 결측 또는 0 이하 값이 있어 계산할 수 없습니다.")
        w = pd.Series(mc.values, index=tickers)
        w = w / w.sum()

    elif scheme == "점수가중":
        if not score_col or score_col not in df.columns:
            raise ValueError("'점수가중'에는 점수 컬럼이 필요합니다.")
        s = pd.to_numeric(df[score_col], errors="coerce")
        if s.isna().any():
            raise ValueError("점수에 결측값이 있어 계산할 수 없습니다.")
        # 점수(z-score 등)는 음수일 수 있어, 가중치로 쓰려면 전부 양수로 이동한다.
        s = s - min(0.0, float(s.min())) + 1e-6
        if s.sum() <= 0:
            raise ValueError("점수 합이 0 이하라 가중치를 만들 수 없습니다.")
        w = pd.Series(s.values, index=tickers)
        w = w / w.sum()

    elif scheme == "역변동성가중":
        if not vol_col or vol_col not in df.columns:
            raise ValueError("'역변동성가중'에는 변동성 컬럼이 필요합니다.")
        v = pd.to_numeric(df[vol_col], errors="coerce")
        if v.isna().any() or (v <= 0).any():
            raise ValueError("변동성에 결측 또는 0 이하 값이 있어 계산할 수 없습니다.")
        w = pd.Series(1.0 / v.values, index=tickers)
        w = w / w.sum()

    elif scheme == "시총점수혼합":
        w_mc = compute_weights(df, ticker_col=ticker_col, scheme="시가총액가중",
                               mktcap_col=mktcap_col)
        w_sc = compute_weights(df, ticker_col=ticker_col, scheme="점수가중",
                               score_col=score_col)
        w = blend_alpha * w_mc + (1 - blend_alpha) * w_sc
        w = w / w.sum()

    else:
        raise ValueError(f"알 수 없는 비중 방식: {scheme}")

    return w


def _iterative_cap(w: pd.Series, cap: float, max_iter=50, tol=1e-9) -> pd.Series:
    """단일 상한을 반복 재배분(iterative proportional capping)으로 만족시킨다."""
    w = w.copy().astype(float)
    if cap * len(w) < 1.0 - tol:
        raise ValueError(f"상한({cap:.2%}) × 종목수({len(w)})가 100% 미만이라 "
                         f"이 상한으로는 비중 합계 100%를 만들 수 없습니다. "
                         f"상한을 완화하거나 종목 수를 늘려주세요.")
    fixed = pd.Series(False, index=w.index)
    for _ in range(max_iter):
        over = (~fixed) & (w > cap + tol)
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        fixed |= over
        room = w[~fixed]
        room_sum = float(room.sum())
        if room_sum <= tol:
            raise ValueError("상한 재배분 중 받아줄 종목이 남지 않았습니다 — "
                             "상한을 완화해주세요.")
        w[~fixed] = room + (room / room_sum) * excess
    else:
        raise ValueError(f"{max_iter}회 반복해도 상한을 만족하지 못했습니다.")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError("재배분 후 비중 합계가 100%가 아닙니다 (내부 오류).")
    return w


def apply_caps(w: pd.Series, *, cap_per_stock: float = 1.0, group: pd.Series = None,
               cap_per_group: float = None, max_iter=50, tol=1e-9) -> pd.Series:
    """
    종목별 상한(cap_per_stock)과, 선택적으로 그룹별(산업·국가 등) 상한을
    함께 만족시킨다. 단순히 상한을 잘라내고 정규화하면 다시 상한을
    초과할 수 있어, 초과분을 규칙대로 반복 재배분한다.

    그룹 상한은 종목 상한과 번갈아 적용하는 근사(alternating projection)다 —
    두 제약을 동시에 만족하는 정확한 최적해를 구하는 것은 아니며, 대부분의
    실무 상황에서는 수렴하지만 극단적인 조합에서는 안 될 수 있다(그 경우
    ValueError).
    """
    w = _iterative_cap(w, cap_per_stock, max_iter=max_iter, tol=tol)
    if group is None or cap_per_group is None:
        return w

    group = group.reindex(w.index)
    for _ in range(max_iter):
        gsum = w.groupby(group).sum()
        over_groups = gsum[gsum > cap_per_group + tol].index
        if len(over_groups) == 0:
            break
        for g in over_groups:
            idx = group[group == g].index
            g_total = float(w[idx].sum())
            excess = g_total - cap_per_group
            w[idx] = w[idx] * (cap_per_group / g_total)
            other_idx = w.index.difference(idx)
            other_sum = float(w[other_idx].sum())
            if other_sum <= tol:
                raise ValueError("산업/국가 상한 재배분 중 받아줄 다른 종목이 "
                                 "남지 않았습니다 — 상한을 완화해주세요.")
            w[other_idx] = w[other_idx] + (w[other_idx] / other_sum) * excess
        w = _iterative_cap(w, cap_per_stock, max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"{max_iter}회 반복해도 그룹 상한을 만족하지 못했습니다.")
    return w
