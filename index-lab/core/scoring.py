"""
종합점수 계산. 필수 적격성 조건(filters.py)과 순위 매기기는 분리한다 —
점수가 낮다고 탈락시키는 게 아니라, 통과한 종목 안에서 순서만 매긴다.
"""
import numpy as np
import pandas as pd


def _standardize(s: pd.Series, method: str, clip=None) -> pd.Series:
    """method: 'zscore' | 'percentile' | 'raw'. 값이 클수록 좋다는 방향으로 맞춘 뒤 호출."""
    if clip is not None:
        s = s.clip(lower=clip[0], upper=clip[1])
    if method == "raw":
        out = s
    elif method == "percentile":
        out = s.rank(pct=True) * 100
    elif method == "zscore":
        std = s.std()
        out = (s - s.mean()) / std if std and np.isfinite(std) and std > 1e-12 \
            else pd.Series(0.0, index=s.index)
    else:
        raise ValueError(f"알 수 없는 점수화 방식: {method}")
    return out


def compute_score(df: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """
    factors: {컬럼명: {"weight": float, "direction": 1|-1, "method": "zscore"|
              "percentile"|"raw", "clip": (lo, hi) 선택, "fill": 결측 대체값 선택}}
    direction=-1 이면 값이 작을수록 유리(예: 변동성)하다는 뜻으로 부호를 뒤집는다.
    결측값은 fill(기본 0, zscore/percentile 기준으로는 '평균'을 의미)로 채운다.
    반환: 원본 df에 '{컬럼}_점수' 열들과 '종합점수' 열을 더한 DataFrame.
    """
    out = df.copy()
    total_w = sum(f["weight"] for f in factors.values())
    if total_w <= 0:
        raise ValueError("요인 가중치 합이 0 이하입니다.")

    score = pd.Series(0.0, index=out.index)
    for col, cfg in factors.items():
        if col not in out.columns:
            raise ValueError(f"'{col}' 컬럼이 없습니다.")
        raw = pd.to_numeric(out[col], errors="coerce")
        n_missing = int(raw.isna().sum())
        std = _standardize(raw, cfg.get("method", "zscore"), cfg.get("clip"))
        fill = cfg.get("fill", 0.0)
        std = std.fillna(fill)
        std = std * cfg.get("direction", 1)
        out[f"{col}_점수"] = std
        out[f"{col}_결측"] = n_missing
        score = score + std * (cfg["weight"] / total_w)

    out["종합점수"] = score
    out["순위"] = out["종합점수"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("종합점수", ascending=False).reset_index(drop=True)
