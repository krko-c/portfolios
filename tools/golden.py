#!/usr/bin/env python3
"""
골든 회귀 테스트 (docs/PLAN.md 0-3).

tests/fixtures/ 의 고정 원시 데이터로 핵심 계산을 실행해 결과를 JSON으로
수집(collect)하고, 두 JSON을 비교(compare)해서 코드 변경 전후로 계산
결과가 바뀌었는지 확인한다. 네트워크 호출이 전혀 없다 — fixture가 없으면
collect가 바로 에러를 낸다.

    python3 tools/golden.py collect --out tools/golden_before.json
    # ... 코드 수정 ...
    python3 tools/golden.py collect --out tools/golden_after.json
    python3 tools/golden.py compare tools/golden_before.json tools/golden_after.json
"""
import argparse
import hashlib
import json
import platform
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _app_extract import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
PRICES_PATH = FIXTURE_DIR / "golden_prices_raw.parquet"
FX_PATH = FIXTURE_DIR / "golden_fx_raw.parquet"

# 고정 입력 세트 (docs/PLAN.md 0-2 — tools/make_fixture.py 와 동일해야 함. 변경 금지)
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "005930.KS"]
WEIGHTS = {"SPY": 30, "QQQ": 20, "TLT": 25, "GLD": 15, "005930.KS": 10}
TICKER_CCY = {"SPY": "USD", "QQQ": "USD", "TLT": "USD", "GLD": "USD", "005930.KS": "KRW"}
START, END = "2018-01-01", "2025-12-31"
BASE_CCY = "KRW"
USE_DIVIDENDS = True
FX_HEDGE = False
COST_BP = 10
RF_RATE = 0.03
REBAL = "분기별"  # REBAL_QUARTER. docs/PLAN.md는 "분기"로 줄여 썼지만 app.py 상수 값은 이것.

# factor 블록에 쓸 요인 — 0순위 고정 입력 세트에는 별도 매크로 팩터 데이터가
# 없어서(그건 1순위 이후), 같은 fixture 안에 있는 두 종목(SPY=주식, TLT=채권)의
# 수익률을 요인으로 쓴다. 서로 상관이 낮아 다중회귀가 잘 조건화된다.
FACTOR_TICKERS = ["SPY", "TLT"]

NEED_FUNCS = {
    "build_price_frame", "portfolio_returns", "start_weights",
    "growth_contribution", "turnover_series", "apply_cost", "annual_turnover",
    "equity_curve", "drawdown_series", "max_drawdown", "cagr", "annual_vol",
    "sharpe_ratio", "sortino_ratio", "perf_row",
    "_dd_arr", "risk_std", "risk_mad", "risk_semisd", "ratio_sortino",
    "ratio_omega", "risk_cdar", "risk_ulcer", "risk_cvar",
    "ann_stats", "port_perf", "risk_contributions", "_as_bounds", "_opt_solve",
    "opt_max_sharpe", "opt_min_vol", "opt_risk_parity",
    "hrp_weights", "factor_betas",
}
NEED_CONSTS = {"RISK_DEFS"}
PRELUDE = """
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
TRADING_DAYS = 252
_EPS_STD = 1e-12
REBAL_DAILY = "매일 (고정비중 유지)"
REBAL_MONTH = "월별"
REBAL_QUARTER = "분기별"
REBAL_YEAR = "연별"
REBAL_NONE = "없음 (첫날 비중으로 매수 후 보유)"
PERIOD_CODE = {REBAL_MONTH: "M", REBAL_QUARTER: "Q", REBAL_YEAR: "Y"}
"""

_FIXTURE_CACHE = {}


def _require_fixture():
    if not PRICES_PATH.exists() or not FX_PATH.exists():
        raise FileNotFoundError(
            f"fixture가 없습니다: {PRICES_PATH} / {FX_PATH}\n"
            "tools/make_fixture.py 를 네트워크가 되는 환경에서 먼저 실행해야 합니다 "
            "(docs/PLAN.md 0-2). 이 sandbox에는 없는 게 정상입니다."
        )


def _fixture_prices() -> pd.DataFrame:
    if "prices" not in _FIXTURE_CACHE:
        _require_fixture()
        _FIXTURE_CACHE["prices"] = pd.read_parquet(PRICES_PATH)
    return _FIXTURE_CACHE["prices"]


def _fixture_fx() -> pd.DataFrame:
    if "fx" not in _FIXTURE_CACHE:
        _require_fixture()
        _FIXTURE_CACHE["fx"] = pd.read_parquet(FX_PATH)
    return _FIXTURE_CACHE["fx"]


def _slice_dates(s: pd.Series, start, end) -> pd.Series:
    return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def _fixture_probe_ticker(candidates):
    t = candidates[0]
    return {"currency": TICKER_CCY[t], "name": t, "quote_type": ""}


def _fixture_load_ticker(ticker, start, end, currency=None, name=None):
    df = _fixture_prices()
    close = _slice_dates(df[f"{ticker}__close"].dropna(), start, end)
    adjclose = _slice_dates(df[f"{ticker}__adjclose"].dropna(), start, end)
    return {"close": close, "adjclose": adjclose,
            "currency": currency or TICKER_CCY[ticker], "name": name or ticker}


def _fixture_load_fx(from_ccy, to_ccy, start, end):
    if from_ccy == to_ccy:
        return None
    fx = _fixture_fx()
    col = f"{from_ccy}{to_ccy}"
    if col in fx.columns:
        s = fx[col].dropna()
    elif f"{to_ccy}{from_ccy}" in fx.columns:
        s = 1.0 / fx[f"{to_ccy}{from_ccy}"].dropna()
    else:
        raise ValueError(f"fixture에 없는 환율 쌍: {from_ccy} -> {to_ccy}")
    return _slice_dates(s, start, end)


def _load_app_pure():
    """
    app.py에서 순수 계산 함수만 뽑는다. probe_ticker/load_ticker/load_fx는
    네트워크를 쓰므로 추출하지 않고, fixture 기반 대체 함수를 미리 넣어둔다 —
    build_price_frame이 전역 이름으로 이 셋을 찾으므로 그대로 fixture를 탄다.
    """
    return extract(
        NEED_FUNCS, NEED_CONSTS, PRELUDE,
        extra_globals={
            "probe_ticker": _fixture_probe_ticker,
            "load_ticker": _fixture_load_ticker,
            "load_fx": _fixture_load_fx,
        },
    )


def _round(x, nd=10):
    """float은 소수점 nd자리로 반올림 (부동소수 노이즈로 인한 오탐 방지)."""
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if np.isnan(f) else round(f, nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _input_hash() -> str:
    payload = {
        "tickers": TICKERS, "weights": WEIGHTS, "start": START, "end": END,
        "base_ccy": BASE_CCY, "use_dividends": USE_DIVIDENDS, "fx_hedge": FX_HEDGE,
        "cost_bp": COST_BP, "rebalance": REBAL, "rf_rate": round(RF_RATE, 10),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return "IH-" + hashlib.sha256(raw.encode()).hexdigest()[:8]


def _run_id() -> str:
    now = datetime.now()
    return f"PF-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.randbits(12):03X}"


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def collect_golden() -> dict:
    """
    고정 fixture로 핵심 계산을 실행하고 결과 dict를 반환한다. 네트워크 호출 없음.
    """
    mod = _load_app_pure()

    prices, meta, fx_used = mod.build_price_frame(
        TICKERS, START, END, BASE_CCY, USE_DIVIDENDS, fx_hedge=FX_HEDGE)

    port_r = mod.portfolio_returns(prices, WEIGHTS, REBAL)
    tno = mod.turnover_series(prices, WEIGHTS, REBAL)
    port_r_cost = mod.apply_cost(port_r, tno, COST_BP)
    contrib = mod.growth_contribution(prices, WEIGHTS, REBAL, port_r_cost)

    data = {
        "price_frame_shape": list(prices.shape),
        "actual_start": str(prices.index.min().date()),
        "actual_end": str(prices.index.max().date()),
        "per_ticker_first_last": {
            t: {"first": str(prices[t].dropna().index.min().date()),
                "last": str(prices[t].dropna().index.max().date()),
                "n_obs": int(prices[t].dropna().shape[0])}
            for t in TICKERS
        },
        "na_ratio": float(prices.isna().mean().mean()),
    }

    perf = mod.perf_row(port_r_cost, RF_RATE)
    risk = {name: float(d["fn"](port_r_cost, rf=RF_RATE))
            for name, d in mod.RISK_DEFS.items()}
    turnover = {
        "mean": float(tno.mean()),
        "sum": float(tno.sum()),
        "annual": float(mod.annual_turnover(tno)),
        "n_rebalance": int((tno > 1e-9).sum()),
    }
    contribution = {t: float(v) for t, v in contrib.items()}

    daily = prices[TICKERS].pct_change().dropna()
    mu, cov = mod.ann_stats(daily)
    optimize = _collect_optimize(mod, mu, cov, daily)

    F = daily[FACTOR_TICKERS]
    fb, fr2, fn = mod.factor_betas(port_r_cost, F)
    factor = {
        "betas": {t: (float(b) if not np.isnan(b) else None) for t, b in fb.items()},
        "r2": float(fr2) if not np.isnan(fr2) else None,
        "n": int(fn),
    }

    golden = {
        "meta": {
            "run_id": _run_id(),
            "input_hash": _input_hash(),
            "git_commit": _git_commit(),
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "data": data,
        "perf": perf,
        "risk": risk,
        "turnover": turnover,
        "contribution": contribution,
        "optimize": optimize,
        "factor": factor,
    }
    return {k: (v if k == "meta" else _round(v)) for k, v in golden.items()}


def _collect_optimize(mod, mu, cov, daily) -> dict:
    """
    목표: min_vol / max_sharpe / risk_parity / hrp.
    objective는 각 최적화가 실제로 최소화하는 값(자연스러운 단위로 부호를
    되돌린 것) — min_vol=변동성, max_sharpe=샤프지수(부호 복원),
    risk_parity=위험기여도 분산(작을수록 균등), hrp는 목적함수가 없는
    휴리스틱이라 None.
    """
    out = {}

    w = mod.opt_min_vol(mu, cov)
    out["min_vol"] = _opt_result(mod, w, mu, cov, lambda r, v, s: v)

    w = mod.opt_max_sharpe(mu, cov, rf=RF_RATE)
    out["max_sharpe"] = _opt_result(mod, w, mu, cov, lambda r, v, s: s)

    w = mod.opt_risk_parity(cov)
    if w is None:
        out["risk_parity"] = {"weights": None, "objective": None, "constraints_ok": False}
    else:
        rc = mod.risk_contributions(w, cov)
        obj = float(((rc - rc.mean()) ** 2).sum())
        out["risk_parity"] = {
            "weights": [float(x) for x in w], "objective": obj,
            "constraints_ok": bool(abs(float(np.sum(w)) - 1.0) < 1e-6),
        }

    w = mod.hrp_weights(daily)
    out["hrp"] = {
        "weights": [float(x) for x in w] if w is not None else None,
        "objective": None,
        "constraints_ok": bool(w is not None and abs(float(np.sum(w)) - 1.0) < 1e-6),
    }
    return out


def _opt_result(mod, w, mu, cov, objective_fn):
    if w is None:
        return {"weights": None, "objective": None, "constraints_ok": False}
    ret, vol, sharpe = mod.port_perf(w, mu, cov, RF_RATE)
    return {
        "weights": [float(x) for x in w],
        "objective": float(objective_fn(ret, vol, sharpe)),
        "constraints_ok": bool(abs(float(np.sum(w)) - 1.0) < 1e-6),
    }


def _flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}[{i}]", v, out)
    else:
        out[prefix] = obj


def compare_golden(before: dict, after: dict,
                   tol_default: float = 1e-9, tol_weights: float = 1e-4) -> list:
    """
    meta 블록은 비교하지 않는다. 나머지를 평평하게 펴서 키별로 비교한다.
    optimize.*.weights 는 tol_weights, 그 외 숫자는 tol_default.
    bool/문자열/None은 정확히 일치해야 한다. 키 추가/삭제도 보고한다.
    """
    b = {k: v for k, v in before.items() if k != "meta"}
    a = {k: v for k, v in after.items() if k != "meta"}
    fb, fa = {}, {}
    _flatten("", b, fb)
    _flatten("", a, fa)

    diffs = []
    for key in sorted(set(fb) - set(fa)):
        diffs.append({"key": key, "kind": "REMOVED", "before": fb[key]})
    for key in sorted(set(fa) - set(fb)):
        diffs.append({"key": key, "kind": "ADDED", "after": fa[key]})

    for key in sorted(set(fb) & set(fa)):
        bv, av = fb[key], fa[key]
        tol = tol_weights if ".weights[" in key else tol_default
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)) and not isinstance(bv, bool):
            if bv is None or av is None:
                changed = bv != av
            else:
                changed = abs(float(bv) - float(av)) > tol
        else:
            changed = bv != av
        if changed:
            diffs.append({"key": key, "kind": "CHANGED", "before": bv, "after": av})
    return diffs


def _cmd_collect(args):
    golden = collect_golden()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, sort_keys=True, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"✅ 골든 수집 완료: {out}")
    return 0


def _cmd_compare(args):
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    diffs = compare_golden(before, after, args.tol_default, args.tol_weights)
    if not diffs:
        print("✅ 차이 없음")
        return 0
    print(f"❌ {len(diffs)}개 차이")
    for d in diffs:
        if d["kind"] == "CHANGED":
            print(f"  [CHANGED] {d['key']}: {d['before']} -> {d['after']}")
        elif d["kind"] == "ADDED":
            print(f"  [ADDED]   {d['key']}: {d['after']}")
        else:
            print(f"  [REMOVED] {d['key']}: {d['before']}")
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="fixture로 계산해 골든 JSON 저장")
    pc.add_argument("--out", default="tools/golden_out.json")
    pc.set_defaults(func=_cmd_collect)

    pk = sub.add_parser("compare", help="골든 JSON 두 개 비교")
    pk.add_argument("before")
    pk.add_argument("after")
    pk.add_argument("--tol-default", type=float, default=1e-9, dest="tol_default")
    pk.add_argument("--tol-weights", type=float, default=1e-4, dest="tol_weights")
    pk.set_defaults(func=_cmd_compare)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as ex:
        print(f"❌ {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
