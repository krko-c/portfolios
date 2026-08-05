#!/usr/bin/env python3
"""
계산 정합성 12항목.

app.py 안의 run_self_tests() 를 그대로 불러 실행합니다.
Streamlit 없이 돌 수 있도록 필요한 함수만 뽑아 씁니다.

    python3 tools/selftest.py [app.py]
"""
import ast
import sys
import types
from pathlib import Path

NEED_FUNCS = {
    "run_self_tests", "portfolio_returns", "start_weights", "weight_drift",
    "growth_contribution", "turnover_series", "apply_cost", "annual_turnover",
    "equity_curve", "drawdown_series", "max_drawdown", "cagr", "annual_vol",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "ulcer_index", "port_perf",
    "risk_contributions", "_as_bounds", "_opt_solve", "opt_max_sharpe",
    "opt_min_vol", "opt_target_return", "ann_stats", "black_litterman",
    "bl_weights", "implied_returns", "shock_asset", "combine_shock",
    "guess_asset_kind", "factor_betas", "factor_vif", "standardize",
    "r2_weight", "r2_grade", "impact_label",
}
NEED_CONSTS = {"SPREAD_DUR", "ASSET_META", "ASSET_KINDS", "R2_GRADE"}

PRELUDE = '''
import numpy as np, pandas as pd, re
from scipy.optimize import minimize
TRADING_DAYS = 252
_EPS_STD = 1e-12
REBAL_DAILY = "매일 (고정비중 유지)"
REBAL_MONTH = "월별"
REBAL_QUARTER = "분기별"
REBAL_YEAR = "연별"
REBAL_NONE = "없음 (첫날 비중으로 매수 후 보유)"
PERIOD_CODE = {REBAL_MONTH: "M", REBAL_QUARTER: "Q", REBAL_YEAR: "Y"}
'''


def main(path="app.py"):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.split("\n")

    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NEED_FUNCS:
            parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
        elif isinstance(node, ast.Assign):
            tgt = getattr(node.targets[0], "id", "")
            if tgt in NEED_CONSTS:
                parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))

    mod = types.ModuleType("selftest")
    try:
        exec(compile(PRELUDE + "\n".join(parts), "selftest", "exec"), mod.__dict__)
    except Exception as ex:
        print(f"❌ 함수 추출 실패: {type(ex).__name__}: {ex}")
        return 1

    if not hasattr(mod, "run_self_tests"):
        print("❌ run_self_tests 를 찾지 못했습니다")
        return 1

    res = mod.run_self_tests()
    print(res[["검사", "결과", "상세"]].to_string(index=False, max_colwidth=44))

    fail = int(res["결과"].astype(str).str.contains("실패").sum())
    print(f"\n{len(res)}개 중 통과 {len(res) - fail} · 실패 {fail}")
    if fail:
        print("\n❌ 실패한 항목의 계산을 확인하세요.")
        for _, r in res[res["결과"].astype(str).str.contains("실패")].iterrows():
            print(f"  [{r['검사']}] {r['상세']}")
            print(f"    확인하는 것: {r['확인하는 것']}")
        return 1
    print("✅ 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "app.py"))
