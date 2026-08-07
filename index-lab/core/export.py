"""방법론 설정 JSON, 구성종목·백테스트 결과 Excel 출력."""
import io
import json

import pandas as pd


def methodology_to_json(config: dict) -> str:
    """dict를 그대로 JSON 문자열로 (날짜·Timestamp는 문자열로 변환)."""
    def _default(o):
        if isinstance(o, pd.Timestamp):
            return o.strftime("%Y-%m-%d")
        return str(o)
    return json.dumps(config, ensure_ascii=False, indent=2, default=_default)


def build_excel(*, constituents: pd.DataFrame, selection_log: pd.DataFrame,
                funnel: pd.DataFrame, backtest_summary: dict,
                constituent_history: pd.DataFrame, methodology: dict) -> bytes:
    """구성종목·선정로그·필터깔때기·백테스트요약·정기변경이력·방법론설정을
    시트별로 담은 Excel 파일(바이트)을 만든다."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        constituents.to_excel(xw, sheet_name="1_구성종목", index=False)
        selection_log.to_excel(xw, sheet_name="2_선정로그", index=False)
        funnel.to_excel(xw, sheet_name="3_필터깔때기", index=False)
        pd.DataFrame(list(backtest_summary.items()), columns=["지표", "값"]) \
            .to_excel(xw, sheet_name="4_백테스트요약", index=False)
        constituent_history.to_excel(xw, sheet_name="5_정기변경이력", index=False)
        pd.DataFrame([{"항목": k, "값": (v if isinstance(v, (str, int, float)) else str(v))}
                     for k, v in methodology.items()]) \
            .to_excel(xw, sheet_name="6_방법론설정", index=False)
    return buf.getvalue()
