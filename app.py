"""
Portfolio Analyzer v3 — Streamlit
=================================
변경점 (v2 대비)
- 포트폴리오를 여러 개(최대 6개) 만들어 한 그래프에서 비교
- 벤치마크도 여러 개 추가 가능 (콤마로 구분)
- 배당 재투자 여부 선택 (수정종가 vs 종가)
- 통화 자동 감지 (야후 메타데이터 사용, 수동 입력 불필요)
- 종목별 개별 다운로드 + 캐싱 → 여러 포트폴리오가 같은 종목을 쓰면 1회만 조회

실행
----
pip install -r requirements.txt
streamlit run app.py
"""

import io
import json
import re
from pathlib import Path

from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

TRADING_DAYS = 252
MAX_PORTFOLIOS = 6

REBAL_DAILY = "매일 (고정비중 유지)"
REBAL_MONTH = "월별"
REBAL_QUARTER = "분기별"
REBAL_YEAR = "연별"
REBAL_NONE = "없음 (첫날 비중으로 매수 후 보유)"
REBAL_OPTIONS = [REBAL_DAILY, REBAL_MONTH, REBAL_QUARTER, REBAL_YEAR, REBAL_NONE]
PERIOD_CODE = {REBAL_MONTH: "M", REBAL_QUARTER: "Q", REBAL_YEAR: "Y"}

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2"]
BENCH_PALETTE = ["#6b7280", "#9ca3af", "#4b5563", "#d1d5db"]

# 클릭 한 번으로 고르는 추천 색 (채도를 낮춘 최신 데이터 시각화 톤)
SWATCHES = [
    ("인디고", "#4f46e5"), ("블루", "#2563eb"), ("시안", "#0891b2"),
    ("틸", "#0d9488"), ("그린", "#16a34a"), ("라임", "#65a30d"),
    ("앰버", "#d97706"), ("오렌지", "#ea580c"), ("로즈", "#e11d48"),
    ("퍼플", "#9333ea"), ("슬레이트", "#64748b"),
]

st.set_page_config(page_title="포트폴리오 분석기", page_icon="📊", layout="wide")


# ======================================================================
# 데이터 로딩 — 종목 단위 캐싱
# ======================================================================
def _suffix_currency(ticker: str) -> str:
    """야후에서 통화를 못 받았을 때만 쓰는 최후의 추측."""
    t = ticker.upper()
    for sfx, ccy in [(".KS", "KRW"), (".KQ", "KRW"), (".T", "JPY"), (".HK", "HKD"),
                     (".TW", "TWD"), (".SS", "CNY"), (".SZ", "CNY"), (".L", "GBP"),
                     (".DE", "EUR"), (".PA", "EUR"), (".AS", "EUR"), (".TO", "CAD")]:
        if t.endswith(sfx):
            return ccy
    return "USD"


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker(ticker: str, start, end):
    """
    종목 1개의 가격 + 통화를 가져온다.
    반환: {"close": Series, "adjclose": Series, "currency": str, "name": str}
    - close    : 액면분할만 보정된 주가 (배당 제외)
    - adjclose : 액면분할 + 배당 재투자까지 보정
    통화는 야후 메타데이터에서 직접 읽는다.
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해주세요.")

    # 타임존 제거 후 날짜만 남김 (국가별 거래시간 차이로 정렬이 깨지는 것 방지)
    hist.index = pd.to_datetime(hist.index.date)

    close = hist["Close"].dropna()
    adj = hist["Adj Close"].dropna() if "Adj Close" in hist.columns else close.copy()

    # --- 통화 자동 감지 ---
    currency, name = None, ticker
    try:
        fi = tk.fast_info
        currency = (fi.get("currency") if hasattr(fi, "get") else getattr(fi, "currency", None))
    except Exception:
        pass
    if not currency:
        try:
            info = tk.info or {}
            currency = info.get("currency")
            name = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            pass
    currency = (currency or _suffix_currency(ticker)).upper()

    return {"close": close, "adjclose": adj, "currency": currency, "name": name}


@st.cache_data(ttl=3600, show_spinner=False)
def load_fx(from_ccy: str, to_ccy: str, start, end):
    """1 from_ccy = ? to_ccy. 직접 페어가 없으면 역페어의 역수를 사용."""
    if from_ccy == to_ccy:
        return None

    def _get(sym):
        h = yf.Ticker(sym).history(start=start, end=end, auto_adjust=False)
        if h.empty:
            return pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index.date)
        return h["Close"].dropna()

    s = _get(f"{from_ccy}{to_ccy}=X")
    if s.empty:
        inv = _get(f"{to_ccy}{from_ccy}=X")
        if inv.empty:
            raise ValueError(f"환율 데이터를 찾을 수 없습니다: {from_ccy} → {to_ccy}")
        s = 1.0 / inv
    return s


@st.cache_data(ttl=86400, show_spinner=False)
def ticker_label(ticker: str) -> str:
    """
    표의 '종목명' 열에 넣을 짧은 문구를 만든다.
    성공: '✅ Samsung Electronics Co., Ltd. · KRW'
    실패: '❌ 확인 불가'
    24시간 캐싱되므로 같은 티커는 두 번째부터 즉시 반환된다.
    """
    t = (ticker or "").strip()
    if not t:
        return ""

    name = ccy = None
    try:
        tk = yf.Ticker(t)

        # info: 종목명 (실패해도 넘어감)
        try:
            info = tk.info or {}
            name = info.get("shortName") or info.get("longName")
            ccy = info.get("currency")
        except Exception:
            pass

        # fast_info: 통화·시세로 유효성 확인
        try:
            fi = tk.fast_info
            g = (lambda k: fi.get(k)) if hasattr(fi, "get") else (lambda k: getattr(fi, k, None))
            ccy = ccy or g("currency")
            if not name and g("last_price") is not None:
                name = "(종목명 없음)"
        except Exception:
            pass

        # 그래도 확인이 안 되면 시세 직접 조회
        if not name:
            h = tk.history(period="5d")
            if not h.empty:
                name = "(종목명 없음)"
    except Exception:
        return "❌ 확인 불가"

    if not name:
        return "❌ 확인 불가"
    return f"✅ {name}" + (f" · {ccy.upper()}" if ccy else "")


BBG_SUFFIX = {
    "KS": ".KS", "KQ": ".KQ", "US": "", "JP": ".T", "TT": ".TW",
    "LN": ".L", "GR": ".DE", "FP": ".PA", "NA": ".AS", "SM": ".MC",
    "IM": ".MI", "CN": ".TO", "AU": ".AX", "IN": ".NS", "SP": ".SI",
}


def normalize_ticker(raw: str):
    """
    입력을 야후 티커 후보 목록으로 바꾼다 (네트워크 조회 없음).
      005930.KS  -> ['005930.KS']          야후 정식
      005930 KS  -> ['005930.KS']          블룸버그
      005930     -> ['005930.KS','.KQ']    숫자만 (코스피 우선)
      NVDA US    -> ['NVDA']
      700 HK     -> ['0700.HK']            홍콩은 4자리로 채움
      600519 CH  -> ['600519.SS']          6으로 시작하면 상하이
    """
    s = (raw or "").strip().upper()
    if not s:
        return []
    s = re.sub(r"\s+EQUITY$", "", s)
    if s.startswith("^") or "." in s:
        return [s]

    m = re.match(r"^([A-Z0-9]+)\s+([A-Z]{2})$", s)
    if m:
        code, ex = m.groups()
        if ex == "CH":
            return [code + (".SS" if code.startswith("6") else ".SZ")]
        if ex == "HK":
            return [code.zfill(4) + ".HK"]
        if ex in BBG_SUFFIX:
            sfx = BBG_SUFFIX[ex]
            return [code + sfx] if sfx else [code]
        return [code]

    if s.isdigit():
        if len(s) == 6:
            return [s + ".KS", s + ".KQ"]
        if len(s) == 4:
            return [s + ".T"]
    return [s]


def _fi_get(fi, *keys):
    """FastInfo는 yfinance 버전마다 dict형/속성형이 달라 양쪽 다 시도한다."""
    for k in keys:
        try:
            if hasattr(fi, "get"):
                v = fi.get(k)
                if v is not None:
                    return v
        except Exception:
            pass
        try:
            v = getattr(fi, k, None)
            if v is not None:
                return v
        except Exception:
            pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def probe_ticker(candidates: tuple):
    """
    후보를 검증해 진짜 그 종목인지 확인한다.

    야후는 없는 심볼을 요청하면 엉뚱한 상품(펀드 등)으로 유사 매칭해 응답하는 경우가
    있다. 그래서 응답 meta 의 symbol 이 요청한 것과 같은지 반드시 대조한다.
    (086520.KS 는 존재하지 않는데 응답이 오던 문제)

    반환: {"ticker","currency","name","rows","first","last","log"}
    """
    log, best = [], None
    for cand in candidates:
        try:
            tk = yf.Ticker(cand)
            h = tk.history(period="1y")
        except Exception as ex:
            log.append(f"{cand}: 조회 실패 — {type(ex).__name__}: {str(ex)[:80]}")
            continue

        if h.empty or "Close" not in h.columns or not h["Close"].notna().any():
            log.append(f"{cand}: 시세 데이터 없음")
            continue

        # --- 응답 meta 확인 (심볼 대조 + 통화) ---
        hm = {}
        try:
            hm = (tk.get_history_metadata() if hasattr(tk, "get_history_metadata")
                  else getattr(tk, "history_metadata", None)) or {}
        except Exception:
            hm = {}

        got = str(hm.get("symbol") or "").upper()
        if got and got != cand.upper():
            log.append(f"{cand}: 야후가 다른 종목({got})으로 응답 → 제외")
            continue

        h = h[h["Close"].notna()]
        ccy = hm.get("currency")
        if not ccy:
            try:
                ccy = _fi_get(tk.fast_info, "currency")
            except Exception:
                ccy = None

        info = {
            "ticker": cand,
            "currency": (ccy or _suffix_currency(cand)).upper(),
            "name": hm.get("longName") or hm.get("shortName") or "",
            "rows": len(h),
            "first": pd.to_datetime(h.index[0]).date(),
            "last": pd.to_datetime(h.index[-1]).date(),
            "log": log,
        }
        log.append(f"{cand}: 확인됨 (최근 1년 {len(h)}일치)")
        if best is None or info["rows"] > best["rows"]:
            best = info

    if best is None:
        return {"ticker": None, "currency": None, "name": "", "rows": 0,
                "first": None, "last": None, "log": log}
    best["log"] = log
    return best



def quick_check(candidates: tuple):
    r = probe_ticker(candidates)
    return r["ticker"], r["currency"]


@st.cache_data(ttl=86400, show_spinner=False)
def get_name(ticker: str) -> str:
    """종목명. 실패해도 계산에는 지장이 없으므로 조용히 빈 문자열을 반환한다."""
    for fn in ("info", "get_info"):
        try:
            info = getattr(yf.Ticker(ticker), fn)
            info = info() if callable(info) else info
            if info:
                nm = info.get("shortName") or info.get("longName")
                if nm:
                    return nm
        except Exception:
            continue
    return ""



def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """무한 재실행을 막기 위한 비교 (문자열로 변환해 안전하게 대조)."""
    try:
        return (a.astype(str).reset_index(drop=True)
                .equals(b.astype(str).reset_index(drop=True)))
    except Exception:
        return False


# ======================================================================
# 포트폴리오 저장 / 불러오기 (app.py 옆의 portfolios.json)
# ======================================================================
SAVE_PATH = Path(__file__).parent / "portfolios.json"
STORE_KEY = "_saved_store"


def _disk_read() -> dict:
    """로컬 실행 시에만 쓰이는 파일 읽기. 클라우드에서는 보통 비어 있다."""
    try:
        if SAVE_PATH.exists():
            return json.loads(SAVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _disk_write(data: dict) -> bool:
    """
    파일 저장 시도. Streamlit Cloud 는 쓰기가 되더라도 서버 재시작 시 사라지므로
    실패해도 문제 삼지 않는다 (세션 보관 + 파일 내려받기로 보완).
    """
    try:
        SAVE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return True
    except Exception:
        return False


def load_saved() -> dict:
    """세션에 보관된 구성. 최초 1회는 로컬 파일에서 가져온다."""
    if STORE_KEY not in st.session_state:
        st.session_state[STORE_KEY] = _disk_read()
    return st.session_state[STORE_KEY]


def write_saved(data: dict) -> bool:
    st.session_state[STORE_KEY] = data
    _disk_write(data)          # 로컬이면 파일로도 남고, 클라우드면 조용히 무시
    return True


def snapshot(port_specs, bench_list, base_ccy, use_div, rf_rate, label="") -> dict:
    """현재 화면 상태를 저장 가능한 형태로 변환."""
    return {
        "label": label,
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "base_ccy": base_ccy,
        "use_dividends": bool(use_div),
        "rf_rate": float(rf_rate),
        "benchmarks": list(bench_list),
        "portfolios": [
            {
                "name": p["name"],
                "rebalance": p["rebalance"],
                "holdings": [
                    {"ticker": str(r["티커"]).strip(),
                     "weight": (None if pd.isna(r["비중(%)"]) else float(r["비중(%)"]))}
                    for _, r in p["holdings"].iterrows()
                    if str(r["티커"]).strip()
                ],
            }
            for p in port_specs
        ],
    }


# ======================================================================
# 엑셀 내보내기 (네이티브 차트 포함)
# ======================================================================
def build_excel(series: dict, comp: pd.DataFrame, prices: pd.DataFrame,
                meta: dict, settings: dict, colors: dict) -> bytes:
    """
    화면 구성과 동일한 순서로 시트를 만든다.
      1_포트폴리오구성 · 2_성과차트 · 3_Drawdown · 4_종합비교
      5_<포트명>(상세) · 6_설정
    - 성과차트 / Drawdown : 엑셀 네이티브 라인차트
    - 상세 시트 : 월별 수익률에 조건부 서식, 비중 추이에 누적 영역 차트
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        f_num = wb.add_format({"num_format": "0.00"})
        f_date = wb.add_format({"num_format": "yyyy-mm-dd"})
        f_head = wb.add_format({"bold": True, "bg_color": "#eef2ff",
                                "border": 1, "align": "center"})
        f_title = wb.add_format({"bold": True, "font_size": 14})
        f_sec = wb.add_format({"bold": True, "font_size": 11,
                               "bg_color": "#f1f5f9", "border": 1})
        # 연간 열 전용 (흰 배경 + 글씨 색)
        f_tot_pos = wb.add_format({"num_format": "0.00", "bold": True,
                                   "font_color": "#0f766e", "bg_color": "#ffffff"})
        f_tot_neg = wb.add_format({"num_format": "0.00", "bold": True,
                                   "font_color": "#dc2626", "bg_color": "#ffffff"})

        def _hex(name, fallback):
            c = colors.get(name, fallback)
            return c if isinstance(c, str) and c.startswith("#") else fallback

        def _safe(s, limit=24):
            return "".join(ch for ch in str(s) if ch not in set('[]:*?/\\'))[:limit]

        ports = {n: v for n, v in series.items() if v["kind"] == "portfolio"}

        # ---------------- 1. 포트폴리오 구성 (벤치마크 포함) ----------------
        rows = []
        for name, v in ports.items():
            total = sum(v["weights"].values()) or 1.0
            for t, w in v["weights"].items():
                m = meta.get(t, {})
                rows.append({
                    "구분": "포트폴리오", "이름": name, "리밸런싱": v["rebalance"],
                    "티커": t, "종목명": m.get("name", t),
                    "거래통화": m.get("currency", "-"),
                    "입력 비중(%)": w,
                    "정규화 비중(%)": round(w / total * 100, 2),
                    "데이터 일수": m.get("rows", ""),
                })
        for name, v in series.items():
            if v["kind"] == "portfolio":
                continue
            m = meta.get(name, {})
            rows.append({
                "구분": "벤치마크", "이름": name, "리밸런싱": "-",
                "티커": name, "종목명": m.get("name", name),
                "거래통화": m.get("currency", "-"),
                "입력 비중(%)": 100.0, "정규화 비중(%)": 100.0,
                "데이터 일수": m.get("rows", ""),
            })
        df1 = pd.DataFrame(rows)
        df1.to_excel(xw, sheet_name="1_포트폴리오구성", index=False, startrow=1)
        ws = xw.sheets["1_포트폴리오구성"]
        ws.write(0, 0, "포트폴리오 구성", f_title)
        for c, col in enumerate(df1.columns):
            ws.write(1, c, col, f_head)
        for c, wdt in enumerate([11, 16, 20, 13, 32, 10, 13, 15, 11]):
            ws.set_column(c, c, wdt, f_num if c in (6, 7) else None)
        ws.freeze_panes(2, 0)
        ws.autofilter(1, 0, 1 + len(df1), len(df1.columns) - 1)

        # ---------------- 2. 성과 차트 ----------------
        curves = pd.DataFrame({n: equity_curve(v["returns"]) for n, v in series.items()})
        curves.index.name = "날짜"
        curves.to_excel(xw, sheet_name="2_성과차트")
        ws = xw.sheets["2_성과차트"]
        ws.set_column(0, 0, 12, f_date)
        ws.set_column(1, len(curves.columns), 14, f_num)
        ws.freeze_panes(1, 1)

        ch = wb.add_chart({"type": "line"})
        for j, nm in enumerate(curves.columns, start=1):
            ch.add_series({
                "name":       ["2_성과차트", 0, j],
                "categories": ["2_성과차트", 1, 0, len(curves), 0],
                "values":     ["2_성과차트", 1, j, len(curves), j],
                "line": {"width": 1.75, "color": _hex(nm, "#2563eb")},
            })
        ch.set_title({"name": "성과 차트"})
        ch.set_size({"width": 940, "height": 450})
        ch.set_legend({"position": "bottom"})
        ws.insert_chart(1, len(curves.columns) + 2, ch)

        # ---------------- 3. Drawdown ----------------
        dds = pd.DataFrame({n: drawdown_series(v["returns"]) * 100
                            for n, v in series.items()})
        dds.index.name = "날짜"
        dds.to_excel(xw, sheet_name="3_Drawdown")
        ws = xw.sheets["3_Drawdown"]
        ws.set_column(0, 0, 12, f_date)
        ws.set_column(1, len(dds.columns), 14, f_num)
        ws.freeze_panes(1, 1)

        ch2 = wb.add_chart({"type": "line"})
        for j, nm in enumerate(dds.columns, start=1):
            ch2.add_series({
                "name":       ["3_Drawdown", 0, j],
                "categories": ["3_Drawdown", 1, 0, len(dds), 0],
                "values":     ["3_Drawdown", 1, j, len(dds), j],
                "line": {"width": 1.5, "color": _hex(nm, "#dc2626")},
            })
        ch2.set_title({"name": "Drawdown (%)"})
        ch2.set_size({"width": 940, "height": 400})
        ch2.set_legend({"position": "bottom"})
        ws.insert_chart(1, len(dds.columns) + 2, ch2)

        # ---------------- 4. 종합 비교 ----------------
        comp.to_excel(xw, sheet_name="4_종합비교", startrow=1)
        ws = xw.sheets["4_종합비교"]
        ws.write(0, 0, "종합 성과 비교", f_title)
        ws.set_column(0, 0, 26)
        ws.set_column(1, len(comp.columns), 14, f_num)
        for c, col in enumerate(comp.columns, start=1):
            ws.write(1, c, col, f_head)
        ws.freeze_panes(2, 1)

        # ---------------- 5. 포트폴리오별 상세 ----------------
        for name, v in ports.items():
            sname = f"5_{_safe(name)}"
            ws = wb.add_worksheet(sname)
            xw.sheets[sname] = ws
            # 첫 열에 날짜 서식을 걸면 월별표의 '연도(정수)'가 1905년으로 깨지므로 폭만 지정
            ws.set_column(0, 0, 14)
            ws.set_column(1, 16, 13, f_num)
            r = 0
            ws.write(r, 0, f"{name}  ({v['rebalance']})", f_title); r += 2

            # (1) 최악의 낙폭 구간
            ws.write(r, 0, "최악의 낙폭 구간", f_sec); r += 1
            t1 = worst_drawdowns(v["returns"], top_n=10)
            t1.to_excel(xw, sheet_name=sname, startrow=r, index=False)
            for c, col in enumerate(t1.columns):
                ws.write(r, c, col, f_head)
            r += len(t1) + 3

            # (2) 월별 수익률 + 조건부 서식
            ws.write(r, 0, "월별 수익률 (%)", f_sec); r += 1
            t2 = monthly_table(v["returns"])
            t2.to_excel(xw, sheet_name=sname, startrow=r)
            hdr_row, first_row = r, r + 1
            last_row = r + len(t2)
            cols = list(t2.columns)
            for c, col in enumerate(cols, start=1):
                ws.write(hdr_row, c, col, f_head)

            if "연간" in cols:
                tot_c = cols.index("연간") + 1
                month_last = tot_c - 1
            else:
                tot_c, month_last = None, len(cols)

            if month_last >= 1 and last_row >= first_row:
                ws.conditional_format(first_row, 1, last_row, month_last, {
                    "type": "3_color_scale",
                    "min_type": "num", "min_value": -10, "min_color": "#f8696b",
                    "mid_type": "num", "mid_value": 0,   "mid_color": "#ffffff",
                    "max_type": "num", "max_value": 10,  "max_color": "#63be7b",
                })
                if tot_c is not None:
                    ws.conditional_format(first_row, tot_c, last_row, tot_c, {
                        "type": "cell", "criteria": ">=", "value": 0,
                        "format": f_tot_pos})
                    ws.conditional_format(first_row, tot_c, last_row, tot_c, {
                        "type": "cell", "criteria": "<", "value": 0,
                        "format": f_tot_neg})
            r = last_row + 3

            # (3) 보유 비중 추이 + 누적 영역 차트
            ws.write(r, 0, "보유 비중 추이 (%)", f_sec); r += 1
            t3 = (weight_drift(prices, v["weights"], v["rebalance"])
                  .loc[v["returns"].index] * 100).round(2)
            t3.index.name = "날짜"
            t3.to_excel(xw, sheet_name=sname, startrow=r)
            for c, col in enumerate(t3.columns, start=1):
                ws.write(r, c, col, f_head)
            wd_hdr, wd_first, wd_last = r, r + 1, r + len(t3)
            for rr in range(wd_first, wd_last + 1):
                pass  # 날짜 셀은 pandas가 datetime 서식으로 기록

            ch3 = wb.add_chart({"type": "area", "subtype": "stacked"})
            for j, tk in enumerate(t3.columns, start=1):
                ch3.add_series({
                    "name":       [sname, wd_hdr, j],
                    "categories": [sname, wd_first, 0, wd_last, 0],
                    "values":     [sname, wd_first, j, wd_last, j],
                })
            ch3.set_title({"name": "보유 비중 추이 (%)"})
            ch3.set_y_axis({"max": 100, "min": 0})
            ch3.set_size({"width": 860, "height": 360})
            ch3.set_legend({"position": "bottom"})
            ws.insert_chart(wd_hdr, len(t3.columns) + 2, ch3)

        # ---------------- 6. 설정 ----------------
        df6 = pd.DataFrame(list(settings.items()), columns=["항목", "값"])
        df6.to_excel(xw, sheet_name="6_설정", index=False, startrow=1)
        ws = xw.sheets["6_설정"]
        ws.write(0, 0, "분석 설정", f_title)
        for c, col in enumerate(df6.columns):
            ws.write(1, c, col, f_head)
        ws.set_column(0, 0, 32)
        ws.set_column(1, 1, 64)

    return buf.getvalue()


def build_opt_excel(alloc, t_oos, t_ins, oos_rows, bdf, rdf, corr,
                    wd, settings, opt_date) -> bytes:
    """최적화 결과를 화면과 같은 순서로 엑셀에 담는다."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        f_num = wb.add_format({"num_format": "0.00"})
        f_date = wb.add_format({"num_format": "yyyy-mm-dd"})
        f_head = wb.add_format({"bold": True, "bg_color": "#eef2ff",
                                "border": 1, "align": "center"})
        f_title = wb.add_format({"bold": True, "font_size": 14})

        def _sheet(df, name, index=True, title=None, widths=None):
            df.to_excel(xw, sheet_name=name, index=index, startrow=1)
            ws = xw.sheets[name]
            ws.write(0, 0, title or name, f_title)
            off = 1 if index else 0
            for c, col in enumerate(df.columns):
                ws.write(1, c + off, str(col), f_head)
            ws.set_column(0, 0, 26)
            ws.set_column(off, off + len(df.columns), 15, f_num)
            return ws

        _sheet(alloc, "1_최적자산배분", False, "최적 자산배분 (Optimal Allocation)")
        _sheet(t_oos, "2_표본외성과", True, "표본외 성과 (Out-of-Sample)")
        _sheet(t_ins, "3_표본내성과", True, "표본내 성과 (In-Sample)")

        curves = pd.DataFrame({k: equity_curve(v) for k, v in oos_rows.items()
                               if v is not None})
        curves.index.name = "날짜"
        curves.to_excel(xw, sheet_name="4_성과추이")
        ws = xw.sheets["4_성과추이"]
        ws.write(0, 0, "", f_title)
        ws.set_column(0, 0, 12, f_date)
        ws.set_column(1, len(curves.columns), 15, f_num)
        ch = wb.add_chart({"type": "line"})
        for j, nm in enumerate(curves.columns, start=1):
            ch.add_series({"name": ["4_성과추이", 0, j],
                           "categories": ["4_성과추이", 1, 0, len(curves), 0],
                           "values": ["4_성과추이", 1, j, len(curves), j],
                           "line": {"width": 1.75}})
        ch.set_title({"name": "포트폴리오 성과"})
        ch.set_size({"width": 900, "height": 430})
        ch.set_legend({"position": "bottom"})
        ws.insert_chart(1, len(curves.columns) + 2, ch)

        if bdf is not None and not bdf.empty:
            _sheet(bdf, "5_벤치마크지표", True, "벤치마크 대비 지표")
        _sheet(rdf, "6_위험기여도", False, "위험 기여도 (Risk Contribution)")

        if wd is not None and not wd.empty:
            wd2 = wd.copy(); wd2.index.name = "날짜"
            wd2.to_excel(xw, sheet_name="7_비중추이")
            ws = xw.sheets["7_비중추이"]
            ws.set_column(0, 0, 12, f_date)
            ws.set_column(1, len(wd2.columns), 13, f_num)
            ca = wb.add_chart({"type": "area", "subtype": "stacked"})
            for j, nm in enumerate(wd2.columns, start=1):
                ca.add_series({"name": ["7_비중추이", 0, j],
                               "categories": ["7_비중추이", 1, 0, len(wd2), 0],
                               "values": ["7_비중추이", 1, j, len(wd2), j]})
            ca.set_title({"name": "시간에 따른 비중 변화"})
            ca.set_y_axis({"min": 0, "max": 100})
            ca.set_size({"width": 900, "height": 380})
            ca.set_legend({"position": "bottom"})
            ws.insert_chart(1, len(wd2.columns) + 2, ca)

        _sheet(corr, "8_자산상관관계", True, "자산 상관관계 (Asset Correlations)")
        pd.DataFrame(list(settings.items()), columns=["항목", "값"]).to_excel(
            xw, sheet_name="9_설정", index=False, startrow=1)
        ws = xw.sheets["9_설정"]
        ws.write(0, 0, "최적화 설정", f_title)
        ws.set_column(0, 0, 30); ws.set_column(1, 1, 64)
    return buf.getvalue()


def build_price_frame(tickers, start, end, base_ccy: str, use_dividends: bool,
                      fx_hedge: bool = False, gap_fill: bool = False):
    """
    여러 종목을 기준통화로 환산한 가격 DataFrame.
    fx_hedge=True 면 환율 변동을 제거(환헤지 가정)하고 종목 자체 수익률만 본다.
    gap_fill=True 면 국가별 휴장일 차이를 직전 종가로 메워 공통 거래일 손실을 줄인다.
    반환: (prices, meta, fx_used)
    """
    series, meta, fx_cache = {}, {}, {}

    for t in tickers:
        d = load_ticker(t, start, end)
        px = d["adjclose"] if use_dividends else d["close"]
        ccy = d["currency"]
        meta[t] = {"currency": ccy, "name": d["name"], "rows": len(px)}

        if ccy != base_ccy and not fx_hedge:
            if ccy not in fx_cache:
                fx_cache[ccy] = load_fx(ccy, base_ccy, start, end)
            fx = fx_cache[ccy].reindex(px.index).ffill().bfill()
            px = px * fx
        series[t] = px

    df = pd.DataFrame(series)
    if gap_fill:
        df = df.ffill()
    return df.dropna(), meta, fx_cache


# ======================================================================
# 포트폴리오 계산
# ======================================================================
def portfolio_returns(prices: pd.DataFrame, weights: dict, rebalance: str) -> pd.Series:
    tickers = list(weights.keys())
    px = prices[tickers].dropna()
    if len(px) < 2:
        raise ValueError("공통 거래일이 부족합니다. 기간이나 종목 구성을 확인해주세요.")
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    daily = px.pct_change().dropna()

    if rebalance == REBAL_DAILY:
        out = (daily * w).sum(axis=1)
    elif rebalance == REBAL_NONE:
        growth = (1 + daily).cumprod()
        first = pd.DataFrame([np.ones(len(tickers))], columns=tickers,
                             index=[daily.index[0] - pd.Timedelta(days=1)])
        growth = pd.concat([first, growth])
        out = (growth * w).sum(axis=1).pct_change().dropna()
    else:
        pid = daily.index.to_period(PERIOD_CODE[rebalance])
        out = pd.Series(index=daily.index, dtype=float)
        for _, idx in daily.groupby(pid).groups.items():
            pr = daily.loc[idx]
            growth = (1 + pr).cumprod()
            wr = (growth * w).sum(axis=1).pct_change()
            wr.iloc[0] = (pr.iloc[0] * w).sum()
            out.loc[idx] = wr.values
    return out.dropna()


def weight_drift(prices: pd.DataFrame, weights: dict, rebalance: str) -> pd.DataFrame:
    tickers = list(weights.keys())
    px = prices[tickers].dropna()
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    if rebalance == REBAL_DAILY:
        return pd.DataFrame(np.tile(w, (len(px), 1)), index=px.index, columns=tickers)

    if rebalance == REBAL_NONE:
        val = (px / px.iloc[0]) * w
    else:
        pid = px.index.to_period(PERIOD_CODE[rebalance])
        val = pd.DataFrame(index=px.index, columns=tickers, dtype=float)
        for _, idx in px.groupby(pid).groups.items():
            sub = px.loc[idx]
            val.loc[idx] = (sub / sub.iloc[0]).values * w
    return val.div(val.sum(axis=1), axis=0)


# ---------------------- 거래비용 · 회전율 ----------------------
def turnover_series(prices: pd.DataFrame, weights: dict, rebalance: str) -> pd.Series:
    """
    각 거래일의 회전율 Σ|바뀐 비중|.
    '오늘 시작 비중'과 '어제 종료 비중(가격 변동으로 흘러간 상태)'의 차이가 실제 거래량이다.
    Buy&Hold 는 거래가 없으므로 0, 매일 리밸런싱은 매일 발생한다.
    """
    tickers = list(weights)
    d = prices[tickers].dropna().pct_change().dropna()
    ws = start_weights(prices, weights, rebalance).reindex(d.index)
    v = ws * (1 + d)
    w_end_prev = v.div(v.sum(axis=1), axis=0).shift(1)
    tno = (ws - w_end_prev).abs().sum(axis=1)
    tno.iloc[0] = 1.0                      # 최초 매수
    return tno.fillna(0.0)


def apply_cost(r: pd.Series, tno: pd.Series, cost_bp: float) -> pd.Series:
    """편도 거래비용(bp)을 회전율에 곱해 일별 수익률에서 차감."""
    if not cost_bp:
        return r
    return r - tno.reindex(r.index).fillna(0.0) * (cost_bp / 10000.0)


def annual_turnover(tno: pd.Series) -> float:
    if len(tno) < 2:
        return np.nan
    yrs = (tno.index[-1] - tno.index[0]).days / 365.25
    body = tno.iloc[1:]                    # 최초 매수 제외
    return float(body.sum() / yrs) if yrs > 0 else np.nan


# ---------------------- 적립식 (DCA) ----------------------
def xirr(cashflows, dates, lo=-0.95, hi=10.0):
    """현금흐름 내부수익률(연율). 이분법으로 안정적으로 해를 찾는다."""
    if not cashflows or len(cashflows) < 2:
        return np.nan
    t0 = dates[0]

    def npv(rate):
        return sum(cf / ((1 + rate) ** ((d - t0).days / 365.25))
                   for cf, d in zip(cashflows, dates))
    try:
        if npv(lo) * npv(hi) > 0:
            return np.nan
        for _ in range(200):
            mid = (lo + hi) / 2
            if npv(lo) * npv(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    except Exception:
        return np.nan


def dca_result(r: pd.Series, monthly: float, start_value: float = 0.0):
    """
    매월 첫 거래일에 monthly 만큼 추가 납입.
    반환: (평가액 시계열, 누적 원금 시계열, 납입일 목록)
    """
    curve = (1 + r).cumprod()
    idx = curve.index
    first_of_month = pd.Series(idx, index=idx).groupby(
        [idx.year, idx.month]).first().values
    units, invested = 0.0, 0.0
    if start_value > 0:
        units += start_value / float(curve.iloc[0])
        invested += start_value
    vals, invs, buys = [], [], []
    contrib = {pd.Timestamp(d) for d in first_of_month}
    for dt, c in curve.items():
        if dt in contrib and monthly > 0:
            units += monthly / float(c)
            invested += monthly
            buys.append(dt)
        vals.append(units * float(c))
        invs.append(invested)
    return (pd.Series(vals, index=idx), pd.Series(invs, index=idx), buys)


# ---------------------- 성과 지표
def equity_curve(r, start_value=1.0):
    """
    투자 직전 시점(=start_value)을 맨 앞에 붙여서 곡선이 1.00에서 출발하게 한다.
    이걸 빼면 첫날 수익률이 이미 반영된 값부터 그려지고,
    CAGR 계산 기간도 하루 짧아지며 첫날 하락이 MDD에 반영되지 않는다.
    """
    if len(r) == 0:
        return pd.Series(dtype=float)
    ec = start_value * (1 + r).cumprod()
    base = pd.Series([float(start_value)], index=[r.index[0] - pd.Timedelta(days=1)])
    return pd.concat([base, ec])


def drawdown_series(r):
    c = equity_curve(r)
    return c / c.cummax() - 1.0


def max_drawdown(r):
    return drawdown_series(r).min()


def cagr(r):
    c = equity_curve(r)
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    return c.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan


def annual_vol(r):
    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(r, rf=0.0):
    e = r - rf / TRADING_DAYS
    return np.nan if e.std() == 0 else e.mean() / e.std() * np.sqrt(TRADING_DAYS)


def sortino_ratio(r, rf=0.0):
    e = r - rf / TRADING_DAYS
    d = e[e < 0].std()
    return np.nan if (d == 0 or np.isnan(d)) else e.mean() / d * np.sqrt(TRADING_DAYS)


def calmar_ratio(r):
    m = abs(max_drawdown(r))
    return np.nan if m == 0 else cagr(r) / m


def ulcer_index(r):
    dd = drawdown_series(r) * 100
    return np.sqrt((dd ** 2).mean())


def martin_ratio(r):
    ui = ulcer_index(r)
    return np.nan if ui == 0 else (cagr(r) * 100) / ui


def worst_drawdowns(r, top_n=5):
    c = equity_curve(r)
    dd = c / c.cummax() - 1.0
    recs, in_dd = [], False
    peak, trough, tval = c.index[0], None, 0.0
    for date, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, peak, trough, tval = True, c.loc[:date].idxmax(), date, v
        elif v < 0 and in_dd and v < tval:
            tval, trough = v, date
        elif v == 0 and in_dd:
            recs.append({"peak": peak, "trough": trough, "recovery": date, "depth": tval})
            in_dd = False
    if in_dd:
        recs.append({"peak": peak, "trough": trough, "recovery": None, "depth": tval})

    rows = []
    for x in sorted(recs, key=lambda z: z["depth"])[:top_n]:
        if x["recovery"] is not None:
            rec = f"{(x['recovery'] - x['trough']).days}일"
            uw = f"{(x['recovery'] - x['peak']).days}일"
        else:
            rec, uw = "미회복", f"{(c.index[-1] - x['peak']).days}일+ (진행중)"
        rows.append({
            "하락폭": f"{x['depth']*100:.2f}%",
            "고점일": x["peak"].date(), "저점일": x["trough"].date(),
            "하락기간": f"{(x['trough'] - x['peak']).days}일",
            "회복기간": rec, "총 침체기간": uw,
        })
    return pd.DataFrame(rows)


def _heat_color(v, lo=-10.0, hi=10.0):
    """
    월별 수익률 셀 배경색. 음수는 붉게, 양수는 푸르게.
    matplotlib 없이 순수 파이썬으로 색을 계산한다.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""

    if x >= 0:
        t = min(x / hi, 1.0) if hi > 0 else 0.0          # 0(흰색) → 1(초록)
        r, g, b = 255 - int(121 * t), 255 - int(16 * t), 255 - int(103 * t)
    else:
        t = min(x / lo, 1.0) if lo < 0 else 0.0          # 0(흰색) → 1(빨강)
        r, g, b = 255 - int(3 * t), 255 - int(90 * t), 255 - int(90 * t)
    return f"background-color: rgb({r},{g},{b})"


def _corr_color(v):
    """상관관계 셀 색상. 높으면 붉게(분산 효과 낮음), 낮거나 음수면 푸르게."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    x = max(-1.0, min(1.0, x))
    if x >= 0:
        r, g, b = 255 - int(3 * x), 255 - int(90 * x), 255 - int(90 * x)
    else:
        t2 = -x
        r, g, b = 255 - int(121 * t2), 255 - int(16 * t2), 255 - int(103 * t2)
    return f"background-color: rgb({r},{g},{b})"


def _total_color(v):
    """연간(Total) 열: 배경은 흰색, 숫자 색으로만 구분."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    col = "#0f766e" if x >= 0 else "#dc2626"
    return f"background-color: #ffffff; color: {col}; font-weight: 700"


def style_monthly(df: pd.DataFrame):
    """월별 수익률 표: 월 칸은 배경 히트맵, 연간 칸은 흰 배경 + 글씨 색."""
    sty = df.style.format("{:.2f}", na_rep="-")
    apply_fn = sty.map if hasattr(sty, "map") else sty.applymap

    month_cols = [c for c in df.columns if c != "연간"]
    if month_cols:
        sty = apply_fn(_heat_color, subset=month_cols)
        apply_fn = sty.map if hasattr(sty, "map") else sty.applymap
    if "연간" in df.columns:
        sty = apply_fn(_total_color, subset=["연간"])
    return sty


def monthly_table(r):
    m = (1 + r).resample("ME").prod() - 1
    df = m.to_frame("ret")
    df["Year"], df["Month"] = df.index.year, df.index.month
    pv = df.pivot(index="Year", columns="Month", values="ret")
    pv.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in pv.columns]
    pv["연간"] = (1 + r).groupby(r.index.year).prod() - 1
    return (pv * 100).round(2)



# ======================================================================
# 성장 기여도 분해
# ======================================================================
def start_weights(prices: pd.DataFrame, weights: dict, rebalance: str) -> pd.DataFrame:
    """
    각 거래일이 '시작될 때' 보유 중인 비중. 그날 수익률에 실제로 적용되는 값이다.
    (하루가 끝난 뒤의 비중을 쓰면 리밸런싱이 있는 경우 합계가 어긋난다)
    """
    tickers = list(weights)
    px = prices[tickers].dropna()
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()
    d = px.pct_change().dropna()

    if rebalance == REBAL_DAILY:
        return pd.DataFrame(np.tile(w, (len(d), 1)), index=d.index, columns=tickers)

    if rebalance == REBAL_NONE:
        g = (1 + d).cumprod().shift(1)
        g.iloc[0] = 1.0
        val = g * w
    else:
        pid = d.index.to_period(PERIOD_CODE[rebalance])
        val = pd.DataFrame(index=d.index, columns=tickers, dtype=float)
        for _, ix in d.groupby(pid).groups.items():
            pr = d.loc[ix]
            g = (1 + pr).cumprod().shift(1)
            g.iloc[0] = 1.0
            val.loc[ix] = (g * w).values
    return val.div(val.sum(axis=1), axis=0)


def growth_contribution(prices: pd.DataFrame, weights: dict, rebalance: str,
                        port_r: pd.Series) -> pd.Series:
    """
    종목별 성장 기여도.  기여도_i = Σ_t V(t-1) × 시작비중_i(t) × 수익률_i(t)
    이 값들의 합은 포트폴리오 총수익과 정확히 일치한다.
    """
    tickers = list(weights)
    d = prices[tickers].dropna().pct_change().dropna()
    d = d.reindex(port_r.index).dropna()
    ws = start_weights(prices, weights, rebalance).reindex(d.index)
    V_prev = equity_curve(port_r).shift(1).reindex(d.index)
    return (d * ws).mul(V_prev, axis=0).sum()


# ======================================================================
# 최적화 엔진
# ======================================================================
# ---------------------- 위험 지표 (경험적) ----------------------
def _dd_arr(r):
    c = (1 + r).cumprod()
    return (c / c.cummax() - 1).values


def risk_std(r, **k):
    return float(r.std() * np.sqrt(TRADING_DAYS))


def risk_mad(r, **k):
    """평균절대편차 — 제곱을 쓰지 않아 극단값에 덜 민감하다."""
    return float((r - r.mean()).abs().mean() * np.sqrt(TRADING_DAYS))


def risk_semisd(r, thr=0.0, **k):
    """준표준편차 — 기준선 아래 움직임만 측정."""
    d = np.minimum(r.values - thr / TRADING_DAYS, 0.0)
    return float(np.sqrt((d ** 2).mean()) * np.sqrt(TRADING_DAYS))


def risk_cvar(r, q=0.05, **k):
    """조건부 VaR — 최악 q% 구간의 평균 손실."""
    v = np.sort(r.values)
    n = max(1, int(np.ceil(q * len(v))))
    return float(-v[:n].mean() * np.sqrt(TRADING_DAYS))


def risk_cdar(r, q=0.05, **k):
    """조건부 낙폭 — 최악 q% 낙폭의 평균."""
    d = np.sort(_dd_arr(r))
    n = max(1, int(np.ceil(q * len(d))))
    return float(-d[:n].mean())


def risk_ulcer(r, **k):
    return float(np.sqrt(((_dd_arr(r) * 100) ** 2).mean()))


def ratio_sortino(r, rf=0.0, **k):
    e = r - rf / TRADING_DAYS
    d = e[e < 0].std()
    return -1e6 if (d == 0 or np.isnan(d)) else float(e.mean() / d * np.sqrt(TRADING_DAYS))


def ratio_omega(r, thr=0.0, **k):
    """오메가 비율 — 기준선 위 이익총합 / 아래 손실총합."""
    x = r.values - thr / TRADING_DAYS
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return 1e6 if l <= 0 else float(g / l)


RISK_DEFS = {
    "표준편차 (Standard Deviation)": dict(group="변동성 기반", fn=risk_std, kind="risk",
                 desc="가장 표준적인 위험 척도. 상승·하락을 구분하지 않습니다."),
    "평균절대편차 (MAD)": dict(group="변동성 기반", fn=risk_mad, kind="risk",
                          desc="제곱을 쓰지 않아 극단적 하루에 덜 흔들립니다."),
    "준표준편차 (Semi Std Dev)": dict(group="하방위험", fn=risk_semisd, kind="risk",
                  desc="기준선 아래로 내려간 움직임만 위험으로 봅니다."),
    "소르티노 비율 (Sortino Ratio)": dict(group="하방위험", fn=ratio_sortino, kind="ratio",
                    desc="하방 변동성만으로 나눈 위험조정수익. 클수록 좋습니다."),
    "오메가 비율 (Omega Ratio)": dict(group="하방위험", fn=ratio_omega, kind="ratio",
                   desc="이익총합을 손실총합으로 나눈 값. 분포 전체 모양을 봅니다."),
    "조건부 낙폭 (CDaR)": dict(group="낙폭 기반", fn=risk_cdar, kind="risk",
                          desc="최악 5% 낙폭의 평균. 깊은 하락을 집중적으로 억제합니다."),
    "얼서지수 (Ulcer Index)": dict(group="낙폭 기반", fn=risk_ulcer, kind="risk",
                 desc="낙폭의 깊이와 지속기간을 함께 반영합니다."),
    "조건부 손실 (CVaR)": dict(group="꼬리위험", fn=risk_cvar, kind="risk",
                          desc="최악 5% 일간 손실의 평균. 꼬리 위험에 민감합니다."),
}


def _series(R: pd.DataFrame, w) -> pd.Series:
    """비중을 적용한 일별 수익률 (최적화 계산용, 일별 리밸런싱 가정)."""
    return pd.Series(R.values @ np.asarray(w, dtype=float), index=R.index)


def eval_risk(R, w, name, rf=0.0):
    d = RISK_DEFS[name]
    return float(d["fn"](_series(R, w), rf=rf))


def ann_stats(returns: pd.DataFrame):
    """연율화 기대수익률과 공분산."""
    return (returns.mean() * TRADING_DAYS).values, (returns.cov() * TRADING_DAYS).values


def port_perf(w, mu, cov, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, ((ret - rf) / vol if vol > 0 else np.nan)


def risk_contributions(w, cov):
    """자산별 위험 기여도. 합은 포트폴리오 변동성과 같다."""
    vol = np.sqrt(w @ cov @ w)
    return np.zeros_like(w) if vol == 0 else w * (cov @ w) / vol


def _opt_solve(fn, n, wmin, wmax, extra=()):
    """SLSQP. 시작점을 여러 개 시도해 국소해에 갇히는 것을 줄인다."""
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(extra)
    bounds = tuple((wmin, wmax) for _ in range(n))
    best, best_val = None, np.inf
    rng = np.random.default_rng(0)
    starts = [np.ones(n) / n] + [rng.random(n) for _ in range(6)]
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), wmin, wmax)
        if x0.sum() > 0:
            x0 = x0 / x0.sum()
        try:
            res = minimize(fn, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-12})
        except Exception:
            continue
        if res.success and res.fun < best_val:
            best, best_val = res.x, res.fun
    if best is None:
        return np.ones(n) / n
    best = np.clip(best, wmin, wmax)
    return best / best.sum()


def opt_max_sharpe(mu, cov, rf=0.0, wmin=0.0, wmax=1.0):
    def neg(w):
        v = np.sqrt(w @ cov @ w)
        return 1e6 if v <= 0 else -(w @ mu - rf) / v
    return _opt_solve(neg, len(mu), wmin, wmax)


def opt_min_vol(mu, cov, wmin=0.0, wmax=1.0):
    return _opt_solve(lambda w: w @ cov @ w, len(mu), wmin, wmax)


def opt_risk_parity(cov, wmin=0.0, wmax=1.0):
    def obj(w):
        rc = risk_contributions(w, cov)
        return float(((rc - rc.mean()) ** 2).sum()) * 1e4
    return _opt_solve(obj, len(cov), max(wmin, 1e-4), wmax)


def opt_target_return(mu, cov, target, wmin=0.0, wmax=1.0):
    extra = [{"type": "eq", "fun": lambda w: float(w @ mu) - target}]
    return _opt_solve(lambda w: w @ cov @ w, len(mu), wmin, wmax, extra)


def efficient_frontier(mu, cov, n_points=30, wmin=0.0, wmax=1.0):
    lo = float(mu @ opt_min_vol(mu, cov, wmin, wmax))
    hi = float(mu.max()) if wmax >= 1.0 else float(mu @ opt_max_sharpe(mu, cov, 0, wmin, wmax))
    hi = max(hi, lo + 1e-6)
    pts = []
    for tgt in np.linspace(lo, hi, n_points):
        w = opt_target_return(mu, cov, tgt, wmin, wmax)
        r, v, _ = port_perf(w, mu, cov)
        if abs(r - tgt) < 5e-3:
            pts.append((v, r, w))
    return pts


# ======================================================================
# 목표 제약이 있는 일반 최적화
# ======================================================================
# ---------------------- 벤치마크 대비 지표 ----------------------
def bench_metrics(rp: pd.Series, rb: pd.Series, rf=0.0) -> dict:
    """알파(연)·베타·R²·상승/하락장 포착률."""
    df = pd.concat([rp, rb], axis=1).dropna()
    if len(df) < 20:
        return {k: np.nan for k in
                ("알파(연,%)", "베타", "R²", "상승장 포착률(%)", "하락장 포착률(%)")}
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    var = b.var()
    beta = float(a.cov(b) / var) if var > 0 else np.nan
    alpha = float(((a.mean() - rf / TRADING_DAYS)
                   - beta * (b.mean() - rf / TRADING_DAYS)) * TRADING_DAYS * 100)
    r2 = float(a.corr(b) ** 2)

    def cap(mask):
        k = int(mask.sum())
        if k < 2:
            return np.nan
        pa = float((1 + a[mask]).prod() ** (1 / k) - 1)
        pb = float((1 + b[mask]).prod() ** (1 / k) - 1)
        return np.nan if abs(pb) < 1e-12 else pa / pb * 100

    return {"알파(연,%)": alpha, "베타": beta, "R²": r2,
            "상승장 포착률(%)": cap(b > 0), "하락장 포착률(%)": cap(b < 0)}


# ---------------------- HRP (계층적 위험 분산) ----------------------
def hrp_weights(R: pd.DataFrame) -> np.ndarray:
    """
    상관관계로 종목을 계층 군집화한 뒤, 트리를 따라 역분산 배분한다.
    기대수익률을 전혀 추정하지 않으므로 과최적화에 상대적으로 강하다.
    """
    n = R.shape[1]
    if n == 1:
        return np.array([1.0])
    corr = R.corr().values
    cov = R.cov().values
    d = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), "single")
    root, _ = to_tree(Z, rd=True)
    order = root.pre_order(lambda x: x.id)

    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            left, right = c[:h], c[h:]

            def cvar(items):
                sub = cov[np.ix_(items, items)]
                iv = 1 / np.diag(sub)
                iv = iv / iv.sum()
                return float(iv @ sub @ iv)

            vl, vr = cvar(left), cvar(right)
            a = 1 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
            w[left] *= a
            w[right] *= (1 - a)
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


# ---------------------- 주기적 재최적화 (워크포워드) ----------------------
REOPT_FREQ = {"한 번만": None, "분기마다": "QS", "매년": "YS"}


def walk_forward(prices, tickers, start, freq_code, train_win, rebal,
                 goal, risk_name, rf, wmin, wmax, min_ret, max_vol,
                 progress=None):
    """
    각 재최적화 시점까지의 데이터로만 비중을 구해 다음 구간에 적용하고 이어붙인다.
    실제로 운용했다면 어땠을지를 재현한다.
    """
    dates = [pd.Timestamp(d) for d in
             pd.date_range(start, prices.index[-1], freq=freq_code)]
    if not dates:
        return None, []
    segs, hist = [], []
    for i, d in enumerate(dates):
        tr_start = prices.index[0] if train_win is None else max(prices.index[0], d - train_win)
        tr = prices.loc[tr_start:d]
        if len(tr) < MIN_TRAIN_DAYS:
            continue
        R = tr[tickers].pct_change().dropna()
        try:
            if goal.startswith("HRP"):
                w = hrp_weights(R)
            else:
                w, _, _ = optimize(R, goal, risk_name, rf, wmin, wmax, min_ret, max_vol)
        except Exception:
            continue
        end = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
        seg = prices.loc[d:end]
        if len(seg) < 2:
            continue
        W = {t: float(x) for t, x in zip(tickers, w)}
        try:
            segs.append(portfolio_returns(seg, W, rebal))
        except Exception:
            continue
        hist.append((d, w))
        if progress:
            progress((i + 1) / len(dates))
    if not segs:
        return None, []
    full = pd.concat(segs)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full, hist


OPT_GOALS = {
    "위험 대비 수익 최대화": "선택한 위험 지표로 나눈 초과수익을 극대화합니다. "
                     "위험 지표를 표준편차로 두면 샤프지수 최대화와 같습니다.",
    "위험 최소화": "수익률과 무관하게 선택한 위험 지표를 최소화합니다.",
    "수익률 최대화": "제약 조건 안에서 기대수익률만 극대화합니다.",
    "위험균형 (Risk Parity)": "각 종목의 위험 기여도가 같아지도록 배분합니다. "
                        "표준편차(공분산) 기반으로 계산합니다.",
    "계층적 위험 분산 (HRP)": "상관관계로 종목을 묶은 뒤 트리를 따라 역분산 배분합니다. "
                       "기대수익률을 추정하지 않아 과최적화에 상대적으로 강합니다.",
}


def feasibility(mu, cov, wmin, wmax, min_ret=None, max_vol=None):
    """
    목표 제약이 달성 가능한지 미리 확인한다.
    반환: (가능여부, 안내문, 달성가능 최대수익률, 달성가능 최소변동성)
    """
    n = len(mu)
    w_hi = _opt_solve(lambda w: -float(w @ mu), n, wmin, wmax)
    ret_max = float(w_hi @ mu)
    w_lo = opt_min_vol(mu, cov, wmin, wmax)
    vol_min = float(np.sqrt(w_lo @ cov @ w_lo))

    if min_ret is not None and min_ret > ret_max + 1e-9:
        return (False, f"목표수익률 {min_ret*100:.2f}% 는 달성할 수 없습니다. "
                       f"현재 종목과 제약으로 가능한 최대 수익률은 **{ret_max*100:.2f}%** 입니다.",
                ret_max, vol_min)
    if max_vol is not None and max_vol < vol_min - 1e-9:
        return (False, f"목표변동성 {max_vol*100:.2f}% 는 달성할 수 없습니다. "
                       f"현재 종목과 제약으로 가능한 최소 변동성은 **{vol_min*100:.2f}%** 입니다.",
                ret_max, vol_min)

    if min_ret is not None and max_vol is not None:
        w_t = opt_target_return(mu, cov, min_ret, wmin, wmax)
        v_at = float(np.sqrt(w_t @ cov @ w_t))
        if v_at > max_vol + 1e-6:
            return (False,
                    f"두 목표를 동시에 만족할 수 없습니다. 수익률 {min_ret*100:.2f}% 를 "
                    f"달성하려면 변동성이 최소 **{v_at*100:.2f}%** 는 되어야 합니다 "
                    f"(목표 {max_vol*100:.2f}%). 목표수익률을 낮추거나 변동성 한도를 높여주세요.",
                    ret_max, vol_min)
    return True, "", ret_max, vol_min


def optimize(R: pd.DataFrame, goal: str, risk_name: str, rf=0.0,
             wmin=0.0, wmax=1.0, min_ret=None, max_vol=None):
    """
    goal 과 risk_name 에 따라 최적 비중을 계산한다.
    min_ret / max_vol 은 부등식 제약 (≥ / ≤).
    """
    mu, cov = ann_stats(R)
    n = R.shape[1]
    d = RISK_DEFS[risk_name]

    extra = []
    if min_ret is not None:
        extra.append({"type": "ineq", "fun": lambda w: float(w @ mu) - min_ret})
    if max_vol is not None:
        extra.append({"type": "ineq",
                      "fun": lambda w: max_vol - float(np.sqrt(w @ cov @ w))})

    if goal.startswith("계층적"):
        return hrp_weights(R), mu, cov

    if goal.startswith("위험균형"):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + extra

        def rp_obj(w):
            rc = risk_contributions(w, cov)
            return float(((rc - rc.mean()) ** 2).sum()) * 1e4
        return _opt_solve(rp_obj, n, max(wmin, 1e-4), wmax, extra), mu, cov

    if goal.startswith("수익률"):
        return _opt_solve(lambda w: -float(w @ mu), n, wmin, wmax, extra), mu, cov

    if goal.startswith("위험 최소화"):
        if d["kind"] == "ratio":
            fn = lambda w: -float(d["fn"](_series(R, w), rf=rf))
        else:
            fn = lambda w: float(d["fn"](_series(R, w), rf=rf))
        return _opt_solve(fn, n, wmin, wmax, extra), mu, cov

    # 위험 대비 수익 최대화
    if d["kind"] == "ratio":
        fn = lambda w: -float(d["fn"](_series(R, w), rf=rf))
    else:
        def fn(w):
            s = _series(R, w)
            risk = float(d["fn"](s, rf=rf))
            if risk <= 1e-12:
                return 1e6
            ann_ret = float(np.asarray(w) @ mu)
            return -(ann_ret - rf) / risk
    return _opt_solve(fn, n, wmin, wmax, extra), mu, cov



# ======================================================================
# 최적화 화면
# ======================================================================
OPT_COLS = ["티커", "종목명", "현재 비중(%)", "최소(%)", "최대(%)"]

TRAIN_WINDOWS = {
    "6개월": pd.DateOffset(months=6), "1년": pd.DateOffset(years=1),
    "2년": pd.DateOffset(years=2), "3년": pd.DateOffset(years=3),
    "5년": pd.DateOffset(years=5), "전체 기간": None,
}
CUTOFFS = {
    "3개월 전": pd.DateOffset(months=3), "6개월 전": pd.DateOffset(months=6),
    "1년 전": pd.DateOffset(years=1), "2년 전": pd.DateOffset(years=2),
    "3년 전": pd.DateOffset(years=3), "직접 지정": None,
}
MIN_TRAIN_DAYS, MIN_OOS_DAYS = 60, 20


def _opt_blank():
    return {"티커": "", "종목명": "", "현재 비중(%)": np.nan,
            "최소(%)": 0.0, "최대(%)": 100.0}


def perf_row(r: pd.Series, rf: float) -> dict:
    """성과 지표 한 줄. 데이터가 모자라면 NaN."""
    if r is None or len(r) < 5:
        return {k: np.nan for k in
                ("수익률(연,%)", "변동성(연,%)", "샤프지수", "소르티노", "최대낙폭(%)")}
    return {
        "수익률(연,%)": cagr(r) * 100,
        "변동성(연,%)": annual_vol(r) * 100,
        "샤프지수": sharpe_ratio(r, rf),
        "소르티노": sortino_ratio(r, rf),
        "최대낙폭(%)": max_drawdown(r) * 100,
    }


def _cmp_table(rows: dict, rf: float) -> pd.DataFrame:
    """원본·최적화·벤치마크 성과표 + 변화율."""
    df = pd.DataFrame({k: perf_row(v, rf) for k, v in rows.items()}).T
    if "원본 포트폴리오" in df.index and "최적화 포트폴리오" in df.index:
        o, n = df.loc["원본 포트폴리오"], df.loc["최적화 포트폴리오"]
        chg = {}
        for c in df.columns:
            if pd.isna(o[c]) or pd.isna(n[c]) or abs(o[c]) < 1e-12:
                chg[c] = np.nan
            else:
                chg[c] = (n[c] - o[c]) / abs(o[c]) * 100
        df.loc["변화율 (Change, %)"] = chg
    return df


def render_optimizer(base_ccy, start_date, end_date, use_div, rf_rate):
    key = "_opt_df"
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame([
            {"티커": t, "종목명": "", "현재 비중(%)": w, "최소(%)": 0.0, "최대(%)": 100.0}
            for t, w in [("AAPL", 25.0), ("NVDA", 25.0), ("SCHD", 25.0), ("SPMO", 25.0)]
        ])[OPT_COLS]

    # ------------------------------------------------------------------
    st.subheader("1️⃣ 현재 포트폴리오 (Current Holdings)")
    st.caption("지금 보유 중인 종목과 비중을 넣으세요. 최적화 결과는 **이 구성과 비교**해서 보여드립니다. "
               "특정 종목을 제한하려면 최소·최대 편입비중을 조정하세요.")

    c1, c2, c3 = st.columns([1, 1.2, 4])
    n = c1.number_input("종목 수", 2, 20, len(st.session_state[key]), step=1, key="_opt_n")
    if int(n) != len(st.session_state[key]):
        cur = st.session_state[key]
        cur = (pd.concat([cur, pd.DataFrame([_opt_blank()] * (int(n) - len(cur)))],
                         ignore_index=True) if int(n) > len(cur)
               else cur.iloc[:int(n)].reset_index(drop=True))
        st.session_state[key] = cur[OPT_COLS]
        st.session_state.pop("_opt_editor", None)
        st.rerun()
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    eq_now = c2.checkbox("비중 균등", key="_opt_eq")

    ed = st.data_editor(
        st.session_state[key], num_rows="fixed", width="stretch",
        key="_opt_editor", column_order=OPT_COLS, hide_index=True,
        column_config={
            "티커": st.column_config.TextColumn("티커", width="small"),
            "종목명": st.column_config.TextColumn("종목명 (자동)", disabled=True, width="medium"),
            "현재 비중(%)": st.column_config.NumberColumn(
                "현재 비중(%)", min_value=0.0, max_value=100.0, step=0.01,
                format="%.2f", width="small", disabled=eq_now),
            "최소(%)": st.column_config.NumberColumn(
                "최소(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small"),
            "최대(%)": st.column_config.NumberColumn(
                "최대(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small"),
        })

    f = ed.copy()
    for c in OPT_COLS:
        if c not in f.columns:
            f[c] = np.nan
    f = f[OPT_COLS]
    f["티커"] = f["티커"].fillna("").astype(str).str.strip()
    f["현재 비중(%)"] = pd.to_numeric(f["현재 비중(%)"], errors="coerce").round(2)
    f["최소(%)"] = pd.to_numeric(f["최소(%)"], errors="coerce").fillna(0.0)
    f["최대(%)"] = pd.to_numeric(f["최대(%)"], errors="coerce").fillna(100.0)

    res_t, res_l, bad = [], [], []
    if any(t for t in f["티커"]):
        with st.spinner("티커 확인 중..."):
            for t in f["티커"]:
                if not t:
                    res_t.append(""); res_l.append(""); continue
                r = probe_ticker(tuple(normalize_ticker(t)))
                if r["ticker"] is None:
                    res_t.append(t); res_l.append("❌ 확인 불가"); bad.append(t)
                else:
                    res_t.append(r["ticker"])
                    res_l.append(f"✅ {r.get('name') or '(종목명 없음)'} · {r['currency']}")
    else:
        res_t, res_l = list(f["티커"]), [""] * len(f)
    f["티커"], f["종목명"] = res_t, res_l

    if eq_now:
        m = f["티커"] != ""
        k = int(m.sum())
        if k:
            base = round(100.0 / k, 2)
            vals = [base] * k
            vals[-1] = round(100.0 - base * (k - 1), 2)
            f.loc[m, "현재 비중(%)"] = vals
        f.loc[~m, "현재 비중(%)"] = np.nan

    if not _frames_equal(f, st.session_state[key]):
        st.session_state[key] = f
        st.session_state.pop("_opt_editor", None)
        st.rerun()

    live = f[f["티커"] != ""].copy()
    wsum = float(pd.to_numeric(live["현재 비중(%)"], errors="coerce").fillna(0).sum())
    st.caption(f"현재 비중 합계 **{wsum:.2f}%**" +
               ("" if abs(wsum - 100) < 0.01 else " · 100%가 아니면 자동 정규화됩니다"))
    if bad:
        st.error(f"확인되지 않는 티커: {', '.join(bad)}")

    # ------------------------------------------------------------------
    st.subheader("2️⃣ 최적화 설정 (Settings)")
    o1, o2 = st.columns([1, 1])
    goal = o1.selectbox("목적", list(OPT_GOALS))
    o1.caption(OPT_GOALS[goal])

    groups = {}
    for k, v in RISK_DEFS.items():
        groups.setdefault(v["group"], []).append(k)
    risk_labels = [f"{g} · {k}" for g in groups for k in groups[g]]
    risk_pick = o2.selectbox("위험을 정의하는 방법", risk_labels,
                             disabled=goal.startswith(("위험균형", "계층적")),
                             help="목적 계산에 쓰이는 위험 척도입니다.")
    risk_name = risk_pick.split(" · ", 1)[1]
    o2.caption(RISK_DEFS[risk_name]["desc"]
               if not goal.startswith(("위험균형", "계층적"))
               else "이 방식은 위험 지표 선택과 무관하게 공분산 구조로 계산됩니다.")

    st.markdown("**목표 제약** — 비워두면 제약 없이 계산합니다")
    g1, g2 = st.columns(2)
    use_ret = g1.checkbox("목표수익률 지정")
    min_ret = (g1.number_input("연 수익률 이상 (%)", -20.0, 200.0, 10.0, step=0.5,
                               label_visibility="collapsed") / 100) if use_ret else None
    use_vol = g2.checkbox("목표변동성 지정")
    max_vol = (g2.number_input("연 변동성 이하 (%)", 0.5, 100.0, 5.0, step=0.5,
                               label_visibility="collapsed") / 100) if use_vol else None
    if use_ret or use_vol:
        st.caption("두 값은 **이상 / 이하** 조건으로 적용됩니다. "
                   "동시에 만족할 수 없으면 달성 가능한 수치를 알려드립니다.")

    p1, p2, p3, p4 = st.columns(4)
    reopt = p4.selectbox("재최적화 주기", list(REOPT_FREQ), index=0,
                         help="'한 번만'은 기준일에 정한 비중을 끝까지 유지합니다. "
                              "분기·매년을 고르면 그 주기마다 그 시점까지의 데이터로 "
                              "비중을 다시 계산해, 실제 운용을 재현합니다.")
    cut_label = p1.selectbox("최적화 기준일 (Optimization Date)", list(CUTOFFS), index=2,
                             help="이 시점까지의 데이터로만 비중을 계산하고, "
                                  "이후 구간으로 실제 성과를 채점합니다.")
    train_label = p2.selectbox("학습 기간 (Training Period)", list(TRAIN_WINDOWS), index=1,
                               help="비중을 계산하는 데 쓸 과거 데이터의 길이입니다.")
    rebal = p3.selectbox("리밸런싱", REBAL_OPTIONS, index=2)

    custom_date = None
    if cut_label == "직접 지정":
        custom_date = st.date_input("기준일 직접 지정",
                                    pd.Timestamp.today() - pd.DateOffset(years=1))

    bench_tk = st.text_input("벤치마크", value="^GSPC",
                             help="비워두면 벤치마크 없이 비교합니다.").strip()

    run_opt = st.button("🎯 포트폴리오 최적화", type="primary", width="stretch",
                        disabled=bool(bad) or live.empty)
    if run_opt:
        st.session_state["_opt_has_run"] = True
    if not st.session_state.get("_opt_has_run"):
        st.info("👆 종목과 현재 비중을 넣고 **포트폴리오 최적화**를 눌러주세요.")
        return

    tickers = list(live["티커"])
    if len(tickers) < 2:
        st.error("최소 2개 종목이 필요합니다.")
        return
    if wsum <= 0:
        st.error("현재 비중을 입력해주세요. 원본 포트폴리오와 비교하는 것이 이 도구의 핵심입니다.")
        return

    bench_norm = normalize_ticker(bench_tk)[0] if bench_tk else None
    need = tickers + ([bench_norm] if bench_norm else [])

    try:
        with st.spinner("데이터 수집 중..."):
            prices, meta, fx_used = build_price_frame(
                need, start_date, end_date, base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    if prices.empty:
        st.error("공통 거래일이 없습니다.")
        return

    # ---------------- 기간 분할 ----------------
    last = prices.index[-1]
    if cut_label == "직접 지정":
        opt_date = pd.Timestamp(custom_date)
    else:
        opt_date = last - CUTOFFS[cut_label]
    tw = TRAIN_WINDOWS[train_label]
    train_start = prices.index[0] if tw is None else max(prices.index[0], opt_date - tw)

    train_px = prices.loc[train_start:opt_date]
    oos_px = prices.loc[opt_date:]

    if len(train_px) < MIN_TRAIN_DAYS:
        st.error(f"학습 구간이 {len(train_px)}일뿐입니다 (최소 {MIN_TRAIN_DAYS}일 필요). "
                 f"학습 기간을 늘리거나 분석 시작일을 앞당겨주세요.")
        return
    if len(oos_px) < MIN_OOS_DAYS:
        st.error(f"검증 구간이 {len(oos_px)}일뿐입니다 (최소 {MIN_OOS_DAYS}일 필요). "
                 f"최적화 기준일을 더 과거로 옮겨주세요.")
        return

    # ---------------- 최적화 (학습 구간만 사용) ----------------
    R_train = train_px[tickers].pct_change().dropna()
    mu, cov = ann_stats(R_train)
    lo = live["최소(%)"].values / 100
    hi = live["최대(%)"].values / 100
    if lo.sum() > 1.0:
        st.error(f"최소 편입비중 합계가 {lo.sum()*100:.0f}%로 100%를 넘습니다.")
        return
    if hi.sum() < 1.0:
        st.error(f"최대 편입비중 합계가 {hi.sum()*100:.0f}%로 100%에 못 미칩니다.")
        return
    wmin, wmax = float(lo.max()), float(hi.min())

    ok, msg, ret_max, vol_min = feasibility(mu, cov, wmin, wmax, min_ret, max_vol)
    if not ok:
        st.error("🚫 " + msg)
        st.caption(f"참고 · 현재 종목·제약으로 달성 가능한 범위: "
                   f"수익률 최대 {ret_max*100:.2f}% / 변동성 최소 {vol_min*100:.2f}%")
        return

    with st.spinner("최적 비중 계산 중..."):
        w_opt, mu, cov = optimize(R_train, goal, risk_name, rf_rate,
                                  wmin, wmax, min_ret, max_vol)

    w_now = pd.to_numeric(live["현재 비중(%)"], errors="coerce").fillna(0).values
    w_now = w_now / w_now.sum()

    W_now = {t: float(x) for t, x in zip(tickers, w_now)}
    W_opt = {t: float(x) for t, x in zip(tickers, w_opt)}

    _rn = "표준편차 (위험균형)" if goal.startswith("위험균형") else risk_name
    _cons = []
    if min_ret is not None:
        _cons.append(f"수익률 ≥ {min_ret*100:.2f}%")
    if max_vol is not None:
        _cons.append(f"변동성 ≤ {max_vol*100:.2f}%")
    st.success(f"✅ 계산 완료 · 목적: **{goal}** · 위험 정의: **{_rn}**"
               + (f" · 제약: {' · '.join(_cons)}" if _cons else "")
               + f"\n\n학습 기간 {train_label} ({train_px.index[0].date()} ~ "
                 f"{opt_date.date()}) · 검증 구간 {opt_date.date()} ~ {last.date()} "
                 f"({len(oos_px):,}일)")

    # ---------------- 최적 자산배분 ----------------
    st.subheader("3️⃣ 최적 자산배분 (Optimal Allocation)")
    alloc = pd.DataFrame({
        "티커": tickers,
        "종목명": [meta.get(t, {}).get("name", t) for t in tickers],
        "원본 비중(%)": w_now * 100,
        "최적 비중(%)": w_opt * 100,
    })
    alloc["변화(%p)"] = alloc["최적 비중(%)"] - alloc["원본 비중(%)"]
    alloc = alloc.sort_values("최적 비중(%)", ascending=False)

    fa = go.Figure()
    fa.add_trace(go.Bar(y=alloc["티커"], x=alloc["원본 비중(%)"], name="원본",
                        orientation="h", marker_color="#cbd5e1",
                        text=[f"{x:.1f}%" for x in alloc["원본 비중(%)"]],
                        textposition="outside"))
    fa.add_trace(go.Bar(y=alloc["티커"], x=alloc["최적 비중(%)"], name="최적화",
                        orientation="h", marker_color="#0d9488",
                        text=[f"{x:.1f}%" for x in alloc["최적 비중(%)"]],
                        textposition="outside"))
    fa.update_layout(barmode="group", height=90 + 56 * len(alloc),
                     margin=dict(l=0, r=50, t=30, b=0),
                     xaxis=dict(title=None, ticksuffix="%"),
                     yaxis=dict(autorange="reversed"),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fa, width="stretch")
    st.dataframe(alloc.style.format({"원본 비중(%)": "{:.2f}", "최적 비중(%)": "{:.2f}",
                                     "변화(%p)": "{:+.2f}"}),
                 width="stretch", hide_index=True)

    with st.expander("🔬 비중 안정성 점검 (Weight Stability)", expanded=False):
        st.caption("학습 기간을 바꿔가며 최적 비중이 얼마나 흔들리는지 봅니다. "
                   "기간을 조금 바꿨는데 비중이 크게 달라진다면, 그 결과는 "
                   "우연에 기댄 것일 수 있습니다.")
        stab_rows = {}
        for lbl in ["6개월", "1년", "2년", "3년"]:
            twx = TRAIN_WINDOWS[lbl]
            tsx = prices.index[0] if twx is None else max(prices.index[0], opt_date - twx)
            tpx = prices.loc[tsx:opt_date]
            if len(tpx) < MIN_TRAIN_DAYS:
                continue
            Rx = tpx[tickers].pct_change().dropna()
            try:
                if goal.startswith("계층적"):
                    wx = hrp_weights(Rx)
                else:
                    wx, _, _ = optimize(Rx, goal, risk_name, rf_rate,
                                        wmin, wmax, min_ret, max_vol)
                stab_rows[lbl] = pd.Series(wx * 100, index=tickers)
            except Exception:
                continue
        if len(stab_rows) >= 2:
            sdf = pd.DataFrame(stab_rows)
            sdf["변동폭(%p)"] = sdf.max(axis=1) - sdf.min(axis=1)
            st.dataframe(sdf.style.format("{:.2f}"), width="stretch")
            worst = float(sdf["변동폭(%p)"].max())
            if worst > 30:
                st.warning(f"학습 기간에 따라 비중이 최대 **{worst:.1f}%p** 까지 달라집니다. "
                           f"결과가 불안정하니 종목 수를 줄이거나 비중 상한을 두는 것을 "
                           f"고려해보세요.")
            else:
                st.success(f"학습 기간을 바꿔도 비중 변동폭이 최대 {worst:.1f}%p 로 "
                           f"비교적 안정적입니다.")
        else:
            st.info("비교할 만큼의 데이터가 부족합니다.")

    if st.button("↗️ 이 비중으로 상세 분석하기", width="stretch",
                 help="최적 비중을 포트폴리오 분석 화면으로 넘겨 자세히 살펴봅니다."):
        st.session_state["df_0"] = pd.DataFrame(
            [{"티커": t, "비중(%)": round(float(x) * 100, 2), "종목명": ""}
             for t, x in zip(tickers, w_opt)])[COLS]
        st.session_state["nrow_0"] = len(tickers)
        st.session_state["name_0"] = f"최적화 ({goal})"
        st.session_state["rebal_0"] = rebal
        st.session_state["_tool"] = "📊 포트폴리오 분석"
        st.session_state["_has_run"] = False
        st.success("포트폴리오 분석 화면으로 옮겼습니다. 왼쪽 도구에서 확인하세요.")
        st.rerun()

    # ---------------- 성과 비교 ----------------
    def _bt(px, W):
        try:
            return portfolio_returns(px, W, rebal)
        except Exception:
            return None

    def _bench(px):
        if not bench_norm or bench_norm not in px.columns:
            return None
        try:
            return portfolio_returns(px, {bench_norm: 100.0}, REBAL_NONE)
        except Exception:
            return None

    st.subheader("4️⃣ 성과 비교 (Performance)")

    st.markdown("#### 표본외 구간 (Out-of-Sample) — 최적화 기준일 이후")
    st.caption("최적화가 **보지 않은** 데이터입니다. 실제 성적표에 해당하며, "
               "최적화가 원본보다 나쁠 수도 있습니다.")
    oos_rows = {"원본 포트폴리오": _bt(oos_px, W_now),
                "최적화 포트폴리오": _bt(oos_px, W_opt)}

    wf_r, wf_hist = None, []
    if REOPT_FREQ[reopt] is not None:
        bar = st.progress(0.0, text="주기적 재최적화 계산 중...")
        try:
            wf_r, wf_hist = walk_forward(
                prices, tickers, opt_date, REOPT_FREQ[reopt], tw, rebal,
                goal, risk_name, rf_rate, wmin, wmax, min_ret, max_vol,
                progress=lambda x: bar.progress(min(1.0, x),
                                                text=f"주기적 재최적화 계산 중... {x*100:.0f}%"))
        except Exception as ex:
            st.warning(f"재최적화 계산 실패: {ex}")
        bar.empty()
        if wf_r is not None and len(wf_r) > 5:
            oos_rows[f"재최적화 ({reopt})"] = wf_r
            st.info(f"🔁 **{reopt}** 재최적화를 {len(wf_hist)}회 수행했습니다. "
                    f"각 시점까지의 데이터만 사용했으므로 미래 정보가 섞이지 않습니다.")

    b_oos = _bench(oos_px)
    if b_oos is not None:
        oos_rows[f"벤치마크 ({bench_norm})"] = b_oos
    t_oos = _cmp_table(oos_rows, rf_rate)
    st.dataframe(t_oos.style.format("{:.2f}", na_rep="-"), width="stretch")

    t_ins = pd.DataFrame()
    with st.expander("표본내 구간 (In-Sample) — 학습에 쓰인 데이터 · 참고용", expanded=False):
        st.caption("최적화가 이 구간을 보고 비중을 정했으므로 좋게 나오는 것이 당연합니다. "
                   "성과 판단의 근거로 삼으면 안 됩니다.")
        ins_rows = {"원본 포트폴리오": _bt(train_px, W_now),
                    "최적화 포트폴리오": _bt(train_px, W_opt)}
        b_ins = _bench(train_px)
        if b_ins is not None:
            ins_rows[f"벤치마크 ({bench_norm})"] = b_ins
        t_ins = _cmp_table(ins_rows, rf_rate)
        st.dataframe(t_ins.style.format("{:.2f}", na_rep="-"), width="stretch")

    # ---------------- 성과 차트 ----------------
    st.subheader("5️⃣ 포트폴리오 성과 (Growth of Investment)")
    unit = 10_000_000 if base_ccy == "KRW" else 10_000
    st.caption(f"최적화 기준일에 {unit:,} {base_ccy}를 투자했다면 어떻게 됐을지 보여줍니다.")
    fp = go.Figure()
    palette = {"원본 포트폴리오": "#94a3b8", "최적화 포트폴리오": "#0d9488"}
    for label, r in oos_rows.items():
        if r is None or len(r) < 2:
            continue
        ec = equity_curve(r) * unit
        col = palette.get(label, "#d97706")
        fp.add_trace(go.Scatter(x=ec.index, y=ec.values, name=label,
                                line=dict(color=col, width=2.2,
                                          dash=None if label in palette else "dash")))
    fp.update_layout(height=420, hovermode="x unified",
                     margin=dict(l=0, r=0, t=30, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                     yaxis=dict(title=None, tickformat=",.0f"))
    st.plotly_chart(fp, width="stretch")

    # ---------------- 벤치마크 대비 지표 ----------------
    if b_oos is not None:
        st.subheader("6️⃣ 벤치마크 대비 지표 (Benchmark Metrics)")
        st.caption("검증 구간 기준입니다. **알파(Alpha)** 는 시장 노출만으로 설명되지 않는 "
                   "초과수익, **베타(Beta)** 는 시장 대비 민감도, **R²** 는 시장으로 설명되는 "
                   "비율입니다. **포착률(Capture Ratio)** 은 시장이 오를 때/내릴 때 "
                   "내 포트폴리오가 그중 몇 %를 따라갔는지 보여줍니다 — "
                   "상승은 높고 하락은 낮을수록 유리합니다.")
        brows = {}
        for label, r in oos_rows.items():
            if r is None or label.startswith("벤치마크"):
                continue
            brows[label] = bench_metrics(r, b_oos, rf_rate)
        bdf = pd.DataFrame(brows).T
        st.dataframe(bdf.style.format("{:.2f}", na_rep="-"), width="stretch")

    # ---------------- 위험 기여도 · 효율적 투자선 ----------------
    st.subheader("7️⃣ 위험 기여도 (Risk Contribution)")
    rc = risk_contributions(w_opt, cov)
    rc_pct = rc / rc.sum() * 100 if rc.sum() else np.zeros_like(rc)
    rdf = pd.DataFrame({"티커": tickers, "최적 비중(%)": w_opt * 100,
                        "위험 기여도(%)": rc_pct}).sort_values("위험 기여도(%)", ascending=False)
    st.dataframe(rdf.style.format({"최적 비중(%)": "{:.2f}", "위험 기여도(%)": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("전체 변동성에서 각 종목이 차지하는 몫입니다. 비중이 낮아도 변동이 크면 높게 나옵니다. "
               "학습 구간 기준으로 계산했습니다.")

    with st.expander("효율적 투자선 (Efficient Frontier) — 학습 구간 기준", expanded=False):
        ef = efficient_frontier(mu, cov, 25, wmin, wmax)
        if ef:
            fe = go.Figure()
            fe.add_trace(go.Scatter(x=[v*100 for v, _, _ in ef], y=[r*100 for _, r, _ in ef],
                                    mode="lines", name="효율적 투자선",
                                    line=dict(color="#94a3b8", width=2)))
            for i, t in enumerate(tickers):
                fe.add_trace(go.Scatter(x=[np.sqrt(cov[i, i])*100], y=[mu[i]*100],
                                        mode="markers+text", text=[t],
                                        textposition="top center", showlegend=False,
                                        marker=dict(size=8, color="#cbd5e1")))
            r_o, v_o, _ = port_perf(w_now, mu, cov, rf_rate)
            r_n, v_n, _ = port_perf(w_opt, mu, cov, rf_rate)
            fe.add_trace(go.Scatter(x=[v_o*100], y=[r_o*100], mode="markers+text",
                                    text=["원본"], textposition="bottom center",
                                    marker=dict(size=13, color="#64748b", symbol="diamond"),
                                    name="원본"))
            fe.add_trace(go.Scatter(x=[v_n*100], y=[r_n*100], mode="markers+text",
                                    text=["최적"], textposition="top center",
                                    marker=dict(size=16, color="#0d9488", symbol="star"),
                                    name="최적화"))
            fe.update_layout(height=430, xaxis=dict(title="변동성 (연, %)"),
                             yaxis=dict(title="기대수익률 (연, %)"),
                             margin=dict(l=0, r=0, t=20, b=0),
                             legend=dict(orientation="h", y=1.02, yanchor="bottom"))
            st.plotly_chart(fe, width="stretch")
        else:
            st.info("제약 조건이 빡빡해 투자선을 그릴 수 없습니다.")

    # ---------------- 추가 분석 ----------------
    st.subheader("8️⃣ 시간에 따른 비중 변화 (Allocation Over Time)")
    st.caption("검증 구간에서 실제 보유 비중이 어떻게 흘러갔는지 보여줍니다. "
               "리밸런싱 주기에 따라 모양이 달라집니다.")
    wd = pd.DataFrame()
    try:
        r_opt_oos = oos_rows["최적화 포트폴리오"]
        wd = (weight_drift(oos_px, W_opt, rebal).loc[r_opt_oos.index] * 100)
        fw2 = go.Figure()
        for c in wd.columns:
            fw2.add_trace(go.Scatter(x=wd.index, y=wd[c], name=c, stackgroup="one",
                                     mode="lines", line=dict(width=0.5)))
        fw2.update_layout(height=320, hovermode="x unified", yaxis_range=[0, 100],
                          yaxis=dict(ticksuffix="%"), margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fw2, width="stretch")
    except Exception as ex:
        st.warning(f"비중 추이를 그릴 수 없습니다: {ex}")

    win = st.select_slider("롤링 구간", options=[63, 126, 252], value=126,
                           format_func=lambda x: {63: "3개월", 126: "6개월", 252: "1년"}[x],
                           help="아래 세 차트의 이동 계산 구간입니다.")

    st.subheader("9️⃣ 샤프지수 추이 (Rolling Sharpe)")
    st.caption(f"{'3개월' if win==63 else '6개월' if win==126 else '1년'} 이동 샤프지수입니다. "
               "특정 시기에만 좋았는지, 꾸준했는지를 보여줍니다.")
    fs = go.Figure()
    for label, r in oos_rows.items():
        if r is None or len(r) < win + 5:
            continue
        roll = (r.rolling(win).mean() - rf_rate / TRADING_DAYS) / r.rolling(win).std() \
               * np.sqrt(TRADING_DAYS)
        fs.add_trace(go.Scatter(x=roll.index, y=roll.values, name=label,
                                line=dict(color=palette.get(label, "#d97706"), width=1.8,
                                          dash=None if label in palette else "dash")))
    fs.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fs.update_layout(height=320, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fs, width="stretch")

    st.subheader("🔟 낙폭 (Drawdowns)")
    st.caption("직전 고점 대비 얼마나 내려와 있는지를 매일 표시한 것입니다. "
               "가장 깊이 파인 지점이 곧 위 비교표의 **최대낙폭(MDD)** 이며, "
               "0으로 돌아오는 데 걸린 기간이 원금 회복에 걸린 시간입니다.")
    fd = go.Figure()
    for label, r in oos_rows.items():
        if r is None or len(r) < 2:
            continue
        dd = drawdown_series(r) * 100
        fd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=label,
                                line=dict(color=palette.get(label, "#d97706"), width=1.8,
                                          dash=None if label in palette else "dash")))
    fd.update_layout(height=320, hovermode="x unified", yaxis=dict(ticksuffix="%"),
                     margin=dict(l=0, r=0, t=20, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fd, width="stretch")

    st.subheader("1️⃣1️⃣ 변동성 추이 (Rolling Volatility)")
    st.caption(f"일별 수익률의 이동 표준편차를 연율화(×√252)한 값입니다. "
               f"위 슬라이더에서 고른 {'3개월' if win==63 else '6개월' if win==126 else '1년'} "
               f"구간을 뒤돌아보며 계산하므로, 값이 치솟은 시점은 그 직전에 큰 등락이 "
               f"있었다는 뜻입니다.")
    fv = go.Figure()
    for label, r in oos_rows.items():
        if r is None or len(r) < win + 5:
            continue
        rv = r.rolling(win).std() * np.sqrt(TRADING_DAYS) * 100
        fv.add_trace(go.Scatter(x=rv.index, y=rv.values, name=label,
                                line=dict(color=palette.get(label, "#d97706"), width=1.8,
                                          dash=None if label in palette else "dash")))
    fv.update_layout(height=320, hovermode="x unified", yaxis=dict(ticksuffix="%"),
                     margin=dict(l=0, r=0, t=20, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fv, width="stretch")

    st.subheader("1️⃣2️⃣ 자산 상관관계 (Asset Correlations)")
    st.caption("검증 구간의 일별 수익률 기준입니다. 1에 가까우면 같이 움직이고, "
               "0에 가까우면 서로 무관하며, 음수면 반대로 움직입니다. "
               "낮은 값이 많을수록 분산 효과가 큽니다.")
    corr_cols = list(tickers)
    R_oos = oos_px[corr_cols].pct_change().dropna()
    extra_cols = {}
    if oos_rows.get("최적화 포트폴리오") is not None:
        extra_cols["최적화 포트폴리오"] = oos_rows["최적화 포트폴리오"]
    if bench_norm and bench_norm in oos_px.columns:
        extra_cols[f"벤치마크({bench_norm})"] = oos_px[bench_norm].pct_change()
    C = R_oos.copy()
    for k, v in extra_cols.items():
        C[k] = v.reindex(C.index)
    corr = C.dropna().corr().round(2)
    st.dataframe(corr.style.format("{:.2f}").map(_corr_color), width="stretch")

    st.divider()
    st.subheader("📥 결과 내보내기")
    opt_settings = {
        "목적": goal,
        "위험 정의": _rn,
        "목표 제약": " · ".join(_cons) if _cons else "없음",
        "재최적화 주기": reopt,
        "최적화 기준일": str(opt_date.date()),
        "학습 기간": f"{train_label} ({train_px.index[0].date()} ~ {opt_date.date()})",
        "검증 구간": f"{opt_date.date()} ~ {last.date()} ({len(oos_px):,}일)",
        "리밸런싱": rebal,
        "기준 통화": base_ccy,
        "무위험 수익률": f"{rf_rate*100:.2f}%",
        "벤치마크": bench_norm or "-",
        "배당 처리": "재투자 (총수익)" if use_div else "주가만 (배당 제외)",
        "생성 시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    x1, x2 = st.columns(2)
    try:
        xb = build_opt_excel(alloc, t_oos, t_ins, oos_rows,
                             bdf if b_oos is not None else None,
                             rdf, corr, wd, opt_settings, opt_date)
        x1.download_button("📊 엑셀 파일 받기 (차트 포함)", xb,
                           f"optimization_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")
    except Exception as ex:
        x1.error(f"엑셀 생성 실패: {ex}")
    x2.download_button("📄 최적 비중 CSV", alloc.to_csv(index=False).encode("utf-8-sig"),
                       f"optimal_weights_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                       "text/csv", width="stretch")
    st.caption("엑셀은 화면과 같은 순서(최적자산배분 → 표본외성과 → 표본내성과 → 성과추이 → "
               "벤치마크지표 → 위험기여도 → 비중추이 → 자산상관관계 → 설정)로 구성되며, "
               "성과추이와 비중추이에는 엑셀 네이티브 차트가 포함됩니다.")
    st.caption("최적화는 학습 구간의 평균·변동성·상관관계가 앞으로도 유지된다고 가정합니다. "
               "특히 기대수익률 추정은 오차가 커서, 표본외 성과가 원본보다 나쁜 경우도 흔합니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")



# ======================================================================
# 사이드바
# ======================================================================
with st.sidebar:
    tool = st.radio("도구", ["📊 포트폴리오 분석", "🎯 포트폴리오 최적화"],
                    label_visibility="collapsed", key="_tool")
    st.divider()
    st.header("⚙️ 공통 설정")
    base_ccy = st.selectbox("기준 통화", ["KRW", "USD", "JPY", "EUR", "GBP"], index=0,
                            help="모든 자산을 이 통화로 환산해 비교합니다.")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("시작일", pd.Timestamp("2018-01-01"))
    end_date = c2.date_input("종료일", pd.Timestamp.today())

    st.divider()
    div_mode = st.radio(
        "배당 처리",
        ["배당 재투자 (총수익)", "주가만 (배당 제외)"],
        index=0,
        help="재투자를 선택하면 받은 배당으로 같은 종목을 다시 산다고 가정합니다. "
             "두 경우 모두 액면분할은 보정됩니다.",
    )
    use_div = div_mode.startswith("배당 재투자")

    cost_bp = st.number_input("거래비용 (편도, bp)", 0.0, 200.0, 0.0, step=1.0,
                              help="1bp = 0.01%. 매수·매도 각각에 적용됩니다. "
                                   "국내주식 위탁수수료+세금은 보통 20~30bp 수준입니다. "
                                   "0이면 비용을 반영하지 않습니다.")
    fx_hedge = st.checkbox("환헤지 가정 (환율 고정)", value=False,
                           help="체크하면 환율 변동을 제거하고 종목 자체 수익률만 봅니다.")
    gap_fill = st.checkbox("휴장일 직전값으로 채우기", value=False,
                           help="국가별 휴장일이 다를 때 공통 거래일이 크게 줄어드는 것을 "
                                "막습니다. 체크하면 없는 날은 직전 종가로 채웁니다.")

    rf_rate = st.number_input("무위험 수익률 (연, %)", value=3.0, step=0.25,
                              help="Sharpe·Sortino 계산에만 쓰입니다. "
                                   "기준 통화에 맞는 금리를 넣으세요.") / 100

    st.divider()
    st.subheader("벤치마크")
    bench_raw = st.text_area(
        "벤치마크 티커 (콤마 또는 줄바꿈으로 구분)",
        value="^KS11, ^GSPC",
        height=80,
        help="^KS11 코스피 · ^KQ11 코스닥 · ^GSPC S&P500 · ^N225 닛케이225 · ^IXIC 나스닥",
    )
    bench_list = []
    for b in bench_raw.replace("\n", ",").split(","):
        b = b.strip()
        if b:
            cands = normalize_ticker(b)
            bench_list.append(cands[0] if cands else b)

    if st.button("🔍 벤치마크 티커 확인", width="stretch"):
        if not bench_list:
            st.caption("확인할 티커가 없습니다.")
        else:
            with st.spinner("확인 중..."):
                results = []
                for b in bench_list:
                    tk, ccy = quick_check(tuple(normalize_ticker(b)))
                    results.append((b, tk, ccy, get_name(tk) if tk else ""))
            for b, tk, ccy, nm in results:
                if tk:
                    st.markdown(
                        f"<div style='font-size:0.8rem;line-height:1.35'>"
                        f"<b>{tk}</b><br>{nm or '(종목명 없음)'} · {ccy}</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='font-size:0.8rem;line-height:1.35;color:#dc2626'>"
                        f"<b>{b}</b><br>❌ 확인 불가</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("💾 저장 / 불러오기")
    saved = load_saved()
    if saved:
        pick = st.selectbox("저장된 구성", ["(선택)"] + sorted(saved.keys()))
        lc1, lc2 = st.columns(2)
        if lc1.button("📂 불러오기", width="stretch", disabled=(pick == "(선택)")):
            st.session_state["_pending_load"] = pick
            st.rerun()
        if lc2.button("🗑 삭제", width="stretch", disabled=(pick == "(선택)")):
            saved.pop(pick, None)
            write_saved(saved)
            st.rerun()
    else:
        st.caption("아직 저장된 구성이 없습니다.")

    save_name = st.text_input("저장할 이름", key="_save_name",
                              placeholder="예: 한미일 분산 3종")
    if st.button("💾 현재 구성 저장", width="stretch",
                 disabled=not save_name.strip()):
        # 표 갱신으로 화면이 다시 그려져도 유실되지 않도록 세션에 보관
        st.session_state["_pending_save"] = save_name.strip()

    st.caption("저장한 구성은 이 브라우저 세션에 보관됩니다. "
               "다음에도 쓰려면 아래에서 파일로 내려받아 두세요.")

    dl1, dl2 = st.columns(2)
    if saved:
        dl1.download_button(
            "📥 내보내기", json.dumps(saved, ensure_ascii=False, indent=2)
            .encode("utf-8"), "portfolios.json", "application/json",
            width="stretch", help="저장된 구성 전체를 파일로 내려받습니다.")
    up = dl2.file_uploader("📤 가져오기", type=["json"],
                           label_visibility="collapsed",
                           help="내려받아 둔 portfolios.json 을 올리면 복원됩니다.")
    if up is not None and not st.session_state.get("_imported_once"):
        try:
            incoming = json.loads(up.getvalue().decode("utf-8"))
            if isinstance(incoming, dict):
                merged = dict(load_saved()); merged.update(incoming)
                write_saved(merged)
                st.session_state["_imported_once"] = True
                st.success(f"{len(incoming)}개 구성을 불러왔습니다.")
                st.rerun()
        except Exception as ex:
            st.error(f"파일을 읽지 못했습니다: {ex}")

    st.divider()
    st.caption("**티커 예시**\n\n"
               "한국 `005930.KS` `086520.KQ`\n\n"
               "미국 `NVDA` `SPY` `QQQ`\n\n"
               "일본 `7203.T`  홍콩 `0700.HK`")


IS_OPT = tool.endswith("최적화")

if IS_OPT:
    st.title("🎯 포트폴리오 최적화")
    st.caption("종목을 넣으면 목적에 맞는 최적 비중을 계산합니다. "
               "과거 데이터 기반이며 미래 성과를 보장하지 않습니다.")
    render_optimizer(base_ccy, start_date, end_date, use_div, rf_rate)
    st.stop()

st.title("📊 포트폴리오 분석")
st.caption("여러 포트폴리오와 벤치마크를 한 화면에서 비교합니다. 통화는 자동으로 인식됩니다.")


# ======================================================================
# 포트폴리오 입력
# ======================================================================
COLS = ["티커", "비중(%)", "종목명"]
DEFAULT_ROWS = 3
MIN_ROWS, MAX_ROWS = 1, 30
PRESETS = [
    [("005930.KS", 60.0), ("SPY", 40.0)],
    [("SPY", 60.0), ("AGG", 40.0)],
]


def blank_row():
    return {"티커": "", "비중(%)": np.nan, "종목명": ""}


def default_holdings(i: int) -> pd.DataFrame:
    rows = list(PRESETS[i]) if i < len(PRESETS) else []
    data = [{"티커": t, "비중(%)": w, "종목명": ""} for t, w in rows]
    while len(data) < DEFAULT_ROWS:
        data.append(blank_row())
    return pd.DataFrame(data)[COLS]


def fit_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """행 수를 n개로 맞춘다 (모자라면 빈 행 추가, 남으면 뒤에서 자름)."""
    df = df.copy()
    if len(df) < n:
        df = pd.concat([df, pd.DataFrame([blank_row()] * (n - len(df)))],
                       ignore_index=True)
    elif len(df) > n:
        df = df.iloc[:n].reset_index(drop=True)
    return df[COLS]


# --- 저장된 구성 불러오기 (위젯이 그려지기 전에 적용해야 함) ---
if "_pending_load" in st.session_state:
    key = st.session_state.pop("_pending_load")
    cfg = load_saved().get(key)
    if cfg:
        for k in [k for k in st.session_state
                  if k.startswith(("df_", "hold_", "name_", "rebal_", "eqw_", "nrow_"))]:
            st.session_state.pop(k, None)
        ports = cfg.get("portfolios", [])
        st.session_state["_n_port"] = max(1, min(MAX_PORTFOLIOS, len(ports)))
        for i, p in enumerate(ports[:MAX_PORTFOLIOS]):
            rows = [{"티커": h["ticker"], "비중(%)": (np.nan if h.get("weight") is None
                                                   else h["weight"]), "종목명": ""}
                    for h in p.get("holdings", [])]
            if not rows:
                rows = [blank_row()]
            st.session_state[f"df_{i}"] = pd.DataFrame(rows)[COLS]
            st.session_state[f"nrow_{i}"] = len(rows)
            st.session_state[f"name_{i}"] = p.get("name", f"포트{i+1}")
            st.session_state[f"rebal_{i}"] = p.get("rebalance", REBAL_QUARTER)
        st.success(f"**{cfg.get('label', key)}** 구성을 불러왔습니다.")

st.subheader("1️⃣ 포트폴리오 구성 (Holdings)")
st.caption("티커는 `005930.KS` · `005930 KS`(블룸버그) · `005930`(숫자만) 모두 인식합니다. "
           "엑셀에서 **티커·비중 두 열**을 복사해 첫 칸에 붙여넣으면 한 번에 채워집니다.")

n_port = st.number_input("포트폴리오 개수", 1, MAX_PORTFOLIOS,
                         int(st.session_state.pop("_n_port", 2)), step=1)

port_specs = []
tabs = st.tabs([f"포트{i+1}" for i in range(int(n_port))])

for i, tab in enumerate(tabs):
    with tab:
        c1, c2, c3, c4 = st.columns([2.2, 2.2, 1.2, 1.4])
        name = c1.text_input("이름", value=f"포트{i+1}", key=f"name_{i}")
        rebal = c2.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key=f"rebal_{i}")

        dfkey = f"df_{i}"
        if dfkey not in st.session_state:
            st.session_state[dfkey] = default_holdings(i)

        n_rows = c3.number_input("종목 수", MIN_ROWS, MAX_ROWS,
                                 len(st.session_state[dfkey]), step=1, key=f"nrow_{i}")
        if int(n_rows) != len(st.session_state[dfkey]):
            st.session_state[dfkey] = fit_rows(st.session_state[dfkey], int(n_rows))
            st.session_state.pop(f"hold_{i}", None)
            st.rerun()

        c4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        equal_w = c4.checkbox("균등 비중", key=f"eqw_{i}",
                              help="체크하면 입력된 종목에 100%를 균등 배분합니다.")

        edited = st.data_editor(
            st.session_state[dfkey],
            num_rows="fixed", width="stretch", key=f"hold_{i}",
            column_order=COLS, hide_index=True,
            column_config={
                "티커": st.column_config.TextColumn(
                    "티커", width="small",
                    help="005930.KS · 005930 KS · 005930 · NVDA 모두 가능"),
                "비중(%)": st.column_config.NumberColumn(
                    "비중(%)", min_value=0.0, max_value=100.0,
                    step=0.01, format="%.2f", width="small", disabled=equal_w),
                "종목명": st.column_config.TextColumn(
                    "종목명 (자동)", disabled=True, width="large"),
            },
        )

        # ---- 정규화 ----
        filled = edited.copy()
        for c in COLS:
            if c not in filled.columns:
                filled[c] = np.nan
        filled = filled[COLS]
        filled["티커"] = filled["티커"].fillna("").astype(str).str.strip()
        filled["비중(%)"] = pd.to_numeric(filled["비중(%)"], errors="coerce").round(2)

        # ---- 1단계: 티커 형식 변환 + 빠른 검증 ----
        resolved, labels, need_name, thin = [], [], [], []
        raw_list = list(filled["티커"])
        if any(t for t in raw_list):
            with st.spinner("티커 확인 중..."):
                for t in raw_list:
                    if not t:
                        resolved.append(""); labels.append(""); continue
                    r = probe_ticker(tuple(normalize_ticker(t)))
                    if r["ticker"] is None:
                        resolved.append(t); labels.append("❌ 확인 불가")
                    else:
                        resolved.append(r["ticker"])
                        nm = r.get("name") or ""
                        if nm:
                            labels.append(f"✅ {nm} · {r['currency']}")
                        else:
                            labels.append(f"✅ … · {r['currency']}")
                            need_name.append((len(resolved) - 1, r["ticker"], r["currency"]))
                        if r["rows"] < 200:
                            thin.append((r["ticker"], r["rows"], r["first"]))
        else:
            resolved = list(raw_list); labels = [""] * len(raw_list)

        filled["티커"] = resolved
        filled["종목명"] = labels

        # ---- 2단계: 종목명 채우기 ----
        for idx, tk, ccy in need_name:
            nm = get_name(tk)
            filled.iat[idx, COLS.index("종목명")] = (
                f"✅ {nm} · {ccy}" if nm else f"✅ (종목명 없음) · {ccy}")

        # ---- 균등 비중 ----
        if equal_w:
            mask = filled["티커"] != ""
            k = int(mask.sum())
            if k > 0:
                base = round(100.0 / k, 2)
                vals = [base] * k
                vals[-1] = round(100.0 - base * (k - 1), 2)
                filled.loc[mask, "비중(%)"] = vals
            filled.loc[~mask, "비중(%)"] = np.nan

        if not _frames_equal(filled, st.session_state[dfkey]):
            st.session_state[dfkey] = filled
            st.session_state.pop(f"hold_{i}", None)
            st.rerun()

        v = filled[filled["티커"].astype(str).str.strip() != ""].copy()
        tot = float(pd.to_numeric(v["비중(%)"], errors="coerce").fillna(0).sum()) if not v.empty else 0.0
        bad = v[v["종목명"].astype(str).str.startswith("❌")]["티커"].tolist()

        m1, m2 = st.columns([1, 1])
        m1.caption(f"비중 합계 **{tot:.2f}%**"
                   + ("" if abs(tot - 100) < 0.01 else " · 100%가 아니면 자동 정규화됩니다"))
        if bad:
            m2.caption(f"⚠️ 확인 실패: **{', '.join(bad)}**")
        elif not v.empty:
            m2.caption("모든 티커 정상 ✅")

        if thin:
            txt = " · ".join(f"**{tk}** 최근 1년 {n}일 (시작 {f})" for tk, n, f in thin)
            st.warning(f"데이터 이력이 짧은 종목이 있습니다 — {txt}\n\n"
                       f"신규 상장이면 정상이지만, 아니라면 티커가 잘못 잡혔을 수 있습니다. "
                       f"이런 종목이 있으면 **전체 분석 기간이 그 종목에 맞춰 줄어듭니다.**")

        if bad:
            with st.expander("🩺 조회 실패 원인 보기", expanded=False):
                st.caption("각 티커에 대해 시도한 방법과 실패 사유입니다.")
                for t in bad:
                    r = probe_ticker(tuple(normalize_ticker(t)))
                    st.markdown(f"**{t}** → 시도한 후보: `{', '.join(normalize_ticker(t))}`")
                    if r["log"]:
                        st.code("\n".join(r["log"]), language="text")
                    else:
                        st.code("기록된 오류 없음", language="text")
                st.caption(f"yfinance {getattr(yf, '__version__', '버전 확인 불가')} · "
                           "대부분 `pip install -U yfinance` 로 해결됩니다.")

        port_specs.append({"name": name, "rebalance": rebal, "holdings": v})


run = st.button("🚀 분석 실행", type="primary", width="stretch")

# --- 현재 구성 저장 ---
if "_pending_save" in st.session_state:
    nm = st.session_state.pop("_pending_save")
    store = load_saved()
    store[nm] = snapshot(port_specs, bench_list, base_ccy, use_div, rf_rate, label=nm)
    if write_saved(store):
        st.success(f"**{nm}** 으로 저장했습니다. "
                   f"사이드바의 **내보내기**로 파일을 받아두면 다음에도 쓸 수 있습니다.")
    else:
        st.error(f"저장 실패 — {SAVE_PATH} 에 쓸 권한이 있는지 확인해주세요.")

# ======================================================================
# 실행
# ======================================================================
if run:
    st.session_state["_has_run"] = True
if not st.session_state.get("_has_run"):
    st.info("👆 종목과 비중을 입력한 뒤 **분석 실행**을 눌러주세요. 통화는 자동으로 인식합니다.")
    st.stop()

def _weight_sum(df):
    return float(pd.to_numeric(df["비중(%)"], errors="coerce").fillna(0).sum())


active = [p for p in port_specs
          if not p["holdings"].empty and _weight_sum(p["holdings"]) > 0]
if not active:
    st.error("최소 하나의 포트폴리오에 종목과 비중을 입력해주세요.")
    st.stop()

# 잘못된 티커가 있으면 여기서 멈춘다 (그냥 두면 데이터 수집 단계에서 통째로 실패)
invalid = []
for p in active:
    for _, r in p["holdings"].iterrows():
        if str(r.get("종목명", "")).startswith("❌"):
            invalid.append(f"{p['name']} → {str(r['티커']).strip()}")
if invalid:
    st.error("확인되지 않는 티커가 있습니다. 수정 후 다시 실행해주세요.\n\n"
             + "\n".join(f"- {x}" for x in invalid))
    st.stop()

all_tickers = sorted({str(r["티커"]).strip()
                      for p in active for _, r in p["holdings"].iterrows()}
                     | set(bench_list))

# ---------------- 데이터 수집 ----------------
try:
    with st.spinner(f"{len(all_tickers)}개 종목 데이터 수집 및 환율 환산 중..."):
        prices, meta, fx_used = build_price_frame(
            all_tickers, start_date, end_date, base_ccy, use_div,
            fx_hedge, gap_fill)
except Exception as e:
    st.error(f"데이터를 가져오지 못했습니다: {e}")
    st.stop()

if prices.empty:
    st.error("공통 거래일이 없습니다. 기간이나 종목 구성을 확인해주세요.")
    st.stop()

# ---------------- 종목별 데이터 구간 점검 ----------------
# 한 종목의 이력이 짧으면 공통 거래일이 그만큼 잘려나가므로 반드시 눈에 띄게 알린다.
cov = []
for t in prices.columns:
    col = prices[t].dropna()
    if col.empty:
        continue
    cov.append({"티커": t, "종목명": meta.get(t, {}).get("name", t),
                "통화": meta.get(t, {}).get("currency", "-"),
                "시작": col.index[0].date(), "종료": col.index[-1].date(),
                "일수": len(col)})
cov_df = pd.DataFrame(cov).sort_values("일수")

if not cov_df.empty:
    longest = int(cov_df["일수"].max())
    common_n = len(prices)
    binding = cov_df[cov_df["일수"] < longest * 0.9]

    if not binding.empty and common_n < longest * 0.9:
        lost = longest - common_n
        st.error(
            f"⚠️ **기간이 크게 줄었습니다** — 가장 긴 종목은 {longest:,}일치인데 "
            f"공통 거래일은 {common_n:,}일입니다 ({lost:,}일 손실).\n\n"
            f"아래 종목의 이력이 짧아 전체 기간을 제한하고 있습니다. "
            f"티커가 맞는지 확인하거나, 해당 종목을 빼고 다시 실행해보세요."
        )
        st.dataframe(binding, width="stretch", hide_index=True)

# ---------------- 인식된 종목 정보 ----------------
with st.expander("🔍 인식된 종목 정보 (통화·데이터 구간)", expanded=False):
    show = cov_df.copy()
    if not show.empty:
        show["환산"] = [("그대로" if c == base_ccy else f"{c} → {base_ccy}")
                       for c in show["통화"]]
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("통화는 야후 파이낸스가 제공하는 종목 정보에서 직접 읽어옵니다. "
               "'일수'가 유독 적은 종목이 있으면 티커를 다시 확인해주세요.")

# ---------------- 적용된 환율 ----------------
if fx_used and not fx_hedge:
    with st.expander("💱 적용된 환율", expanded=False):
        st.caption(f"각 종목의 거래통화를 {base_ccy}로 환산하는 데 쓴 환율입니다. "
                   f"환율 변동도 수익률에 포함돼 있습니다. "
                   f"환율 영향을 빼고 보려면 사이드바의 **환헤지 가정**을 켜세요.")
        ff = go.Figure()
        for ccy, s in fx_used.items():
            if s is None or s.empty:
                continue
            fx_s = s.reindex(prices.index).ffill().bfill()
            ff.add_trace(go.Scatter(x=fx_s.index, y=fx_s.values,
                                    name=f"1 {ccy} → {base_ccy}"))
        ff.update_layout(height=280, hovermode="x unified",
                         margin=dict(l=0, r=0, t=20, b=0),
                         legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(ff, width="stretch")
        if base_ccy == "KRW" and "JPY" in fx_used:
            st.caption("※ 엔화는 **1엔당 원화**입니다. 뉴스의 '100엔당' 표기와 100배 차이납니다.")
elif fx_hedge:
    st.info("💱 **환헤지 가정**이 켜져 있습니다. 환율 변동을 제거하고 "
            "종목 자체 수익률만 계산했습니다.")

# ---------------- 수익률 계산 ----------------
series, errors = {}, []

for p in active:
    w = {}
    for _, r in p["holdings"].iterrows():
        t = str(r["티커"]).strip()
        wt = pd.to_numeric(r["비중(%)"], errors="coerce")
        if t and pd.notna(wt) and wt > 0:
            w[t] = float(wt)
    if not w:
        continue
    try:
        series[p["name"]] = {
            "returns": portfolio_returns(prices, w, p["rebalance"]),
            "kind": "portfolio", "weights": w, "rebalance": p["rebalance"],
        }
    except Exception as e:
        errors.append(f"{p['name']}: {e}")

for b in bench_list:
    if b not in prices.columns:
        errors.append(f"벤치마크 {b}: 데이터 없음")
        continue
    try:
        series[b] = {
            "returns": portfolio_returns(prices, {b: 100.0}, REBAL_NONE),
            "kind": "benchmark", "weights": {b: 100.0}, "rebalance": REBAL_NONE,
        }
    except Exception as e:
        errors.append(f"벤치마크 {b}: {e}")

for msg in errors:
    st.warning(msg)
if not series:
    st.error("계산 가능한 포트폴리오가 없습니다.")
    st.stop()

# ---------------- 공통 기간으로 정렬 ----------------
common = None
for v in series.values():
    common = v["returns"].index if common is None else common.intersection(v["returns"].index)
for v in series.values():
    v["returns"] = v["returns"].loc[common]

st.success(f"✅ 분석 완료 · 공통 거래일 **{len(common):,}일** "
           f"({common[0].date()} ~ {common[-1].date()}) · "
           f"{'배당 재투자 포함' if use_div else '주가만 (배당 제외)'}")

# ======================================================================
# 결과
# ======================================================================
st.subheader("2️⃣ 성과 차트 (Performance)")

# --- 선 색상 지정 ---
default_color = {}
pi = bi = 0
for name, v in series.items():
    if v["kind"] == "portfolio":
        default_color[name] = PALETTE[pi % len(PALETTE)]; pi += 1
    else:
        default_color[name] = BENCH_PALETTE[bi % len(BENCH_PALETTE)]; bi += 1

with st.expander("🎨 선 색상 변경", expanded=False):
    st.caption("추천 색을 누르면 해당 시계열에 바로 적용됩니다.")
    for name in series:
        st.markdown(f"**{name}**")
        pcols = st.columns(len(SWATCHES) + 2)
        for k, (label, hexv) in enumerate(SWATCHES):
            if pcols[k].button(" ", key=f"sw_{name}_{k}", help=f"{label} {hexv}",
                               width="stretch"):
                st.session_state[f"color_{name}"] = hexv
                st.rerun()
            pcols[k].markdown(
                f"<div style='height:14px;border-radius:4px;background:{hexv};"
                f"margin:-10px 0 6px 0'></div>", unsafe_allow_html=True)
        pcols[-2].color_picker("직접", value=default_color[name], key=f"color_{name}",
                               label_visibility="collapsed")

color_of = {n: st.session_state.get(f"color_{n}", default_color[n]) for n in series}

fig = go.Figure()
for name, v in series.items():
    ec = equity_curve(v["returns"])
    is_port = v["kind"] == "portfolio"
    fig.add_trace(go.Scatter(
        x=ec.index, y=ec.values, name=name,
        line=dict(color=color_of[name], width=2.4 if is_port else 1.5,
                  dash=None if is_port else "dash")))
fig.update_layout(height=460, hovermode="x unified",
                  legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                  margin=dict(l=0, r=0, t=30, b=0))
fig.update_yaxes(title_text=None)

log_scale = st.checkbox("로그 스케일로 보기", value=False,
                        help="기간이 길면 로그 스케일이 초기 구간 변화를 더 잘 보여줍니다.")
if log_scale:
    fig.update_yaxes(type="log")
st.plotly_chart(fig, width="stretch")

# ---------------- Drawdown ----------------
st.subheader("3️⃣ 낙폭 비교 (Drawdowns)")
fig2 = go.Figure()
for name, v in series.items():
    dd = drawdown_series(v["returns"]) * 100
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name=name,
        line=dict(color=color_of[name], width=1.8 if v["kind"] == "portfolio" else 1.2,
                  dash=None if v["kind"] == "portfolio" else "dash")))
fig2.update_layout(height=360, hovermode="x unified", yaxis_title="Drawdown (%)",
                   legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                   margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig2, width="stretch")

# ---------------- 종합 비교표 ----------------
st.subheader("4️⃣ 종합 성과 비교 (Summary)")
rows = []
for name, v in series.items():
    r = v["returns"]
    rows.append({
        "": ("📈 " if v["kind"] == "portfolio" else "📊 ") + name,
        "누적성장배수": equity_curve(r).iloc[-1],
        "CAGR (%)": cagr(r) * 100,
        "변동성 (%)": annual_vol(r) * 100,
        "MDD (%)": max_drawdown(r) * 100,
        "샤프지수": sharpe_ratio(r, rf_rate),
        "소르티노": sortino_ratio(r, rf_rate),
        "칼마": calmar_ratio(r),
        "얼서지수": ulcer_index(r),
        "연 회전율(%)": (v.get("turnover") * 100
                     if v["kind"] == "portfolio" and v.get("turnover") is not None
                     and not pd.isna(v.get("turnover")) else np.nan),
    })
comp = pd.DataFrame(rows).set_index("")
_hi = [c for c in ["누적성장배수", "CAGR (%)", "MDD (%)", "샤프지수", "소르티노", "칼마"]
       if c in comp.columns]
_lo = [c for c in ["변동성 (%)", "얼서지수", "연 회전율(%)"] if c in comp.columns]
st.dataframe(
    comp.style.format("{:.2f}", na_rep="-")
        .highlight_max(subset=_hi, color="#dcfce7")
        .highlight_min(subset=_lo, color="#dcfce7"),
    width="stretch")
st.caption("초록색 = 각 지표에서 가장 좋은 값 (MDD는 0에 가까울수록 좋음). "
           "📈 포트폴리오 · 📊 벤치마크 · **연 회전율**은 1년에 자산의 몇 %를 "
           "사고파는지를 뜻하며, 거래비용을 입력했다면 성과에 이미 반영돼 있습니다.")

# ---------------- 포트폴리오별 상세 ----------------
st.subheader("5️⃣ 연도별 성과 (Calendar Year Returns)")
st.caption("연도별로 잘라 보면 특정 해에만 좋았는지, 꾸준했는지가 드러납니다. "
           "약세장이 있던 해를 눈여겨보세요.")
yr_rows = {}
for name, v in series.items():
    yr = (1 + v["returns"]).groupby(v["returns"].index.year).prod() - 1
    yr_rows[name] = yr * 100
ydf = pd.DataFrame(yr_rows)
st.dataframe(ydf.style.format("{:.2f}", na_rep="-").map(_heat_color),
             width="stretch")

st.subheader("6️⃣ 적립식 투자 (Dollar-Cost Averaging)")
st.caption("목돈을 한 번에 넣는 대신 매달 일정액을 넣었다면 어땠을지 계산합니다. "
           "나중에 넣은 돈일수록 투자 기간이 짧으므로, 단순 수익률보다 "
           "**내부수익률(IRR)** 이 올바른 비교 기준입니다.")
d1, d2 = st.columns(2)
_unit = 1_000_000 if base_ccy == "KRW" else 1_000
init_amt = d1.number_input(f"최초 투자금 ({base_ccy})", 0.0, 1e12,
                           float(_unit * 10), step=float(_unit))
mon_amt = d2.number_input(f"매월 납입액 ({base_ccy})", 0.0, 1e12,
                          float(_unit), step=float(_unit))
if mon_amt > 0 or init_amt > 0:
    drows, fig_d = [], go.Figure()
    for name, v in series.items():
        val, inv, buys = dca_result(v["returns"], mon_amt, init_amt)
        if val.empty:
            continue
        cf = ([-init_amt] if init_amt > 0 else []) + [-mon_amt] * len(buys) + [float(val.iloc[-1])]
        dts = ([val.index[0]] if init_amt > 0 else []) + list(buys) + [val.index[-1]]
        irr = xirr(cf, dts)
        drows.append({"": name, f"총 납입금 ({base_ccy})": float(inv.iloc[-1]),
                      f"최종 평가액 ({base_ccy})": float(val.iloc[-1]),
                      "단순 수익률(%)": (float(val.iloc[-1]) / float(inv.iloc[-1]) - 1) * 100
                      if inv.iloc[-1] > 0 else np.nan,
                      "내부수익률 IRR(%)": irr * 100 if not pd.isna(irr) else np.nan})
        fig_d.add_trace(go.Scatter(x=val.index, y=val.values, name=name,
                                   line=dict(color=color_of.get(name), width=2)))
    if drows:
        _inv = pd.DataFrame({"납입원금": inv})
        fig_d.add_trace(go.Scatter(x=inv.index, y=inv.values, name="납입 원금",
                                   line=dict(color="#94a3b8", width=1.5, dash="dot")))
        fig_d.update_layout(height=380, hovermode="x unified",
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(tickformat=",.0f"),
                            legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fig_d, width="stretch")
        st.dataframe(pd.DataFrame(drows).set_index("")
                     .style.format({f"총 납입금 ({base_ccy})": "{:,.0f}",
                                    f"최종 평가액 ({base_ccy})": "{:,.0f}",
                                    "단순 수익률(%)": "{:.2f}",
                                    "내부수익률 IRR(%)": "{:.2f}"}, na_rep="-"),
                     width="stretch")

st.subheader("7️⃣ 포트폴리오별 상세 (Details)")
for name, v in series.items():
    if v["kind"] != "portfolio":
        continue
    with st.expander(f"📈 {name} — {v['rebalance']}", expanded=False):
        r = v["returns"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("연평균 수익률 (CAGR)", f"{cagr(r)*100:.2f}%")
        m2.metric("최대낙폭 (MDD)", f"{max_drawdown(r)*100:.2f}%")
        m3.metric("샤프지수 (Sharpe)", f"{sharpe_ratio(r, rf_rate):.2f}")
        m4.metric("현재 낙폭 (Current DD)", f"{drawdown_series(r).iloc[-1]*100:.2f}%")

        st.markdown("**최대 낙폭 구간 (Worst Drawdowns)**")
        st.dataframe(worst_drawdowns(r), width="stretch", hide_index=True)

        st.markdown("**성장 기여도 (Contribution to Growth)**")
        st.caption("각 종목이 포트폴리오 총 성장률에 얼마나 기여했는지 보여줍니다. "
                   "막대의 합이 곧 포트폴리오 총 성장률입니다.")
        try:
            contrib = growth_contribution(prices, v["weights"], v["rebalance"], r) * 100
            contrib = contrib.sort_values(ascending=False)
            total_g = float(contrib.sum())

            labels = list(contrib.index) + ["Portfolio"]
            values = list(contrib.values) + [total_g]
            measures = ["relative"] * len(contrib) + ["total"]

            fw = go.Figure(go.Waterfall(
                orientation="v", measure=measures, x=labels, y=values,
                text=[f"{x:+.2f}%" for x in values],
                textposition="outside",
                connector={"line": {"color": "#d1d5db", "width": 1}},
                increasing={"marker": {"color": "#0d9488"}},
                decreasing={"marker": {"color": "#dc2626"}},
                totals={"marker": {"color": "#c8b89a"}},
            ))
            fw.update_layout(height=340, showlegend=False,
                             margin=dict(l=0, r=0, t=20, b=0),
                             yaxis=dict(title=None, ticksuffix="%"))
            st.plotly_chart(fw, width="stretch", key=f"wf_{name}")

            cdf = pd.DataFrame({
                "종목": contrib.index,
                "기여도(%p)": contrib.values,
                "비중(%)": [v["weights"].get(t, np.nan) / sum(v["weights"].values()) * 100
                           for t in contrib.index],
            })
            cdf["총성장 대비"] = cdf["기여도(%p)"] / total_g * 100 if total_g else np.nan
            st.dataframe(cdf.style.format({"기여도(%p)": "{:+.2f}", "비중(%)": "{:.2f}",
                                           "총성장 대비": "{:.1f}%"}),
                         width="stretch", hide_index=True)
            st.caption(f"기여도 합계 **{total_g:+.2f}%** = 포트폴리오 총 성장률")
        except Exception as ex:
            st.warning(f"기여도를 계산하지 못했습니다: {ex}")

        st.markdown("**보유 비중 추이 (Allocation Drift)**")
        wd = weight_drift(prices, v["weights"], v["rebalance"]).loc[r.index] * 100
        f3 = go.Figure()
        for t in wd.columns:
            f3.add_trace(go.Scatter(x=wd.index, y=wd[t], name=t, stackgroup="one",
                                    mode="lines", line=dict(width=0.5)))
        f3.update_layout(height=260, hovermode="x unified", yaxis_title="비중 (%)",
                         yaxis_range=[0, 100], margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f3, width="stretch", key=f"wd_{name}")

        st.markdown("**월별 수익률 (Monthly Returns, %)**")
        st.dataframe(style_monthly(monthly_table(r)), width="stretch")

# ---------------- 다운로드 ----------------
st.divider()
st.subheader("📥 결과 내보내기")

settings_info = {
    "기준 통화": base_ccy,
    "분석 기간": f"{common[0].date()} ~ {common[-1].date()}",
    "공통 거래일": f"{len(common):,}일",
    "배당 처리": "재투자 (총수익)" if use_div else "주가만 (배당 제외)",
    "무위험 수익률": f"{rf_rate*100:.2f}%",
    "벤치마크": ", ".join(bench_list) if bench_list else "-",
    "생성 시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
for p in active:
    hold = p["holdings"]
    txt = ", ".join(f"{str(r['티커']).strip()} {r['비중(%)']:.2f}%"
                    for _, r in hold.iterrows()
                    if str(r["티커"]).strip() and pd.notna(r["비중(%)"]))
    settings_info[f"[{p['name']}] {p['rebalance']}"] = txt

e1, e2 = st.columns(2)
try:
    xlsx_bytes = build_excel(series, comp, prices, meta, settings_info, color_of)
    e1.download_button(
        "📊 엑셀 파일 받기 (차트 포함)", xlsx_bytes,
        f"portfolio_analysis_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch")
except Exception as e:
    e1.error(f"엑셀 생성 실패: {e}")

csv_df = pd.DataFrame()
for n, v in series.items():
    ec = equity_curve(v["returns"])
    csv_df[f"{n}_성과"] = ec
    csv_df[f"{n}_수익률(%)"] = (v["returns"] * 100).round(4)
    csv_df[f"{n}_낙폭(%)"] = (drawdown_series(v["returns"]) * 100).round(4)
csv_df.index.name = "날짜"
e2.download_button("📄 성과 차트 CSV", csv_df.to_csv().encode("utf-8-sig"),
                   f"portfolio_data_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                   "text/csv", width="stretch")

st.caption("엑셀은 화면과 같은 순서(1_포트폴리오구성 → 2_성과차트 → 3_Drawdown → "
           "4_종합비교 → 5_포트별상세 → 6_설정)로 구성됩니다. 성과차트·Drawdown·비중추이에는 "
           "엑셀에서 바로 편집 가능한 네이티브 차트가, 월별 수익률에는 조건부 서식이 적용됩니다.")
st.caption("교육·참고용이며 투자 자문이 아닙니다. 과거 성과는 미래 수익을 보장하지 않습니다. "
           "거래비용·세금은 반영되지 않았습니다.")
