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

st.set_page_config(page_title="Portfolio Analyzer", page_icon="📊", layout="wide")


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


def build_price_frame(tickers, start, end, base_ccy: str, use_dividends: bool):
    """
    여러 종목을 받아 기준통화로 환산된 가격 DataFrame을 만든다.
    반환: (prices DataFrame, meta dict)
    """
    series, meta = {}, {}
    fx_cache = {}

    for t in tickers:
        d = load_ticker(t, start, end)
        px = d["adjclose"] if use_dividends else d["close"]
        ccy = d["currency"]
        meta[t] = {"currency": ccy, "name": d["name"], "rows": len(px)}

        if ccy != base_ccy:
            if ccy not in fx_cache:
                fx_cache[ccy] = load_fx(ccy, base_ccy, start, end)
            fx = fx_cache[ccy].reindex(px.index).ffill().bfill()
            px = px * fx
        series[t] = px

    return pd.DataFrame(series).dropna(), meta


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


# ---------------------- 성과 지표 ----------------------
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
# 사이드바
# ======================================================================
st.title("📊 Portfolio Analyzer")
st.caption("여러 포트폴리오와 벤치마크를 한 화면에서 비교합니다. 통화는 자동으로 인식됩니다.")

with st.sidebar:
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

st.subheader("1️⃣ 포트폴리오 구성")
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
        prices, meta = build_price_frame(all_tickers, start_date, end_date,
                                         base_ccy, use_div)
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
st.subheader("2️⃣ 성과 차트")

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
st.subheader("3️⃣ Drawdown 비교")
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
st.subheader("4️⃣ 종합 성과 비교")
rows = []
for name, v in series.items():
    r = v["returns"]
    rows.append({
        "": ("📈 " if v["kind"] == "portfolio" else "📊 ") + name,
        "최종배수": equity_curve(r).iloc[-1],
        "CAGR (%)": cagr(r) * 100,
        "변동성 (%)": annual_vol(r) * 100,
        "MDD (%)": max_drawdown(r) * 100,
        "Sharpe": sharpe_ratio(r, rf_rate),
        "Sortino": sortino_ratio(r, rf_rate),
        "Calmar": calmar_ratio(r),
        "Ulcer": ulcer_index(r),
    })
comp = pd.DataFrame(rows).set_index("")
st.dataframe(
    comp.style.format("{:.2f}")
        .highlight_max(subset=["최종배수", "CAGR (%)", "MDD (%)", "Sharpe",
                               "Sortino", "Calmar"], color="#dcfce7")
        .highlight_min(subset=["변동성 (%)", "Ulcer"], color="#dcfce7"),
    width="stretch")
st.caption("초록색 = 각 지표에서 가장 좋은 값 (MDD는 0에 가까울수록 좋음). 📈 포트폴리오 · 📊 벤치마크")

# ---------------- 포트폴리오별 상세 ----------------
st.subheader("5️⃣ 포트폴리오별 상세")
for name, v in series.items():
    if v["kind"] != "portfolio":
        continue
    with st.expander(f"📈 {name} — {v['rebalance']}", expanded=False):
        r = v["returns"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f"{cagr(r)*100:.2f}%")
        m2.metric("MDD", f"{max_drawdown(r)*100:.2f}%")
        m3.metric("Sharpe", f"{sharpe_ratio(r, rf_rate):.2f}")
        m4.metric("현재 낙폭", f"{drawdown_series(r).iloc[-1]*100:.2f}%")

        st.markdown("**최악의 낙폭 구간**")
        st.dataframe(worst_drawdowns(r), width="stretch", hide_index=True)

        st.markdown("**보유 비중 추이**")
        wd = weight_drift(prices, v["weights"], v["rebalance"]).loc[r.index] * 100
        f3 = go.Figure()
        for t in wd.columns:
            f3.add_trace(go.Scatter(x=wd.index, y=wd[t], name=t, stackgroup="one",
                                    mode="lines", line=dict(width=0.5)))
        f3.update_layout(height=260, hovermode="x unified", yaxis_title="비중 (%)",
                         yaxis_range=[0, 100], margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f3, width="stretch", key=f"wd_{name}")

        st.markdown("**월별 수익률 (%)**")
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
