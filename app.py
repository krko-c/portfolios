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
import os
import re
import subprocess
from functools import partial
from pathlib import Path

from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

TRADING_DAYS = 252
MAX_PORTFOLIOS = 6
# 공개 배포(예: Streamlit Cloud)는 접속자 전원이 프로세스 하나를 공유하므로,
# 서버 파일에 저장하면 다른 세션의 구성이 섞여 보일 수 있다.
# 본인 컴퓨터에서 혼자 실행할 때만 명시적으로 켜서 쓴다.
ENABLE_LOCAL_PERSISTENCE = os.environ.get("ENABLE_LOCAL_PERSISTENCE", "").lower() == "true"

# 사이드바 메뉴 — 성격별로 묶어 표시한다
TOOL_GROUPS = [
    ("분석", ["📊 포트폴리오 분석", "🗞 시장동향", "📉 시장·거시 국면", "🔗 자산 상관관계",
              "🌩 스트레스 테스트"]),
    ("배분", ["🎯 포트폴리오 최적화", "🧭 뷰 기반 자산배분", "➕ 자산 추가 효과"]),
    ("결정", ["⚖️ 최종 대안 비교"]),
    ("", ["📖 도움말"]),
]
TOOL_DEFAULT = "📊 포트폴리오 분석"

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
def load_ticker(ticker: str, start, end, currency: str = None, name: str = None):
    """
    종목 1개의 가격을 가져온다.
    반환: {"close": Series, "adjclose": Series, "currency": str, "name": str}
    - close    : 액면분할만 보정된 주가 (배당 제외)
    - adjclose : 액면분할 + 배당 재투자까지 보정

    currency/name 을 인자로 주면(보통 probe_ticker 결과) 그대로 쓰고 야후에
    다시 묻지 않는다 — 안 주면 여기서 새로 감지한다(단독 호출 대비).
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해주세요.")

    # 타임존 제거 후 날짜만 남김 (국가별 거래시간 차이로 정렬이 깨지는 것 방지)
    hist.index = pd.to_datetime(hist.index.date)

    close = hist["Close"].dropna()
    adj = hist["Adj Close"].dropna() if "Adj Close" in hist.columns else close.copy()

    if not currency:
        try:
            fi = tk.fast_info
            currency = (fi.get("currency") if hasattr(fi, "get") else getattr(fi, "currency", None))
        except Exception:
            pass
        if not currency:
            try:
                info = tk.info or {}
                currency = info.get("currency")
                name = name or info.get("shortName") or info.get("longName")
            except Exception:
                pass
    currency = (currency or _suffix_currency(ticker)).upper()
    name = name or ticker

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


def _is_symbol_list(name, ticker=None) -> bool:
    """
    야후가 종목명 대신 '유사 심볼 목록'을 돌려준 경우인지 판별한다.
    예: "240810.KS,0P00017YB3,330566"  ← 원익IPS 인데 심볼 나열이 옴
    회사명에도 콤마가 흔하므로("HANMI Semiconductor Co., Lt"),
    조각이 전부 '숫자를 포함하거나 거래소 접미사가 붙은 심볼' 형태일 때만 걸러낸다.
    """
    if not name:
        return False
    s = str(name).strip()
    if ticker and s.upper().startswith(str(ticker).upper() + ","):
        return True
    if re.search(r"\b0P[0-9A-Z]{8}\b", s):          # 모닝스타 펀드 코드
        return True
    parts = [x.strip() for x in s.split(",")]
    if len(parts) < 2:
        return False

    def symbolish(x):
        if not x or " " in x:
            return False
        if not re.fullmatch(r"[0-9A-Z.^-]{2,15}", x):
            return False
        return bool(re.search(r"\d", x)) or bool(re.search(r"\.[A-Z]{1,3}$", x))

    return all(symbolish(x) for x in parts)


def _clean_name(name, ticker=None) -> str:
    """오염된 종목명은 버린다."""
    return "" if _is_symbol_list(name, ticker) else (str(name).strip() if name else "")


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

        nm_ = _clean_name(hm.get("longName"), cand) or \
            _clean_name(hm.get("shortName"), cand)
        if not nm_:
            # meta 가 오염됐거나 비어 있으면 info 로 한 번 더 시도한다
            try:
                _inf = tk.info or {}
                nm_ = (_clean_name(_inf.get("longName"), cand) or
                       _clean_name(_inf.get("shortName"), cand))
            except Exception:
                nm_ = ""
        # 자산 유형 판정에 쓸 상품 구분. 추가 조회 없이 meta 에서 가져온다.
        # EQUITY / ETF / MUTUALFUND / INDEX / CURRENCY / FUTURE 등
        qtype = str(hm.get("instrumentType") or "").upper()
        info = {
            "ticker": cand,
            "currency": (ccy or _suffix_currency(cand)).upper(),
            "quote_type": qtype,
            "name": nm_,
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
                nm = (_clean_name(info.get("shortName"), ticker) or
                      _clean_name(info.get("longName"), ticker))
                if nm:
                    return nm
        except Exception:
            continue
    return ""



@st.cache_data(ttl=86400, show_spinner=False)
def load_etf_holdings(ticker: str):
    """
    ETF 상위 보유종목을 조회한다.
    반환: (DataFrame|None, 안내문)
    야후는 상위 10개 내외만 제공하며, 국내 ETF는 대부분 데이터가 없다.
    """
    try:
        fd = yf.Ticker(ticker).funds_data
        df = fd.top_holdings
    except Exception as ex:
        return None, f"보유종목 정보를 제공하지 않는 종목입니다. ({type(ex).__name__})"

    if df is None or len(df) == 0:
        return None, "보유종목 정보가 비어 있습니다."

    pct_col = next((c for c in df.columns if "percent" in str(c).lower()), None)
    name_col = next((c for c in df.columns if "name" in str(c).lower()), None)
    if pct_col is None:
        return None, "비중 정보를 찾을 수 없습니다."

    w = pd.to_numeric(df[pct_col], errors="coerce")
    if w.max() is not None and w.max() <= 1.5:      # 0~1 소수로 오는 경우
        w = w * 100
    _syms = [str(x).strip() for x in df.index]
    _nms = ([_clean_name(v, s) for v, s in zip(df[name_col].astype(str).values, _syms)]
            if name_col else [""] * len(_syms))
    out = pd.DataFrame({
        "티커": _syms,
        "종목명": _nms,
        "비중(%)": w.round(2).values,
    })
    out = out[out["티커"].str.strip() != ""].reset_index(drop=True)
    if out.empty:
        return None, "유효한 보유종목이 없습니다."
    return out, ""


def render_etf_loader(target_key: str, gen_key: str, editor_key: str,
                      cols: list, weight_col: str, extra_defaults: dict = None):
    """
    ETF 구성종목 불러오기 UI. 조회 후 버튼을 누르면 대상 표를 덮어쓴다.
    target_key : 종목 표를 담고 있는 session_state 키
    gen_key    : 행 수 위젯 재생성용 세대 키
    """
    with st.expander("📦 ETF 구성종목 불러오기", expanded=False):
        st.caption("ETF 티커를 넣으면 상위 보유종목과 비중을 가져와 아래 표에 채웁니다. "
                   "**야후가 제공하는 범위가 상위 10개 내외**라 전체 구성종목은 아니며, "
                   "**국내 ETF는 대부분 지원되지 않습니다.** "
                   "미국 ETF에서 주로 동작합니다.")
        e1, e2 = st.columns([3, 1])
        etf = e1.text_input("ETF 티커", key=f"etf_in_{target_key}",
                            placeholder="예: QQQ, SPY, SCHD, XLK")
        e2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if e2.button("조회", key=f"etf_go_{target_key}", width="stretch",
                     disabled=not etf.strip()):
            cands = normalize_ticker(etf)
            sym = cands[0] if cands else etf.strip().upper()
            with st.spinner(f"{sym} 구성종목 조회 중..."):
                hold, msg = load_etf_holdings(sym)
            st.session_state[f"etf_res_{target_key}"] = (sym, hold, msg)

        res = st.session_state.get(f"etf_res_{target_key}")
        if not res:
            return
        sym, hold, msg = res
        if hold is None:
            st.warning(f"**{sym}** — {msg}\n\n"
                       f"국내 ETF이거나 야후가 구성 정보를 제공하지 않는 상품일 수 있습니다. "
                       f"이 경우 종목을 직접 입력해주세요.")
            return

        tot = float(hold["비중(%)"].sum())
        st.dataframe(hold.style.format({"비중(%)": "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.caption(f"**{sym}** 상위 {len(hold)}종목 · 합계 **{tot:.2f}%** "
                   f"(나머지 {max(0.0, 100-tot):.2f}%는 그 외 종목)")

        mode = None
        if weight_col:
            mode = st.radio(
                "나머지 비중 처리",
                ["상위 종목만으로 100% 재조정", f"나머지는 {sym} 자체로 채우기"],
                key=f"etf_mode_{target_key}", horizontal=False,
                help="상위 종목의 합이 100%에 못 미칩니다. 부족분을 어떻게 다룰지 정하세요.")

        if st.button("⬇️ 아래 표에 채우기", key=f"etf_fill_{target_key}",
                     type="primary", width="stretch"):
            rows = []
            if weight_col is None:
                rows = [{"티커": r["티커"]} for _, r in hold.iterrows()]
            elif mode.startswith("상위"):
                scale = 100.0 / tot if tot > 0 else 1.0
                for _, r in hold.iterrows():
                    rows.append({"티커": r["티커"],
                                 weight_col: round(float(r["비중(%)"]) * scale, 2)})
            else:
                for _, r in hold.iterrows():
                    rows.append({"티커": r["티커"], weight_col: round(float(r["비중(%)"]), 2)})
                rest = round(100.0 - tot, 2)
                if rest > 0.01:
                    rows.append({"티커": sym, weight_col: rest})
            if weight_col:
                s = sum(r[weight_col] for r in rows)
                if rows and abs(s - 100) > 1e-9:
                    rows[-1][weight_col] = round(rows[-1][weight_col] + (100 - s), 2)

            base = {c: (np.nan if c == weight_col else "") for c in cols}
            if extra_defaults:
                base.update(extra_defaults)
            full = [{**base, **r} for r in rows]
            st.session_state[target_key] = pd.DataFrame(full)[cols]
            st.session_state[gen_key] = st.session_state.get(gen_key, 0) + 1
            st.session_state.pop(editor_key, None)
            st.session_state.pop(f"etf_res_{target_key}", None)
            st.success(f"{len(rows)}종목을 표에 채웠습니다.")
            st.rerun()


# ======================================================================
# 종목 입력 표 (모든 화면 공통)
# ======================================================================
CLIP_KEY = "_clipboard"          # 화면 사이 복사용 임시 보관함
PLANS_KEY = "_plans"             # 최종 비교용 후보안 모음
MAX_PLANS = 8


def load_plans() -> dict:
    """비교 화면에서 쓸 후보안 모음. {이름: {rows, from, rebalance, note}}"""
    if PLANS_KEY not in st.session_state:
        st.session_state[PLANS_KEY] = {}
    return st.session_state[PLANS_KEY]


def save_plan(name: str, rows: list, source: str, rebalance=None, note="") -> bool:
    """
    후보안을 담는다. rows 는 [{"티커":.., "비중":..}, ...] 형태.
    같은 이름이면 덮어쓴다.
    """
    rows = [r for r in rows if str(r.get("티커") or "").strip()]
    if not rows:
        return False
    tot = sum(float(r.get("비중") or 0) for r in rows)
    if tot > 0:                       # 합계 100%로 정규화
        rows = [{"티커": r["티커"], "비중": float(r.get("비중") or 0) / tot * 100}
                for r in rows]
    else:                             # 비중이 없으면 균등
        rows = [{"티커": r["티커"], "비중": 100.0 / len(rows)} for r in rows]
    plans = load_plans()
    if name not in plans and len(plans) >= MAX_PLANS:
        return False
    plans[name] = {"rows": rows, "from": source,
                   "rebalance": rebalance, "note": note,
                   "saved_at": pd.Timestamp.now().strftime("%m-%d %H:%M")}
    st.session_state[PLANS_KEY] = plans
    return True


def plan_button(label: str, rows: list, source: str, key: str,
                default_name: str = "", rebalance=None, note=""):
    """'비교 후보로 담기' 버튼. 이름을 받아 후보함에 넣는다."""
    c1, c2 = st.columns([2, 1])
    nm = c1.text_input("이 안의 이름", value=default_name, key=f"{key}_name",
                       placeholder="예: 최적화안 · 금 5% 편입안")
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button(label, key=f"{key}_btn", width="stretch",
                 disabled=not nm.strip()):
        if save_plan(nm.strip(), rows, source, rebalance, note):
            st.success(f"**{nm.strip()}** 을 비교 후보에 담았습니다. "
                       f"⚖️ 최종 비교 화면에서 확인하세요.")
        else:
            st.error(f"담지 못했습니다. 후보는 최대 {MAX_PLANS}개까지이며, "
                     f"종목이 하나 이상 있어야 합니다.")



def clip_summary() -> str:
    """보관함에 담긴 구성을 한 줄로 요약한다."""
    c = st.session_state.get(CLIP_KEY)
    if not c or not c.get("rows"):
        return ""
    rows = c["rows"]
    head = " · ".join(
        f"{r['티커']}" + (f" {r['비중']:.1f}%" if r.get("비중") is not None else "")
        for r in rows[:3])
    more = f" 외 {len(rows)-3}개" if len(rows) > 3 else ""
    return f"{head}{more} ({len(rows)}종목)"


def render_ticker_table(*, state_key, editor_key, gen_key, cols, weight_col=None,
                        title="종목", min_rows=1, max_rows=30, default_rows=None,
                        extra_config=None, extra_defaults=None,
                        show_etf=True, show_equal=False, equal_key=None,
                        count_label="종목 수", help_text=None):
    """
    티커 입력 표를 그린다. 모든 화면이 이 함수를 쓴다.

    처리하는 것
      · 종목 수 조절 (세대 번호로 위젯을 새로 만들어 값이 확실히 따라오게 함)
      · 행 삭제 (체크 즉시)
      · 티커 형식 변환 + 자동 확인 (005930 → 005930.KS)
      · ETF 구성종목 불러오기
      · 화면 사이 복사 / 붙여넣기
      · 비중 균등 배분(선택)

    반환: (정리된 DataFrame, 유효 티커 목록, 확인 실패 목록)
    """
    st.session_state.setdefault(gen_key, 0)

    def _blank():
        r = {c: ("" if c != weight_col else np.nan) for c in cols}
        if extra_defaults:
            r.update(extra_defaults)
        return r

    if state_key not in st.session_state:
        rows = default_rows or [{"티커": ""}]
        base = []
        for r in rows:
            d = _blank(); d.update(r); base.append(d)
        st.session_state[state_key] = pd.DataFrame(base)[cols]

    if help_text:
        st.caption(help_text)

    # ---------------- ETF 불러오기 ----------------
    if show_etf:
        render_etf_loader(target_key=state_key, gen_key=gen_key,
                          editor_key=editor_key, cols=cols, weight_col=weight_col)

    # ---------------- 상단 조작줄 ----------------
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 3])
    cur_n = len(st.session_state[state_key])
    # 저장된 행 수가 허용 범위를 벗어나면 먼저 맞춘다.
    # (min_rows 를 바꿨거나 다른 화면에서 붙여넣은 뒤 생길 수 있다)
    if cur_n < min_rows or cur_n > max_rows:
        fixed = st.session_state[state_key]
        if cur_n < min_rows:
            fixed = pd.concat([fixed, pd.DataFrame([_blank()] * (min_rows - cur_n))],
                              ignore_index=True)
        else:
            fixed = fixed.iloc[:max_rows].reset_index(drop=True)
        st.session_state[state_key] = fixed[cols]
        st.session_state[gen_key] += 1
        st.session_state.pop(editor_key, None)
        st.rerun()

    n = c1.number_input(count_label, min_rows, max_rows, cur_n, step=1,
                        key=f"{gen_key}_n_{st.session_state[gen_key]}")
    if int(n) != cur_n:
        cur = st.session_state[state_key]
        cur = (pd.concat([cur, pd.DataFrame([_blank()] * (int(n) - len(cur)))],
                         ignore_index=True) if int(n) > len(cur)
               else cur.iloc[:int(n)].reset_index(drop=True))
        st.session_state[state_key] = cur[cols]
        st.session_state[gen_key] += 1
        st.session_state.pop(editor_key, None)
        st.rerun()

    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button("📋 복사", key=f"{gen_key}_copy", width="stretch",
                 help="이 표의 구성을 보관함에 담습니다. 다른 화면에서 붙여넣을 수 있습니다."):
        cur = st.session_state[state_key]
        rows = []
        for _, r in cur.iterrows():
            tk = str(r.get("티커") or "").strip()
            if not tk:
                continue
            wv = None
            if weight_col:
                wv = pd.to_numeric(r.get(weight_col), errors="coerce")
                wv = None if pd.isna(wv) else float(wv)
            rows.append({"티커": tk, "비중": wv})
        if rows:
            st.session_state[CLIP_KEY] = {"rows": rows, "from": title}
            st.rerun()

    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    clip = st.session_state.get(CLIP_KEY)
    if c3.button("📥 붙여넣기", key=f"{gen_key}_paste", width="stretch",
                 disabled=not (clip and clip.get("rows")),
                 help="보관함에 담긴 구성을 이 표에 채웁니다."):
        rows = clip["rows"]
        new = []
        for r in rows:
            d = _blank()
            d["티커"] = r["티커"]
            if weight_col and r.get("비중") is not None:
                d[weight_col] = round(float(r["비중"]), 2)
            new.append(d)
        while len(new) < min_rows:
            new.append(_blank())
        st.session_state[state_key] = pd.DataFrame(new[:max_rows])[cols]
        st.session_state[gen_key] += 1
        st.session_state.pop(editor_key, None)
        st.rerun()

    equal_w = False
    if show_equal and weight_col:
        c4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        equal_w = c4.checkbox("비중 균등", key=equal_key or f"{gen_key}_eq",
                              help="입력된 종목에 100%를 균등 배분합니다.")
    elif clip and clip.get("rows"):
        c4.caption(f"📋 보관함 · {clip.get('from', '')} — {clip_summary()}")

    # ---------------- 표 ----------------
    conf = {
        "티커": st.column_config.TextColumn(
            "티커", width="small",
            help="005930.KS · 005930 KS · 005930 · NVDA 모두 인식합니다."),
        "종목명": st.column_config.TextColumn("종목명 (자동)", disabled=True,
                                          width="large" if len(cols) <= 3 else "medium"),
        "🗑": st.column_config.CheckboxColumn("삭제", width="small",
                                            help="체크하면 해당 행이 삭제됩니다."),
    }
    if weight_col:
        conf[weight_col] = st.column_config.NumberColumn(
            weight_col, min_value=0.0, max_value=100.0, step=0.01,
            format="%.2f", width="small", disabled=equal_w)
    if extra_config:
        conf.update(extra_config)

    disp = st.session_state[state_key].copy()
    disp["🗑"] = False
    edited = st.data_editor(disp, num_rows="fixed", width="stretch",
                            key=editor_key, column_order=cols + ["🗑"],
                            hide_index=True, column_config=conf)

    dele = edited["🗑"].fillna(False).astype(bool)
    if dele.any():
        kept = edited.loc[~dele, cols].reset_index(drop=True)
        if kept.empty:
            kept = pd.DataFrame([_blank()])[cols]
        st.session_state[state_key] = kept
        st.session_state[gen_key] += 1
        st.session_state.pop(editor_key, None)
        st.rerun()

    # ---------------- 정규화 + 티커 확인 ----------------
    f = edited[cols].copy()
    f["티커"] = f["티커"].fillna("").astype(str).str.strip()
    if weight_col:
        f[weight_col] = pd.to_numeric(f[weight_col], errors="coerce").round(2)

    bad = []
    if any(t for t in f["티커"]):
        with st.spinner("티커 확인 중..."):
            res, labs = [], []
            for t in f["티커"]:
                if not t:
                    res.append(""); labs.append(""); continue
                r = probe_ticker(tuple(normalize_ticker(t)))
                if r["ticker"] is None:
                    res.append(t); labs.append("❌ 확인 불가"); bad.append(t)
                else:
                    res.append(r["ticker"])
                    _nm = r.get("name") or f"{r['ticker']} (종목명 미제공)"
                    labs.append(f"✅ {_nm} · {r['currency']}")
        f["티커"], f["종목명"] = res, labs
    else:
        f["종목명"] = ""

    if equal_w and weight_col:
        mask = f["티커"] != ""
        k = int(mask.sum())
        if k > 0:
            base_w = round(100.0 / k, 2)
            vals = [base_w] * k
            vals[-1] = round(100.0 - base_w * (k - 1), 2)
            f.loc[mask, weight_col] = vals
        f.loc[~mask, weight_col] = np.nan

    if not _frames_equal(f, st.session_state[state_key]):
        st.session_state[state_key] = f
        st.session_state.pop(editor_key, None)
        st.rerun()

    live = f[f["티커"] != ""].copy()

    # 정규화 후 같은 티커가 두 번 이상 있으면 이후 set_index/reindex 가 깨지거나
    # 최소·최대 비중 같은 행별 조건이 모호해진다. 자동 합산 대신 차단한다.
    _tk_list = list(live["티커"])
    _dup_tk = sorted({t for t in _tk_list if _tk_list.count(t) > 1})
    if _dup_tk:
        st.error(f"같은 종목이 여러 행에 중복 입력됐습니다: **{', '.join(_dup_tk)}**\n\n"
                 f"`005930`과 `005930.KS`처럼 표기가 달라도 정규화하면 같은 "
                 f"티커가 되면 중복으로 처리됩니다. 자동으로 합치지 않으니, "
                 f"위 표에서 중복된 행을 지우거나 한 행으로 정리해주세요.")
        st.stop()

    return live, list(live["티커"]), bad


@st.cache_data(show_spinner=False)
def _git_commit() -> str:
    """배포된 코드의 git commit 짧은 해시. 재현성 확인용 (결과 캡처 시점의 코드 버전)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=3)
        h = out.stdout.strip()
        return h if h else "확인 불가"
    except Exception:
        return "확인 불가"


def assumptions_panel(*, rebalance=None, cost=None, initial_cost=True,
                      period=None, bench=None, extra=None, path_note=False,
                      data_source="Yahoo Finance"):
    """
    화면 하단에 이 결과의 계산 가정을 모아 보여준다.
    가정이 화면 곳곳에 흩어져 있으면 사용자가 결과를 오해하기 쉽다.
    """
    with st.expander("📋 이 결과의 가정", expanded=False):
        rows = [
            ("실행 시각", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("코드 버전 (git commit)", _git_commit()),
            ("가격 데이터 출처", data_source),
        ]
        if period:
            rows.append(("분석 기간", period))
        if rebalance:
            rows.append(("리밸런싱", rebalance))
        if cost is not None:
            rows.append(("거래비용", f"편도 {cost:.0f}bp"
                         + (" · 반영 안 함" if not cost else "")))
            rows.append(("최초 구축비용",
                         "포함 (첫날 전량 매수 가정)" if initial_cost
                         else "제외 (이전부터 보유 가정)"))
            rows.append(("회전율 표기", "매도+매수를 합한 총 매매비중"))
        rows.append(("기준 통화", base_ccy))
        rows.append(("환율", "제외 (현지통화 기준)" if fx_hedge
                     else "포함 (기준통화 환산)"))
        rows.append(("배당", "재투자 (총수익)" if use_div else "주가만 (배당 제외)"))
        rows.append(("휴장일", "직전 종가로 채움" if gap_fill else "공통 거래일만 사용"))
        if bench:
            rows.append(("벤치마크", bench))
        rows.append(("무위험 수익률", f"{rf_rate*100:.2f}%"))
        if extra:
            extra = list(extra)
            # 화면별로 사이드바 설정을 강제로 덮어쓰는 경우(예: 요인 추정 시
            # 환율을 강제 제외) extra 로 그 사실을 알린다. 자동으로 채운
            # 행과 항목명이 겹치면 화면이 명시한 쪽을 우선하고 자동 행은
            # 빼서, 같은 항목이 서로 다른 값으로 두 번 뜨는 걸 막는다.
            override_labels = {k for k, _ in extra}
            rows = [r for r in rows if r[0] not in override_labels]
            rows.extend(extra)
        st.dataframe(pd.DataFrame(rows, columns=["항목", "설정"]),
                     width="stretch", hide_index=True)
        notes = ["세금·호가스프레드·시장충격은 반영하지 않습니다.",
                 "상장폐지된 종목은 데이터에 없어 과거 성과가 실제보다 좋아 보일 수 "
                 "있습니다(생존편향)."]
        if path_note:
            notes.insert(0, "**경로의존 지표**(조건부 낙폭·얼서지수·오메가)는 "
                            "계산 편의를 위해 **매일 고정비중**을 가정해 구합니다. "
                            "위에 선택한 리밸런싱 주기와 다를 수 있습니다.")
        st.caption("· " + "\n\n· ".join(notes))


def validate_setup(tickers, bad=None, weights=None, min_n=1,
                   need_weight=False, label="종목"):
    """
    화면마다 흩어져 있던 입력 검증을 한 곳에서 처리한다.
    반환: (진행 가능 여부, 안내문 리스트)
    """
    msgs = []
    ok = True
    if bad:
        msgs.append(("error", f"확인되지 않는 티커: {', '.join(bad)}"))
        ok = False
    n = len([t for t in tickers if str(t).strip()])
    if n < min_n:
        msgs.append(("error", f"{label}을 최소 {min_n}개 이상 입력해주세요. (현재 {n}개)"))
        ok = False
    if need_weight:
        s = float(pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0).sum()) \
            if weights is not None else 0.0
        if s <= 0:
            msgs.append(("error", "비중을 입력해주세요. 합계가 0입니다."))
            ok = False
        elif abs(s - 100) > 0.01:
            msgs.append(("info", f"비중 합계가 {s:.2f}% 입니다. 100%로 자동 정규화합니다."))
    return ok, msgs


def show_msgs(msgs):
    for kind, m in msgs:
        getattr(st, kind)(m)


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
    """
    로컬 파일 읽기. ENABLE_LOCAL_PERSISTENCE=true 로 명시적으로 켠 경우에만
    동작한다. 공개 배포 기본값은 세션 저장만 쓴다 (여러 사용자가 프로세스를
    공유해 서버 파일이 섞일 수 있기 때문).
    """
    if not ENABLE_LOCAL_PERSISTENCE:
        return {}
    try:
        if SAVE_PATH.exists():
            return json.loads(SAVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _disk_write(data: dict) -> bool:
    """ENABLE_LOCAL_PERSISTENCE=true 인 로컬 실행에서만 파일로 저장한다."""
    if not ENABLE_LOCAL_PERSISTENCE:
        return False
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


MAX_IMPORT_BYTES = 2_000_000  # 저장 파일은 보통 수십~수백 KB 수준이면 충분하다


def _safe_import_json(uploaded_file):
    """
    가져오기 파일을 검증한다 (크기 + 최소 형식).
    반환: (data, None) 성공 / (None, 에러 메시지) 실패
    """
    size = getattr(uploaded_file, "size", None)
    if size is not None and size > MAX_IMPORT_BYTES:
        return None, (f"파일이 너무 큽니다 ({size / 1e6:.1f}MB). "
                      f"{MAX_IMPORT_BYTES / 1e6:.0f}MB 이하만 허용합니다.")
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as ex:
        return None, f"JSON으로 읽지 못했습니다: {ex}"
    if not isinstance(data, dict):
        return None, "파일 형식이 올바르지 않습니다 (저장 이름 → 구성 객체 형태가 아닙니다)."
    for k, v in data.items():
        if not isinstance(v, dict):
            return None, f"'{k}' 항목의 형식이 올바르지 않습니다."
    return data, None


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


OPT_STORE_KEY = "_opt_saved_store"
OPT_SAVE_PATH = Path(__file__).parent / "optimizations.json"


def load_opt_saved() -> dict:
    if OPT_STORE_KEY not in st.session_state:
        st.session_state[OPT_STORE_KEY] = {}
        if ENABLE_LOCAL_PERSISTENCE:
            try:
                st.session_state[OPT_STORE_KEY] = (
                    json.loads(OPT_SAVE_PATH.read_text(encoding="utf-8"))
                    if OPT_SAVE_PATH.exists() else {})
            except Exception:
                st.session_state[OPT_STORE_KEY] = {}
    return st.session_state[OPT_STORE_KEY]


def write_opt_saved(data: dict) -> bool:
    st.session_state[OPT_STORE_KEY] = data
    if not ENABLE_LOCAL_PERSISTENCE:
        return True
    try:
        OPT_SAVE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception:
        pass
    return True


def opt_snapshot(df, goal, risk_name, cut_label, train_label, rebal,
                 bench_tk, min_ret, max_vol, reopt, label="") -> dict:
    return {
        "label": label,
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "goal": goal, "risk": risk_name, "cutoff": cut_label,
        "train": train_label, "rebalance": rebal, "benchmark": bench_tk,
        "reopt": reopt,
        "min_ret": None if min_ret is None else float(min_ret),
        "max_vol": None if max_vol is None else float(max_vol),
        "holdings": [
            {"ticker": str(r["티커"]).strip(),
             "weight": (None if pd.isna(r["현재 비중(%)"]) else float(r["현재 비중(%)"])),
             "min": float(r["최소(%)"]), "max": float(r["최대(%)"])}
            for _, r in df.iterrows() if str(r["티커"]).strip()
        ],
    }


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
        # 통화·종목명·상품구분은 probe_ticker 한 번으로 같이 얻는다.
        # (예전에는 load_ticker 가 통화·이름을 따로 조회해 야후를 두 번 불렀고,
        #  그마저도 fast_info 가 통화를 주면 이름 조회를 건너뛰어 이름이 티커
        #  그대로 남는 경우가 많았다.)
        try:
            _pr = probe_ticker((t,))
        except Exception:
            _pr = {"currency": None, "name": "", "quote_type": ""}
        _qt = str(_pr.get("quote_type") or "")
        _nm = _pr.get("name") or None
        _ccy = _pr.get("currency") or None
        d = load_ticker(t, start, end, currency=_ccy, name=_nm)
        px = d["adjclose"] if use_dividends else d["close"]
        ccy = d["currency"]
        meta[t] = {"currency": ccy, "name": d["name"], "rows": len(px),
                   "quote_type": _qt}

        if ccy != base_ccy and not fx_hedge:
            if ccy not in fx_cache:
                fx_cache[ccy] = load_fx(ccy, base_ccy, start, end)
            # ffill 만 사용한다. bfill 을 쓰면 분석 시작 시점에 환율이 없을 때
            # 미래의 환율이 과거로 채워져 미래정보가 섞인다.
            fx = fx_cache[ccy].reindex(px.index).ffill()
            px = (px * fx).dropna()
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
def turnover_series(prices: pd.DataFrame, weights: dict, rebalance: str,
                    initial_trade: bool = True, prev_w: dict = None) -> pd.Series:
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
    # 최초일은 포트폴리오를 새로 구축하는 비용.
    # 이어붙이는 구간(워크포워드 등)에서는 prev_w 를 넘겨 실제 교체분만 계산한다.
    if prev_w is None:
        tno.iloc[0] = 1.0 if initial_trade else 0.0
    else:
        first = ws.iloc[0]
        pw = pd.Series({k: float(prev_w.get(k, 0.0)) for k in ws.columns})
        s = pw.sum()
        if s > 0:
            pw = pw / s
        tno.iloc[0] = float((first - pw).abs().sum())
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
    for d, c in curve.items():
        if d in contrib and monthly > 0:
            units += monthly / float(c)
            invested += monthly
            buys.append(d)
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


# 변동성이 사실상 0일 때 비율 지표가 폭주하지 않도록 두는 임계값.
# 부동소수점 오차 때문에 std()가 정확히 0이 아닌 10^-18 같은 값이 될 수 있다.
_EPS_STD = 1e-12


def sharpe_ratio(r, rf=0.0):
    e = r - rf / TRADING_DAYS
    s = e.std()
    if not np.isfinite(s) or s < _EPS_STD:
        return np.nan
    return e.mean() / s * np.sqrt(TRADING_DAYS)


def sortino_ratio(r, rf=0.0):
    """
    하방편차는 '0(목표) 대비 편차'를 전체 표본 수로 나눠야 한다.
    음수인 날들끼리의 표준편차(그 부분집합 평균 대비, 개수-1로 나눔)를 쓰면
    전체 표본 크기에 둔감해지고 값이 왜곡된다 (risk_semisd 와 같은 정의여야 함).
    """
    e = r - rf / TRADING_DAYS
    d = np.sqrt(np.mean(np.minimum(e.values, 0.0) ** 2))
    if not np.isfinite(d) or d < _EPS_STD:
        return np.nan
    return e.mean() / d * np.sqrt(TRADING_DAYS)


def calmar_ratio(r):
    m = abs(max_drawdown(r))
    return np.nan if (not np.isfinite(m) or m < _EPS_STD) else cagr(r) / m


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
    """sortino_ratio 와 동일한 하방편차 정의(전체 표본 기준)를 쓴다."""
    e = r - rf / TRADING_DAYS
    d = np.sqrt(np.mean(np.minimum(e.values, 0.0) ** 2))
    return -1e6 if (d < _EPS_STD or np.isnan(d)) else float(e.mean() / d * np.sqrt(TRADING_DAYS))


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
                          desc="최악 5% 손실의 평균 (연율화). 꼬리 위험에 민감합니다."),
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


def _as_bounds(v, n, default):
    """스칼라 또는 종목별 배열을 길이 n 배열로 맞춘다."""
    if v is None:
        return np.full(n, float(default))
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"제약 길이 불일치: {arr.size} vs 종목 {n}")
    return arr


def _opt_solve(fn, n, wmin, wmax, extra=(), strict=False):
    """
    SLSQP. 시작점을 여러 개 시도해 국소해에 갇히는 것을 줄인다.
    wmin·wmax 는 스칼라뿐 아니라 **종목별 배열**도 받는다.
    strict=True 이면 해를 못 찾았을 때 None 을 돌려준다
    (조용히 동일가중을 반환하면 실패가 정상 결과처럼 보이기 때문).
    """
    lo = _as_bounds(wmin, n, 0.0)
    hi = _as_bounds(wmax, n, 1.0)
    if (lo > hi + 1e-12).any():
        # 조용히 보정하면 잘못된 입력이 숨겨진다
        raise ValueError("최소 비중이 최대 비중보다 큰 종목이 있습니다.")
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(extra)
    bounds = tuple((float(lo[i]), float(hi[i])) for i in range(n))
    best, best_val = None, np.inf
    rng = np.random.default_rng(0)
    starts = [np.ones(n) / n] + [rng.random(n) for _ in range(6)]
    # 하한을 채운 뒤 잔여를 배분한 시작점도 넣는다 (종목별 하한이 있을 때 유리)
    room = hi - lo
    if room.sum() > 1e-12:
        starts.append(lo + room * max(0.0, 1.0 - lo.sum()) / room.sum())
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        s = x0.sum()
        if s > 0:
            x0 = x0 / s
            x0 = np.clip(x0, lo, hi)
        try:
            res = minimize(fn, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-12})
        except Exception:
            continue
        if res.success and res.fun < best_val:
            best, best_val = res.x, res.fun
    if best is None:
        return None if strict else np.ones(n) / n
    best = np.clip(best, lo, hi)
    s = best.sum()
    return best / s if s > 1e-12 else (None if strict else np.ones(n) / n)


def opt_max_sharpe(mu, cov, rf=0.0, wmin=0.0, wmax=1.0, strict=True):
    def neg(w):
        v = np.sqrt(w @ cov @ w)
        return 1e6 if v <= 0 else -(w @ mu - rf) / v
    return _opt_solve(neg, len(mu), wmin, wmax, strict=strict)


def opt_min_vol(mu, cov, wmin=0.0, wmax=1.0, strict=True):
    return _opt_solve(lambda w: w @ cov @ w, len(mu), wmin, wmax, strict=strict)


def opt_risk_parity(cov, wmin=0.0, wmax=1.0, strict=True):
    def obj(w):
        rc = risk_contributions(w, cov)
        return float(((rc - rc.mean()) ** 2).sum()) * 1e4
    # 위험균형은 비중이 0이면 기여도가 정의되지 않으므로 아주 작은 하한을 둔다
    return _opt_solve(obj, len(cov),
                      np.maximum(_as_bounds(wmin, len(cov), 0.0), 1e-4), wmax,
                      strict=strict)


def opt_target_return(mu, cov, target, wmin=0.0, wmax=1.0, strict=True):
    extra = [{"type": "eq", "fun": lambda w: float(w @ mu) - target}]
    return _opt_solve(lambda w: w @ cov @ w, len(mu), wmin, wmax, extra,
                      strict=strict)


def efficient_frontier(mu, cov, n_points=30, wmin=0.0, wmax=1.0):
    n = len(mu)
    _hi_arr = _as_bounds(wmax, n, 1.0)
    w_lo = opt_min_vol(mu, cov, wmin, wmax)
    if w_lo is None:
        return []
    lo = float(mu @ w_lo)
    if float(_hi_arr.min()) >= 1.0:
        hi = float(mu.max())
    else:
        w_hi = _opt_solve(lambda w: -float(w @ mu), n, wmin, wmax)
        hi = float(mu @ w_hi) if w_hi is not None else lo + 1e-6
    hi = max(hi, lo + 1e-6)
    pts = []
    for tgt in np.linspace(lo, hi, n_points):
        w = opt_target_return(mu, cov, tgt, wmin, wmax)
        if w is None:
            continue
        r, v, _ = port_perf(w, mu, cov)
        if abs(r - tgt) < 5e-3:
            pts.append((v, r, w))
    return pts


# ======================================================================
# 목표 제약이 있는 일반 최적화
# ======================================================================
# ---------------------- 벤치마크 대비 지표 ----------------------
def rel_metrics(rp: pd.Series, rb: pd.Series) -> dict:
    """
    벤치마크 대비 상대성과.
      · 초과수익률 — 연율 수익률의 차이
      · 추적오차   — 액티브 수익(포트 − 벤치)의 연율 변동성
      · 정보비율   — 초과수익률 / 추적오차. 위험 한 단위당 초과수익
      · 승률       — 벤치마크를 이긴 날의 비율
    """
    df = pd.concat([rp, rb], axis=1).dropna()
    if len(df) < 20:
        return {k: np.nan for k in
                ("초과수익률(%p)", "추적오차(%)", "정보비율", "승률(%)")}
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    act = a - b
    te = float(act.std() * np.sqrt(TRADING_DAYS))
    ann_a = float((1 + a).prod() ** (TRADING_DAYS / len(a)) - 1)
    ann_b = float((1 + b).prod() ** (TRADING_DAYS / len(b)) - 1)
    excess = ann_a - ann_b
    return {"초과수익률(%p)": excess * 100,
            "추적오차(%)": te * 100,
            "정보비율": (excess / te) if te > 1e-12 else np.nan,
            "승률(%)": float((act > 0).mean() * 100)}


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
                 progress=None, cost_bp=0.0):
    """
    각 재최적화 시점까지의 데이터로만 비중을 구해 다음 구간에 적용하고 이어붙인다.
    실제로 운용했다면 어땠을지를 재현한다.
    """
    dates = [pd.Timestamp(d) for d in
             pd.date_range(start, prices.index[-1], freq=freq_code)]
    if not dates:
        return None, [], 0
    segs, hist = [], []
    fails = 0                # 최적해를 못 찾아 건너뛴 재최적화 시점 수
    prev_end = None          # 직전 구간 종료 시점의 비중
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
        if w is None:
            # 조용히 균등비중으로 대체하지 않고, 이 시점은 건너뛴다
            fails += 1
            continue
        end = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
        seg = prices.loc[d:end]
        if len(seg) < 2:
            continue
        W = {t: float(x) for t, x in zip(tickers, w)}
        try:
            _sr = portfolio_returns(seg, W, rebal)
            # 재최적화 시점의 '실제 교체분'만 비용으로 잡는다.
            # 직전 구간의 종료 비중을 넘기지 않으면 매 구간 100% 재구축으로
            # 계산돼 비용이 크게 부풀려진다.
            _st = turnover_series(seg, W, rebal,
                                  prev_w=prev_end).reindex(_sr.index).fillna(0)
            segs.append(apply_cost(_sr, _st, cost_bp))
            # 이 구간이 끝난 시점의 비중(가격 변동으로 흘러간 상태)을 다음으로 넘긴다
            _wd = weight_drift(seg, W, rebal)
            prev_end = {k: float(v) for k, v in _wd.iloc[-1].items()} if len(_wd) else W
        except Exception:
            continue
        hist.append((d, w))
        if progress:
            progress((i + 1) / len(dates))
    if not segs:
        return None, [], fails
    full = pd.concat(segs)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full, hist, fails


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
    ret_max = float(w_hi @ mu) if w_hi is not None else float(mu.max())
    w_lo = opt_min_vol(mu, cov, wmin, wmax)
    vol_min = (float(np.sqrt(w_lo @ cov @ w_lo)) if w_lo is not None
               else float(np.sqrt(np.diag(cov)).min()))

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
        if w_t is not None:
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
        # HRP 는 제약을 직접 다루지 못하므로 결과를 제약 안으로 투영한다
        w_h = clip_to_bounds(hrp_weights(R), wmin, wmax)
        return (w_h if w_h is not None else hrp_weights(R)), mu, cov

    if goal.startswith("위험균형"):
        def rp_obj(w):
            rc = risk_contributions(w, cov)
            return float(((rc - rc.mean()) ** 2).sum()) * 1e4
        return (_opt_solve(rp_obj, n,
                           np.maximum(_as_bounds(wmin, n, 0.0), 1e-4),
                           wmax, extra, strict=True), mu, cov)

    if goal.startswith("수익률"):
        return (_opt_solve(lambda w: -float(w @ mu), n, wmin, wmax, extra,
                           strict=True), mu, cov)

    if goal.startswith("위험 최소화"):
        if d["kind"] == "ratio":
            fn = lambda w: -float(d["fn"](_series(R, w), rf=rf))
        else:
            fn = lambda w: float(d["fn"](_series(R, w), rf=rf))
        return _opt_solve(fn, n, wmin, wmax, extra, strict=True), mu, cov

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
    return _opt_solve(fn, n, wmin, wmax, extra, strict=True), mu, cov



# ======================================================================
# 최적화 화면
# ======================================================================

# ======================================================================
# 도움말
# ======================================================================
METRIC_DOCS = {
    "수익률": [
        ("연평균 수익률 (CAGR)",
         "전체 기간의 성장을 '매년 몇 %씩 복리로 불었나'로 환산한 값입니다.",
         "최종자산 / 초기자산 의 (1/연수) 제곱 − 1",
         "기간이 다른 포트폴리오를 비교할 때 총수익률보다 공정합니다. "
         "다만 중간의 등락은 보여주지 않으므로 변동성·MDD와 함께 보세요."),
        ("누적성장배수",
         "1을 넣었을 때 끝에 얼마가 됐는지를 배수로 표시합니다.",
         "(1 + 일별수익률)을 전부 곱한 값",
         "2.5면 원금의 2.5배, 즉 +150% 입니다."),
        ("내부수익률 (IRR)",
         "돈을 여러 번 나눠 넣었을 때의 실질 수익률입니다.",
         "모든 현금흐름의 현재가치 합을 0으로 만드는 할인율",
         "적립식에서는 단순 수익률이 실제보다 낮게 보입니다. "
         "나중에 넣은 돈일수록 투자 기간이 짧기 때문이며, IRR이 올바른 지표입니다."),
    ],
    "위험": [
        ("변동성 (Volatility)",
         "수익률이 평균에서 얼마나 흩어져 있는지를 연 단위로 환산한 값입니다.",
         "일별 수익률의 표준편차 × √252",
         "가장 널리 쓰이지만 상승과 하락을 구분하지 못합니다. "
         "급등이 잦아도 변동성은 커집니다."),
        ("최대낙폭 (MDD)",
         "직전 고점 대비 가장 깊이 떨어졌던 폭입니다.",
         "낙폭 시계열의 최솟값",
         "실제로 견뎌야 했던 최악의 순간을 보여줍니다. "
         "−50%를 회복하려면 +100%가 필요하므로 체감 충격이 큽니다.\n\n"
         "**항상 음수이므로 비교할 때 주의가 필요합니다.** −20%에서 −15%로 바뀌면 "
         "숫자는 커졌지만 낙폭은 얕아진 것이므로 **개선**입니다. 이 도구에서는 "
         "그런 경우를 '개선(%p)'으로 표기해 혼동을 줄였습니다."),
        ("낙폭 (Drawdown)",
         "매 시점의 '직전 고점 대비 하락률'입니다. MDD는 이 시계열의 최저점입니다.",
         "현재자산 / 그때까지의 최고자산 − 1",
         "0으로 돌아오는 데 걸린 기간이 곧 원금 회복 기간입니다."),
        ("얼서지수 (Ulcer Index)",
         "낙폭의 깊이뿐 아니라 얼마나 오래 물려 있었는지까지 반영합니다.",
         "낙폭(%)을 제곱해 평균 낸 뒤 제곱근",
         "MDD가 같아도 오래 물려 있었다면 얼서지수가 큽니다. 낮을수록 좋습니다."),
        ("조건부 손실 (CVaR)",
         "최악의 5% 구간에서 평균적으로 얼마나 잃었는지입니다.",
         "일별 수익률 하위 5%의 평균",
         "'평소'가 아니라 '최악일 때'의 손실 크기를 봅니다. 꼬리위험 측정에 쓰입니다."),
        ("조건부 낙폭 (CDaR)",
         "최악의 5% 낙폭 구간의 평균입니다.",
         "낙폭 시계열 하위 5%의 평균",
         "MDD가 단 하나의 최저점만 보는 것과 달리, 깊은 하락 전반을 봅니다."),
        ("준표준편차 (Semi Std Dev)",
         "기준선 아래로 내려간 움직임만 위험으로 셉니다.",
         "음(−)의 편차만 제곱해 평균 낸 뒤 제곱근 × √252",
         "오르는 변동은 위험이 아니라고 보는 관점입니다."),
        ("평균절대편차 (MAD)",
         "평균에서 떨어진 거리의 평균입니다.",
         "|수익률 − 평균|의 평균 × √252",
         "제곱을 쓰지 않아 극단적인 하루에 덜 휘둘립니다."),
    ],
    "위험조정 수익": [
        ("샤프지수 (Sharpe Ratio)",
         "위험 한 단위당 초과수익입니다. 가장 널리 쓰이는 종합 점수입니다.",
         "(수익률 − 무위험수익률) / 변동성",
         "1을 넘으면 양호, 2를 넘으면 우수하다고 봅니다. "
         "다만 변동성을 쓰므로 상승 변동도 감점 요인이 됩니다."),
        ("소르티노 비율 (Sortino Ratio)",
         "샤프지수의 분모를 하락 변동성만으로 바꾼 것입니다.",
         "(수익률 − 무위험수익률) / 하방편차",
         "상승 변동을 벌하지 않으므로, 오를 때 크게 오르는 자산에 유리합니다."),
        ("칼마 비율 (Calmar Ratio)",
         "수익을 최대낙폭으로 나눈 값입니다.",
         "연평균 수익률 / |최대낙폭|",
         "'최악을 견딘 대가로 얼마를 벌었나'를 봅니다. 클수록 좋습니다."),
        ("마틴 비율 (Martin Ratio)",
         "수익을 얼서지수로 나눈 값입니다.",
         "연평균 수익률(%) / 얼서지수",
         "낙폭의 깊이와 지속기간을 함께 감안한 효율입니다."),
        ("오메가 비율 (Omega Ratio)",
         "기준선 위 이익의 총합을 아래 손실의 총합으로 나눈 값입니다.",
         "Σ(기준선 초과분) / Σ(기준선 미달분)",
         "평균과 표준편차만 보는 지표와 달리 분포 전체 모양을 반영합니다. "
         "1보다 크면 이익 쪽이 우세합니다."),
    ],
    "자산배분 모형": [
        ("균형 기대수익률 (Implied Returns)",
         "시장 비중이 최적이 되려면 각 자산이 얼마를 내야 하는지 역산한 값입니다.",
         "Π = δ × Σ × w(시장비중)",
         "과거 평균을 그대로 쓰지 않고 시장이 암묵적으로 기대하는 수익률을 뽑아냅니다. "
         "추정 오차가 큰 과거 평균의 대안입니다."),
        ("사후 기대수익률 (Posterior Returns)",
         "균형 기대수익률에 내 전망을 확신도만큼 섞은 결과입니다.",
         "[(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]",
         "뷰를 넣지 않으면 균형 값과 같습니다. 확신도가 극단적으로 높으면 뷰 값에 "
         "수렴합니다."),
        ("위험회피계수 (δ)",
         "위험 한 단위를 감수하는 대가로 얼마의 수익을 요구하는지를 나타냅니다.",
         "δ = (시장수익률 − 무위험수익률) / 시장분산",
         "클수록 위험을 싫어해 안전자산 비중이 늘어납니다. 통상 2~3을 씁니다."),
        ("τ (타우)",
         "균형 기대수익률 추정 자체의 불확실성입니다.",
         "공분산에 곱하는 스칼라 (통상 0.025~0.05)",
         "작을수록 시장 균형을 신뢰해 뷰의 영향이 줄어듭니다."),
    ],
    "벤치마크 대비": [
        ("알파 (Alpha)",
         "시장 노출만으로는 설명되지 않는 초과수익입니다.",
         "내 수익률 − [무위험수익률 + 베타 × (시장수익률 − 무위험수익률)]",
         "양수면 시장을 따라간 것 이상의 성과가 있었다는 뜻입니다. "
         "다만 표본이 짧으면 우연일 수 있습니다."),
        ("베타 (Beta)",
         "시장이 1% 움직일 때 내 포트폴리오가 몇 % 움직이는지입니다.",
         "공분산(내 수익률, 시장) / 시장의 분산",
         "1이면 시장과 같은 폭, 1.5면 1.5배 민감, 0.5면 절반만 반응합니다."),
        ("R²",
         "내 움직임 중 시장으로 설명되는 비율입니다.",
         "상관계수의 제곱",
         "0.9면 대부분 시장을 따라간다는 뜻이고, 낮으면 독자적으로 움직입니다. "
         "R²가 낮으면 베타·알파의 신뢰도도 함께 낮아집니다."),
        ("상승장·하락장 포착률 (Capture Ratio)",
         "시장이 오른 날/내린 날에 내가 그중 몇 %를 따라갔는지입니다.",
         "해당 구간의 내 평균수익 / 시장 평균수익 × 100",
         "상승 포착률은 높고 하락 포착률은 낮을수록 유리합니다. "
         "예를 들어 상승 110% · 하락 80%면 좋은 구조입니다."),
    ],
    "구성·비용": [
        ("위험 기여도 (Risk Contribution)",
         "포트폴리오 전체 변동성에서 각 종목이 차지하는 몫입니다.",
         "비중 × (공분산행렬 × 비중) / 포트폴리오 변동성",
         "합치면 포트폴리오 변동성이 됩니다. 비중이 낮아도 변동이 크거나 "
         "다른 자산과 함께 움직이면 위험 기여도는 높게 나옵니다."),
        ("연 회전율 (Turnover)",
         "1년 동안 자산의 몇 %를 사고팔았는지입니다.",
         "매 리밸런싱 시점의 |바뀐 비중| 합계를 연 단위로 환산",
         "매수 후 보유는 0%, 매일 리밸런싱은 200%를 넘기도 합니다. "
         "회전율이 높을수록 거래비용 부담이 커집니다."),
        ("상관관계 (Correlation)",
         "두 자산이 얼마나 함께 움직이는지입니다.",
         "−1에서 +1 사이의 값",
         "1에 가까우면 같이, 0이면 무관하게, 음수면 반대로 움직입니다. "
         "낮은 값이 많을수록 분산 효과가 큽니다."),
        ("실효 종목 수 (Effective N)",
         "분산 효과가 실제로는 몇 종목에 해당하는지를 나타냅니다.",
         "N / (1 + (N−1) × 평균 상관계수)",
         "10종목을 담아도 서로 비슷하게 움직이면 실효는 2~3종목에 그칩니다. "
         "평균 상관이 음수이면 종목 수보다 커질 수 있으며, 서로 헤지된다는 뜻입니다."),
        ("성장 기여도 (Contribution to Growth)",
         "포트폴리오 총 성장률 중 각 종목이 만들어낸 몫입니다.",
         "Σ (그 시점 자산가치 × 시작비중 × 그날 수익률)",
         "모든 종목의 기여도를 더하면 포트폴리오 총수익과 정확히 일치합니다. "
         "비중이 작아도 크게 오른 종목은 기여도가 클 수 있습니다."),
    ],
}


# ======================================================================
# 자체 점검 (계산 정합성)
# ======================================================================
def run_self_tests(prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    핵심 계산이 지켜야 할 성질을 실제로 확인한다.
    실데이터를 주면 그것으로, 없으면 합성 데이터로 검사한다.
    """
    rows = []

    def check(name, fn, note=""):
        try:
            ok, detail = fn()
        except Exception as ex:
            ok, detail = False, f"{type(ex).__name__}: {str(ex)[:60]}"
        rows.append({"검사": name, "결과": "✅ 통과" if ok else "❌ 실패",
                     "상세": detail, "확인하는 것": note})

    if prices is None or prices.empty or prices.shape[1] < 2:
        rng = np.random.default_rng(0)
        n = 900
        idx = pd.date_range("2021-01-01", periods=n, freq="B")
        mk = rng.normal(.0004, .010, n)
        prices = pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(a + b * mk + rng.normal(0, s, n)))
             for t, a, b, s in [("A", .0005, 1.0, .004), ("B", .0001, -.2, .002),
                                ("C", .0003, .1, .008)]}, index=idx)
    tk = list(prices.columns)[:3]
    px = prices[tk].dropna()
    W = {t: 100.0 / len(tk) for t in tk}

    # 1. 기여도 항등식
    def t_contrib():
        worst = 0.0
        for mode in (REBAL_DAILY, REBAL_MONTH, REBAL_QUARTER, REBAL_YEAR, REBAL_NONE):
            r = portfolio_returns(px, W, mode)
            g = growth_contribution(px, W, mode, r)
            worst = max(worst, abs(float(g.sum()) - float((1 + r).prod() - 1)))
        return worst < 1e-9, f"최대 오차 {worst:.2e}"
    check("성장 기여도 = 총수익", t_contrib,
          "종목별 기여도를 더하면 포트폴리오 총수익과 같아야 합니다")

    # 2. 비중 추이 합계
    def t_weight():
        worst = 0.0
        for mode in (REBAL_QUARTER, REBAL_NONE):
            s = weight_drift(px, W, mode).sum(axis=1)
            worst = max(worst, float((s - 1).abs().max()))
        return worst < 1e-9, f"최대 편차 {worst:.2e}"
    check("보유 비중 합계 = 100%", t_weight,
          "어느 시점에도 비중 합이 100%를 벗어나면 안 됩니다")

    # 3. 자산곡선 시작값
    def t_equity():
        v = float(equity_curve(portfolio_returns(px, W, REBAL_QUARTER)).iloc[0])
        return abs(v - 1.0) < 1e-12, f"첫 값 {v:.10f}"
    check("자산곡선이 1.00에서 시작", t_equity,
          "첫날 수익률이 곡선에 미리 반영되면 안 됩니다")

    # 4. 회전율 경계
    def t_turnover():
        a = turnover_series(px, W, REBAL_NONE).iloc[1:].sum()
        b = annual_turnover(turnover_series(px, W, REBAL_DAILY))
        return (a < 1e-9 and b > 0.5), f"매수후보유 {a:.2e} · 매일 {b*100:.0f}%/년"
    check("회전율 경계값", t_turnover,
          "매수 후 보유는 거래가 0, 매일 리밸런싱은 회전율이 커야 합니다")

    # 5. 거래비용 방향
    def t_cost():
        r = portfolio_returns(px, W, REBAL_QUARTER)
        tno = turnover_series(px, W, REBAL_QUARTER).reindex(r.index).fillna(0)
        g, nt = cagr(r), cagr(apply_cost(r, tno, 50))
        return nt < g, f"비용 전 {g*100:.3f}% → 후 {nt*100:.3f}%"
    check("거래비용이 성과를 낮춤", t_cost,
          "비용을 넣으면 수익률이 반드시 줄어야 합니다")

    # 6. 종목별 제약
    def t_bounds():
        mu, cov = ann_stats(px.pct_change().dropna())
        lo = np.array([0.0, 0.0, 0.15][:len(tk)])
        hi = np.array([0.40, 1.0, 1.0][:len(tk)])
        w = opt_max_sharpe(mu, cov, 0.03, lo, hi)
        if w is None:
            return False, "최적해를 찾지 못함"
        ok = ((w >= lo - 1e-6).all() and (w <= hi + 1e-6).all()
              and abs(w.sum() - 1) < 1e-6)
        return ok, " ".join(f"{t}:{x*100:.1f}%" for t, x in zip(tk, w))
    check("종목별 최소·최대 비중 준수", t_bounds,
          "종목마다 다른 한도가 각각 적용되어야 합니다")

    # 7. 잘못된 제약 검출
    def t_badbounds():
        try:
            mu, cov = ann_stats(px.pct_change().dropna())
            opt_min_vol(mu, cov, np.array([0.6] + [0.0] * (len(tk) - 1)),
                        np.array([0.3] + [1.0] * (len(tk) - 1)))
            return False, "예외가 발생하지 않음"
        except ValueError:
            return True, "ValueError 발생 (정상)"
    check("모순된 제약을 걸러냄", t_badbounds,
          "최소가 최대보다 크면 조용히 넘어가면 안 됩니다")

    # 8. 0분모 방어
    def t_zerodiv():
        flat = pd.Series(0.0, index=px.index[:200])
        # 샤프·칼마는 분모(표준편차·MDD)가 항상 0이라 무위험 수준과 무관하게 NaN.
        base_nan = pd.isna(sharpe_ratio(flat, 0.03)) and pd.isna(calmar_ratio(flat))
        # 소르티노는 초과수익이 정확히 0일 때만(무위험도 0) 진짜 0분모라 NaN.
        sortino_zero = pd.isna(sortino_ratio(flat, 0.0))
        # 무위험보다 낮은 채로 flat하면 실제 손실이 있으므로 유한한 값이 정상이다
        # (여기서 NaN이 나오면 이전의 '음수 부분집합 표준편차' 버그가 재발한 것).
        s_neg = sortino_ratio(flat, 0.03)
        sortino_finite = np.isfinite(s_neg) and s_neg < 0
        ok = base_nan and sortino_zero and sortino_finite
        return ok, (f"샤프·칼마 NaN · 소르티노(무위험=0) NaN · "
                    f"소르티노(무위험=3%) {s_neg:.2f} (유한)")
    check("변동성 0일 때 비율 폭주 방지", t_zerodiv,
          "실제 초과수익이 0일 때만 NaN, 무위험보다 못한 채 flat하면 유한한 나쁜 값이어야 합니다")

    # 9. BL — 뷰가 없으면 기준 비중
    def t_bl():
        mu, cov = ann_stats(px.pct_change().dropna())
        wm = np.array([0.5, 0.3, 0.2][:len(tk)])
        wm = wm / wm.sum()
        _, post = black_litterman(cov, wm, 2.5, 0.05, None, None, None)
        w = bl_weights(post, cov, 2.5, -1.0, 1.0, allow_short=True)
        d = float(np.abs(w - wm).max())
        return d < 1e-6, f"최대 차이 {d:.2e}"
    check("뷰가 없으면 기준 비중 그대로", t_bl,
          "블랙-리터만의 핵심 성질입니다")

    # 10. 상대 부 기준 초과수익
    def t_excess():
        rp = portfolio_returns(px, W, REBAL_QUARTER)
        rb = px[tk[0]].pct_change().dropna()
        a, b = rp.align(rb, join="inner")
        pe, be = (1 + a).cumprod(), (1 + b).cumprod()
        exact = float(pe.iloc[-1]) / float(be.iloc[-1]) - 1
        shown = float((pe / be - 1).iloc[-1])
        return abs(exact - shown) < 1e-12, f"오차 {abs(exact-shown):.2e}"
    check("누적 초과수익 = 상대 부 변화", t_excess,
          "일별 차이를 복리로 굴리면 실제와 벌어집니다")

    # 11. 채권 충격 방향
    def t_bond():
        up = shock_asset("국채", 16.0, 250, 0, 0, 300, 0, 0, 0)["금리"]
        dn = shock_asset("국채", 16.0, 250, 0, 0, -300, 0, 0, 0)["금리"]
        hy = shock_asset("하이일드", 3.5, 25, 0, 0, 0, 300, 0, 0)["신용"]
        ig = shock_asset("회사채", 8.4, 120, 0, 0, 0, 300, 0, 0)["신용"]
        ok = (up < 0 < dn) and abs(dn) > abs(up) and hy < ig < 0
        return ok, f"금리+300 {up:.1f}% · −300 {dn:+.1f}% · 신용 HY {hy:.1f}% IG {ig:.1f}%"
    check("채권 충격 방향과 볼록성", t_bond,
          "금리 상승은 손실, 하락 이익이 더 커야 하고, 하이일드가 신용에 더 민감해야 합니다")

    # 12. 환율 곱셈 적용
    def t_fx():
        v = combine_shock({"주식": -35.0, "금리": 0, "신용": 0, "원자재": 0}, 12.0)
        exact = ((1 - 0.35) * (1 + 0.12) - 1) * 100
        return abs(v - exact) < 1e-9, f"{v:.2f}% (덧셈이면 {-35+12}%)"
    check("환율을 곱셈으로 적용", t_fx,
          "단순 덧셈은 충격이 클수록 실제와 벌어집니다")

    return pd.DataFrame(rows)


def render_help():
    st.title("📖 도움말")
    st.caption("이 도구를 쓰는 방법과, 화면에 나오는 숫자들이 무슨 뜻인지 정리했습니다.")

    tab1, tab2, tab3, tab4 = st.tabs(["🚀 사용법", "📊 지표 사전", "🔬 방법론",
                                      "🧪 자체 점검"])

    with tab4:
        st.subheader("계산 정합성 자체 점검")
        st.caption("이 도구의 핵심 계산이 지켜야 할 성질을 실제로 확인합니다. "
                   "기여도 합계가 총수익과 맞는지, 비중 합이 100%인지, 제약이 "
                   "지켜지는지 같은 것들입니다. **하나라도 실패하면 결과를 믿지 "
                   "마시고 알려주세요.**")
        if st.button("🧪 점검 실행", type="primary", width="stretch",
                     key="_selftest_go"):
            with st.spinner("계산 검증 중..."):
                _res = run_self_tests()
            _fail = int((_res["결과"].astype(str).str.contains("실패")).sum())
            if _fail:
                st.error(f"🚫 {_fail}개 항목이 실패했습니다. 아래 상세를 확인해주세요.")
            else:
                st.success(f"✅ {len(_res)}개 항목 모두 통과했습니다.")
            st.dataframe(_res, width="stretch", hide_index=True)
            st.caption("합성 데이터로 검사합니다. 실제 종목 데이터가 아니라 "
                       "**계산 논리 자체**를 확인하는 것입니다.")
        else:
            st.info("👆 **점검 실행**을 누르면 12가지 항목을 확인합니다.")


    # ------------------------------------------------------------------
    with tab1:
        st.subheader("이 도구는 무엇인가")
        st.markdown(
            "종목을 조합해 **과거 성과를 계산하고, 더 나은 배분을 찾고, 위험 구조를 "
            "들여다보는** 도구입니다. 왼쪽 사이드바에서 화면을 고릅니다.\n\n"
            "| 화면 | 답하는 질문 |\n|---|---|\n"
            "| **📊 포트폴리오 분석** | 이 구성은 과거에 어땠나 |\n"
            "| **🎯 포트폴리오 최적화** | 비중을 어떻게 바꾸면 나아지나 |\n"
            "| **➕ 자산 추가 효과** | 무엇을 얼마나 더하면 좋을까 |\n"
            "| **🧭 뷰 기반 자산배분** | 시장 전망의 영향 분석 + 자산 전망 배분 |\n"
            "| **🔗 자산 상관관계** | 이 종목들은 서로 얼마나 겹치나 |\n"
            "| **📉 시장·거시 국면** | 강세·약세 구분 + 경기·물가 거시 국면 |\n""| **🌩 스트레스 테스트** | 과거 위기 재현 + 가상 충격 |\n""| **⚖️ 최종 대안 비교** | 여러 안 중 무엇을 고를까 |\n\n"
            "**사이드바 설정(기준 통화·기간·배당·거래비용·환헤지 등)은 모든 화면에 "
            "공통으로 적용**됩니다. 종목 표와 티커 입력 방식도 화면마다 동일합니다.")

        st.divider()
        st.subheader("1단계 · 종목 입력")
        st.markdown(
            "**티커는 세 가지 형식을 모두 인식합니다.** 어느 것을 넣든 정식 티커로 바뀝니다.\n\n"
            "| 입력 | 결과 | 설명 |\n|---|---|---|\n"
            "| `005930.KS` | `005930.KS` | 야후 정식 형식 |\n"
            "| `005930 KS` | `005930.KS` | 블룸버그 형식 (가장 정확) |\n"
            "| `005930` | `005930.KS` | 숫자만 (코스피→코스닥 순으로 탐색) |\n"
            "| `NVDA` | `NVDA` | 미국은 그대로 |\n\n"
            "입력 후 **Enter**를 누르면 종목명이 자동으로 채워집니다. "
            "❌가 뜨면 티커가 잘못된 것이며, 아래 **조회 실패 원인 보기**에서 이유를 볼 수 있습니다.")
        st.info("**엑셀에서 붙여넣기** — 티커와 비중 두 열을 복사해 표의 첫 칸에 "
                "붙여넣으면 한 번에 채워집니다. 열 순서는 `티커 | 비중` 이어야 합니다. "
                "엑셀에서 `005930`처럼 앞자리가 0인 코드는 텍스트 서식으로 저장해야 "
                "0이 사라지지 않습니다.")

        st.divider()
        st.subheader("2단계 · 공통 설정 (왼쪽 사이드바)")
        st.markdown(
            "| 설정 | 의미 |\n|---|---|\n"
            "| **기준 통화** | 모든 자산을 이 통화로 환산해 비교합니다 |\n"
            "| **시작일 / 종료일** | 분석 기간 |\n"
            "| **배당 처리** | 재투자하면 총수익, 아니면 주가만 봅니다 |\n"
            "| **거래비용** | 편도 bp. 회전율에 곱해 수익률에서 차감합니다 |\n"
            "| **환헤지 가정** | 켜면 환율 변동을 제거하고 종목 자체 수익률만 봅니다 |\n"
            "| **휴장일 처리** | 국가를 섞을 때 공통 거래일이 줄어드는 문제를 다룹니다 |\n"
            "| **무위험 수익률** | 샤프·소르티노 계산의 기준선입니다 |\n\n"
            "사이드바 설정은 **두 화면 모두에 적용**됩니다.")

        st.divider()
        st.subheader("3단계 · 결과 읽기")
        st.markdown(
            "**분석 화면**은 성과 차트 → 낙폭 → 종합 비교표 → 연도별 → 적립식 → "
            "포트폴리오별 상세 순으로 이어집니다. 종합 비교표에서 초록색은 "
            "각 지표의 최고값입니다.\n\n"
            "**최적화 화면**에서 가장 중요한 것은 **표본외 성과**입니다. "
            "최적화가 보지 못한 기간의 성적이라 실제 실력에 가깝습니다. "
            "표본내(학습 구간) 성과는 좋게 나오는 것이 당연하므로 판단 근거로 삼지 "
            "않아야 합니다.")

        st.divider()
        st.subheader("화면별 사용법")
        with st.expander("📊 포트폴리오 분석"):
            st.markdown(
                "종목과 비중을 넣고 **분석 실행**을 누르면 성과 차트·낙폭·종합 비교표가 "
                "나옵니다. 포트폴리오를 여러 개 만들어 나란히 비교할 수 있고, 벤치마크도 "
                "여러 개 넣을 수 있습니다.\n\n"
                "연도별 성과, 적립식 투자(IRR 포함), 포트폴리오별 상세(성장 기여도·비중 "
                "추이·월별 수익률)까지 이어집니다.")
        with st.expander("🎯 포트폴리오 최적화"):
            st.markdown(
                "**현재 비중을 반드시 입력하세요.** 최적화 결과를 그 구성과 비교하는 것이 "
                "이 화면의 핵심입니다.\n\n"
                "목적과 위험 정의를 고르고, **최적화 기준일**과 **학습 기간**을 정합니다. "
                "기준일까지의 데이터로만 비중을 계산하고, 그 이후 구간으로 채점합니다.\n\n"
                "결과에서 **표본외 성과**를 먼저 보세요. 표본내는 좋게 나오는 것이 당연합니다. "
                "**비중 안정성 점검**에서 학습 기간에 따라 비중이 크게 흔들린다면 그 결과는 "
                "신뢰도가 낮습니다.")
        with st.expander("➕ 자산 추가 효과"):
            st.markdown(
                "두 가지 방식이 있습니다.\n\n"
                "- **📌 고정 비중 편입 효과** — 1·3·5·10% 같은 정해진 비중으로 넣어보고 "
                "지표가 어떻게 달라지는지 봅니다. **재원을 어디서 빼느냐**(비례 축소 vs "
                "특정 자산)에 따라 결과가 크게 달라지므로 함께 비교하세요.\n"
                "- **🔎 최적 조합 탐색** — 후보 중 어떤 조합이 가장 좋은지 최적화로 찾습니다. "
                "후보는 **15~30개**, 추가는 **1~3개**가 적당합니다.\n\n"
                "조합이 1,000가지를 넘으면 전수 탐색 대신 단계적 선택으로 자동 전환됩니다. "
                "순위는 학습 구간 기준이고 표본외 성과는 확인용으로 함께 표시됩니다.\n\n"
                "**1등만 보지 마세요.** 상위 조합 전반과 **채택 빈도**를 함께 보는 것이 "
                "중요합니다. 여러 조합에 반복해서 등장하는 종목이 실제로 유용한 후보입니다.\n\n"
                "**반복 검증 기준일 수**를 2 이상으로 두면 기준일을 6개월씩 뒤로 옮겨가며 "
                "같은 탐색을 반복합니다. 여러 시점에서 거듭 뽑히는 후보라야 특정 시기의 "
                "우연이 아니라고 볼 수 있습니다. 한 번만 등장한 후보는 신중하게 보세요.")
        with st.expander("🧭 뷰 기반 자산배분 · 시장·거시 전망"):
            st.markdown(
                "금리·경기·물가·달러에 대한 전망을 넣으면, 각 자산이 그 변수에 얼마나 "
                "민감한지 **과거 데이터로 추정해** 영향을 분석합니다.\n\n"
                "**장기채를 보유하지 않아도 금리 전망을 넣을 수 있습니다.** 전망 대상과 "
                "투자 대상이 분리돼 있어서요. 전망이 지금 포트폴리오에 표현되지 않으면 "
                "그 사실도 알려줍니다.\n\n"
                "**민감도는 다중회귀로 구합니다.** 요인끼리 상관이 있을 때 단순회귀는 "
                "크게 빗나가기 때문입니다. 요인은 기본 4개(금리·경기·물가·달러)이고, "
                "신용·위험선호는 경기와 많이 겹쳐 선택 사항으로 뒀습니다.\n\n"
                "**민감도는 표준화해서 보여줍니다.** 요인마다 평소 움직이는 폭이 달라"
                "(금리는 작고 경기는 큼), 그대로 두면 계수가 스케일 차이만큼 부풀려져 "
                "요인 간 비교가 안 됩니다. 표의 값은 **요인이 평소만큼(1σ) 움직일 때 "
                "자산이 몇 % 반응하는지**입니다. 원래 단위는 접이식에서 볼 수 있고, "
                "거기서는 금리 민감도를 채권 듀레이션과 비교할 수 있습니다.\n\n"
                "**🧪 민감도가 상식과 맞는지 확인** 버튼으로 검증할 수 있습니다. "
                "TLT·SHY·SPY·GLD 같이 성격이 분명한 자산의 민감도가 알려진 범위 안에 "
                "있는지 자동으로 봅니다. 여기서 어긋나면 요인 대용치에 문제가 있다는 "
                "뜻이니 영향 분석을 믿지 마세요.\n\n"
                "**R²(설명력)에 따라 판단 강도가 자동으로 조절됩니다.** 영향 점수에 "
                "√R²를 곱해, 설명력이 낮은 자산은 방향을 약하게만 말합니다. "
                "**R²가 0.10 미만이면 '판정 유보'** 로 두고 아예 방향을 말하지 "
                "않습니다.\n\n"
                "| R² | 표시 |\n|---|---|\n"
                "| 0.40 이상 | 일반 참고 가능 |\n"
                "| 0.20~0.40 | 제한적 참고 |\n"
                "| 0.10~0.20 | 신뢰도 매우 낮음 |\n"
                "| 0.10 미만 | 판정 유보 |\n\n"
                "요인 대용치가 모두 미국 지표라 **한국 개별종목은 설명력이 낮게** "
                "나옵니다. 예를 들어 반도체는 업황·메모리 가격·원달러·중국 경기가 "
                "지배하는데 그 요인들이 여기 없기 때문입니다. 거시와 무관하다는 뜻이 "
                "아니라, **이 모형만으로는 방향을 판단하기 어렵다**는 뜻입니다.\n\n"
                "**📐 기간을 바꿔도 민감도가 유지되는가** 로 계수 안정성도 확인할 수 "
                "있습니다. R²가 높아도 개별 계수는 표본에 따라 흔들릴 수 있어, "
                "1·3·5년에서 부호가 유지되는지 봅니다.\n\n"
                "**📉 시장·거시 국면** 화면에서 현재 국면을 전망 초안으로 가져올 수 "
                "있습니다. 국면은 현재 상태이고 전망은 앞으로이므로, 확정하지 않고 "
                "수정 가능한 초안으로만 넘깁니다.\n\n"
                "**전망 기간**은 가중치로 반영됩니다. 3개월(0.6)은 2년(1.15)보다 약하게 "
                "반영되는데, 가까운 앞을 먼 앞만큼 확신하기 어렵기 때문입니다.\n\n"
                "**정책금리와 장기금리를 함께 넣으면** 같은 금리 요인에 두 번 반영됩니다. "
                "입력 단계에서 경고가 뜨니 하나만 쓰세요. 국면 화면에서 초안을 가져올 "
                "때는 **장기금리만 자동으로 켜집니다.**\n\n"
                "**전망은 켰는데 해당 요인이 꺼져 있으면** 계산에서 조용히 빠집니다. "
                "이 경우에도 경고가 뜨며, 국면 초안을 가져올 때는 필요한 요인이 "
                "자동으로 함께 켜집니다.\n\n"
                "**영향 등급은 상대평가**입니다. 지금 입력한 자산들 사이에서 가장 크게 "
                "반응하는 자산을 기준으로 나머지를 견줍니다. 장기채처럼 민감도가 큰 "
                "자산이 섞이면 다른 자산이 상대적으로 중립에 가깝게 보일 수 있습니다.\n\n"
                "**물가 요인은 원자재(DBC)로 잽니다.** CPI 자체가 아니라 원유·금속·농산물 "
                "가격을 반영하므로, 화면에도 '물가·원자재'로 표기했습니다.\n\n"
                "**달러 민감도는 현지통화 기준**입니다. '달러가 강해질 때 그 자산 자체가 "
                "어떻게 움직이나'를 재는 것이고, 해외자산의 원화 환산 효과는 별개입니다. "
                "그건 🌩 스트레스 테스트의 환율 충격에서 다룹니다.\n\n"
                "이 단계에서는 **비중을 바꾸지 않고 진단만** 합니다.")
        with st.expander("🧭 뷰 기반 자산배분 · 자산수익률 전망"):
            st.markdown(
                "**기준 비중**을 출발점으로 넣고, 전망이 있으면 뷰로 추가합니다. "
                "뷰를 하나도 넣지 않으면 기준 비중이 그대로 결과가 됩니다.\n\n"
                "- **절대 뷰** — 'A가 연 12%를 낼 것이다'\n"
                "- **상대 뷰** — 'A가 B보다 연 3%p 나을 것이다'\n\n"
                "**확신도**가 조절 밸브입니다. 낮으면 기준 비중에서 거의 안 움직이고, "
                "높을수록 뷰 쪽으로 크게 기웁니다. 결과가 지나치게 쏠린다면 확신도를 낮추거나 "
                "최대 비중을 조여보세요.")
        with st.expander("🔗 자산 상관관계"):
            st.markdown(
                "종목들이 얼마나 함께 움직이는지 봅니다. **실효 종목 수**가 핵심 지표인데, "
                "10종목을 담아도 서로 비슷하면 실효는 2~3종목에 불과할 수 있습니다.\n\n"
                "**위험군 묶기**는 비슷하게 움직이는 자산을 하나로 묶어 보여줍니다. "
                "'자산은 7개지만 실질 위험군은 3개' 같은 진단이 나옵니다.\n\n"
                "**국면별·위기별 상관관계**를 꼭 펼쳐보세요. 평상시 0.3이던 상관이 "
                "약세장에 0.8로 오른다면, 기대했던 분산 효과가 정작 필요할 때 "
                "사라진다는 뜻입니다.\n\n"
                "**계산 주기 선택이 중요합니다.** 거래 시간대가 다른 국가를 섞으면 일별 "
                "상관계수가 실제보다 낮게 나옵니다. 한국·미국·일본을 함께 볼 때는 "
                "주별이나 월별을 쓰세요.")
        with st.expander("⚖️ 최종 대안 비교"):
            st.markdown(
                "여러 화면에서 만든 안을 **같은 조건으로 나란히** 비교합니다.\n\n"
                "각 화면 아래의 **⚖️ 비교 후보로 담기** 에 이름을 붙여 담으면 여기 모입니다. "
                "분석·최적화·자산 추가 효과·뷰 기반 배분 네 화면에서 담을 수 있고, "
                "최대 8개까지 보관됩니다.\n\n"
                "**리밸런싱·거래비용·기간을 통일해 다시 계산**하므로, 각 안을 만들 때의 "
                "설정과 결과가 다를 수 있습니다. 그래야 공정한 비교가 됩니다.\n\n"
                "수익·위험·낙폭·회전율에 더해 **위험 집중도**(한 종목이 전체 위험에서 "
                "차지하는 최대 비율)와 **위기별 손실**까지 함께 봅니다.")
        with st.expander("⚡ 가상 충격 시나리오"):
            st.markdown(
                "**🌩 스트레스 테스트** 화면의 두 번째 탭입니다. 과거에 없던 충격을 "
                "가정해 넣어봅니다.\n\n"
                "**주식·금리·신용스프레드·환율·원자재** 다섯 가지를 동시에 움직일 수 "
                "있고, 금융위기형·인플레 재점화 같은 시나리오도 준비돼 있습니다.\n\n"
                "**자산 유형별로 다르게** 계산합니다.\n\n"
                "| 유형 | 적용 요인 |\n|---|---|\n"
                "| 주식 | 시장 베타 × 주식 충격 |\n"
                "| 국채 | 듀레이션·볼록성 × 금리 충격 |\n"
                "| 회사채·하이일드 | 금리 효과 + 신용 스프레드 효과 |\n"
                "| 금 / 원자재 | 각각의 가격 충격 (분리) |\n"
                "| 해외자산 | 마지막에 환율을 곱셈으로 적용 |\n\n"
                "유형은 자동 판정하되 **표에서 직접 고칠 수 있습니다.** 목록에 없는 "
                "ETF는 '확인 필요'로 표시되니 듀레이션을 넣어주세요.\n\n"
                "**베타는 현지통화 기준**으로 추정합니다. 기준통화로 환산하면 환율이 "
                "이미 섞여 들어가 환율 충격이 이중 반영되기 때문입니다.\n\n"
                "**시장·통화가 여럿이면** 금리와 환율을 따로 줄 수 있습니다. "
                "미국채와 한국채는 다르게 움직이니까요.\n\n"
                "요인별 기여도에 **상호작용** 항목이 있는데, 환율이 곱셈으로 적용되며 "
                "생기는 차이입니다. 이 항목까지 더해야 합계가 예상 손익과 맞습니다.\n\n"
                "**선형 근사라는 점을 기억하세요.** 베타는 과거 평균이고, 실제 위기에는 "
                "상관관계가 급등해 추정보다 더 빠지는 경우가 많습니다. 하한선이 아니라 "
                "대략의 크기로 보시면 됩니다.")
        with st.expander("📅 과거 위기 재현"):
            st.markdown(
                "닷컴 붕괴·금융위기·코로나·긴축 발작 같은 **과거 위기 구간만 잘라내어** "
                "포트폴리오가 얼마나 버텼는지 봅니다. 직접 날짜를 지정해 구간을 추가할 수도 "
                "있습니다.\n\n"
                "**구간내 최저**는 위기 시작일 대비 최저 시점의 수익률, "
                "**회복까지**는 저점에서 위기 직전 "
                "수준을 되찾기까지 걸린 기간입니다. '미회복'이면 아직 되찾지 못했다는 "
                "뜻입니다.\n\n"
                "**위기 구간 상관관계**를 꼭 확인하세요. 평상시 낮던 상관이 위기에 함께 "
                "올라간다면, 정작 방어가 필요한 순간에 분산 효과가 줄어든다는 신호입니다.")
        with st.expander("🌡 거시 경기·물가 국면 (FRED)"):
            st.markdown(
                "**📉 시장 국면** 화면의 두 번째 탭입니다. 미국 거시지표"
                "(산업생산·고용·물가)로 **경기 × 물가** 4분면을 판정합니다.\n\n"
                "**발표 지연**을 반드시 감안하세요. 거시지표는 한두 달 뒤에 발표되므로 "
                "기본값 2개월만큼 지표를 뒤로 옮겨 그 시차를 근사적으로 반영합니다.\n\n"
                "다만 **과거 수치의 사후 수정 효과까지 없애지는 못합니다.** 지금 FRED에서 "
                "받는 값은 최신 개정치라서요. 완전히 재현하려면 ALFRED의 발표 당시 "
                "데이터가 필요한데, 이 도구는 거기까지 다루지 않습니다. "
                "그래서 이 화면은 **현재 상황을 읽는 용도**이고 과거 백테스트에는 "
                "쓰지 않습니다.\n\n"
                "**시장 내재 신호**는 야후 가격으로 계산합니다. 거시지표는 실물이고 "
                "시장은 선반영이라, **둘이 엇갈릴 때가 오히려 정보**입니다.\n\n"
                "**과거 국면별 성과**는 접어두었습니다. 표본이 적고 사후 수정 문제가 "
                "있어 통계적 근거가 약하기 때문입니다. 이 화면은 **현재 상황을 읽는 "
                "참고 자료**이지 과거 백테스트 도구가 아닙니다.")
        with st.expander("📉 시장 추세 국면 (가격 기반)"):
            st.markdown(
                "기준 지수를 강세·약세·횡보로 나누고, 국면마다 자산들이 어땠는지 봅니다.\n\n"
                "가장 중요한 부분은 **국면별 상관관계**입니다. 평상시 낮던 상관관계가 "
                "약세장에서 함께 올라간다면, 정작 방어가 필요한 순간에 분산 효과가 "
                "줄어든다는 뜻입니다.\n\n"
                "국면 구분은 **사후적**입니다. 실시간으로는 지금이 어느 국면인지 확정할 수 "
                "없다는 점을 감안해서 보세요.")

        st.divider()
        st.subheader("화면 사이 구성 옮기기")
        st.markdown(
            "모든 표 위에 **📋 복사** 와 **📥 붙여넣기** 버튼이 있습니다. "
            "한 화면에서 복사한 뒤 다른 화면에서 붙여넣으면 종목을 다시 입력할 필요가 "
            "없습니다.\n\n"
            "- **📋 복사** — 그 표의 종목과 비중을 보관함에 담습니다\n"
            "- **📥 붙여넣기** — 보관함 내용을 지금 표에 채웁니다. 비중 칸이 없는 "
            "화면이면 종목만 들어갑니다\n\n"
            "계산 결과도 옮길 수 있습니다. 최적화의 **최적 비중 복사**, 뷰 기반 배분의 "
            "**제안 배분 복사** 를 누르면 그 결과가 보관함에 담깁니다.\n\n"
            "**화면끼리 표를 공유하지는 않습니다.** 각 화면의 표는 독립적이므로, "
            "한쪽을 고쳐도 다른 쪽은 그대로입니다. 옮기고 싶을 때만 복사·붙여넣기를 "
            "쓰시면 됩니다.")

        st.divider()
        st.subheader("계산이 맞는지 확인하려면")
        st.markdown(
            "**🧪 자체 점검** 탭에서 이 도구의 핵심 계산이 지켜야 할 성질 12가지를 "
            "직접 확인할 수 있습니다. 기여도 합계가 총수익과 맞는지, 비중 합이 100%인지, "
            "종목별 제약이 실제로 지켜지는지 같은 것들입니다.\n\n"
            "하나라도 실패하면 그 결과는 믿지 마시고 알려주세요.")

        st.divider()
        st.subheader("저장 · 내보내기")
        st.markdown(
            "- **구성 저장 / 불러오기** — 각 화면 상단의 접이식 칸에 있습니다. "
            "이름을 붙여 저장하면 종목·비중·설정이 함께 보관됩니다.\n"
            "- 저장 내용은 브라우저 세션에 남습니다. 오래 쓰실 구성은 "
            "**📥 내보내기**로 파일을 받아두었다가 **📤 가져오기**로 복원하세요. "
            "이 파일을 다른 사람에게 보내면 같은 구성을 쓸 수 있습니다.\n"
            "- **엑셀 내보내기** — 화면과 같은 순서로 시트가 구성되며, "
            "성과·낙폭·비중 추이에는 엑셀에서 바로 편집할 수 있는 차트가 들어갑니다.")

        st.divider()
        st.subheader("자주 겪는 문제")
        with st.expander("티커가 ❌로 나옵니다"):
            st.markdown(
                "접미사를 확인해보세요 — 코스피 `.KS`, 코스닥 `.KQ`, 일본 `.T`, "
                "홍콩 `.HK`, 대만 `.TW`, 상하이 `.SS`, 선전 `.SZ`, 런던 `.L`, 독일 `.DE`.\n\n"
                "블룸버그 형식(`005930 KS`)을 쓰면 거래소를 직접 알려주므로 가장 확실합니다. "
                "그래도 안 되면 야후 파이낸스에 해당 종목이 없는 경우입니다.")
        with st.expander("분석 기간이 크게 줄었습니다"):
            st.markdown(
                "종목 중 하나의 데이터 이력이 짧으면 **공통 거래일**이 그만큼 잘립니다. "
                "빨간 경고와 함께 원인 종목이 표시되니 확인해보세요.\n\n"
                "국가를 섞어서 생긴 문제라면 사이드바의 **휴장일 처리**를 "
                "'직전 종가로 채우기'로 바꿔보세요. 다만 이 방식은 휴장일 수익률을 "
                "0으로 처리하므로 변동성이 실제보다 낮게 나옵니다.")
        with st.expander("데이터를 못 가져옵니다"):
            st.markdown(
                "야후 파이낸스가 일시적으로 요청을 제한하는 경우가 있습니다. "
                "잠시 후 다시 시도해보세요. 계속되면 `pip install -U yfinance` 로 "
                "최신 버전을 설치하면 해결되는 경우가 많습니다.")

    # ------------------------------------------------------------------
    with tab2:
        st.caption("화면에 나오는 숫자들의 정의와 읽는 법입니다.")
        for group, items in METRIC_DOCS.items():
            st.subheader(group)
            for name, what, how, note in items:
                with st.expander(name):
                    st.markdown(f"**무엇인가** · {what}")
                    st.markdown(f"**계산** · {how}")
                    st.markdown(f"**읽는 법** · {note}")
            st.divider()

    # ------------------------------------------------------------------
    with tab3:
        st.caption("숫자를 계산하는 방식과 그 배경 이론을 정리했습니다. "
                   "순서대로 읽으면 하나의 흐름이 됩니다.")
        th, pr = st.tabs(["📐 이론", "⚙️ 실무 요소"])

        # ============================ 이론 ============================
        with th:
            st.subheader("1. 수익률을 재는 두 가지 방식")
            st.markdown(
                "**산술평균**은 일별 수익률을 단순 평균한 값이고, **기하평균(복리)**은 "
                "자산이 실제로 불어난 속도입니다. 둘은 같지 않으며 항상 기하평균이 작습니다.")
            st.latex(r"g \;\approx\; \mu - \frac{\sigma^{2}}{2}")
            st.markdown(
                "차감되는 항을 **변동성 손실(volatility drag)** 이라고 합니다.\n\n"
                "+50% 뒤 −50%면 산술평균은 0%지만 실제로는 원금의 75%만 남습니다. "
                "변동성이 크면 같은 평균 수익률이어도 실제 실현되는 수익은 줄어듭니다. "
                "**변동성을 줄이는 것이 수익에도 도움이 되는** 이유입니다.")

            st.divider()
            st.subheader("2. 낙폭의 비대칭성")
            st.markdown("$x$만큼 하락한 것을 되돌리려면 이만큼 올라야 합니다.")
            st.latex(r"\text{필요 상승률} = \frac{x}{1-x}")
            st.markdown(
                "| 하락 | 필요한 상승 |\n|---|---|\n| −10% | +11% |\n| −20% | +25% |\n"
                "| −30% | +43% |\n| −50% | +100% |\n| −70% | +233% |\n\n"
                "최대낙폭(MDD)을 단순한 참고 수치가 아니라 **회복 가능성의 문제**로 보아야 하는 "
                "이유입니다. "
                "칼마 비율이 수익을 MDD로 나누는 것도 같은 맥락입니다.")

            st.divider()
            st.subheader("3. 분산투자가 작동하는 수학적 이유")
            st.markdown("두 자산 포트폴리오의 분산은 다음과 같습니다.")
            st.latex(r"\sigma_p^{2} = w_1^{2}\sigma_1^{2} + w_2^{2}\sigma_2^{2} "
                     r"+ 2\,w_1 w_2\,\rho_{12}\,\sigma_1\sigma_2")
            st.markdown(
                "핵심은 마지막 항의 **상관계수** 입니다.\n\n"
                "- 상관계수 1 — 분산 효과 없음. 위험은 그냥 가중평균\n"
                "- 상관계수 0 — 위험이 눈에 띄게 감소\n"
                "- 상관계수 음수 — 서로 상쇄. 이론상 완전 제거도 가능\n\n"
                "분산투자가 **투자에서 유일한 무상의 이득(free lunch)** 으로 불리는 근거가 "
                "여기에 있습니다. 기대수익은 비중의 "
                "가중평균 그대로인데 위험만 가중평균보다 낮아지기 때문입니다. "
                "수익을 포기하지 않고 위험만 줄이는 유일한 방법입니다.")
            st.info("종목을 늘려도 위험이 0이 되지는 않습니다. 개별 종목 고유의 위험"
                    "(비체계적 위험)은 사라지지만 시장 전체가 함께 움직이는 부분"
                    "(체계적 위험)은 남습니다. 베타가 재는 것이 이 남는 부분입니다.")

            st.divider()
            st.subheader("4. 효율적 투자선")
            st.markdown(
                "1952년 마코위츠는 '수익만 보지 말고 수익과 위험을 함께 보라'는 관점을 "
                "정식화했습니다. 가능한 모든 비중 조합을 (위험, 수익) 평면에 찍으면 "
                "위쪽 경계선이 생기는데, 이 선 위의 조합들이 **효율적 투자선**입니다.\n\n"
                "선 위의 각 점은 *그 수익률을 내는 조합 중 위험이 가장 낮은 것*입니다. "
                "선 아래 조합은 같은 위험으로 더 벌 수 있으므로 선택할 이유가 없습니다.\n\n"
                "무위험자산을 더하면, 무위험수익률에서 투자선에 그은 접선이 가장 효율적인 "
                "조합이 됩니다. 그 접점이 **샤프지수가 최대인 포트폴리오**입니다.")
            st.latex(r"\text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}")

            st.divider()
            st.subheader("5. 추정 오차와 비중 쏠림 — 최적화의 구조적 한계")
            st.markdown(
                "평균-분산 최적화를 제약 없이 수행하면 **소수 종목에 비중이 극단적으로 집중되는** "
                "결과가 자주 나타납니다. 위험을 반영하지 않아서가 아니라, **기대수익률 "
                "추정의 정밀도가 낮기 때문**입니다. 평균의 추정 오차는 다음과 같습니다.")
            st.latex(r"SE(\hat{\mu}) = \frac{\sigma}{\sqrt{T}} \qquad (T=\text{관측 연수})")
            st.markdown(
                "변동성 20%인 자산을 **1년** 관찰하면 기대수익률 추정의 표준오차가 "
                "**20%p**입니다. '연 15% 기대'라고 계산해도 참값은 −5%~+35% 어디든 될 수 "
                "있다는 뜻입니다.\n\n"
                "결정적인 점은 이 오차가 **관측 횟수가 아니라 달력상 기간**에만 의존한다는 "
                "것입니다. 일별로 잘게 쪼개 데이터 개수를 늘려도 평균 추정은 나아지지 "
                "않습니다. 반면 **변동성과 상관계수는 잘게 쪼갤수록 빠르게 정확해집니다.**\n\n"
                "그 결과 최적화는 실제로 우수한 자산이 아니라 **추정치가 상방으로 크게 벗어난 "
                "자산**을 선택하는 경향을 보입니다. Michaud는 이러한 성질을 들어 평균-분산 "
                "최적화를 **'오차 극대화기(error maximizer)'** 라고 표현한 바 있습니다.")
            st.warning("**대응 방법 세 가지**\n\n"
                       "1. **최대 편입비중 제약** — 한 종목의 비중 상한을 40% 등으로 설정합니다\n"
                       "2. **학습 기간 확대** — 추정 오차를 줄입니다\n"
                       "3. **기대수익률을 사용하지 않는 방식** — 변동성 최소화 · 위험균형 · HRP는 "
                       "평균을 추정하지 않으므로 결과가 한층 안정적입니다\n\n"
                       "**비중 안정성 점검**에서 학습 기간에 따라 비중이 크게 달라진다면, 해당 결과는 "
                       "추정 오차에 크게 의존하고 있을 가능성이 높습니다.")

            st.divider()
            st.subheader("6. 기대수익률을 쓰지 않는 대안들")
            st.markdown(
                "**변동성 최소화** — 공분산만 사용합니다. 수익을 포기하는 대신 추정 오차 "
                "문제에서 자유롭습니다.\n\n"
                "**위험균형 (Risk Parity)** — 각 자산의 *위험 기여도*가 같아지도록 배분합니다. "
                "비중이 아니라 위험을 균등하게 나눈다는 발상입니다.")
            st.latex(r"RC_i = w_i\,\frac{(\Sigma w)_i}{\sqrt{w^{\top}\Sigma w}}"
                     r"\,,\qquad \sum_i RC_i = \sigma_p")
            st.markdown(
                "변동성이 큰 자산은 비중이 자동으로 낮아집니다. 예를 들어 주식 60 · 채권 40 "
                "구성은 금액 기준으로는 6:4이지만 위험 기여도 기준으로는 주식이 90% 이상을 "
                "차지합니다. 위험균형은 이러한 불균형을 조정합니다.\n\n"
                "**HRP (계층적 위험 분산)** — Lopez de Prado가 2016년 제시했으며 "
                "세 단계로 작동합니다.\n\n"
                "1. 상관계수를 거리로 변환해 종목을 **계층 군집화**\n"
                "2. 비슷하게 움직이는 종목이 이웃하도록 **순서 재배열**\n"
                "3. 트리를 따라 내려가며 **역분산으로 반씩 분할 배분**\n\n"
                "공분산 행렬의 역행렬을 구하지 않아 수치적으로 안정적이고 기대수익률도 "
                "쓰지 않습니다. 종목이 많고 상관구조가 뚜렷할 때 특히 유리합니다.")

            st.divider()
            st.subheader("7. 전망을 반영하는 방법 — 블랙-리터만")
            st.markdown(
                "5번에서 본 문제의 정공법 해법입니다. 과거 평균으로 기대수익률을 추정하는 "
                "대신, **시장이 암묵적으로 기대하는 수익률에서 출발**합니다.\n\n"
                "**1단계 · 역최적화** — 현재의 시장 비중이 최적이 되려면 각 자산이 얼마를 "
                "내야 하는지 거꾸로 계산합니다.")
            st.latex(r"\Pi = \delta\,\Sigma\,w_{\text{시장}}")
            st.markdown(
                "여기서 δ는 위험회피계수입니다. 이렇게 얻은 Π는 과거 평균과 달리 "
                "**추정 오차가 작습니다.** 시장 참여자 전체의 판단이 반영된 값이기 "
                "때문입니다.\n\n"
                "**2단계 · 전망 결합** — 내 뷰를 확신도만큼만 섞습니다.")
            st.latex(r"E[R] = \bigl[(\tau\Sigma)^{-1} + P^{\top}\Omega^{-1}P\bigr]^{-1}"
                     r"\bigl[(\tau\Sigma)^{-1}\Pi + P^{\top}\Omega^{-1}Q\bigr]")
            st.markdown(
                "- **P** — 어느 자산에 대한 뷰인지 나타내는 행렬\n"
                "- **Q** — 뷰의 값 (연 수익률 또는 우위 폭)\n"
                "- **Ω** — 뷰의 불확실성. 확신도가 낮을수록 큰 값을 넣어 영향을 줄입니다\n"
                "- **τ** — 균형 추정 자체의 불확실성\n\n"
                "**세 가지 성질**이 이 모형의 장점입니다.\n\n"
                "1. 뷰가 없으면 결과가 **시장 비중과 정확히 일치**합니다\n"
                "2. 확신도가 낮으면 시장 비중에서 조금만 움직입니다\n"
                "3. 뷰를 준 자산뿐 아니라 **상관관계로 연결된 다른 자산도 함께 조정**됩니다\n\n"
                "특히 3번이 중요합니다. '신흥국이 좋을 것 같다'는 뷰 하나를 넣으면, "
                "신흥국과 함께 움직이는 자산의 비중도 자연스럽게 따라 조정됩니다.")
            st.warning("**한계** — 시장 비중이 합리적인 출발점이라는 가정에 기댑니다. "
                       "시가총액 비중을 쓰기 어려운 자산군이라면 벤치마크나 현재 전략적 "
                       "배분을 대신 넣게 되는데, 그러면 '균형'의 의미가 약해집니다. "
                       "공분산도 여전히 과거에서 추정하므로, 상관 구조가 바뀌면 결과도 "
                       "달라집니다.")

            st.divider()
            st.subheader("8. 시장 국면과 상관관계의 변화")
            st.markdown(
                "3번에서 분산투자의 근거가 **상관계수**라고 했습니다. 그런데 상관계수는 "
                "고정된 값이 아닙니다.\n\n"
                "특히 시장이 급락할 때 대부분의 위험자산이 함께 움직이는 경향이 있습니다. "
                "평상시 0.3이던 상관이 위기 국면에 0.8까지 올라가면, "
                "**분산이 가장 필요한 순간에 효과가 줄어드는** 셈입니다.\n\n"
                "국면은 통용되는 정의를 따릅니다.\n\n"
                "- **약세장** — 고점 대비 −20% 하락. 시작일은 하락을 확인한 날이 아니라 "
                "그 **고점으로 소급**합니다\n"
                "- **강세장** — 저점 대비 +20% 반등. 시작일은 그 **저점으로 소급**합니다\n"
                "- **횡보** — 강세 구간 중 최근 수익률이 일정 범위 안에 머문 기간\n\n"
                "다만 이 구분은 **사후적**입니다. −20%에 도달해야 약세장이었음을 알 수 "
                "있으므로, 실시간으로 '지금이 약세장'이라고 확정할 수는 없습니다.")

            st.divider()
            st.subheader("9. 후보 종목 탐색과 데이터 스누핑")
            st.markdown(
                "'후보 30개 중 가장 좋은 조합'을 찾을 때도 5번과 같은 문제가 생깁니다. "
                "오히려 더 심해집니다. **후보가 많을수록 순전히 우연만으로도 "
                "매우 좋아 보이는 조합이 존재**하기 때문입니다. 이를 "
                "**데이터 스누핑(data snooping)** 이라고 합니다.\n\n"
                "이 도구는 세 가지로 대응합니다.\n\n"
                "1. **순위는 학습 구간으로만** 매깁니다. 표본외 성과로 순위를 매기면 "
                "검증의 의미가 사라집니다\n"
                "2. **1등만 강조하지 않고 상위 10개**를 함께 보여줍니다. 1등과 5등의 "
                "차이가 미미하다면 그 순위는 의미가 없습니다\n"
                "3. **채택 빈도**를 제시합니다. 여러 조합에 반복해서 등장하는 종목이 "
                "특정 조합의 우연이 아니라 실제로 유용한 후보일 가능성이 큽니다\n\n"
                "후보를 15~30개로 좁히는 것을 권하는 것도 같은 이유입니다.")

            st.divider()
            st.subheader("10. 검증 — 표본내와 표본외")
            st.markdown(
                "전체 기간을 모두 활용해 최적 비중을 구한 뒤 같은 기간의 성과를 제시하면 "
                "결과가 좋게 나오는 것이 당연합니다. 이미 답을 알고 있는 상태에서 답을 "
                "고른 것과 다르지 않기 때문입니다. 이러한 현상을 **과최적화**라고 합니다. "
                "이를 피하기 위해 기간을 둘로 나눕니다.")
            st.code("├────── 학습 기간 ──────┤────── 검증 기간 ──────┤\n"
                    "    여기서 비중을 계산          그 비중을 그대로 적용\n"
                    "                       ↑\n"
                    "                  최적화 기준일", language="text")
            st.markdown(
                "**검증 기간(표본외) 성과만이 의미 있는 성적표입니다.** 표본내에서 좋고 "
                "표본외에서 나쁘면 과최적화된 것이며, 이 도구는 그 결과를 숨기지 않습니다.\n\n"
                "**워크포워드(주기적 재최적화)** 는 한 걸음 더 나아갑니다. 분기마다 그 "
                "시점까지의 데이터로만 비중을 다시 계산해 이어붙이므로, *실제로 그렇게 "
                "운용했다면* 어땠을지를 재현합니다. 미래 정보가 단 한 번도 섞이지 않는 것이 "
                "핵심입니다.")

        # ========================== 실무 요소 ==========================
        with pr:
            st.subheader("리밸런싱")
            st.markdown(
                "가격이 움직이면 비중은 자연히 변합니다. 상승한 자산의 비중이 커집니다. "
                "리밸런싱은 이를 목표 비중으로 되돌리는 것입니다.\n\n"
                "| 방식 | 조정 시점 |\n|---|---|\n"
                "| 매일 (고정비중 유지) | 매 거래일. 비중이 전혀 흐르지 않음 |\n"
                "| 월별 / 분기별 / 연별 | 각 구간 첫 거래일에만 되돌림 |\n"
                "| 없음 | 첫날 사고 그대로 보유 |\n\n"
                "**선택에 따라 결과가 크게 달라집니다.** 매수 후 보유는 상승한 자산으로 비중이 "
                "집중되어 최대낙폭이 커지는 경향이 있습니다. 반대로 리밸런싱 주기가 짧을수록 "
                "회전율이 높아져 비용 부담이 늘어납니다. 실무에서는 분기별을 절충안으로 "
                "많이 사용합니다.\n\n"
                "리밸런싱은 본질적으로 **상승한 자산을 매도해 하락한 자산을 매수하는** 방식이므로, "
                "평균회귀 성향이 강한 시장에서는 수익에도 기여하지만 추세가 장기간 지속되는 "
                "시장에서는 불리하게 작용할 수 있습니다.")

            st.divider()
            st.subheader("거래비용")
            st.markdown(
                "**편도 bp**로 입력합니다. 1bp = 0.01%이며 매수·매도 각각에 적용됩니다. "
                "국내주식은 위탁수수료와 증권거래세를 합쳐 보통 20~30bp 수준입니다. "
                "비용은 **회전율에 비례**합니다.")
            st.latex(r"\text{비용}_t=\Bigl(\sum_i \bigl|w_i^{\text{목표}}"
                     r"-w_i^{\text{직전}}\bigr|\Bigr)\times c")
            st.markdown(
                "매수 후 보유는 회전율이 0이라 영향이 없지만, 매일 리밸런싱은 연 200%를 "
                "넘겨 성과가 눈에 띄게 깎입니다. **재최적화를 켤 때는 반드시 비용을 함께 "
                "입력하세요.** 비용 없이 보면 재최적화가 실제보다 유리해 보입니다.")

            st.divider()
            st.subheader("배당 처리")
            st.markdown(
                "**배당 재투자**는 받은 배당으로 같은 종목을 다시 산다고 가정합니다(총수익). "
                "**주가만**은 가격 변동만 봅니다.\n\n"
                "배당수익률 2.5% 종목을 10년 보유하면 두 방식의 최종 자산이 약 28% 차이 "
                "납니다. 장기 분석에서는 재투자 기준이 표준입니다. 두 경우 모두 액면분할은 "
                "보정되므로 분할일에 가짜 폭락이 생기지 않습니다.")

            st.divider()
            st.subheader("통화 환산과 환헤지")
            st.markdown(
                "국가가 다른 종목을 섞으면 기준 통화로 환산해야 비교가 됩니다. 이때 "
                "**환율 변동도 수익률에 포함**됩니다. 원화 기준으로 미국 주식을 보면 주가가 "
                "그대로여도 환율이 오르면 수익이 납니다.\n\n"
                "**환헤지 가정**을 켜면 환율 변동을 제거하고 종목 자체 수익률만 봅니다. "
                "실제 헤지에는 비용이 들지만 이 도구는 비용 없는 완전 헤지를 가정합니다.\n\n"
                "환율은 분산 효과에도 영향을 줍니다. 원화가 위기 때 약세를 보이면 달러 "
                "자산의 원화 환산 수익이 방어 역할을 하기도 합니다.")
            st.warning("엔화 환율은 **1엔당 원화**입니다. 뉴스의 '100엔당' 표기와 100배 "
                       "차이나므로 9~10 정도로 보여도 정상입니다.")

            st.divider()
            st.subheader("위험을 정의하는 여러 방법")
            st.markdown(
                "'위험'은 하나가 아닙니다. 무엇을 위험으로 보느냐에 따라 최적 비중이 "
                "달라집니다.\n\n"
                "| 그룹 | 지표 | 관점 |\n|---|---|---|\n"
                "| 변동성 기반 | 표준편차, 평균절대편차 | 얼마나 흔들리는가 |\n"
                "| 하방위험 | 준표준편차, 소르티노, 오메가 | 내려갈 때만 위험이다 |\n"
                "| 낙폭 기반 | 조건부 낙폭(CDaR), 얼서지수 | 얼마나 깊이·오래 하락하는가 |\n"
                "| 꼬리위험 | 조건부 손실(CVaR) | 최악일 때 얼마나 잃는가 |\n\n"
                "**표준편차는 급등과 급락을 구분하지 못합니다.** 상승이 잦은 자산과 하락이 "
                "잦은 자산의 표준편차는 거의 같게 나타나는 반면, CDaR로 측정하면 큰 폭의 "
                "차이가 나타납니다. 낙폭을 견디기 어려운 성향이라면 낙폭 기반 지표가 목적에 맞습니다.")

            st.divider()
            st.subheader("위기 구간 분석")
            st.markdown(
                "평균적으로 좋아 보이는 구성도 위기에는 전혀 다르게 움직입니다. "
                "전체 기간의 평균 지표만 보면 이런 취약점이 드러나지 않습니다.\n\n"
                "이 도구는 널리 쓰이는 위기 구간을 **S&P 500의 고점 → 저점** 기준으로 "
                "미리 넣어두었습니다. 시장 국면 분석이 −20% 규칙으로 국면을 자동 구분하는 "
                "것과 달리, 여기서는 **사건 단위로 날짜를 고정**해 서로 다른 포트폴리오를 "
                "같은 잣대로 비교할 수 있게 했습니다.\n\n"
                "**회복 기간**이 특히 중요합니다. 같은 −30% 하락이어도 6개월 만에 회복하는 "
                "것과 3년이 걸리는 것은 전혀 다른 경험입니다. 낙폭의 비대칭성(이론 2번)과 "
                "함께 보시면 이해가 분명해집니다.")

            st.divider()
            st.subheader("적립식과 내부수익률")
            st.markdown(
                "목돈을 한 번에 넣는 경우와 매달 나눠 넣는 경우는 결과가 다릅니다. "
                "적립식에서는 **나중에 넣은 돈일수록 투자 기간이 짧으므로** 단순 "
                "수익률(평가액÷납입원금−1)이 실제 성과를 과소평가합니다.\n\n"
                "**내부수익률(IRR)** 은 각 납입 시점을 고려해 '연 몇 %짜리 상품에 넣은 "
                "셈인가'를 계산합니다. 적립식 성과를 비교할 때는 IRR을 기준으로 삼는 것이 "
                "적절합니다.")

            st.divider()
            st.subheader("이 도구의 한계")
            st.markdown(
                "- **과거가 미래를 보장하지 않습니다.** 특히 상관관계는 위기에 급등하는 "
                "경향이 있어, 정작 분산이 필요한 순간에 효과가 사라지곤 합니다.\n"
                "- **기대수익률 추정 오차** — 이론 5번에서 설명한 구조적 한계입니다.\n"
                "- **세금 미반영** — 거래비용만 반영되며 양도소득세·배당소득세는 빠져 있습니다.\n"
                "- **생존편향** — 상장폐지된 종목은 데이터에 없어 과거 성과가 실제보다 좋아 "
                "보일 수 있습니다.\n"
                "- **유동성·체결 가정** — 종가에 원하는 수량이 체결된다고 가정합니다.\n"
                "- **데이터 출처는 야후 파이낸스**이며, 국내 종목의 배당 조정이 부정확한 "
                "경우가 있습니다.")
            st.error("이 도구는 **교육·참고용**이며 투자 자문이 아닙니다. "
                     "투자 판단과 그 결과는 이용자 본인의 책임입니다.")


# ======================================================================
# 자산 상관관계 화면
# ======================================================================
# ======================================================================
# 자산 추가 효과
# ======================================================================
# ======================================================================
# 시장 국면 분석
# ======================================================================
# ======================================================================
# 뷰 기반 자산배분 (블랙-리터만)
# ======================================================================
# ======================================================================
# 스트레스 테스트
# ======================================================================
# ---------------------- 가상 충격 ----------------------
# 자산 유형: 충격이 전이되는 방식이 유형마다 다르다
ASSET_KINDS = ["주식", "국채", "회사채", "하이일드", "금", "원자재", "현금", "기타·혼합"]

# 대표 ETF의 (유형, 듀레이션, 볼록성).
# 볼록성 단위는 연² 이며 ΔP ≈ −DΔy + ½C(Δy)² 에 그대로 들어간다.
ASSET_META = {
    # 국채
    "SHY": ("국채", 1.9, 5), "IEI": ("국채", 4.3, 25), "IEF": ("국채", 7.5, 70),
    "TLH": ("국채", 11.5, 150), "TLT": ("국채", 16.0, 250), "EDV": ("국채", 23.0, 600),
    "GOVT": ("국채", 6.0, 50),
    "TIP": ("국채", 6.8, 60), "SCHO": ("국채", 1.9, 5), "SPTL": ("국채", 15.5, 240),
    "148070.KS": ("국채", 7.0, 60), "152380.KS": ("국채", 4.5, 25),
    "114260.KS": ("국채", 2.8, 10), "304660.KS": ("국채", 17.0, 280),
    # 회사채·종합
    "LQD": ("회사채", 8.4, 120), "VCIT": ("회사채", 6.3, 55),
    "VCSH": ("회사채", 2.7, 10), "AGG": ("회사채", 6.0, 50),
    "BND": ("회사채", 6.0, 50), "MBB": ("회사채", 5.5, 40),
    "IGSB": ("회사채", 2.6, 10), "USIG": ("회사채", 7.0, 80),
    # 하이일드
    "HYG": ("하이일드", 3.5, 25), "JNK": ("하이일드", 3.4, 24),
    "SHYG": ("하이일드", 1.9, 6), "SJNK": ("하이일드", 2.0, 7),
    # 금·원자재
    "GLD": ("금", None, None), "IAU": ("금", None, None), "SGOL": ("금", None, None),
    "132030.KS": ("금", None, None),
    "SLV": ("원자재", None, None), "DBC": ("원자재", None, None),
    "GSG": ("원자재", None, None), "PDBC": ("원자재", None, None),
    "USO": ("원자재", None, None), "COMT": ("원자재", None, None),
    "DBA": ("원자재", None, None), "COPX": ("원자재", None, None),
    # 대표 주식형 ETF — quoteType 이 ETF 라도 유형이 분명하다
    "SPY": ("주식", None, None), "VOO": ("주식", None, None),
    "IVV": ("주식", None, None), "QQQ": ("주식", None, None),
    "VTI": ("주식", None, None), "DIA": ("주식", None, None),
    "IWM": ("주식", None, None), "EFA": ("주식", None, None),
    "EEM": ("주식", None, None), "VEA": ("주식", None, None),
    "VWO": ("주식", None, None), "ACWI": ("주식", None, None),
    "SCHD": ("주식", None, None), "VIG": ("주식", None, None),
    "VNQ": ("주식", None, None), "XLK": ("주식", None, None),
    "XLF": ("주식", None, None), "XLE": ("주식", None, None),
    "XLY": ("주식", None, None), "XLP": ("주식", None, None),
    "XLV": ("주식", None, None), "XLI": ("주식", None, None),
    "XLU": ("주식", None, None), "SOXX": ("주식", None, None),
    "SMH": ("주식", None, None), "SPMO": ("주식", None, None),
    "069500.KS": ("주식", None, None), "229200.KS": ("주식", None, None),
    "371460.KS": ("주식", None, None), "379800.KS": ("주식", None, None),
    "133690.KS": ("주식", None, None), "360750.KS": ("주식", None, None),
    # 현금성
    "SHV": ("현금", 0.3, 1), "BIL": ("현금", 0.2, 1),
}

# 유형별 신용 스프레드 민감도 (스프레드 듀레이션, 연)
# 2008·2020 실제 낙폭에서 역산했다.
SPREAD_DUR = {"하이일드": 2.6, "회사채": 2.2, "국채": 0.0,
              "주식": 0.0, "금": 0.0, "원자재": 0.0, "현금": 0.0, "기타·혼합": 0.0}

SHOCK_PRESETS = {
    "직접 입력": {},
    "주식 급락 (−20%)": {"eq": -20.0, "dy": -50.0, "cs": 150.0, "fx": 5.0,
                    "gold": 5.0, "cm": -10.0},
    "금리 급등 (+100bp)": {"eq": -8.0, "dy": 100.0, "cs": 50.0, "fx": 3.0,
                      "gold": -5.0, "cm": 0.0},
    "인플레 재점화": {"eq": -12.0, "dy": 150.0, "cs": 80.0, "fx": 2.0,
                "gold": 8.0, "cm": 15.0},
    "경기 침체": {"eq": -25.0, "dy": -100.0, "cs": 250.0, "fx": 8.0,
              "gold": 10.0, "cm": -20.0},
    "금융위기형 충격": {"eq": -35.0, "dy": -150.0, "cs": 400.0, "fx": 12.0,
                 "gold": 15.0, "cm": -25.0},
}

# 통화 → 금리 시장. 금리와 환율은 시장마다 따로 움직이므로 구분한다.
CCY_MARKET = {"USD": "미국", "KRW": "한국", "JPY": "일본", "EUR": "유럽",
              "GBP": "영국", "CNY": "중국", "HKD": "홍콩", "TWD": "대만"}


def asset_market(ticker: str, ccy: str) -> str:
    """자산이 어느 금리 시장에 속하는지. 통화를 기준으로 본다."""
    return CCY_MARKET.get(str(ccy).upper(), str(ccy).upper() or "기타")


def guess_asset_kind(ticker: str, name: str = "", quote_type: str = "") -> tuple:
    """
    티커·종목명으로 자산 유형을 추정한다.
    반환: (유형, 듀레이션, 볼록성, 자동판정 여부)
    확실하지 않으면 '기타·혼합'으로 두고 사용자가 직접 고르게 한다.
    """
    key = str(ticker).upper().strip()
    if key in ASSET_META:
        k, d, c = ASSET_META[key]
        return k, d, c, True
    nm = f"{ticker} {name}".upper()
    # 이름에서 유추 (확실한 신호만)
    if any(x in nm for x in ("TREASURY", "국고", "국채", "GOVT BOND")):
        return "국채", 5.0, 40, False
    if any(x in nm for x in ("HIGH YIELD", "하이일드", "HIGH-YIELD")):
        return "하이일드", 3.5, 25, False
    if any(x in nm for x in ("CORPORATE BOND", "회사채", "CREDIT BOND")):
        return "회사채", 6.0, 55, False
    if any(x in nm for x in ("GOLD", "골드", "금현물")):
        return "금", None, None, False
    if any(x in nm for x in ("COMMODITY", "원자재", "CRUDE", "OIL", "SILVER")):
        return "원자재", None, None, False
    if any(x in nm for x in ("BOND", "채권", "FIXED INCOME")):
        return "기타·혼합", 5.0, 40, False
    # 상품 구분이 있으면 그걸 우선한다.
    #   EQUITY  → 개별 주식이 확실하다
    #   ETF·펀드 → 안에 무엇이 들었는지 모르므로 기타·혼합으로 두고 확인받는다
    qt = str(quote_type or "").upper()
    if qt == "EQUITY":
        return "주식", None, None, True
    if qt in ("ETF", "MUTUALFUND", "FUND"):
        return "기타·혼합", None, None, False
    if qt in ("INDEX", "CURRENCY", "FUTURE", "CRYPTOCURRENCY"):
        return "기타·혼합", None, None, False

    # 상품 구분을 못 받았을 때만 티커 형식으로 추정한다.
    # 이름을 못 받아온 채권 ETF가 주식으로 처리되면 금리 충격이 통째로 빠진다.
    if re.fullmatch(r"[A-Z]{1,5}", key) or re.fullmatch(r"\d{6}\.(KS|KQ)", key):
        return "주식", None, None, False
    return "기타·혼합", None, None, False


def shock_asset(kind, dur, conv, beta, eq_bp, dy_bp, cs_bp, gold_pct, cm_pct,
                spread_dur=None):
    """
    자산 유형별로 충격을 적용한다. 반환은 요인별 변화율(%) 딕셔너리.
    유형마다 적용 요인이 다르며, 겹쳐 더하지 않는다.
      · 주식        → 시장 베타 × 주식 충격
      · 국채        → 듀레이션·볼록성 × 금리 충격
      · 회사채/하이일드 → 금리 효과 + 스프레드 효과 (별도 표시)
      · 금 / 원자재   → 해당 가격 충격
      · 현금        → 없음
      · 기타·혼합    → 사용자가 지정한 베타·듀레이션을 그대로 사용
    """
    out = {"주식": 0.0, "금리": 0.0, "신용": 0.0, "원자재": 0.0}
    dy = (dy_bp or 0.0) / 10000.0
    cs = (cs_bp or 0.0) / 10000.0

    if kind == "주식":
        out["주식"] = (beta or 0.0) * (eq_bp or 0.0)
    elif kind in ("국채", "회사채", "하이일드", "기타·혼합"):
        if dur:
            out["금리"] = (-dur * dy + 0.5 * (conv or 0.0) * dy ** 2) * 100
        sd = SPREAD_DUR.get(kind, 0.0) if spread_dur is None else float(spread_dur)
        if sd:
            out["신용"] = -sd * cs * 100
        # 기타·혼합은 주식 성격이 섞였을 수 있어 사용자가 준 베타를 그대로 쓴다
        if kind == "기타·혼합" and beta:
            out["주식"] = beta * (eq_bp or 0.0)
    elif kind == "금":
        out["원자재"] = gold_pct or 0.0
    elif kind == "원자재":
        out["원자재"] = cm_pct or 0.0
    elif kind == "현금":
        pass
    return out


def combine_shock(factors: dict, fx_pct: float) -> float:
    """
    요인별 효과를 합치고 환율을 곱셈으로 적용한다.
      기준통화 수익률 = (1 + 현지자산 충격) × (1 + 환율 충격) − 1
    단순 덧셈보다 정확하며, 충격이 클수록 차이가 커진다.
    """
    local = sum(factors.values()) / 100.0
    return ((1 + local) * (1 + (fx_pct or 0.0) / 100.0) - 1) * 100


def estimate_betas(R: pd.DataFrame, mkt: pd.Series) -> dict:
    """
    각 자산의 시장 베타. 종목마다 겹치는 날짜만 써서 따로 계산한다.
    한 종목의 데이터가 짧다고 다른 종목의 표본까지 줄어들면 안 된다.
    """
    out = {}
    for c in R.columns:
        df = pd.concat([R[c], mkt], axis=1).dropna()
        if len(df) < 10:
            out[c] = 0.0
            continue
        v = float(df.iloc[:, 1].var())
        out[c] = float(df.iloc[:, 0].cov(df.iloc[:, 1]) / v) if v > 1e-18 else 0.0
    return out


def beta_quality(R: pd.DataFrame, mkt: pd.Series) -> dict:
    """베타의 설명력(R²)과 표본 수. 신뢰도 판단에 쓴다."""
    out = {}
    for c in R.columns:
        df = pd.concat([R[c], mkt], axis=1).dropna()
        n_ = len(df)
        r2 = float(df.iloc[:, 0].corr(df.iloc[:, 1]) ** 2) if n_ > 5 else np.nan
        out[c] = {"R2": r2, "표본": n_}
    return out


STRESS_COLS = ["티커", "종목명", "비중(%)"]

# 널리 쓰이는 위기 구간 (S&P 500 고점 → 저점 기준)
CRISIS_PRESETS = {
    "닷컴 버블 붕괴 (2000~02)": ("2000-03-24", "2002-10-09"),
    "글로벌 금융위기 (2007~09)": ("2007-10-09", "2009-03-09"),
    "유럽 재정위기 (2011)": ("2011-04-29", "2011-10-03"),
    "중국 쇼크·유가 급락 (2015~16)": ("2015-05-21", "2016-02-11"),
    "2018년 4분기 조정": ("2018-09-20", "2018-12-24"),
    "코로나 폭락 (2020)": ("2020-02-19", "2020-03-23"),
    "긴축 발작 (2022)": ("2022-01-03", "2022-10-12"),
}


# ======================================================================
# 최종 대안 비교
# ======================================================================
def render_compare(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate,
                   cost_bp):
    st.title("⚖️ 최종 대안 비교")
    st.caption("여러 화면에서 만든 안을 **같은 조건으로 나란히** 비교합니다. "
               "각 화면 아래의 **비교 후보로 담기**로 안을 모아두세요.")

    plans = load_plans()
    if not plans:
        st.info("아직 담긴 안이 없습니다.\n\n"
                "**📊 포트폴리오 분석** · **🎯 최적화** · **➕ 자산 추가 효과** · "
                "**🧭 뷰 기반 자산배분** 화면에서 결과를 낸 뒤 "
                "**⚖️ 비교 후보로 담기** 를 누르면 여기 모입니다.")
        return

    # ---------------- 담긴 안 ----------------
    st.subheader("1️⃣ 담긴 안 (Candidates)")
    prows = []
    for nm, pl in plans.items():
        prows.append({"이름": nm, "출처": pl.get("from", "-"),
                      "종목 수": len(pl["rows"]),
                      "구성": " · ".join(f"{r['티커']} {r['비중']:.1f}%"
                                       for r in pl["rows"][:4])
                              + (" …" if len(pl["rows"]) > 4 else ""),
                      "담은 시각": pl.get("saved_at", "-")})
    st.dataframe(pd.DataFrame(prows), width="stretch", hide_index=True)

    d1, d2 = st.columns([2, 1])
    drop = d1.multiselect("뺄 안 고르기", list(plans), key="_cmp_drop")
    d2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if d2.button("🗑 선택 안 빼기", width="stretch", disabled=not drop):
        for k in drop:
            plans.pop(k, None)
        st.session_state[PLANS_KEY] = plans
        st.rerun()

    picked = st.multiselect("비교할 안", list(plans), default=list(plans),
                            key="_cmp_pick")
    if len(picked) < 1:
        st.warning("최소 1개를 선택해주세요.")
        return

    # ---------------- 조건 ----------------
    st.subheader("2️⃣ 비교 조건 (Settings)")
    st.caption("모든 안에 **같은 조건**을 적용해야 공정한 비교가 됩니다. "
               "각 안을 만들 때의 설정과 다를 수 있습니다.")
    e1, e2, e3 = st.columns(3)
    rebal_c = e1.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key="_cmp_rebal")
    bench_c = e2.text_input("벤치마크", value="^GSPC", key="_cmp_bench")
    crisis_c = e3.multiselect("위기 구간", list(CRISIS_PRESETS),
                              default=["글로벌 금융위기 (2007~09)", "코로나 폭락 (2020)",
                                       "긴축 발작 (2022)"], key="_cmp_crisis")

    run_c = st.button("⚖️ 비교 실행", type="primary", width="stretch")
    if run_c:
        st.session_state["_cmp_has_run"] = True
    if not st.session_state.get("_cmp_has_run"):
        st.info("👆 조건을 정한 뒤 **비교 실행**을 눌러주세요.")
        return

    # ---------------- 데이터 ----------------
    tks = sorted({r["티커"] for k in picked for r in plans[k]["rows"]})
    bench_n = normalize_ticker(bench_c)[0] if bench_c.strip() else None
    need = tks + ([bench_n] if bench_n else [])
    lo_c = (min(pd.Timestamp(CRISIS_PRESETS[k][0]) for k in crisis_c)
            if crisis_c else pd.Timestamp(start_date))
    fetch_start = min(pd.Timestamp(start_date), lo_c - pd.DateOffset(months=6))
    try:
        with st.spinner(f"{len(need)}개 종목 데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(need, fetch_start, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    if prices.empty:
        st.error("공통 거래일이 없습니다.")
        return

    main_px = prices.loc[pd.Timestamp(start_date):]
    if len(main_px) < 60:
        st.error("분석 기간이 너무 짧습니다.")
        return

    def _run(rows, px):
        W = {r["티커"]: float(r["비중"]) for r in rows if r["티커"] in px.columns}
        if not W or sum(W.values()) <= 0:
            return None, np.nan
        try:
            rr = portfolio_returns(px, W, rebal_c)
            tno = turnover_series(px, W, rebal_c).reindex(rr.index).fillna(0)
            return apply_cost(rr, tno, cost_bp), annual_turnover(tno)
        except Exception:
            return None, np.nan

    br = None
    if bench_n and bench_n in main_px.columns:
        try:
            br = portfolio_returns(main_px, {bench_n: 100.0}, REBAL_NONE)
        except Exception:
            br = None

    st.success(f"✅ 비교 완료 · {len(picked)}개 안 · "
               f"{main_px.index[0].date()} ~ {main_px.index[-1].date()} · "
               f"리밸런싱 {rebal_c} · 거래비용 {cost_bp:.0f}bp")

    # ---------------- 종합 비교표 ----------------
    st.subheader("3️⃣ 종합 비교 (Summary)")
    rows_m, curves = [], {}
    for nm in picked:
        rr, tno = _run(plans[nm]["rows"], main_px)
        if rr is None:
            continue
        curves[nm] = rr
        m_ = perf_row(rr, rf_rate)
        row = {"안": nm, **m_, "연 회전율(%)": tno * 100 if not pd.isna(tno) else np.nan}
        # 위험 집중도
        try:
            _t = [r["티커"] for r in plans[nm]["rows"] if r["티커"] in main_px.columns]
            _w = np.array([r["비중"] for r in plans[nm]["rows"]
                           if r["티커"] in main_px.columns], dtype=float)
            _w = _w / _w.sum()
            _cov = (main_px[_t].pct_change().dropna().cov() * TRADING_DAYS).values
            _rc = risk_contributions(_w, _cov)
            row["위험 집중도(%)"] = float(_rc.max() / _rc.sum() * 100) if _rc.sum() > 0 else np.nan
        except Exception:
            row["위험 집중도(%)"] = np.nan
        if br is not None:
            row.update({k: v for k, v in rel_metrics(rr, br).items()
                        if k in ("초과수익률(%p)", "추적오차(%)", "정보비율")})
        rows_m.append(row)

    if br is not None:
        bm_ = perf_row(br, rf_rate)
        rows_m.append({"안": f"벤치마크 ({bench_n})", **bm_,
                       "연 회전율(%)": 0.0, "위험 집중도(%)": np.nan})
        curves[f"벤치마크 ({bench_n})"] = br

    if not rows_m:
        st.error("계산 가능한 안이 없습니다.")
        return
    cmp_df = pd.DataFrame(rows_m).set_index("안")
    _hi = [c for c in ["수익률(연,%)", "최대낙폭(%)", "샤프지수", "소르티노",
                       "초과수익률(%p)", "정보비율"] if c in cmp_df.columns]
    _lo = [c for c in ["변동성(연,%)", "연 회전율(%)", "위험 집중도(%)", "추적오차(%)"]
           if c in cmp_df.columns]
    st.dataframe(cmp_df.style.format("{:.2f}", na_rep="-")
                 .highlight_max(subset=_hi, color="#dcfce7")
                 .highlight_min(subset=_lo, color="#dcfce7"),
                 width="stretch")
    st.caption("초록색 = 각 지표에서 가장 좋은 값. **최대낙폭은 음수**이므로 0에 가까울수록, "
               "**위험 집중도**는 한 종목이 전체 위험에서 차지하는 최대 비율이라 낮을수록 "
               "좋습니다.")

    # ---------------- 성과 곡선 ----------------
    st.subheader("4️⃣ 성과 곡선 (Growth)")
    fg = go.Figure()
    for nm, rr in curves.items():
        ec = equity_curve(rr)
        fg.add_trace(go.Scatter(x=ec.index, y=ec.values, name=nm,
                                line=dict(width=1.6 if nm.startswith("벤치마크") else 2.2,
                                          dash="dash" if nm.startswith("벤치마크") else None)))
    fg.update_layout(height=400, hovermode="x unified",
                     margin=dict(l=0, r=0, t=30, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fg, width="stretch")

    fd = go.Figure()
    for nm, rr in curves.items():
        dd = drawdown_series(rr) * 100
        fd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=nm,
                                line=dict(width=1.6 if nm.startswith("벤치마크") else 2)))
    fd.update_layout(height=300, hovermode="x unified", yaxis=dict(ticksuffix="%"),
                     margin=dict(l=0, r=0, t=20, b=0),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fd, width="stretch")
    st.caption("아래는 낙폭입니다. 위기 때 어느 안이 덜 빠졌는지 보세요.")

    # ---------------- 위기 손실 ----------------
    if crisis_c:
        st.subheader("5️⃣ 위기 구간 손실 (Crisis Loss)")
        st.caption("과거 위기마다 각 안이 얼마나 빠졌는지입니다. "
                   "평균 성과가 비슷해도 위기 대응은 크게 다를 수 있습니다.")
        crows = []
        for nm in picked:
            row = {"안": nm}
            for ck in crisis_c:
                a_, b_ = CRISIS_PRESETS[ck]
                seg = prices.loc[pd.Timestamp(a_):pd.Timestamp(b_)]
                rr, _ = _run(plans[nm]["rows"], seg)
                if rr is None or len(rr) < 3:
                    row[ck] = np.nan
                    continue
                eq = (1 + rr).cumprod()
                row[ck] = (float(eq.min()) - 1) * 100
            crows.append(row)
        if br is not None:
            row = {"안": f"벤치마크 ({bench_n})"}
            for ck in crisis_c:
                a_, b_ = CRISIS_PRESETS[ck]
                seg = prices.loc[pd.Timestamp(a_):pd.Timestamp(b_)]
                if bench_n in seg.columns and len(seg) > 3:
                    rb_ = portfolio_returns(seg, {bench_n: 100.0}, REBAL_NONE)
                    eq = (1 + rb_).cumprod()
                    row[ck] = (float(eq.min()) - 1) * 100
                else:
                    row[ck] = np.nan
            crows.append(row)
        cr_df = pd.DataFrame(crows).set_index("안")
        st.dataframe(cr_df.style.format("{:.2f}", na_rep="-").map(_heat_color),
                     width="stretch")
        st.caption("구간 시작일 대비 최저 시점의 수익률입니다. 0에 가까울수록 잘 버틴 것입니다.")

    # ---------------- 비중 비교 ----------------
    st.subheader("6️⃣ 비중 비교 (Allocation)")
    wide = {}
    for nm in picked:
        wide[nm] = {r["티커"]: r["비중"] for r in plans[nm]["rows"]}
    wdf = pd.DataFrame(wide).fillna(0.0)
    wdf.index.name = "티커"
    fw = go.Figure()
    for nm in picked:
        fw.add_trace(go.Bar(name=nm, x=wdf.index, y=wdf[nm]))
    fw.update_layout(barmode="group", height=340,
                     margin=dict(l=0, r=0, t=30, b=0),
                     yaxis=dict(ticksuffix="%"),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fw, width="stretch")
    st.dataframe(wdf.style.format("{:.2f}"), width="stretch")

    # ---------------- 요약 ----------------
    st.subheader("📝 요약 (Summary)")
    only = cmp_df[~cmp_df.index.str.startswith("벤치마크")]
    if len(only):
        lines = []
        b_shp = only["샤프지수"].idxmax()
        b_mdd = only["최대낙폭(%)"].idxmax()
        b_ret = only["수익률(연,%)"].idxmax()
        lines.append(f"**샤프지수**가 가장 높은 안은 **{b_shp}** "
                     f"({only.loc[b_shp, '샤프지수']:.2f})입니다.")
        lines.append(f"**낙폭이 가장 얕은** 안은 **{b_mdd}** "
                     f"({only.loc[b_mdd, '최대낙폭(%)']:.2f}%), "
                     f"**수익률이 가장 높은** 안은 **{b_ret}** "
                     f"({only.loc[b_ret, '수익률(연,%)']:.2f}%)입니다.")
        if b_shp != b_ret:
            lines.append("수익률 1위와 위험조정 1위가 다릅니다. "
                         "감내할 수 있는 변동성 수준에 따라 선택이 갈립니다.")
        if "연 회전율(%)" in only.columns and only["연 회전율(%)"].notna().any():
            hi_t = only["연 회전율(%)"].idxmax()
            lines.append(f"회전율이 가장 높은 안은 **{hi_t}** "
                         f"({only.loc[hi_t, '연 회전율(%)']:.0f}%)로 거래비용 부담이 큽니다.")
        st.markdown("\n\n".join(lines))

    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            cmp_df.to_excel(xw, sheet_name="1_종합비교")
            wdf.to_excel(xw, sheet_name="2_비중")
            if crisis_c:
                cr_df.to_excel(xw, sheet_name="3_위기손실")
            pd.DataFrame({k: equity_curve(v) for k, v in curves.items()}).to_excel(
                xw, sheet_name="4_성과추이")
            pd.DataFrame(list({
                "비교 기간": f"{main_px.index[0].date()} ~ {main_px.index[-1].date()}",
                "리밸런싱": rebal_c, "기준 통화": base_ccy,
                "거래비용": f"{cost_bp:.0f}bp", "벤치마크": bench_n or "-",
                "배당 처리": "재투자" if use_div else "주가만",
                "생성 시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            }.items()), columns=["항목", "값"]).to_excel(xw, sheet_name="5_설정",
                                                       index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"compare_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_compare", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")
    assumptions_panel(rebalance=rebal_c, cost=cost_bp,
                      period=f"{main_px.index[0].date()} ~ {main_px.index[-1].date()}",
                      bench=bench_n or "-")
    st.caption("모든 안을 같은 조건(리밸런싱·거래비용·기간)으로 다시 계산한 결과입니다. "
               "각 안을 만들 때의 설정과 다를 수 있습니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")


def _stress_blank():
    return {"티커": "", "종목명": "", "비중(%)": np.nan}


def crisis_stats(r: pd.Series, s, e) -> dict:
    """
    위기 구간 성과와 회복 기간.
      · 저점은 반드시 위기 구간 [s, e] 안에서 찾는다
      · 회복은 그 저점 이후, 위기 직전 수준을 되찾은 첫날로 본다
    """
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    seg = r.loc[s:e]
    if len(seg) < 3:
        return None
    eq = (1 + seg).cumprod()
    trough_d = eq.idxmin()
    out = {
        "구간 수익률(%)": (float(eq.iloc[-1]) - 1) * 100,
        "구간내 최저(%)": (float(eq.min()) - 1) * 100,
        "최대낙폭(%)": float((eq / eq.cummax() - 1).min()) * 100,
        "거래일": int(len(seg)),
        "기간(개월)": round((e - s).days / 30.44, 1),
        "저점일": trough_d.date(),
    }
    after = r.loc[s:]
    eq_a = (1 + after).cumprod()
    post = eq_a[eq_a.index > trough_d]
    rec = post[post >= 1.0]
    if len(rec):
        d = rec.index[0]
        out["회복일"] = d.date()
        out["회복까지(개월)"] = round((d - trough_d).days / 30.44, 1)
    else:
        out["회복일"] = None
        out["회복까지(개월)"] = np.nan
    return out


def render_hypo_stress(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill):
    """가정한 충격이 포트폴리오에 미치는 영향을 자산 유형별로 추정한다."""
    st.caption("과거에 없던 충격을 **가정해서** 넣어봅니다. 자산 유형마다 충격이 다르게 "
               "전이됩니다 — 주식은 베타, 채권은 듀레이션, 회사채는 신용스프레드까지요.")

    key = "_hypo_df"
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame([
            {"티커": t, "종목명": "", "비중(%)": w}
            for t, w in [("SPY", 50.0), ("TLT", 25.0), ("GLD", 15.0), ("HYG", 10.0)]
        ])[STRESS_COLS]
    st.session_state.setdefault("_hypo_gen", 0)

    st.subheader("1️⃣ 포트폴리오")
    live, tickers, bad = render_ticker_table(
        state_key=key, editor_key="_hypo_editor", gen_key="_hypo_gen",
        cols=STRESS_COLS, weight_col="비중(%)", title="가상 충격",
        min_rows=1, max_rows=30, show_equal=True, equal_key="_hypo_eq")
    ok_in, msgs = validate_setup(tickers, bad, live["비중(%)"], min_n=1,
                                 need_weight=True)
    show_msgs(msgs)
    if not ok_in:
        return

    # ---------------- 베타 추정 설정 ----------------
    st.subheader("2️⃣ 민감도 추정 (Sensitivity)")
    b1, b2, b3 = st.columns(3)
    beta_yrs = b1.selectbox("베타 추정 기간", [1, 3, 5], index=1, key="_hypo_byrs",
                            format_func=lambda x: f"{x}년")
    beta_freq = b2.selectbox("계산 주기", ["주별", "일별"], index=0, key="_hypo_bfreq",
                             help="거래 시간대가 다른 국가를 섞으면 일별 베타가 낮게 "
                                  "추정됩니다. 국가를 섞었다면 주별을 권합니다.")
    bench_h = b3.text_input("베타 기준 지수", value="^GSPC", key="_hypo_bench")

    # 종목 구성이 바뀌면 다시 확인해야 하지만, 베타 설정만 바꾼 경우에는
    # 화면이 스캔 이전으로 되돌아가지 않도록 상태를 유지한다.
    _scan_sig = tuple(tickers)
    if st.session_state.get("_hypo_scan_sig") != _scan_sig:
        st.session_state["_hypo_scan_done"] = False
        st.session_state["_hypo_scan_sig"] = _scan_sig
    if st.button("🔎 자산 유형·민감도 확인", width="stretch", key="_hypo_scan"):
        st.session_state["_hypo_scan_done"] = True
    if not st.session_state.get("_hypo_scan_done"):
        st.info("👆 먼저 **자산 유형·민감도 확인**을 눌러 각 종목의 성격을 확인하세요.")
        return

    bench_n = normalize_ticker(bench_h)[0] if bench_h.strip() else "^GSPC"
    need = list(dict.fromkeys(tickers + [bench_n]))
    _bstart = pd.Timestamp(end_date) - pd.DateOffset(years=int(beta_yrs))
    try:
        with st.spinner("민감도 추정용 데이터 조회 중..."):
            # 베타는 현지통화 기준으로 추정한다. 기준통화로 환산하면 환율이
            # 이미 섞여 들어가, 뒤에서 환율 충격을 또 더할 때 이중 반영된다.
            px, meta, _f = build_price_frame(need, _bstart, end_date,
                                             base_ccy, use_div,
                                             fx_hedge=True, gap_fill=gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    use_tk = [t for t in tickers if t in px.columns]
    if not use_tk or bench_n not in px.columns:
        st.error("종목 또는 기준 지수 데이터를 가져오지 못했습니다.")
        return

    _p = px if beta_freq == "일별" else px.resample("W-FRI").last()
    # 종목별로 결측을 떨어내야 한 종목의 데이터 부족이 나머지 표본을 깎지 않는다.
    # 시장수익률을 fillna(0) 하면 휴장일을 '0% 수익'으로 보아 베타가 낮아진다.
    R = _p[use_tk].pct_change()
    mkt = _p[bench_n].pct_change()
    betas = estimate_betas(R, mkt)
    bq = beta_quality(R, mkt)

    # ---------------- 자산 속성표 ----------------
    st.markdown("**자산별 스트레스 속성** — 자동 판정 결과를 확인하고 필요하면 고치세요")
    akey = "_hypo_attr"
    guess = []
    for t_ in use_tk:
        _mt = meta.get(t_, {})
        k_, d_, c_, auto = guess_asset_kind(t_, _mt.get("name", ""),
                                            _mt.get("quote_type", ""))
        q = bq.get(t_, {})
        guess.append({
            "티커": t_, "유형": k_,
            "주식 베타": round(betas.get(t_, 0.0), 3) if k_ in ("주식", "기타·혼합") else 0.0,
            "듀레이션": float(d_) if d_ else 0.0,
            "볼록성": float(c_) if c_ else 0.0,
            "통화": meta.get(t_, {}).get("currency", base_ccy),
            "R²": round(q.get("R2", np.nan), 2) if q else np.nan,
            "자동판정": "✅" if auto else "⚠️ 확인 필요",
        })
    gdf = pd.DataFrame(guess)
    # 티커뿐 아니라 베타 추정 조건이 바뀌어도 다시 계산해야 한다.
    # 그러지 않으면 화면에는 '5년·일별'인데 값은 이전 '3년·주별'이 남는다.
    _attr_sig = (tuple(use_tk), int(beta_yrs), beta_freq, bench_n,
                 str(end_date), bool(use_div))
    if (akey not in st.session_state or
            st.session_state.get("_hypo_attr_sig") != _attr_sig):
        st.session_state[akey] = gdf
        st.session_state["_hypo_attr_sig"] = _attr_sig
    elif st.button("🔄 추정값 다시 적용", key="_hypo_reattr",
                   help="자동 추정한 베타·듀레이션으로 되돌립니다. "
                        "직접 고친 값은 사라집니다."):
        st.session_state[akey] = gdf
        st.rerun()

    attr = st.data_editor(
        st.session_state[akey], num_rows="fixed", width="stretch", key="_hypo_attr_ed",
        hide_index=True,
        column_config={
            "티커": st.column_config.TextColumn("티커", disabled=True, width="small"),
            "유형": st.column_config.SelectboxColumn("유형", options=ASSET_KINDS,
                                                  width="medium"),
            "주식 베타": st.column_config.NumberColumn("주식 베타", step=0.05,
                                                  format="%.3f", width="small"),
            "듀레이션": st.column_config.NumberColumn("듀레이션", min_value=0.0,
                                                 max_value=30.0, step=0.5,
                                                 format="%.1f", width="small"),
            "볼록성": st.column_config.NumberColumn("볼록성", min_value=0.0,
                                                max_value=800.0, step=10.0,
                                                format="%.0f", width="small"),
            "통화": st.column_config.TextColumn("통화", disabled=True, width="small"),
            "R²": st.column_config.NumberColumn("R²", disabled=True, format="%.2f",
                                                width="small"),
            "자동판정": st.column_config.SelectboxColumn(
                "확인 상태", options=["✅", "⚠️ 확인 필요", "👤 확인함"],
                width="medium",
                help="자동 판정이 맞으면 그대로, 직접 고쳤으면 '👤 확인함'으로 "
                     "바꿔주세요."),
        })
    st.session_state[akey] = attr
    A = attr.set_index("티커")

    _need_check = attr[attr["자동판정"].astype(str).str.contains("확인")]
    if len(_need_check):
        st.warning(f"**{', '.join(_need_check['티커'])}** 는 목록에 없어 이름으로 "
                   f"추정했습니다. 유형·듀레이션이 맞는지 확인해주세요. "
                   f"틀리면 결과가 크게 달라집니다.")
    _lowr2 = attr[(attr["유형"] == "주식") & (attr["R²"] < 0.3)]
    if len(_lowr2):
        st.caption(f"⚠️ {', '.join(_lowr2['티커'])} 는 기준 지수로 설명되는 비율(R²)이 "
                   f"낮아 베타 신뢰도가 떨어집니다.")

    # ---------------- 충격 ----------------
    st.subheader("3️⃣ 충격 설정 (Shock)")
    preset = st.selectbox("시나리오", list(SHOCK_PRESETS), key="_hypo_preset")
    pre = SHOCK_PRESETS.get(preset, {})
    s1, s2, s3 = st.columns(3)
    eq_shock = s1.number_input("주식 시장 (%)", -60.0, 30.0,
                               float(pre.get("eq", -20.0)), step=1.0,
                               key=f"_hypo_eq_{preset}")
    dy_shock = s2.number_input("금리 (bp)", -300.0, 400.0,
                               float(pre.get("dy", 0.0)), step=10.0,
                               key=f"_hypo_dy_{preset}",
                               help="국채 금리 변화. 듀레이션·볼록성만큼 반영됩니다.")
    cs_shock = s3.number_input("신용 스프레드 (bp)", -100.0, 600.0,
                               float(pre.get("cs", 0.0)), step=10.0,
                               key=f"_hypo_cs_{preset}",
                               help="회사채·하이일드에만 추가로 적용됩니다.")
    s4, s5, s6 = st.columns(3)
    fx_shock = s4.number_input(f"환율 ({base_ccy} 절하, %)", -20.0, 30.0,
                               float(pre.get("fx", 0.0)), step=1.0,
                               key=f"_hypo_fx_{preset}",
                               help=f"{base_ccy} 외 통화 자산에 곱셈으로 적용됩니다. "
                                    f"아래에서 통화별로 다르게 줄 수도 있습니다.")
    gold_shock = s5.number_input("금 (%)", -40.0, 50.0,
                                 float(pre.get("gold", 0.0)), step=1.0,
                                 key=f"_hypo_gold_{preset}",
                                 help="금은 위기에 오르는 경우가 많아 원자재와 "
                                      "분리했습니다.")
    cm_shock = s6.number_input("원자재 (%)", -50.0, 50.0,
                               float(pre.get("cm", 0.0)), step=1.0,
                               key=f"_hypo_cm_{preset}")

    # ---------------- 시장·통화별 세부 조정 ----------------
    _ccys = sorted({str(A.loc[t_, "통화"]).upper() for t_ in use_tk})
    _mkts = sorted({asset_market(t_, A.loc[t_, "통화"]) for t_ in use_tk
                    if str(A.loc[t_, "유형"]) in ("국채", "회사채", "하이일드",
                                                "기타·혼합")})
    dy_by_mkt, fx_by_ccy = {}, {}
    if len(_mkts) > 1 or len([c for c in _ccys if c != base_ccy]) > 1:
        with st.expander("🌍 시장·통화별로 다르게 주기", expanded=False):
            st.caption("금리와 환율은 나라마다 따로 움직입니다. 비워두면 위에서 정한 "
                       "값을 모든 시장·통화에 동일하게 적용합니다.")
            if len(_mkts) > 1:
                st.markdown("**금리 (bp)**")
                _cols = st.columns(len(_mkts))
                for c_, mk in zip(_cols, _mkts):
                    dy_by_mkt[mk] = c_.number_input(
                        f"{mk} 금리", -300.0, 400.0, float(dy_shock), step=10.0,
                        key=f"_hypo_dy_{mk}_{preset}")
            _fcys = [c for c in _ccys if c != base_ccy]
            if len(_fcys) > 1:
                st.markdown(f"**환율 ({base_ccy} 대비 절하, %)**")
                _cols2 = st.columns(len(_fcys))
                for c_, cy in zip(_cols2, _fcys):
                    fx_by_ccy[cy] = c_.number_input(
                        f"{cy}/{base_ccy}", -20.0, 30.0, float(fx_shock), step=1.0,
                        key=f"_hypo_fx_{cy}_{preset}")

    # 충격 설정이 바뀌면 결과를 무효화하고, 그 외 재실행에서는 결과를 유지한다.
    # (버튼은 누른 그 실행에서만 True 라, 아래 위젯을 건드리면 결과가 사라진다)
    _sig = (preset, eq_shock, dy_shock, cs_shock, fx_shock, gold_shock, cm_shock,
            tuple(sorted(dy_by_mkt.items())), tuple(sorted(fx_by_ccy.items())),
            tuple(use_tk), tuple(np.round(live["비중(%)"].fillna(0).values, 4)),
            tuple(map(str, attr.values.ravel())),
            tuple(sorted(st.session_state.get("_hypo_sd", {}).items())))
    if st.session_state.get("_hypo_sig") != _sig:
        st.session_state["_hypo_run"] = False
        st.session_state["_hypo_sig"] = _sig
    if st.button("⚡ 충격 적용", type="primary", width="stretch", key="_hypo_go"):
        st.session_state["_hypo_run"] = True
    if not st.session_state.get("_hypo_run"):
        st.info("👆 충격을 정한 뒤 **충격 적용**을 눌러주세요.")
        return

    # ---------------- 계산 ----------------
    w = live.set_index("티커")["비중(%)"].reindex(use_tk).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=use_tk)
    w = w / w.sum() * 100

    def run_shock(eq_, dy_, cs_, fx_, gold_, cm_):
        rows, total = [], 0.0
        for t_ in use_tk:
            a = A.loc[t_]
            kind = str(a["유형"])
            ccy = str(a["통화"]).upper()
            _sd = st.session_state.get("_hypo_sd", {}).get(t_)
            # 시장·통화별로 다르게 지정했으면 그 값을 쓴다
            _mk = asset_market(t_, ccy)
            _dy = dy_by_mkt.get(_mk, dy_) if dy_by_mkt else dy_
            f = shock_asset(kind, float(a["듀레이션"]) or None,
                            float(a["볼록성"]) or None, float(a["주식 베타"]),
                            eq_, _dy, cs_, gold_, cm_, spread_dur=_sd)
            _fx = fx_by_ccy.get(ccy, fx_) if fx_by_ccy else fx_
            fx_e = _fx if (not fx_hedge and ccy != base_ccy) else 0.0
            tot_e = combine_shock(f, fx_e)
            contrib = tot_e * float(w[t_]) / 100
            total += contrib
            rows.append({"티커": t_, "유형": kind, "시장": _mk,
                         "비중(%)": float(w[t_]),
                         "주식효과(%)": f["주식"], "금리효과(%)": f["금리"],
                         "신용효과(%)": f["신용"], "원자재효과(%)": f["원자재"],
                         "환율효과(%)": fx_e, "합계(%)": tot_e,
                         "기여(%p)": contrib})
        return pd.DataFrame(rows), total

    idf, total = run_shock(eq_shock, dy_shock, cs_shock, fx_shock,
                           gold_shock, cm_shock)
    idf = idf.sort_values("기여(%p)")

    st.subheader("4️⃣ 자산별 영향 (Impact by Asset)")
    st.dataframe(idf.style.format({"비중(%)": "{:.2f}", "주식효과(%)": "{:+.2f}",
                                   "금리효과(%)": "{:+.2f}", "신용효과(%)": "{:+.2f}",
                                   "원자재효과(%)": "{:+.2f}", "환율효과(%)": "{:+.2f}",
                                   "합계(%)": "{:+.2f}", "기여(%p)": "{:+.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("**환율효과**는 곱셈으로 적용됩니다 — 기준통화 수익률 = "
               "(1 + 자산 충격) × (1 + 환율 충격) − 1. 따라서 합계는 각 열의 "
               "단순 합과 조금 다를 수 있습니다.")

    # ---------------- 결과 ----------------
    st.subheader("5️⃣ 포트폴리오 영향")
    k1, k2, k3 = st.columns(3)
    k1.metric("예상 손익", f"{total:+.2f}%")
    _w = idf.iloc[0]
    k2.metric("가장 큰 손실 기여", _w["티커"], delta=f"{_w['기여(%p)']:+.2f}%p")
    _pos = idf[idf["기여(%p)"] > 0]
    if len(_pos):
        _b = _pos.iloc[-1]
        k3.metric("가장 큰 완충 기여", _b["티커"], delta=f"{_b['기여(%p)']:+.2f}%p")
    else:
        _d = idf.loc[idf["합계(%)"].idxmax()]
        k3.metric("가장 방어적인 자산", _d["티커"], delta=f"{_d['합계(%)']:+.2f}%")

    fb = go.Figure(go.Bar(
        x=idf["기여(%p)"], y=idf["티커"], orientation="h",
        marker_color=["#dc2626" if v < 0 else "#0d9488" for v in idf["기여(%p)"]],
        text=[f"{v:+.2f}%p" for v in idf["기여(%p)"]], textposition="outside"))
    fb.update_layout(height=80 + 40 * len(idf), margin=dict(l=0, r=60, t=30, b=0),
                     xaxis=dict(title="포트폴리오 손익 기여 (%p)"),
                     yaxis=dict(autorange="reversed"))
    st.plotly_chart(fb, width="stretch")

    _by = {k: float((idf[f"{k}효과(%)"] * idf["비중(%)"] / 100).sum())
           for k in ("주식", "금리", "신용", "원자재", "환율")}
    # 환율은 곱셈으로 적용되므로 단순 가중합과 실제 합계가 어긋난다.
    # 그 차이를 '상호작용'으로 명시해 합계가 정확히 맞도록 한다.
    _by["상호작용"] = total - sum(_by.values())
    st.markdown(f"가정한 충격에서 포트폴리오는 **{total:+.2f}%** 움직일 것으로 "
                f"추정됩니다.")
    st.dataframe(pd.DataFrame([_by]).T.rename(columns={0: "기여(%p)"})
                 .style.format("{:+.2f}"), width="stretch")
    st.caption(f"합계 **{sum(_by.values()):+.2f}%p** = 예상 손익. "
               f"**상호작용**은 환율이 곱셈으로 적용되면서 생기는 차이입니다 — "
               f"자산이 크게 빠진 상태에서 환율이 움직이면 그 효과가 줄거나 커집니다.")

    # 스프레드 듀레이션은 자산별로 다르다. 유형 기본값을 쓰되 직접 고칠 수 있게 한다.
    with st.expander("신용 스프레드 민감도 조정", expanded=False):
        st.caption("종합채권(AGG·BND)이나 MBS는 순수 회사채보다 신용 민감도가 낮습니다. "
                   "유형별 기본값이 맞지 않으면 여기서 고치세요. "
                   "단위는 **스프레드 듀레이션(연)** 입니다.")
        _sd_key = "_hypo_sd"
        _sd_rows = [{"티커": t_, "유형": str(A.loc[t_, "유형"]),
                     "스프레드 듀레이션": float(
                         st.session_state.get(_sd_key, {}).get(
                             t_, SPREAD_DUR.get(str(A.loc[t_, "유형"]), 0.0)))}
                    for t_ in use_tk]
        _sd_df = st.data_editor(
            pd.DataFrame(_sd_rows), num_rows="fixed", width="stretch",
            key="_hypo_sd_ed", hide_index=True,
            column_config={
                "티커": st.column_config.TextColumn("티커", disabled=True),
                "유형": st.column_config.TextColumn("유형", disabled=True),
                "스프레드 듀레이션": st.column_config.NumberColumn(
                    "스프레드 듀레이션", min_value=0.0, max_value=15.0, step=0.1,
                    format="%.1f")})
        st.session_state[_sd_key] = dict(zip(_sd_df["티커"],
                                             _sd_df["스프레드 듀레이션"]))
    _big = sorted(_by.items(), key=lambda x: x[1])
    if _big and _big[0][1] < -0.01:
        st.caption(f"손실의 가장 큰 원인은 **{_big[0][0]}** 요인입니다 "
                   f"({_big[0][1]:+.2f}%p).")

    # ---------------- 커버리지 ----------------
    _wmap = {t_: float(w[t_]) for t_ in use_tk}
    _auto_w = sum(_wmap.get(r["티커"], 0) for _, r in attr.iterrows()
                  if str(r["자동판정"]) == "✅")
    _chk_w = sum(_wmap.get(r["티커"], 0) for _, r in attr.iterrows()
                 if str(r["자동판정"]).startswith("👤"))
    _man_w = 100.0 - _auto_w - _chk_w
    c1, c2, c3 = st.columns(3)
    c1.metric("자동 판정", f"{_auto_w:.1f}%")
    c2.metric("사용자 확인", f"{_chk_w:.1f}%")
    c3.metric("미확인", f"{_man_w:.1f}%")
    st.caption("**종목 수가 아니라 비중 기준**입니다. 미확인 자산이 5%인지 70%인지가 "
               "결과 신뢰도를 좌우하기 때문입니다.")
    if _man_w > 20:
        st.warning(f"유형이 확인되지 않은 자산이 **{_man_w:.1f}%** 입니다. "
                   f"위 속성표에서 유형·듀레이션을 확인해주세요. "
                   f"확인 후 '자동판정' 열을 👤 로 바꾸면 이 경고가 사라집니다.")

    # ---------------- 민감도 ----------------
    st.subheader("6️⃣ 민감도 (Sensitivity)")
    st.caption("**선택한 요인만** 바꾸고 나머지 충격은 위 설정값으로 고정합니다. "
               "따라서 현재 설정 지점의 값은 위 결과와 정확히 일치합니다.")
    axis = st.radio("변화시킬 요인", ["주식", "금리", "신용 스프레드", "환율"],
                    horizontal=True, key="_hypo_axis")
    rng = {"주식": np.arange(-60, 31, 5), "금리": np.arange(-300, 401, 50),
           "신용 스프레드": np.arange(-100, 601, 50),
           "환율": np.arange(-20, 31, 5)}[axis]
    ys = []
    for g in rng:
        _, tt = run_shock(g if axis == "주식" else eq_shock,
                          g if axis == "금리" else dy_shock,
                          g if axis == "신용 스프레드" else cs_shock,
                          g if axis == "환율" else fx_shock,
                          gold_shock, cm_shock)
        ys.append(tt)
    cur = {"주식": eq_shock, "금리": dy_shock,
           "신용 스프레드": cs_shock, "환율": fx_shock}[axis]
    fs = go.Figure(go.Scatter(x=rng, y=ys, mode="lines",
                              line=dict(color="#2563eb", width=2), name="손익"))
    fs.add_trace(go.Scatter(x=[cur], y=[total], mode="markers+text",
                            text=["현재"], textposition="top center",
                            marker=dict(size=13, color="#dc2626"), name="현재 설정"))
    fs.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fs.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
                     xaxis=dict(title=f"{axis} 충격 "
                                      f"({'bp' if axis in ('금리', '신용 스프레드') else '%'})"),
                     yaxis=dict(title="포트폴리오 손익 (%)", ticksuffix="%"))
    st.plotly_chart(fs, width="stretch")

    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            idf.to_excel(xw, sheet_name="1_자산별영향", index=False)
            attr.to_excel(xw, sheet_name="2_자산속성", index=False)
            pd.DataFrame([_by]).T.rename(columns={0: "기여(%p)"}).to_excel(
                xw, sheet_name="3_요인별")
            pd.DataFrame({f"{axis} 충격": rng, "손익(%)": ys}).to_excel(
                xw, sheet_name="4_민감도", index=False)
            pd.DataFrame(list({
                "시나리오": preset, "주식": f"{eq_shock:+.0f}%",
                "금리": f"{dy_shock:+.0f}bp", "신용 스프레드": f"{cs_shock:+.0f}bp",
                "환율": f"{fx_shock:+.0f}%", "금": f"{gold_shock:+.0f}%",
                "원자재": f"{cm_shock:+.0f}%", "베타 기준": bench_n,
                "베타 추정": f"{beta_yrs}년 · {beta_freq}",
                "예상 손익": f"{total:+.2f}%",
            }.items()), columns=["항목", "값"]).to_excel(xw, sheet_name="5_설정",
                                                       index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"hypo_stress_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_hypo", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")

    with st.expander("이 계산의 가정", expanded=False):
        st.markdown(
            f"- **베타** — {bench_n} 기준 · 최근 {beta_yrs}년 · {beta_freq} 수익률. "
            f"환율이 섞이지 않도록 **현지통화 기준**으로 추정합니다\\n"
            f"- **채권** — 듀레이션·볼록성 근사 `ΔP ≈ −DΔy + ½C(Δy)²`. "
            f"볼록성 단위는 연²이며 표에서 직접 고칠 수 있습니다\\n"
            f"- **신용 스프레드** — 회사채·하이일드에만 스프레드 듀레이션만큼 적용\\n"
            f"- **환율** — 곱셈 적용 `(1+자산)(1+환율)−1`. "
            f"{'사이드바에서 환율효과를 제외해 적용되지 않습니다' if fx_hedge else '기준통화 외 자산에만 적용됩니다'}\\n"
            f"- **금과 원자재** — 위기에 다르게 움직이므로 충격을 분리했습니다\\n"
            f"- 하나의 금리·환율 충격을 모든 통화·시장에 동일하게 적용합니다")

    st.warning("**선형 근사입니다.** 베타는 과거 평균이라 실제 위기에는 상관관계가 "
               "급등해 추정보다 더 빠지는 경우가 많습니다. 채권은 듀레이션 근사라 "
               "금리 변동이 클수록 오차가 커집니다. **하한선이 아니라 대략의 크기**로 "
               "보세요. 교육·참고용이며 투자 자문이 아닙니다.")


def render_stress(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate):
    st.title("🌩 스트레스 테스트")
    _s1, _s2 = st.tabs(["📅 과거 위기 재현", "⚡ 가상 충격 시나리오"])
    with _s2:
        render_hypo_stress(base_ccy, start_date, end_date, use_div,
                           fx_hedge, gap_fill)
    with _s1:
        _render_hist_stress(base_ccy, start_date, end_date, use_div,
                            fx_hedge, gap_fill, rf_rate)


def _render_hist_stress(base_ccy, start_date, end_date, use_div,
                        fx_hedge, gap_fill, rf_rate):
    st.caption("과거 위기 국면만 잘라내어, 그 시기에 포트폴리오가 어떻게 버텼는지 봅니다. "
               "평균적으로 좋아 보이는 구성도 위기에는 전혀 다르게 움직일 수 있습니다.")

    live, tickers, bad = render_ticker_table(
        state_key="_stress_df", editor_key="_stress_editor", gen_key="_stress_gen",
        cols=STRESS_COLS, weight_col="비중(%)", title="스트레스 테스트",
        min_rows=1, max_rows=30, show_equal=True, equal_key="_stress_eq",
        default_rows=[{"티커": "SPY", "비중(%)": 60.0},
                      {"티커": "AGG", "비중(%)": 40.0}])

    ok_in, msgs = validate_setup(tickers, bad, live["비중(%)"], min_n=1,
                                 need_weight=True)
    show_msgs(msgs)

    # ---------------- 구간 ----------------
    st.subheader("2️⃣ 위기 구간 (Crisis Periods)")
    st.caption("널리 쓰이는 구간을 고르거나, 직접 날짜를 지정할 수 있습니다. "
               "구간은 S&P 500의 고점 → 저점을 기준으로 잡았습니다.")
    picked = st.multiselect("분석할 구간", list(CRISIS_PRESETS),
                            default=["글로벌 금융위기 (2007~09)", "코로나 폭락 (2020)",
                                     "긴축 발작 (2022)"], key="_stress_pick")
    with st.expander("직접 구간 추가", expanded=False):
        u1, u2, u3 = st.columns([2, 1.5, 1.5])
        cname = u1.text_input("구간 이름", key="_stress_cname",
                              placeholder="예: 2024년 8월 엔캐리 청산")
        cs = u2.date_input("시작일", pd.Timestamp("2024-07-31"), key="_stress_cs")
        ce = u3.date_input("종료일", pd.Timestamp("2024-08-05"), key="_stress_ce")
        if st.button("구간 추가", width="stretch",
                     disabled=not cname.strip()):
            cust = dict(st.session_state.get("_stress_custom", {}))
            cust[cname.strip()] = (str(cs), str(ce))
            st.session_state["_stress_custom"] = cust
            st.rerun()
        cust = st.session_state.get("_stress_custom", {})
        if cust:
            st.caption("추가한 구간 · " + " · ".join(
                f"{k} ({v[0]}~{v[1]})" for k, v in cust.items()))
            if st.button("추가 구간 모두 지우기", width="stretch"):
                st.session_state.pop("_stress_custom", None)
                st.rerun()

    bench_s = st.text_input("벤치마크", value="^GSPC", key="_stress_bench")
    sc1, sc2 = st.columns(2)
    rebal_s = sc1.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key="_stress_rebal")
    init_cost = sc2.checkbox("위기 시작일 매수비용 포함", value=False,
                             key="_stress_initcost",
                             help="기본은 **위기 이전부터 보유 중**이었다고 봅니다. "
                                  "체크하면 위기 시작일에 전량 새로 매수한 것으로 보고 "
                                  "그 비용까지 차감합니다.")

    all_c = dict(CRISIS_PRESETS)
    all_c.update(st.session_state.get("_stress_custom", {}))
    use_c = {k: all_c[k] for k in picked if k in all_c}
    use_c.update(st.session_state.get("_stress_custom", {}))

    run_s = st.button("🌩 스트레스 테스트 실행", type="primary", width="stretch",
                      disabled=not ok_in or not use_c)
    if run_s:
        st.session_state["_stress_has_run"] = True
    if not st.session_state.get("_stress_has_run"):
        st.info("👆 포트폴리오와 위기 구간을 정한 뒤 **스트레스 테스트 실행**을 눌러주세요.")
        return
    if not use_c:
        st.error("분석할 구간을 하나 이상 선택해주세요.")
        return

    # ---------------- 데이터 ----------------
    bench_n = normalize_ticker(bench_s)[0] if bench_s.strip() else None
    need = tickers + ([bench_n] if bench_n else [])
    # 위기 구간이 분석 기간 밖일 수 있으므로 넉넉히 받는다
    lo = min(pd.Timestamp(v[0]) for v in use_c.values())
    fetch_start = min(pd.Timestamp(start_date), lo - pd.DateOffset(months=6))
    try:
        with st.spinner("데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(need, fetch_start, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    use_tk = [t for t in tickers if t in prices.columns]
    if not use_tk:
        st.error("종목 데이터를 가져오지 못했습니다.")
        return

    w = live.set_index("티커")["비중(%)"].reindex(use_tk).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=use_tk)
    W = {t: float(v) for t, v in w.items()}

    try:
        pr = portfolio_returns(prices, W, rebal_s)
        tno = turnover_series(prices, W, rebal_s).reindex(pr.index).fillna(0)
        if not init_cost and len(tno):
            # 위기 이전부터 보유 중이었다고 보고 최초 매수비용은 빼지 않는다
            tno.iloc[0] = 0.0
        pr = apply_cost(pr, tno, cost_bp)
    except Exception as ex:
        st.error(f"포트폴리오를 계산하지 못했습니다: {ex}")
        return
    br = None
    if bench_n and bench_n in prices.columns:
        try:
            br = portfolio_returns(prices, {bench_n: 100.0}, REBAL_NONE)
        except Exception:
            br = None

    st.success(f"✅ 실행 완료 · {len(use_tk)}종목 · 구간 {len(use_c)}개 · "
               f"데이터 {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # ---------------- 구간별 성과 ----------------
    st.subheader("3️⃣ 위기별 성과 (Performance in Crises)")
    st.caption("**구간내 최저**는 위기 시작일 대비 그 구간에서 가장 낮았던 시점의 "
               "수익률입니다(구간 이전 고점 대비가 아닙니다). "
               "**회복까지**는 저점에서 위기 직전 수준을 되찾기까지 걸린 기간입니다.")
    rows, skipped = [], []
    for nm, (a, b) in use_c.items():
        x = crisis_stats(pr, a, b)
        if x is None:
            skipped.append(nm); continue
        row = {"위기": nm, "시작": a, "종료": b,
               "구간내 최저(%)": x["구간내 최저(%)"],
               "구간 수익률(%)": x["구간 수익률(%)"],
               "기간(개월)": x["기간(개월)"],
               "저점일": x["저점일"],
               "회복까지(개월)": x["회복까지(개월)"]}
        if br is not None:
            xb = crisis_stats(br, a, b)
            row["벤치마크 구간내 최저(%)"] = xb["구간내 최저(%)"] if xb else np.nan
            row["초과(%p)"] = (row["구간내 최저(%)"] - row["벤치마크 구간내 최저(%)"]
                              if xb else np.nan)
        rows.append(row)
    if skipped:
        st.warning(f"데이터가 없어 건너뛴 구간: {', '.join(skipped)}")
    if not rows:
        st.error("분석 가능한 구간이 없습니다. 사이드바의 시작일을 앞당겨보세요.")
        return

    cdf = pd.DataFrame(rows)
    fmt = {"구간내 최저(%)": "{:.2f}", "구간 수익률(%)": "{:.2f}",
           "기간(개월)": "{:.1f}", "회복까지(개월)": "{:.1f}"}
    if "벤치마크 구간내 최저(%)" in cdf.columns:
        fmt.update({"벤치마크 구간내 최저(%)": "{:.2f}", "초과(%p)": "{:+.2f}"})
    st.dataframe(cdf.style.format(fmt, na_rep="미회복")
                 .highlight_max(subset=["구간내 최저(%)"], color="#dcfce7"),
                 width="stretch", hide_index=True)
    st.caption("초록색 = 가장 덜 빠진 구간. **낙폭은 음수이므로 0에 가까울수록 좋습니다.** "
               "회복까지가 '미회복'이면 아직 위기 이전 수준을 되찾지 못했다는 뜻입니다.")

    fbar = go.Figure()
    fbar.add_trace(go.Bar(x=cdf["위기"], y=cdf["구간내 최저(%)"], name="포트폴리오",
                          marker_color="#dc2626",
                          text=[f"{v:.1f}%" for v in cdf["구간내 최저(%)"]],
                          textposition="outside"))
    if "벤치마크 구간내 최저(%)" in cdf.columns:
        fbar.add_trace(go.Bar(x=cdf["위기"], y=cdf["벤치마크 구간내 최저(%)"],
                              name=f"벤치마크 ({bench_n})", marker_color="#cbd5e1",
                              text=[f"{v:.1f}%" for v in cdf["벤치마크 구간내 최저(%)"]],
                              textposition="outside"))
    fbar.update_layout(barmode="group", height=380,
                       margin=dict(l=0, r=0, t=30, b=0),
                       yaxis=dict(title=None, ticksuffix="%"),
                       legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fbar, width="stretch")

    # ---------------- 자산별 ----------------
    st.subheader("4️⃣ 위기별 자산 성과 (By Asset)")
    st.caption("각 자산이 위기 구간에서 어떻게 움직였는지입니다. "
               "**무엇이 방어 역할을 했는지** 확인해보세요.")
    arows = []
    R_all = prices[use_tk].pct_change().dropna()
    for nm, (a, b) in use_c.items():
        seg = R_all.loc[pd.Timestamp(a):pd.Timestamp(b)]
        if len(seg) < 3:
            continue
        for t in use_tk:
            eqx = (1 + seg[t]).cumprod()
            arows.append({"위기": nm, "티커": t,
                          "구간 수익률(%)": (float(eqx.iloc[-1]) - 1) * 100,
                          "구간내 최저(%)": (float(eqx.min()) - 1) * 100})
    if arows:
        adf = pd.DataFrame(arows)
        piv = adf.pivot(index="티커", columns="위기", values="구간 수익률(%)")
        piv = piv[[c for c in cdf["위기"] if c in piv.columns]]
        st.dataframe(piv.style.format("{:.2f}", na_rep="-").map(_heat_color),
                     width="stretch")
        worst_c = cdf.loc[cdf["구간내 최저(%)"].idxmin(), "위기"]
        if worst_c in piv.columns:
            best_a = piv[worst_c].idxmax()
            st.info(f"가장 큰 낙폭을 기록한 **{worst_c}** 구간에서 가장 잘 버틴 자산은 "
                    f"**{best_a}** ({piv.loc[best_a, worst_c]:+.2f}%)입니다.")

    # ---------------- 위기 구간 상관관계 ----------------
    st.subheader("5️⃣ 위기 구간 상관관계 (Correlation in Crises)")
    st.caption("**분산투자가 가장 시험받는 지점입니다.** 평상시 낮던 상관관계가 위기에 "
               "함께 올라가면, 정작 방어가 필요한 순간에 분산 효과가 줄어듭니다.")
    if len(use_tk) < 2:
        st.info("상관관계를 보려면 2개 이상의 종목이 필요합니다.")
    else:
        mask = pd.Series(False, index=R_all.index)
        for nm, (a, b) in use_c.items():
            mask |= (R_all.index >= pd.Timestamp(a)) & (R_all.index <= pd.Timestamp(b))
        crows = {}
        for lbl, sel in [("평상시", ~mask), ("위기 구간", mask), ("전체", None)]:
            sub = R_all if sel is None else R_all[sel]
            if len(sub) < 20:
                continue
            cm = sub.corr()
            rho, eff = diversification_stats(cm)
            crows[lbl] = {"표본(거래일)": len(sub), "평균 상관계수": rho,
                          "실효 종목 수": eff}
        if crows:
            st.dataframe(pd.DataFrame(crows).T.style.format(
                {"표본(거래일)": "{:.0f}", "평균 상관계수": "{:.3f}",
                 "실효 종목 수": "{:.2f}"}), width="stretch")
            if "평상시" in crows and "위기 구간" in crows:
                d = crows["위기 구간"]["평균 상관계수"] - crows["평상시"]["평균 상관계수"]
                if d > 0.1:
                    st.warning(f"위기 구간 평균 상관계수가 평상시보다 **{d:+.2f}** 높습니다. "
                               f"실효 종목 수도 {crows['평상시']['실효 종목 수']:.1f}개 → "
                               f"**{crows['위기 구간']['실효 종목 수']:.1f}개** 로 줄어듭니다. "
                               f"위기에 분산 효과가 약해진다는 뜻입니다.")
                elif d < -0.05:
                    st.success(f"위기 구간 평균 상관계수가 평상시보다 **{d:+.2f}** 낮습니다. "
                               f"하락 국면에서도 분산이 유지되는 구성입니다.")
        tabs_s = st.tabs(["평상시 행렬", "위기 구간 행렬"])
        for tb, sel in zip(tabs_s, [~mask, mask]):
            with tb:
                sub = R_all[sel]
                if len(sub) >= 20:
                    st.dataframe(sub.corr().round(2).style.format("{:.2f}")
                                 .map(_corr_color), width="stretch")
                else:
                    st.info("표본이 부족합니다.")

    # ---------------- 회복 경로 ----------------
    st.subheader("6️⃣ 회복 경로 (Recovery Path)")
    pick_c2 = st.selectbox("구간 선택", list(cdf["위기"]), key="_stress_path")
    a, b = use_c[pick_c2]
    seg_r = pr.loc[pd.Timestamp(a):]
    if len(seg_r) > 5:
        eqp = (1 + seg_r).cumprod()
        fpath = go.Figure()
        fpath.add_trace(go.Scatter(x=eqp.index, y=(eqp.values - 1) * 100,
                                   name="포트폴리오", line=dict(color="#2563eb", width=2)))
        if br is not None:
            bb = (1 + br.loc[pd.Timestamp(a):]).cumprod()
            fpath.add_trace(go.Scatter(x=bb.index, y=(bb.values - 1) * 100,
                                       name=f"벤치마크 ({bench_n})",
                                       line=dict(color="#9ca3af", width=1.5, dash="dash")))
        fpath.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
        fpath.add_vrect(x0=pd.Timestamp(a), x1=pd.Timestamp(b),
                        fillcolor="#fee2e2", opacity=0.5, line_width=0, layer="below")
        fpath.update_layout(height=380, hovermode="x unified",
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(title=None, ticksuffix="%"),
                            legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fpath, width="stretch")
        st.caption("붉은 음영이 위기 구간입니다. 0%선을 다시 넘는 시점이 회복 시점입니다.")

    # ---------------- 요약 ----------------
    st.subheader("📝 요약 (Summary)")
    worst = cdf.loc[cdf["구간내 최저(%)"].idxmin()]
    slines = [f"선택한 {len(cdf)}개 구간 중 가장 큰 낙폭은 **{worst['위기']}** 의 "
              f"**{worst['구간내 최저(%)']:.2f}%** 였습니다."]
    if not pd.isna(worst["회복까지(개월)"]):
        slines.append(f"저점에서 위기 이전 수준을 되찾기까지 "
                      f"**{worst['회복까지(개월)']:.1f}개월**이 걸렸습니다.")
    else:
        slines.append("이 구간은 아직 위기 이전 수준을 되찾지 못했습니다.")
    if "초과(%p)" in cdf.columns and cdf["초과(%p)"].notna().any():
        avg_ex = float(cdf["초과(%p)"].mean())
        slines.append(f"벤치마크 대비 평균 **{avg_ex:+.2f}%p** "
                      f"{'덜 빠졌습니다' if avg_ex > 0 else '더 빠졌습니다'}.")
    unrec = cdf[cdf["회복까지(개월)"].isna()]
    if len(unrec):
        slines.append(f"아직 회복되지 않은 구간: **{', '.join(unrec['위기'])}**")
    st.markdown("\n\n".join(slines))

    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            cdf.to_excel(xw, sheet_name="1_위기별성과", index=False)
            if arows:
                pd.DataFrame(arows).to_excel(xw, sheet_name="2_자산별", index=False)
            if len(use_tk) >= 2 and crows:
                pd.DataFrame(crows).T.to_excel(xw, sheet_name="3_상관요약")
            pd.DataFrame([{"구간": k, "시작": v[0], "종료": v[1]}
                          for k, v in use_c.items()]).to_excel(
                xw, sheet_name="4_구간정의", index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"stress_test_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_stress", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")
    assumptions_panel(rebalance=rebal_s, cost=cost_bp, initial_cost=init_cost,
                      period=f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                      bench=bench_n or "-")
    st.caption("과거의 위기가 같은 형태로 반복된다는 보장은 없습니다. "
               "특히 자산 간 상관관계는 위기마다 다르게 나타납니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")


# ======================================================================
# 시장·거시 전망 (요인 민감도)
# ======================================================================
# 요인 대용치. 두 번째가 있으면 비율(A/B)의 변화를 요인으로 쓴다.
FACTOR_PROXY = {
    "장기금리": ("^TNX", None, "10년 국채 금리", "금리가 오르면 +"),
    "경기": ("XLY", "XLP", "경기소비재 / 필수소비재", "경기가 좋아지면 +"),
    # TIP/IEF 는 둘 다 금리에 민감해 물가가 아니라 잔여 금리를 재는 문제가 있었다.
    # 원자재로 바꾸면 물가 압력을 더 직접 반영한다(경기와 겹칠 수 있으니 VIF 확인).
    "물가·원자재": ("DBC", None, "원자재 지수 ETF", "원자재·물가 압력이 커지면 +"),
    "글로벌 달러 강세": ("UUP", None, "달러지수 ETF", "달러가 강해지면 +"),
    "신용": ("HYG", "LQD", "하이일드 / 우량회사채", "신용환경이 좋아지면 +"),
    "위험선호": ("SPY", "TLT", "주식 / 장기채", "위험선호가 커지면 +"),
}

# 민감도가 상식과 맞는지 자동 확인할 때 쓰는 기대치.
# (자산, 요인) → (하한, 상한). None 이면 부호만 본다.
FACTOR_EXPECT = {
    ("TLT", "장기금리"): (-20.0, -11.0),
    ("IEF", "장기금리"): (-9.5, -5.5),
    ("SHY", "장기금리"): (-3.5, -0.8),
    ("SPY", "경기"): (0.4, 2.0),
    ("SPY", "장기금리"): (None, 0.0),
    ("GLD", "글로벌 달러 강세"): (None, 0.0),
    ("XLE", "물가·원자재"): (0.0, None),
    ("XLP", "경기"): (None, 0.6),
}
FACTOR_CHECK_TK = ["TLT", "IEF", "SHY", "SPY", "GLD", "XLE", "XLP"]
# 기본 4개. 신용·위험선호는 경기와 많이 겹쳐 선택 사항으로 둔다.
FACTOR_DEFAULT = ["장기금리", "경기", "물가·원자재", "글로벌 달러 강세"]

VIEW_DIRS = {"강한 상승": 2.0, "상승": 1.0, "보합": 0.0, "하락": -1.0, "강한 하락": -2.0}
VIEW_HORIZON = ["3개월", "6개월", "1년", "2년"]
# 기간이 짧을수록 같은 확신도라도 약하게 반영한다.
# 3개월 앞을 2년 앞만큼 확신하기는 어렵기 때문이다.
VIEW_HZ_WEIGHT = {"3개월": 0.6, "6개월": 0.8, "1년": 1.0, "2년": 1.15}

# 시장전망 변수 → 요인 매핑. 사용자가 익숙한 말로 묻고 내부에서 요인으로 옮긴다.
MARKET_VIEWS = {
    "정책금리": ("장기금리", 0.6, "정책금리는 장기금리에 부분적으로 전이됩니다"),
    "장기금리": ("장기금리", 1.0, ""),
    "경기": ("경기", 1.0, ""),
    "물가": ("물가·원자재", 1.0, "원자재 가격으로 물가 압력을 대신 잽니다"),
    "달러": ("글로벌 달러 강세", 1.0, ""),
    "신용환경": ("신용", 1.0, ""),
    "위험선호": ("위험선호", 1.0, ""),
}


@st.cache_data(ttl=3600, show_spinner=False)
def build_factors(start, end, freq="주별", picks=tuple(FACTOR_DEFAULT)):
    """요인 시계열을 만든다. 반환: (DataFrame, 실패한 요인 목록)"""
    need = []
    for f in picks:
        a, b, *_ = FACTOR_PROXY[f]
        need += [a] + ([b] if b else [])
    need = list(dict.fromkeys(need))
    try:
        px, _m, _f = build_price_frame(need, start, end, "USD", True, True, True)
    except Exception:
        return None, list(picks)
    if px.empty:
        return None, list(picks)
    if freq == "주별":
        px = px.resample("W-FRI").last()
    cols, bad = {}, []
    for f in picks:
        a, b, *_ = FACTOR_PROXY[f]
        if a not in px.columns or (b and b not in px.columns):
            bad.append(f)
            continue
        if f == "장기금리":
            # 금리 지수는 가격이 아니라 수익률 자체다. 변화폭(%p)을 쓴다.
            cols[f] = px[a].diff() / 100.0
        elif b:
            cols[f] = (px[a] / px[b]).pct_change()
        else:
            cols[f] = px[a].pct_change()
    if not cols:
        return None, list(picks)
    return pd.DataFrame(cols).dropna(), bad


def standardize(F: pd.DataFrame):
    """
    요인을 표준편차 1로 맞춘다.
    요인마다 변동성이 크게 다르면(금리 0.0015 vs 경기 0.010) 회귀 계수가
    스케일 차이만큼 부풀려져 요인 간 비교가 불가능해진다.
    반환: (표준화된 F, 요인별 표준편차)
    """
    sd = F.std().replace(0, np.nan)
    return (F - F.mean()) / sd, sd


def factor_betas(r: pd.Series, F: pd.DataFrame):
    """
    다중회귀로 요인별 민감도를 구한다.
    요인끼리 상관이 있으면 단순회귀는 크게 빗나가므로 반드시 다중회귀를 쓴다.
    반환: (민감도 dict, R², 표본 수)
    """
    df = pd.concat([r, F], axis=1).dropna()
    if len(df) < 30:
        return {c: np.nan for c in F.columns}, np.nan, len(df)
    y = df.iloc[:, 0].values
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].values])
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {c: np.nan for c in F.columns}, np.nan, len(df)
    pred = X @ b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
    return dict(zip(F.columns, b[1:])), r2, len(df)


def factor_vif(F: pd.DataFrame) -> dict:
    """
    요인끼리 얼마나 겹치는지(분산팽창계수).
    10을 넘으면 그 요인의 민감도를 따로 떼어 해석하기 어렵다.
    """
    out = {}
    for c in F.columns:
        X = F.drop(columns=[c])
        if X.empty:
            out[c] = 1.0
            continue
        X1 = np.column_stack([np.ones(len(X)), X.values])
        y = F[c].values
        try:
            b, *_ = np.linalg.lstsq(X1, y, rcond=None)
            pred = X1 @ b
            ss = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss > 1e-18 else 0.0
            out[c] = 1 / (1 - r2) if r2 < 0.9999 else np.inf
        except Exception:
            out[c] = np.nan
    return out


# R² 구간별 판단 강도. 설명력이 낮으면 결과를 단정하지 않는다.
R2_GRADE = [(0.40, "일반 참고 가능"), (0.20, "제한적 참고"),
            (0.10, "신뢰도 매우 낮음"), (0.0, "판정 유보")]


def r2_grade(r2: float) -> str:
    if not np.isfinite(r2):
        return "판정 유보"
    for th, lbl in R2_GRADE:
        if r2 >= th:
            return lbl
    return "판정 유보"


def r2_weight(r2: float) -> float:
    """
    설명력이 낮은 자산의 판단 강도를 낮추는 가중치.
    √R² 는 상관계수와 같은 척도라 해석이 자연스럽고,
    낮은 자산은 확실히 줄이면서 정상 자산은 과하게 깎지 않는다.
    """
    return float(np.sqrt(max(r2, 0.0))) if np.isfinite(r2) else 0.0


def impact_label(score: float, scale: float = 1.0) -> str:
    """
    영향 점수를 말로 옮긴다. 없는 정밀도를 만들지 않도록 5단계로만 나눈다.
    scale 은 포트폴리오 안에서의 상대 기준(가장 큰 절대 점수)으로,
    자산군마다 민감도 크기가 달라 절대 기준을 쓰면 구분이 되지 않는다.
    """
    if not np.isfinite(score):
        return "판정 불가"
    s = score / scale if scale and np.isfinite(scale) and scale > 1e-9 else score
    if s > 0.55:
        return "긍정"
    if s > 0.18:
        return "소폭 긍정"
    if s >= -0.18:
        return "중립"
    if s >= -0.55:
        return "소폭 부정"
    return "부정"


BL_COLS = ["티커", "종목명", "기준 비중(%)"]
BL_VIEW_COLS = ["유형", "자산 A", "자산 B", "연 수익률(%)", "확신도"]
BL_CONF = {"상": 0.25, "중": 1.0, "하": 4.0}

# 간단 모드 — 전망 강도를 자산 변동성의 배수로 환산한다.
# 변동성 16%인 주식의 '강한 긍정'은 +3.2%p, 변동성 6%인 채권은 +1.2%p 가 되어
# 자산 성격에 맞는 크기가 자동으로 잡힌다.
BL_SIMPLE = {"강한 긍정": 0.20, "긍정": 0.10, "중립": 0.0,
             "부정": -0.10, "강한 부정": -0.20}
BL_SIMPLE_COLS = ["티커", "전망"]


def _bl_blank():
    return {"티커": "", "종목명": "", "기준 비중(%)": np.nan}


def _bl_view_blank():
    return {"유형": "절대", "자산 A": "", "자산 B": "", "연 수익률(%)": np.nan, "확신도": "중"}


def implied_returns(delta: float, cov: np.ndarray, w_mkt: np.ndarray) -> np.ndarray:
    """역최적화. 시장 비중이 최적이 되게 하는 균형 기대수익률 Π = δΣw."""
    return delta * (cov @ w_mkt)


def black_litterman(cov, w_mkt, delta, tau, P, Q, conf_scale):
    """
    사후 기대수익률
      E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
      Ω = diag(확신도계수 × P(τΣ)Pᵀ)
    뷰가 없으면 균형 수익률을 그대로 돌려준다.
    """
    pi = implied_returns(delta, cov, w_mkt)
    if P is None or len(P) == 0:
        return pi, pi.copy()
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).reshape(-1)
    tS = tau * cov
    omega = np.diag(np.maximum(
        np.diag(P @ tS @ P.T) * np.asarray(conf_scale, dtype=float), 1e-12))
    inv_tS = np.linalg.inv(tS)
    inv_om = np.linalg.inv(omega)
    A = inv_tS + P.T @ inv_om @ P
    b = inv_tS @ pi + P.T @ inv_om @ Q
    return pi, np.linalg.solve(A, b)


def bl_weights(mu, cov, delta, wmin=0.0, wmax=1.0, allow_short=False):
    """사후 기대수익률로 비중 산출. 제약이 있으면 최적화로, 없으면 해석해로."""
    n = len(mu)
    lo_arr = _as_bounds(wmin, n, 0.0)
    if allow_short and float(lo_arr.min()) <= -1:
        w = np.linalg.solve(delta * cov, mu)
        s = w.sum()
        return w / s if abs(s) > 1e-12 else np.ones(n) / n

    def obj(w):
        return float(0.5 * delta * (w @ cov @ w) - w @ mu)
    # 공매도를 허용하면 음수 하한을 그대로 쓴다.
    # (예전에는 항상 0 이상으로 잘라 공매도 설정이 무시됐다)
    lo = lo_arr if allow_short else np.maximum(lo_arr, 0.0)
    return _opt_solve(obj, n, lo, wmax, strict=True)


MV_COLS = ["전망 변수", "방향", "확신도(%)", "전망 기간", "사용", "메모"]


def _mv_default():
    return pd.DataFrame([
        {"전망 변수": k, "방향": "보합", "확신도(%)": 50, "전망 기간": "6개월",
         "사용": False, "메모": ""} for k in MARKET_VIEWS])[MV_COLS]


def render_macro_view(base_ccy, start_date, end_date, use_div,
                      fx_hedge, gap_fill, rf_rate):
    """시장·거시 전망을 넣고, 그것이 지금 포트폴리오에 미치는 영향을 본다."""
    st.caption("금리·경기·물가·달러에 대한 전망을 넣으면, 각 자산이 그 변수에 얼마나 "
               "민감한지 과거 데이터로 추정해 **영향을 분석**합니다. "
               "이 단계에서는 비중을 바꾸지 않고 진단만 합니다.")

    # ---------------- 포트폴리오 ----------------
    st.subheader("1️⃣ 현재 포트폴리오")
    st.caption("전망을 넣을 대상이 아니라, **전망의 영향을 받을 투자 대상**입니다.")
    key = "_mv_df"
    live, tickers, bad = render_ticker_table(
        state_key=key, editor_key="_mv_editor", gen_key="_mv_gen",
        cols=STRESS_COLS, weight_col="비중(%)", title="시장·거시 전망",
        min_rows=1, max_rows=30, show_equal=True, equal_key="_mv_eq",
        default_rows=[{"티커": t_, "비중(%)": w_} for t_, w_ in
                      [("SPY", 45.0), ("QQQ", 15.0), ("TLT", 20.0),
                       ("GLD", 10.0), ("EEM", 10.0)]])
    ok_in, msgs = validate_setup(tickers, bad, live["비중(%)"], min_n=1,
                                 need_weight=True)
    show_msgs(msgs)
    if not ok_in:
        return

    # ---------------- 전망 입력 ----------------
    st.subheader("2️⃣ 시장 전망")
    st.caption("**보유 자산과 무관하게** 전망을 넣을 수 있습니다. 장기채가 없어도 "
               "금리 전망을 넣을 수 있고, 그 전망이 다른 자산에 어떤 영향을 주는지 "
               "보여드립니다.")

    if "_mv_views" not in st.session_state:
        st.session_state["_mv_views"] = _mv_default()
    # 국면 화면에서 넘어온 초안이 있으면 반영
    if st.session_state.pop("_mv_draft_flag", False):
        st.info("📉 **시장·거시 국면** 화면의 판정을 초안으로 가져왔습니다. "
                "국면은 **현재 상태**이고 전망은 **앞으로**이므로, 확인하고 고쳐주세요.")

    mv = st.data_editor(
        st.session_state["_mv_views"], num_rows="fixed", width="stretch",
        key="_mv_view_editor", column_order=MV_COLS, hide_index=True,
        column_config={
            "전망 변수": st.column_config.TextColumn("전망 변수", disabled=True,
                                                width="small"),
            "방향": st.column_config.SelectboxColumn("방향", options=list(VIEW_DIRS),
                                                  width="medium"),
            "확신도(%)": st.column_config.NumberColumn(
                "확신도(%)", min_value=0, max_value=100, step=10, width="small",
                help="낮으면 영향을 작게 반영합니다."),
            "전망 기간": st.column_config.SelectboxColumn("전망 기간",
                                                     options=VIEW_HORIZON,
                                                     width="small"),
            "사용": st.column_config.CheckboxColumn("사용", width="small"),
            "메모": st.column_config.TextColumn("메모 · 근거", width="large"),
        })
    st.session_state["_mv_views"] = mv
    _used = mv[mv["사용"].fillna(False).astype(bool) &
               (mv["방향"] != "보합")]
    # 전망표에는 있는데 회귀 요인에 없으면 계산에서 조용히 빠진다
    _need = {MARKET_VIEWS[str(v)][0] for v in _used["전망 변수"]
             if str(v) in MARKET_VIEWS} - set(FACTOR_DEFAULT)
    _miss = sorted(_need - set(st.session_state.get("_mv_extra", [])))
    if _miss:
        st.warning(f"**{', '.join(_miss)}** 전망을 넣으셨는데 해당 요인이 꺼져 "
                   f"있습니다. 이대로는 계산에 반영되지 않습니다. "
                   f"아래 **추가 요인**에서 켜주세요.")
    # 같은 요인에 두 전망이 겹치는지 입력 단계에서 미리 알린다
    _fmap = {}
    for v in _used["전망 변수"]:
        f_ = MARKET_VIEWS.get(str(v), (None,))[0]
        if f_:
            _fmap.setdefault(f_, []).append(str(v))
    _pre_dup = {k: v for k, v in _fmap.items() if len(v) > 1}
    if _pre_dup:
        st.warning("같은 요인에 전망이 겹칩니다 — "
                   + " · ".join(f"**{k}** ← {', '.join(v)}" for k, v in _pre_dup.items())
                   + "\n\n둘 다 켜두면 그 요인이 두 번 반영돼 영향이 부풀려집니다. "
                     "**하나만 사용**하시길 권합니다. "
                     "(정책금리는 장기금리에 0.6배로 전이되어 들어갑니다)")
    if _used.empty:
        st.info("전망을 하나 이상 **사용**으로 체크하고 방향을 정해주세요. "
                "예를 들어 `장기금리 · 하락 · 확신도 70%` 처럼요.")

    use_hz_weight = st.checkbox(
        "⏱ 전망 기간에 따라 강도 조정 (기간조정 사용)", value=False, key="_mv_hz_on",
        help="켜면 짧은 기간(3개월)의 전망은 약하게, 긴 기간(2년)은 강하게 반영합니다 "
             "(3개월 0.6배 ~ 2년 1.15배). 이 배수는 이론적으로 정해진 값이 아니라 "
             "휴리스틱이라 **기본은 꺼져 있습니다**. 꺼두면 전망 기간은 기록용으로만 "
             "쓰이고, 점수는 전망 방향·확신도·요인 민감도만으로 계산됩니다.")

    # ---------------- 요인 설정 ----------------
    st.subheader("3️⃣ 민감도 추정 설정")
    f1, f2, f3 = st.columns(3)
    f_yrs = f1.selectbox("추정 기간", [1, 3, 5], index=1, key="_mv_yrs",
                         format_func=lambda x: f"{x}년")
    f_freq = f2.selectbox("계산 주기", ["주별", "일별"], index=0, key="_mv_freq",
                          help="거래 시간대가 다른 국가를 섞으면 일별은 민감도가 "
                               "낮게 추정됩니다. 주별을 권합니다.")
    # 세션에 미리 넣은 값(국면 초안 등)이 있으면 그것을 기본값으로 삼는다.
    # default= 를 함께 주면 위젯이 그 값으로 덮어써 버린다.
    st.session_state.setdefault("_mv_extra", [])
    extra_f = f3.multiselect("추가 요인", ["신용", "위험선호"],
                             key="_mv_extra",
                             help="경기와 많이 겹쳐 기본에서 제외했습니다. "
                                  "넣으면 설명력은 오르지만 요인별 해석이 "
                                  "흐려질 수 있습니다.")
    picks = FACTOR_DEFAULT + list(extra_f)

    if base_ccy != "USD":
        st.info(f"**달러 전망을 읽는 법** — 민감도는 **현지통화 기준**으로 잽니다. "
                f"기준통화({base_ccy})로 환산한 값에는 환율이 이미 섞여 있어, "
                f"달러 요인을 따로 떼어낼 수 없기 때문입니다.\n\n"
                f"따라서 여기서 말하는 **달러 민감도는 '달러가 강해질 때 그 자산 자체가 "
                f"현지통화로 어떻게 움직이나'** 입니다. 달러 강세가 "
                f"{base_ccy} 환산 수익에 주는 효과(해외자산 평가액 상승)는 "
                f"**별도**이며, 그건 🌩 스트레스 테스트의 환율 충격에서 다룹니다.")

    with st.expander("요인을 무엇으로 재는지", expanded=False):
        st.dataframe(pd.DataFrame(
            [{"요인": k, "대용치": FACTOR_PROXY[k][0] +
              (f" / {FACTOR_PROXY[k][1]}" if FACTOR_PROXY[k][1] else ""),
              "설명": FACTOR_PROXY[k][2], "부호": FACTOR_PROXY[k][3]}
             for k in picks]), width="stretch", hide_index=True)
        st.caption("모두 **미국 시장 지표**입니다. 한국 자산의 민감도는 이 지표로 "
                   "설명되는 부분만 잡히므로, 설명력(R²)을 함께 확인하세요.")

    if st.button("🔬 민감도 추정", type="primary", width="stretch", key="_mv_go"):
        st.session_state["_mv_run"] = True
    if not st.session_state.get("_mv_run"):
        st.info("👆 **민감도 추정**을 눌러 각 자산이 시장 변수에 얼마나 반응하는지 "
                "확인하세요.")
        return

    # ---------------- 데이터 ----------------
    _start = pd.Timestamp(end_date) - pd.DateOffset(years=int(f_yrs))
    with st.spinner("요인 데이터 조회 중..."):
        F, bad_f = build_factors(_start, end_date, f_freq, tuple(picks))
    if F is None or F.empty:
        st.error("🚫 요인 데이터를 가져오지 못했습니다. 기간을 늘리거나 "
                 "추가 요인을 빼고 다시 시도해주세요.")
        return
    if bad_f:
        st.warning(f"데이터를 못 받은 요인은 제외했습니다: {', '.join(bad_f)}")

    # 달러 요인을 따로 재려면 자산 수익률에서 환율이 빠져 있어야 한다.
    # 기준통화로 환산한 값에는 이미 환율이 섞여 있어 달러 민감도가 이중으로 잡힌다.
    try:
        with st.spinner("자산 데이터 조회 중..."):
            px, meta, _f = build_price_frame(tickers, _start, end_date, base_ccy,
                                             use_div, True, gap_fill)
    except Exception as ex:
        st.error(f"자산 데이터를 가져오지 못했습니다: {ex}")
        return
    use_tk = [t for t in tickers if t in px.columns]
    if not use_tk:
        st.error("자산 데이터를 가져오지 못했습니다.")
        return
    _p = px[use_tk] if f_freq == "일별" else px[use_tk].resample("W-FRI").last()
    R = _p.pct_change()

    # ---------------- 민감도 ----------------
    st.subheader("4️⃣ 자산별 민감도 (Factor Betas)")
    # 요인마다 변동성이 크게 달라(금리는 작고 경기는 큼) 원단위 계수는
    # 요인 간 비교가 불가능하다. 표준화한 값을 기본으로 쓰고 원단위는 함께 보여준다.
    Fz, f_sd = standardize(F)
    vif = factor_vif(Fz)
    _hi_vif = [k for k, v in vif.items() if np.isfinite(v) and v > 10]
    rows, raws = [], []
    for t_ in use_tk:
        bz, r2, n_ = factor_betas(R[t_], Fz)
        br, _, _ = factor_betas(R[t_], F)
        rows.append({"티커": t_,
                     **{k: bz.get(k, np.nan) * 100 for k in Fz.columns},
                     "R²": r2, "표본": n_,
                     "신뢰도": r2_grade(r2 if r2 is not None else np.nan)})
        raws.append({"티커": t_, **{k: br.get(k, np.nan) for k in F.columns}})
    bdf = pd.DataFrame(rows)
    rawdf = pd.DataFrame(raws)
    _fmt = {k: "{:+.2f}" for k in Fz.columns}
    _fmt.update({"R²": "{:.2f}", "표본": "{:.0f}"})
    st.dataframe(bdf.style.format(_fmt, na_rep="-"), width="stretch",
                 hide_index=True)
    st.caption("**민감도(%)** 는 그 요인이 **평소 움직이는 만큼(1σ)** 변할 때 자산이 몇 % "
               "반응하는지입니다. 요인마다 변동폭이 달라서, 이렇게 맞춰야 요인 간 "
               "크기를 비교할 수 있습니다.\n\n"
               "**R²는 설명력**입니다. 낮으면 그 자산의 움직임이 이 요인들로 잘 설명되지 "
               "않는다는 뜻이니, 아래 영향 분석도 신중하게 보세요.")

    with st.expander("원래 단위로 보기 (금리는 듀레이션으로 읽힘)", expanded=False):
        st.dataframe(rawdf.style.format({k: "{:+.2f}" for k in F.columns},
                                        na_rep="-"),
                     width="stretch", hide_index=True)
        st.caption("장기금리 −16.0이면 금리가 1%p 오를 때 약 16% 하락한다는 뜻으로, "
                   "채권 듀레이션과 비교해볼 수 있습니다. "
                   "다른 요인은 대용치의 변화율 기준이라 직접 해석이 어렵습니다.")
        st.dataframe(pd.DataFrame({"요인": list(F.columns),
                                   "1σ (요인 변동폭)": [float(f_sd[c]) for c in F.columns]})
                     .style.format({"1σ (요인 변동폭)": "{:.5f}"}),
                     width="stretch", hide_index=True)

    _low = bdf[bdf["신뢰도"].astype(str).isin(["판정 유보", "신뢰도 매우 낮음"])
               ]["티커"].tolist()
    if _low:
        st.warning(f"**{', '.join(_low)}** 는 지금 요인 세트(금리·경기·물가·달러)로 "
                   f"움직임이 잘 설명되지 않습니다.\n\n"
                   f"이 자산들이 거시와 무관하다는 뜻은 아닙니다. 예를 들어 한국 "
                   f"반도체는 **업황·메모리 가격·원달러·중국 경기** 같은 다른 요인이 "
                   f"지배할 수 있는데, 그 요인들이 여기 없을 뿐입니다. "
                   f"**이 모형만으로는 방향을 판단하기 어렵다**고 보시면 됩니다.\n\n"
                   f"아래 영향 분석에서는 설명력만큼 판단 강도를 낮춰 반영합니다.")
    if _hi_vif:
        st.warning(f"요인끼리 많이 겹칩니다: {', '.join(_hi_vif)} — "
                   f"각 요인의 민감도를 따로 떼어 해석하기 어렵습니다. "
                   f"추가 요인을 빼보세요.")
    # ---------------- 계수 안정성 ----------------
    with st.expander("📐 기간을 바꿔도 민감도가 유지되는가", expanded=False):
        st.caption("R²가 높아도 **개별 요인 계수는 표본에 따라 흔들릴 수 있습니다.** "
                   "추정 기간을 바꿔가며 부호가 유지되는지 확인합니다. "
                   "부호가 뒤집히면 그 요인은 판정을 유보하는 게 안전합니다.\n\n"
                   "자산 가격은 여기서 **최대 5년치를 따로 조회**해 1·3·5년 구간을 "
                   "정확히 잘라 씁니다. 위 4️⃣ 표(추정 기간이 짧으면)와는 표본이 "
                   "달라 값이 다를 수 있습니다.")
        if st.button("📐 안정성 확인", width="stretch", key="_mv_stab"):
            srows, r2rows = [], []
            with st.spinner("기간별로 다시 추정 중..."):
                per = {}
                _stab_start = pd.Timestamp(end_date) - pd.DateOffset(years=5)
                try:
                    stab_px, _sm, _sf = build_price_frame(
                        use_tk, _stab_start, end_date, base_ccy, use_div,
                        True, gap_fill)
                except Exception as ex:
                    st.error(f"안정성 검사용 자산 데이터를 가져오지 못했습니다: {ex}")
                    stab_px = None
                if stab_px is not None and not stab_px.empty:
                    _stab_p = (stab_px if f_freq == "일별"
                              else stab_px.resample("W-FRI").last())
                    Rall = _stab_p.pct_change()
                    for yrs in (1, 3, 5):
                        s_ = pd.Timestamp(end_date) - pd.DateOffset(years=yrs)
                        Fy, _bad = build_factors(s_, end_date, f_freq, tuple(picks))
                        if Fy is None or Fy.empty:
                            continue
                        Fyz, _ = standardize(Fy)
                        Ry = Rall.loc[Fyz.index[0]:]
                        per[yrs] = {t_: factor_betas(Ry[t_], Fyz) for t_ in use_tk
                                    if t_ in Ry.columns}
            for t_ in use_tk:
                r2row = {"티커": t_}
                for y in (1, 3, 5):
                    rec = per.get(y, {}).get(t_)
                    r2row[f"{y}년 R²"] = rec[1] if rec else np.nan
                    r2row[f"{y}년 표본"] = rec[2] if rec else np.nan
                r2rows.append(r2row)
                for fac in F.columns:
                    year_vals = {}
                    for y in per:
                        rec = per[y].get(t_)
                        if rec is None:
                            continue
                        v = rec[0].get(fac)
                        if v is not None and np.isfinite(v):
                            year_vals[y] = v * 100
                    if len(year_vals) < 2:
                        continue
                    vals = list(year_vals.values())
                    signs = {np.sign(v) for v in vals if abs(v) > 1e-9}
                    stable = len(signs) <= 1
                    srows.append({
                        "티커": t_, "요인": fac,
                        **{f"{y}년": year_vals.get(y, np.nan) for y in (1, 3, 5)},
                        "부호 유지": "✅" if stable else "❌ 뒤집힘",
                        "변동폭": float(max(vals) - min(vals))})
            if srows:
                sdf = pd.DataFrame(srows)
                _flip = sdf[sdf["부호 유지"].astype(str).str.startswith("❌")]
                if len(_flip):
                    st.warning(f"부호가 뒤집힌 조합이 **{len(_flip)}개** 있습니다. "
                               f"해당 요인은 기간에 따라 방향이 달라지므로, "
                               f"그 자산의 영향 분석을 신중하게 보세요.")
                else:
                    st.success("모든 조합에서 부호가 유지됩니다.")
                st.dataframe(sdf.style.format(
                    {c: "{:+.2f}" for c in sdf.columns if c.endswith("년")}
                    | {"변동폭": "{:.2f}"}, na_rep="-"),
                    width="stretch", hide_index=True)
                st.caption("**변동폭**이 클수록 기간에 따라 민감도가 크게 달라진다는 "
                           "뜻입니다. 부호가 유지되더라도 변동폭이 크면 크기를 "
                           "그대로 믿기는 어렵습니다. 값은 각 기간의 요인을 그 "
                           "기간 안에서 다시 표준화한 뒤 비교한 것입니다(1σ 기준 %).")
                st.dataframe(pd.DataFrame(r2rows).style.format(
                    {c: "{:.2f}" for c in pd.DataFrame(r2rows).columns
                     if c.endswith("R²")}
                    | {c: "{:.0f}" for c in pd.DataFrame(r2rows).columns
                       if c.endswith("표본")}, na_rep="-"),
                    width="stretch", hide_index=True)
                st.caption("기간이 짧을수록 표본(주별 기준 1년≈52개)이 적어 R²가 "
                           "불안정할 수 있습니다.")
            else:
                st.warning("안정성 검사에 필요한 데이터가 부족합니다.")

    # ---------------- 자동 검증 ----------------
    with st.expander("🧪 민감도가 상식과 맞는지 확인", expanded=False):
        st.caption("성격이 분명한 자산으로 민감도를 뽑아 **알려진 방향·크기와 맞는지** "
                   "확인합니다. 여기서 어긋나면 요인 대용치나 추정 방식에 문제가 "
                   "있다는 뜻이라, 위 영향 분석도 믿기 어렵습니다.")
        if st.button("🧪 검증 실행", width="stretch", key="_mv_check"):
            try:
                with st.spinner("검증용 자산 조회 중..."):
                    cpx, _cm, _cf = build_price_frame(
                        FACTOR_CHECK_TK, _start, end_date, "USD",
                        use_div, True, gap_fill)
                ctk = [x for x in FACTOR_CHECK_TK if x in cpx.columns]
                _cp = cpx[ctk] if f_freq == "일별" else cpx[ctk].resample("W-FRI").last()
                CR = _cp.pct_change()
                crows = []
                for t_ in ctk:
                    br_, r2_, n2_ = factor_betas(CR[t_], F)
                    for fac in F.columns:
                        exp = FACTOR_EXPECT.get((t_, fac))
                        if exp is None:
                            continue
                        lo_, hi_ = exp
                        v = br_.get(fac, np.nan)
                        if not np.isfinite(v):
                            ok_ = "– 계산 불가"
                        elif (lo_ is None or v >= lo_) and (hi_ is None or v <= hi_):
                            ok_ = "✅"
                        else:
                            ok_ = "❌"
                        rng_ = ("음수" if (hi_ == 0.0 and lo_ is None) else
                                "양수" if (lo_ == 0.0 and hi_ is None) else
                                f"{lo_:.1f} ~ {hi_:.1f}" if lo_ is not None
                                and hi_ is not None else
                                f"{hi_:.1f} 이하")
                        crows.append({"자산": t_, "요인": fac,
                                      "추정(원단위)": v, "기대": rng_,
                                      "판정": ok_, "R²": r2_})
                if crows:
                    cdf2 = pd.DataFrame(crows)
                    nfail = int((cdf2["판정"] == "❌").sum())
                    if nfail:
                        st.error(f"🚫 {nfail}개 항목이 기대와 어긋납니다. "
                                 f"요인 대용치를 손봐야 할 수 있습니다.")
                    else:
                        st.success("✅ 모든 항목이 기대 범위 안에 있습니다.")
                    st.dataframe(cdf2.style.format({"추정(원단위)": "{:+.2f}",
                                                    "R²": "{:.2f}"}, na_rep="-"),
                                 width="stretch", hide_index=True)
                    st.caption("금리 민감도는 채권 듀레이션과 비교할 수 있습니다. "
                               "TLT는 −12~−18, SHY는 −1~−3 정도가 정상입니다.")
                else:
                    st.warning("검증용 자산 데이터를 가져오지 못했습니다.")
            except Exception as ex:
                st.error(f"검증에 실패했습니다: {ex}")

    with st.expander("요인 간 상관·중복 정도", expanded=False):
        st.dataframe(Fz.corr().round(2).style.format("{:.2f}").map(_corr_color),
                     width="stretch")
        st.dataframe(pd.DataFrame({"요인": list(vif), "VIF": list(vif.values())})
                     .style.format({"VIF": "{:.2f}"}), width="stretch",
                     hide_index=True)
        st.caption("VIF가 10을 넘으면 그 요인은 다른 요인들로 대부분 설명된다는 뜻입니다.")

    # ---------------- 영향 분석 ----------------
    if _used.empty:
        st.info("전망을 입력하면 여기에 영향 분석이 나타납니다.")
        return
    if _pre_dup:
        st.error("같은 요인에 전망이 겹쳐 있어 영향 분석을 진행할 수 없습니다 — "
                 + " · ".join(f"**{k}** ← {', '.join(v)}" for k, v in _pre_dup.items())
                 + "\n\n두 전망을 그대로 두면 같은 요인이 두 번 반영됩니다. "
                   "위 **2️⃣ 시장 전망**에서 하나만 **사용** 체크하거나 "
                   "방향을 **보합**으로 되돌린 뒤 다시 시도해주세요.")
        return

    st.subheader("5️⃣ 포트폴리오 영향 분석")
    st.info("**영향 등급(긍정·중립·부정)은 지금 입력한 자산들 사이의 상대평가**입니다. "
            "가장 크게 반응하는 자산을 기준으로 나머지를 견줍니다. 따라서 장기채처럼 "
            "민감도가 큰 자산이 섞이면 다른 자산이 상대적으로 중립에 가깝게 보일 수 "
            "있습니다.")
    w = live.set_index("티커")["비중(%)"].reindex(use_tk).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=use_tk)
    w = w / w.sum() * 100
    B = bdf.set_index("티커")

    # 전망을 요인 방향으로 옮긴다
    fdir = {}
    vdesc = []
    _dbl = {}          # 같은 요인에 여러 전망이 몰렸는지 확인용
    for _, r_ in _used.iterrows():
        var = str(r_["전망 변수"])
        fac, mult, note = MARKET_VIEWS.get(var, (None, 0, ""))
        if fac is None or fac not in F.columns:
            continue
        d = VIEW_DIRS.get(str(r_["방향"]), 0.0)
        conf = float(r_["확신도(%)"] or 0) / 100.0
        hz = VIEW_HZ_WEIGHT.get(str(r_["전망 기간"]), 1.0) if use_hz_weight else 1.0
        fdir[fac] = fdir.get(fac, 0.0) + d * conf * mult * hz
        _dbl.setdefault(fac, []).append(var)
        vdesc.append(f"**{var}** {r_['방향']} · 확신도 {r_['확신도(%)']:.0f}% · "
                     f"{r_['전망 기간']}"
                     + (f"(가중 {hz:.2f})" if use_hz_weight else "(기간조정 미사용)")
                     + (f" · {note}" if note else ""))
    if not fdir:
        st.warning("입력한 전망을 지금 요인 목록으로 표현할 수 없습니다. "
                   "추가 요인을 켜보세요.")
        return

    st.markdown("**반영한 전망**\n\n" + "\n".join(f"- {v}" for v in vdesc))
    _dup = {k: v for k, v in _dbl.items() if len(v) > 1}
    if _dup:
        st.warning("같은 요인에 전망이 여러 개 겹쳤습니다 — "
                   + " · ".join(f"**{k}** ← {', '.join(v)}" for k, v in _dup.items())
                   + "\n\n두 전망이 더해져 영향이 실제보다 크게 잡힐 수 있습니다. "
                   "예를 들어 정책금리와 장기금리를 함께 하락으로 두면 금리 요인이 "
                   "두 번 반영됩니다. 하나만 쓰시거나 확신도를 낮춰주세요.")
    st.dataframe(pd.DataFrame([{"요인": k, "합산 방향": v,
                                "반영된 전망": ", ".join(_dbl.get(k, []))}
                               for k, v in fdir.items()])
                 .style.format({"합산 방향": "{:+.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("**합산 방향**은 방향 × 확신도 × 기간가중을 더한 값입니다. "
               "이 값이 각 자산의 민감도와 곱해져 영향 점수가 됩니다.")

    irows = []
    for t_ in use_tk:
        eff, tot = {}, 0.0
        for fac, dir_ in fdir.items():
            b = float(B.loc[t_, fac]) if fac in B.columns else np.nan
            e = b * dir_ if np.isfinite(b) else np.nan
            eff[f"{fac} 영향"] = e
            if np.isfinite(e):
                tot += e
        _r2 = float(B.loc[t_, "R²"])
        _wt = r2_weight(_r2)
        irows.append({"티커": t_, "비중(%)": float(w[t_]), **eff,
                      "원점수": tot, "R²": _r2, "가중(√R²)": _wt,
                      "조정점수": tot * _wt, "신뢰도": B.loc[t_, "신뢰도"]})
    idf = pd.DataFrame(irows).sort_values("조정점수", ascending=False)
    # 자산군마다 민감도 크기가 달라(채권은 듀레이션 탓에 큼) 절대 기준으로는
    # 전부 '긍정'이 된다. 포트폴리오 안에서의 상대 크기로 등급을 매긴다.
    # 판정 유보(R²<0.10) 자산은 계수가 우연히 클 수 있어 기준에서 뺀다 —
    # 안 빼면 판정 가능한 자산들의 등급 기준 자체가 왜곡된다.
    _valid_r2 = idf["R²"].notna() & (idf["R²"] >= 0.10)
    _scale = (float(idf.loc[_valid_r2, "조정점수"].abs().max())
              if _valid_r2.any() else 1.0) or 1.0
    idf["종합 영향"] = [
        "판정 유보" if (not np.isfinite(r2_) or r2_ < 0.10)
        else impact_label(v, _scale)
        for v, r2_ in zip(idf["조정점수"], idf["R²"])]
    idf = idf[[c for c in idf.columns if c not in ("종합 영향", "신뢰도")]
              + ["종합 영향", "신뢰도"]]
    _f2 = {"비중(%)": "{:.2f}", "원점수": "{:+.2f}", "R²": "{:.2f}",
           "가중(√R²)": "{:.2f}", "조정점수": "{:+.2f}"}
    _f2.update({c: "{:+.2f}" for c in idf.columns if c.endswith("영향")
                and c != "종합 영향"})
    st.dataframe(idf.style.format(_f2, na_rep="-")
                 .map(_heat_color, subset=[c for c in idf.columns
                                           if c.endswith("영향") and c != "종합 영향"]),
                 width="stretch", hide_index=True)
    st.caption("**원점수(%)** 는 민감도 × 전망 방향 × 확신도를 더한 값이고, "
               "**조정점수**는 거기에 설명력 가중(√R²)을 곱한 값입니다.\n\n"
               "설명력이 낮은 자산은 민감도 자체가 우연일 수 있어, 그만큼 판단 강도를 "
               "낮춥니다. **R²가 0.10 미만이면 '판정 유보'** 로 두고 방향을 말하지 "
               "않습니다. 아래 유불리 비중도 조정점수 기준입니다.\n\n"
               "절대적인 수익률 예측이 아니라 **자산 간 상대적인 유불리**를 가늠하는 "
               "지표로 보세요.")

    # 포트폴리오 노출
    _ok = idf["종합 영향"] != "판정 유보"
    _pos = float(idf[_ok & (idf["조정점수"] > 0.18 * _scale)]["비중(%)"].sum())
    _neg = float(idf[_ok & (idf["조정점수"] < -0.18 * _scale)]["비중(%)"].sum())
    _hold = float(idf[~_ok]["비중(%)"].sum())
    _neu = 100.0 - _pos - _neg - _hold
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("전망에 유리", f"{_pos:.1f}%")
    k2.metric("중립", f"{_neu:.1f}%")
    k3.metric("불리", f"{_neg:.1f}%")
    k4.metric("판정 유보", f"{_hold:.1f}%",
              help="설명력이 낮아 방향을 판단하지 않은 자산의 비중입니다.")
    if _hold > 20:
        st.info(f"포트폴리오의 **{_hold:.1f}%** 는 지금 요인 세트로 설명되지 않아 "
                f"판단을 유보했습니다. 이 부분은 다른 근거로 판단하셔야 합니다.")

    # 가중 평균은 판정 가능한 자산만으로 계산한다 (판정 유보는 비중까지 왜곡시킴)
    _ok_df = idf[_ok]
    _ok_wsum = float(_ok_df["비중(%)"].sum())
    _wavg = (float((_ok_df["조정점수"] * _ok_df["비중(%)"]).sum() / _ok_wsum)
             if _ok_wsum > 1e-9 else np.nan)
    _wavg_txt = (f"{_wavg:+.2f} (판정 가능한 자산 기준)" if np.isfinite(_wavg)
                 else "계산 불가 (판정 가능한 자산 없음)")
    if not np.isfinite(_wavg):
        st.info("모든 자산이 판정 유보라 가중 평균 점수를 계산할 수 없습니다.")
    elif _neg > 50:
        st.warning(f"입력한 전망이 맞다면 포트폴리오의 **{_neg:.1f}%** 가 불리한 "
                   f"자산입니다. 가중 평균 점수는 **{_wavg_txt}** 입니다.")
    elif _pos > 50:
        st.success(f"전망에 유리한 자산이 **{_pos:.1f}%** 입니다. "
                   f"가중 평균 점수 **{_wavg_txt}**.")
    else:
        st.info(f"유불리가 뚜렷하지 않습니다. 가중 평균 점수 **{_wavg_txt}**.")

    fb = go.Figure(go.Bar(
        x=idf["조정점수"], y=idf["티커"], orientation="h",
        marker_color=["#94a3b8" if l == "판정 유보" else
                      ("#0d9488" if v > 0 else "#dc2626")
                      for v, l in zip(idf["조정점수"], idf["종합 영향"])],
        text=[f"{v:+.2f} ({l})" for v, l in zip(idf["조정점수"], idf["종합 영향"])],
        textposition="outside"))
    fb.update_layout(height=90 + 42 * len(idf), margin=dict(l=0, r=90, t=30, b=0),
                     xaxis=dict(title="조정 영향 점수 (설명력 가중 후)"),
                     yaxis=dict(autorange="reversed"))
    st.plotly_chart(fb, width="stretch")

    # ---------------- 표현되지 않은 전망 ----------------
    st.subheader("6️⃣ 전망이 포트폴리오에 표현되었는가")
    urows = []
    for fac, dir_ in fdir.items():
        expo = float(sum(float(B.loc[t_, fac]) * float(w[t_]) / 100
                         for t_ in use_tk
                         if fac in B.columns and np.isfinite(B.loc[t_, fac])))
        # 전망 방향과 노출이 같은 쪽인지
        aligned = expo * dir_
        urows.append({"요인": fac, "전망 방향": f"{dir_:+.2f}",
                      "포트폴리오 노출": expo,
                      "정렬": "✅ 유리한 방향" if aligned > 0.05
                      else ("⚠️ 반대 방향" if aligned < -0.05 else "– 거의 없음")})
    udf = pd.DataFrame(urows)
    st.dataframe(udf.style.format({"포트폴리오 노출": "{:+.2f}"}),
                 width="stretch", hide_index=True)
    _none = udf[udf["정렬"].astype(str).str.startswith("–")]["요인"].tolist()
    if _none:
        st.info(f"**{', '.join(_none)}** 전망은 지금 포트폴리오에 거의 표현되지 "
                f"않습니다. 그 전망을 실제로 반영하려면 해당 요인에 민감한 자산이 "
                f"필요합니다.\n\n"
                f"다만 **자동으로 편입하지는 않습니다.** 어떤 자산을 넣을지는 "
                f"➕ 자산 추가 효과 화면에서 검토해보세요.")

    # ---------------- 내보내기 ----------------
    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            mv.to_excel(xw, sheet_name="1_전망", index=False)
            bdf.to_excel(xw, sheet_name="2_민감도(표준화)", index=False)
            rawdf.to_excel(xw, sheet_name="2b_민감도(원단위)", index=False)
            idf.to_excel(xw, sheet_name="3_영향분석", index=False)
            udf.to_excel(xw, sheet_name="4_전망표현", index=False)
            Fz.corr().round(3).to_excel(xw, sheet_name="5_요인상관")
            pd.DataFrame(list({
                "추정 기간": f"{f_yrs}년", "계산 주기": f_freq,
                "요인": ", ".join(F.columns),
                "표본": f"{len(F):,}개",
                "기간": f"{F.index[0].date()} ~ {F.index[-1].date()}",
                "기준 통화": base_ccy,
                "기간조정 사용": "예" if use_hz_weight else "아니오",
            }.items()), columns=["항목", "값"]).to_excel(xw, sheet_name="6_설정",
                                                       index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"macro_view_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_mv", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")

    assumptions_panel(period=f"{F.index[0].date()} ~ {F.index[-1].date()}",
                      extra=[("민감도 추정", f"{f_yrs}년 · {f_freq} · 다중회귀 · 표준화"),
                             ("요인", ", ".join(F.columns)),
                             ("환율", "제외 (현지통화 기준) — 달러 요인을 따로 재기 위함"),
                             ("전망 기간", "가중치로 반영 (3개월 0.6 ~ 2년 1.15)"
                              if use_hz_weight else
                              "기록용만 — 기간조정 미사용, 점수엔 반영 안 함")])
    st.caption("민감도는 **과거 평균**입니다. 실제로는 시기에 따라 크게 달라지고, "
               "위기에는 모든 자산이 함께 움직여 요인 구분이 흐려집니다. "
               "이 화면은 **방향을 가늠하는 진단**이지 수익률 예측이 아닙니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")


def render_view_alloc(base_ccy, start_date, end_date, use_div,
                      fx_hedge, gap_fill, rf_rate, cost_bp):
    """전망을 배분에 반영하는 두 가지 방식을 한 화면에서 고르게 한다."""
    st.title("🧭 뷰 기반 자산배분")
    mode = st.radio(
        "전망을 어떤 방식으로 반영할까요?",
        ["🌍 시장·거시 전망", "📈 자산수익률 전망 (Black-Litterman)"],
        key="_va_mode",
        captions=["금리·경기·물가·달러 등 **시장환경**에 대한 전망을 넣고, "
                  "그것이 지금 포트폴리오에 어떤 영향을 주는지 봅니다. "
                  "장기채를 보유하지 않아도 금리 전망을 넣을 수 있습니다.",
                  "특정 자산의 **예상 수익률**이나 자산 간 우위를 직접 넣습니다. "
                  "블랙-리터만으로 배분을 산출합니다."])
    st.divider()
    if mode.startswith("🌍"):
        render_macro_view(base_ccy, start_date, end_date, use_div,
                          fx_hedge, gap_fill, rf_rate)
    else:
        render_black_litterman(base_ccy, start_date, end_date, use_div,
                               fx_hedge, gap_fill, rf_rate, cost_bp)


def render_black_litterman(base_ccy, start_date, end_date, use_div,
                           fx_hedge, gap_fill, rf_rate, cost_bp):
    st.subheader("📈 자산수익률 전망 (Black-Litterman)")
    st.caption("시장 균형에서 출발해, 내가 가진 전망(뷰)을 **확신도만큼만** 반영해 "
               "자산배분을 산출합니다. 뷰를 넣지 않으면 기준 비중이 그대로 나옵니다.")

    vkey = "_bl_views"
    if vkey not in st.session_state:
        st.session_state[vkey] = pd.DataFrame([_bl_view_blank()])[BL_VIEW_COLS]

    st.subheader("1️⃣ 기준 포트폴리오 (Reference Portfolio)")
    live, tickers, bad = render_ticker_table(
        state_key="_bl_df", editor_key="_bl_editor", gen_key="_bl_gen",
        cols=BL_COLS, weight_col="기준 비중(%)", title="뷰 기반 자산배분",
        min_rows=2, max_rows=20, count_label="자산 수",
        help_text="**기준 비중**은 출발점이 되는 배분입니다. 시가총액 비중이 "
                  "이상적이지만, 실무에서는 벤치마크 비중이나 현재 전략적 배분을 넣어도 "
                  "됩니다. 뷰가 없으면 이 비중이 그대로 결과가 됩니다.",
        default_rows=[{"티커": t_, "기준 비중(%)": w_} for t_, w_ in
                      [("SPY", 35.0), ("EFA", 18.0), ("EEM", 10.0),
                       ("SHY", 12.0), ("IEF", 12.0), ("TLT", 8.0),
                       ("GLD", 5.0)]])
    wsum = float(pd.to_numeric(live["기준 비중(%)"], errors="coerce").fillna(0).sum())

    ok_in, msgs = validate_setup(tickers, bad, live["기준 비중(%)"], min_n=2,
                                 need_weight=True, label="자산")
    show_msgs(msgs)

    # ---------------- 뷰 ----------------
    st.subheader("2️⃣ 전망 입력 (Views)")
    mode = st.radio("입력 방식", ["간단 모드", "전문가 모드"], horizontal=True,
                    key="_bl_mode",
                    help="간단 모드는 자산별로 긍정·중립·부정만 고르면 됩니다. "
                         "전문가 모드는 수치와 상대 뷰를 직접 지정합니다.")

    ved = st.session_state[vkey]
    simple_df = None
    conf_pct = 60
    strength = 1.0

    if mode == "간단 모드":
        st.caption("각 자산의 전망만 고르세요. 수치는 **자산의 변동성에 맞춰 자동 환산**됩니다. "
                   "변동성이 큰 주식의 '긍정'과 변동성이 작은 채권의 '긍정'은 "
                   "다른 크기로 반영됩니다.\n\n"
                   "⚠️ 이 환산 규칙(강한 긍정 = 변동성의 20% 등)은 **사용 편의를 위해 "
                   "정한 내부 기준**이며, 블랙-리터만 이론이 정한 값이 아닙니다. "
                   "정확한 수치를 넣으려면 전문가 모드를 쓰세요.")
        with st.expander("💡 금리 전망은 어떻게 넣나요?", expanded=False):
            st.markdown(
                "금리 자체(`^TNX` 등)를 자산으로 넣으면 계산이 왜곡됩니다. 금리가 4%에서 "
                "5%로 오르면 '25% 상승'으로 잡히지만, 채권 투자자에게는 오히려 손실이기 "
                "때문입니다.\n\n"
                "**대신 만기가 다른 채권 ETF의 전망으로 표현하세요.**\n\n"
                "| 전망 | 단기 `SHY` | 중기 `IEF` | 장기 `TLT` |\n|---|---|---|---|\n"
                "| 금리 인하 예상 | 중립 | 긍정 | **강한 긍정** |\n"
                "| 금리 인상 예상 | 긍정 | 중립 | **강한 부정** |\n"
                "| 장단기 역전 해소 | 부정 | 중립 | 긍정 |\n\n"
                "만기가 길수록 금리 변화에 민감하므로(듀레이션), 같은 금리 전망이라도 "
                "**장기채에 더 강한 전망**을 주는 것이 자연스럽습니다. "
                "기본 구성에 `SHY`·`IEF`·`TLT` 를 넣어둔 이유입니다.")
        skey = "_bl_simple"
        if skey not in st.session_state or \
                list(st.session_state[skey]["티커"]) != tickers:
            st.session_state[skey] = pd.DataFrame(
                [{"티커": t, "전망": "중립"} for t in tickers])[BL_SIMPLE_COLS]
        simple_df = st.data_editor(
            st.session_state[skey], num_rows="fixed", width="stretch",
            key="_bl_simple_editor", column_order=BL_SIMPLE_COLS, hide_index=True,
            column_config={
                "티커": st.column_config.TextColumn("티커", disabled=True, width="small"),
                "전망": st.column_config.SelectboxColumn(
                    "전망", options=list(BL_SIMPLE), width="medium"),
            })
        st.session_state[skey] = simple_df
        f1, f2 = st.columns(2)
        conf_pct = f1.slider("전망 확신도 (%)", 20, 100, 60, step=5, key="_bl_conf_pct",
                             help="낮으면 기준 비중에서 거의 움직이지 않고, "
                                  "높을수록 전망 쪽으로 크게 기울어집니다.")
        strength = f2.slider("전망 강도", 0.5, 2.0, 1.0, step=0.1, key="_bl_strength",
                             help="1.0 기준으로 '강한 긍정'은 해당 자산 변동성의 20% "
                                  "만큼 초과수익을 기대한다는 뜻입니다.")
    else:
        st.caption("**절대** — 특정 자산이 연 몇 %를 낼 것이다.  |  "
                   "**상대** — 자산 A가 자산 B보다 연 몇 %p 나을 것이다.\n\n"
                   "확신도가 낮으면 기준 비중에서 거의 움직이지 않고, 높을수록 뷰 쪽으로 "
                   "크게 기울어집니다. 비워두면 뷰 없이 계산합니다.")
        ved = st.data_editor(
            st.session_state[vkey], num_rows="dynamic", width="stretch",
            key="_bl_view_editor", column_order=BL_VIEW_COLS, hide_index=True,
            column_config={
                "유형": st.column_config.SelectboxColumn("유형", options=["절대", "상대"],
                                                       width="small"),
                "자산 A": st.column_config.SelectboxColumn("자산 A", options=tickers,
                                                        width="medium"),
                "자산 B": st.column_config.SelectboxColumn(
                    "자산 B (상대일 때)", options=tickers, width="medium"),
                "연 수익률(%)": st.column_config.NumberColumn(
                    "연 수익률(%)", step=0.5, format="%.2f", width="small",
                    help="절대: 예상 연 수익률 · 상대: A가 B보다 얼마나 나은지(%p)"),
                "확신도": st.column_config.SelectboxColumn("확신도", options=list(BL_CONF),
                                                        width="small"),
            })
        st.session_state[vkey] = ved

    # ---------------- 설정 ----------------
    st.subheader("3️⃣ 모형 설정 (Model Settings)")
    h1, h2 = st.columns([1, 3])
    horizon = h1.selectbox("전망 기간", ["6개월", "1년", "2년", "3년"], index=1,
                           key="_bl_horizon",
                           help="이 전망이 어느 기간을 내다본 것인지입니다.")
    use_bl_hz = h1.checkbox(
        "⏱ 기간조정 사용", value=False, key="_bl_hz_on",
        help="켜면 짧은 기간(6개월)의 전망은 확신도를 더 크게 낮추고, 긴 "
             "기간(3년)은 덜 낮춥니다 (6개월 1.6배 ~ 3년 0.7배). 이 배수는 "
             "이론적으로 정해진 값이 아니라 휴리스틱이라 **기본은 꺼져 "
             "있습니다**. 꺼두면 전망 기간은 기록용으로만 쓰이고, 확신도는 "
             "위에서 고른 값 그대로 반영됩니다.")
    if use_bl_hz:
        h2.caption("전망 기간이 짧으면 **같은 확신도라도 반영 강도를 낮춥니다** "
                   "(6개월 1.6배 ~ 3년 0.7배, 휴리스틱). 6개월 앞을 3년 앞만큼 "
                   "확신하기는 어렵다는 가정입니다. "
                   "입력하신 수익률은 **연율 기준**으로 해석합니다.")
    else:
        h2.caption("**기간조정이 꺼져 있어** 전망 기간은 기록용으로만 쓰이고 "
                   "확신도 반영 강도에는 영향을 주지 않습니다. "
                   "입력하신 수익률은 **연율 기준**으로 해석합니다.")

    p1, p2, p3 = st.columns(3)
    auto_delta = p1.checkbox("위험회피계수 자동 추정", value=True, key="_bl_autod",
                             help="시장 포트폴리오의 과거 초과수익과 변동성에서 "
                                  "δ = (수익률 − 무위험) / 변동성² 로 추정합니다.")
    delta_in = p1.number_input("위험회피계수 δ", 0.5, 10.0, 2.5, step=0.1,
                               key="_bl_delta", disabled=auto_delta,
                               help="클수록 위험을 싫어해 보수적인 배분이 나옵니다. "
                                    "통상 2~3을 씁니다.")
    # τ 는 이 구현에서 Ω = 확신도 × P(τΣ)Pᵀ 로 두기 때문에 사후 기대수익률에서
    # 상쇄된다. 조절해도 결과가 같은 값을 입력받으면 오해를 주므로 고정값으로 둔다.
    tau = 0.05
    p2.caption("**뷰 반영 강도는 확신도로 조절합니다.**\n\n"
               "블랙-리터만의 τ는 이 구현에서 뷰 불확실성 Ω와 함께 상쇄되어 "
               "결과에 영향을 주지 않으므로 입력을 두지 않았습니다 (0.05 고정).")
    lookback = p3.selectbox("공분산 추정 기간", list(TRAIN_WINDOWS), index=4,
                            key="_bl_look",
                            help="상관·변동성을 추정할 과거 구간입니다. "
                                 "길수록 안정적이지만 최근 변화를 늦게 반영합니다.")
    q1, q2 = st.columns(2)
    wmax_bl = q1.slider("자산별 최대 비중 (%)", 10, 100, 60, step=5, key="_bl_wmax")
    allow_short = q2.checkbox("공매도 허용", value=False, key="_bl_short",
                              help="허용하면 음(−)의 비중이 나올 수 있습니다.")

    run_bl = st.button("🧭 자산배분 계산", type="primary", width="stretch",
                       disabled=bool(bad) or len(tickers) < 2 or wsum <= 0)
    if run_bl:
        st.session_state["_bl_has_run"] = True
    if not st.session_state.get("_bl_has_run"):
        st.info("👆 자산과 기준 비중을 입력한 뒤 **자산배분 계산**을 눌러주세요. "
                "뷰는 비워두셔도 됩니다.")
        return

    # ---------------- 데이터 ----------------
    try:
        with st.spinner("데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(tickers, start_date, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    use_tk = [t for t in tickers if t in prices.columns]
    if len(use_tk) < 2:
        st.error("최소 2개 자산의 데이터가 필요합니다.")
        return
    miss = [t for t in tickers if t not in use_tk]
    if miss:
        st.warning(f"데이터가 없어 제외합니다: {', '.join(miss)}")

    lb = TRAIN_WINDOWS[lookback]
    px = prices[use_tk] if lb is None else \
        prices[use_tk].loc[max(prices.index[0], prices.index[-1] - lb):]
    R = px.pct_change().dropna()
    if len(R) < MIN_TRAIN_DAYS:
        st.error(f"공분산 추정 표본이 {len(R)}일뿐입니다. 기간을 늘려주세요.")
        return
    cov = (R.cov() * TRADING_DAYS).values

    w_mkt = live.set_index("티커")["기준 비중(%)"].reindex(use_tk).fillna(0).values
    if w_mkt.sum() <= 0:
        w_mkt = np.ones(len(use_tk))
    w_mkt = w_mkt / w_mkt.sum()

    if auto_delta:
        r_mkt = (R.values @ w_mkt)
        mu_m = float(np.mean(r_mkt) * TRADING_DAYS)
        var_m = float(np.var(r_mkt) * TRADING_DAYS)
        delta = float(np.clip((mu_m - rf_rate) / var_m, 0.5, 10.0)) if var_m > 1e-12 else 2.5
    else:
        delta = float(delta_in)

    # ---------------- 뷰 파싱 ----------------
    P, Q, conf, vdesc = [], [], [], []
    sig = np.sqrt(np.diag(cov))          # 자산별 연 변동성

    # 전망 기간이 짧을수록 불확실성을 키운다 (1년 기준 1.0) — 기간조정을 켰을 때만
    HZ_MULT = {"6개월": 1.6, "1년": 1.0, "2년": 0.8, "3년": 0.7}
    hz = HZ_MULT.get(horizon, 1.0) if use_bl_hz else 1.0

    if mode == "간단 모드" and simple_df is not None:
        scale = float(np.clip((100 - conf_pct) / max(conf_pct, 1) * hz, 0.02, 12.0))
        for _, r in simple_df.iterrows():
            a = str(r.get("티커") or "").strip()
            v = str(r.get("전망") or "중립")
            k = BL_SIMPLE.get(v, 0.0) * float(strength)
            if a not in use_tk or abs(k) < 1e-9:
                continue
            i = use_tk.index(a)
            row = np.zeros(len(use_tk)); row[i] = 1.0
            # 균형 수익률 대비 ±(변동성 × k) 만큼 초과/미달을 기대
            base_pi = float(implied_returns(delta, cov, w_mkt)[i])
            q = base_pi + k * float(sig[i])
            P.append(row); Q.append(q); conf.append(scale)
            vdesc.append(f"**{a}** {v} — 균형 {base_pi*100:.2f}% 대비 "
                         f"{k*sig[i]*100:+.2f}%p (확신도 {conf_pct}%)")
        vv = pd.DataFrame()
    else:
        vv = ved.copy()

    for _, r in vv.iterrows():
        a = str(r.get("자산 A") or "").strip()
        val = pd.to_numeric(r.get("연 수익률(%)"), errors="coerce")
        if a not in use_tk or pd.isna(val):
            continue
        row = np.zeros(len(use_tk))
        if str(r.get("유형")) == "상대":
            b = str(r.get("자산 B") or "").strip()
            if b not in use_tk or b == a:
                continue
            row[use_tk.index(a)] = 1.0
            row[use_tk.index(b)] = -1.0
            vdesc.append(f"{a} 가 {b} 보다 연 {val:+.2f}%p 우위 (확신 {r.get('확신도')})")
        else:
            row[use_tk.index(a)] = 1.0
            vdesc.append(f"{a} 연 수익률 {val:.2f}% (확신 {r.get('확신도')})")
        P.append(row); Q.append(float(val) / 100.0)
        conf.append(BL_CONF.get(str(r.get("확신도")), 1.0) * hz)

    pi, post = black_litterman(cov, w_mkt, delta, tau,
                               P if P else None, Q if Q else None,
                               conf if conf else None)
    _bl_lo = -wmax_bl / 100.0 if allow_short else 0.0
    try:
        w_bl = bl_weights(post, cov, delta, _bl_lo, wmax_bl / 100.0, allow_short)
    except Exception as ex:
        st.error(f"🚫 배분을 계산하지 못했습니다: {ex}")
        return
    if w_bl is None:
        st.error("🚫 제약을 만족하는 배분을 찾지 못했습니다. "
                 "자산별 최대 비중을 높여보세요.")
        return

    st.success(f"✅ 계산 완료 · 자산 {len(use_tk)}개 · 뷰 **{len(P)}개** · "
               f"전망 기간 **{horizon}** · "
               f"위험회피계수 δ = **{delta:.2f}**{' (자동 추정)' if auto_delta else ''} · "
               f"공분산 {lookback} ({R.index[0].date()} ~ {R.index[-1].date()}, {len(R):,}일)")
    if vdesc:
        st.markdown("**반영된 전망**\n\n" + "\n".join(f"- {v}" for v in vdesc))
    else:
        st.info("뷰가 없어 **기준 비중이 그대로** 결과가 됩니다. "
                "위 표에 전망을 넣으면 배분이 조정됩니다.")

    # ---------------- 기대수익률 ----------------
    st.subheader("4️⃣ 기대수익률 (Expected Returns)")
    st.caption("**균형**은 기준 비중이 최적이 되도록 역산한 값이고, "
               "**사후**는 거기에 내 전망을 확신도만큼 섞은 값입니다.")
    edf = pd.DataFrame({
        "티커": use_tk,
        "종목명": [meta.get(t, {}).get("name", t) for t in use_tk],
        "균형 기대수익률(%)": pi * 100,
        "사후 기대수익률(%)": post * 100,
        "변화(%p)": (post - pi) * 100,
        "변동성(연,%)": np.sqrt(np.diag(cov)) * 100,
    })
    st.dataframe(edf.style.format({"균형 기대수익률(%)": "{:.2f}",
                                   "사후 기대수익률(%)": "{:.2f}",
                                   "변화(%p)": "{:+.2f}",
                                   "변동성(연,%)": "{:.2f}"}),
                 width="stretch", hide_index=True)

    # ---------------- 배분 ----------------
    st.subheader("5️⃣ 제안 배분 (Suggested Allocation)")
    adf = pd.DataFrame({
        "티커": use_tk,
        "기준 비중(%)": w_mkt * 100,
        "제안 비중(%)": w_bl * 100,
    })
    adf["변화(%p)"] = adf["제안 비중(%)"] - adf["기준 비중(%)"]
    adf = adf.sort_values("제안 비중(%)", ascending=False)

    fa = go.Figure()
    fa.add_trace(go.Bar(y=adf["티커"], x=adf["기준 비중(%)"], name="기준 비중",
                        orientation="h", marker_color="#cbd5e1",
                        text=[f"{v:.1f}%" for v in adf["기준 비중(%)"]],
                        textposition="outside"))
    fa.add_trace(go.Bar(y=adf["티커"], x=adf["제안 비중(%)"], name="제안 비중",
                        orientation="h", marker_color="#4f46e5",
                        text=[f"{v:.1f}%" for v in adf["제안 비중(%)"]],
                        textposition="outside"))
    fa.update_layout(barmode="group", height=90 + 56 * len(adf),
                     margin=dict(l=0, r=60, t=30, b=0),
                     xaxis=dict(title=None, ticksuffix="%"),
                     yaxis=dict(autorange="reversed"),
                     legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fa, width="stretch")
    st.dataframe(adf.style.format({"기준 비중(%)": "{:.2f}", "제안 비중(%)": "{:.2f}",
                                   "변화(%p)": "{:+.2f}"}),
                 width="stretch", hide_index=True)

    # ---------------- 결과 해석 ----------------
    st.subheader("6️⃣ 이렇게 바뀐 이유 (What Changed and Why)")
    if not P:
        st.info("전망을 넣지 않아 기준 비중이 그대로 유지됐습니다.")
    else:
        chg = pd.Series(w_bl - w_mkt, index=use_tk) * 100
        viewed = set()
        for row in P:
            for i, x in enumerate(row):
                if abs(x) > 1e-9:
                    viewed.add(use_tk[i])

        lines = []
        lines.append(f"전망 **{len(P)}개**를 반영한 결과, 배분이 다음과 같이 바뀌었습니다.")
        up = chg[chg > 0.05].sort_values(ascending=False)
        dn = chg[chg < -0.05].sort_values()
        if len(up):
            lines.append("**늘어난 자산** · " + " · ".join(
                f"{k} {w_mkt[use_tk.index(k)]*100:.1f}% → "
                f"{w_bl[use_tk.index(k)]*100:.1f}% ({v:+.1f}%p)"
                for k, v in up.head(4).items()))
        if len(dn):
            lines.append("**줄어든 자산** · " + " · ".join(
                f"{k} {w_mkt[use_tk.index(k)]*100:.1f}% → "
                f"{w_bl[use_tk.index(k)]*100:.1f}% ({v:+.1f}%p)"
                for k, v in dn.head(4).items()))

        spill = [k for k, v in chg.items() if abs(v) > 0.5 and k not in viewed]
        if spill:
            lines.append(f"전망을 직접 주지 않은 **{', '.join(spill[:4])}** 도 함께 "
                         f"움직였습니다. 전망을 준 자산과 상관관계로 연결돼 있기 때문입니다.")

        r_m, v_m, s_m = port_perf(w_mkt, post, cov, rf_rate)
        r_b, v_b, s_b = port_perf(w_bl, post, cov, rf_rate)
        lines.append(f"그 결과 기대수익률은 {r_m*100:.2f}% → **{r_b*100:.2f}%**, "
                     f"예상 변동성은 {v_m*100:.2f}% → **{v_b*100:.2f}%**, "
                     f"예상 샤프는 {s_m:.2f} → **{s_b:.2f}** 가 됐습니다.")
        st.markdown("\n\n".join(lines))

        top_chg = chg.abs().max()
        if top_chg > 25:
            st.warning(f"비중 변화가 최대 **{top_chg:.1f}%p** 로 큽니다. "
                       f"전망이 빗나가면 손실도 그만큼 커집니다. "
                       f"확신도를 낮추거나 자산별 최대 비중을 조이면 변화폭이 줄어듭니다.")

        with st.expander("전망별 영향 자세히 보기", expanded=False):
            st.caption("각 전망을 **하나만** 반영했을 때의 비중 변화입니다. "
                       "여러 전망이 서로 상쇄되거나 증폭될 수 있어, 합이 전체 결과와 "
                       "정확히 일치하지는 않습니다.")
            rows_v = []
            for i in range(len(P)):
                _, po_i = black_litterman(cov, w_mkt, delta, tau,
                                          [P[i]], [Q[i]], [conf[i]])
                w_i = bl_weights(po_i, cov, delta, _bl_lo, wmax_bl / 100.0, allow_short)
                d_i = (w_i - w_mkt) * 100
                j = int(np.argmax(np.abs(d_i)))
                rows_v.append({"전망": re.sub(r"\*\*", "", vdesc[i]),
                               "가장 크게 바뀐 자산": use_tk[j],
                               "변화(%p)": float(d_i[j]),
                               "총 변화량(%p)": float(np.abs(d_i).sum() / 2)})
            st.dataframe(pd.DataFrame(rows_v).style.format(
                {"변화(%p)": "{:+.2f}", "총 변화량(%p)": "{:.2f}"}),
                width="stretch", hide_index=True)

    # ---------------- 기대 성과 ----------------
    st.subheader("7️⃣ 기대 성과 비교 (Expected Performance)")
    st.caption("사후 기대수익률과 공분산으로 계산한 **예상치**입니다. "
               "과거 실현 성과가 아니라 모형이 내다본 값입니다.")
    rows = []
    for lbl, w_ in [("기준 비중", w_mkt), ("제안 배분", w_bl)]:
        r_, v_, s_ = port_perf(w_, post, cov, rf_rate)
        rows.append({"구성": lbl, "기대수익률(%)": r_ * 100,
                     "예상 변동성(%)": v_ * 100, "예상 샤프": s_})
    pdf_ = pd.DataFrame(rows).set_index("구성")
    if len(pdf_) == 2:
        pdf_.loc["차이"] = pdf_.iloc[1] - pdf_.iloc[0]
    st.dataframe(pdf_.style.format("{:.2f}"), width="stretch")

    with st.expander("과거 성과로 확인해보기 (참고용)", expanded=False):
        st.caption("두 배분을 **과거에 적용했다면** 어땠을지입니다. 지금의 전망을 과거에 "
                   "적용한 것이므로 검증이 아니라 참고 자료로만 보세요. "
                   "리밸런싱은 **분기별**로 고정해 계산합니다.")
        try:
            W_m = {t: float(x) * 100 for t, x in zip(use_tk, w_mkt)}
            # 공매도를 허용했으면 음수 비중도 그대로 넘긴다.
            # 예전에는 0으로 잘라내어 화면의 제안 비중과 검증 결과가 어긋났다.
            W_b = ({t: float(x) * 100 for t, x in zip(use_tk, w_bl)} if allow_short
                   else {t: float(x) * 100 for t, x in zip(use_tk, np.clip(w_bl, 0, None))})
            rows2 = {}
            for lbl, W in [("기준 비중", W_m), ("제안 배분", W_b)]:
                if sum(W.values()) <= 0:
                    continue
                rr = portfolio_returns(px, W, REBAL_QUARTER)
                _t2 = turnover_series(px, W, REBAL_QUARTER).reindex(rr.index).fillna(0)
                rr = apply_cost(rr, _t2, cost_bp)
                rows2[lbl] = perf_row(rr, rf_rate)
            if rows2:
                st.dataframe(pd.DataFrame(rows2).T.style.format("{:.2f}", na_rep="-"),
                             width="stretch")
                fh = go.Figure()
                for lbl, W in [("기준 비중", W_m), ("제안 배분", W_b)]:
                    if sum(W.values()) <= 0:
                        continue
                    _r3 = portfolio_returns(px, W, REBAL_QUARTER)
                    _t3 = turnover_series(px, W, REBAL_QUARTER).reindex(_r3.index).fillna(0)
                    ec = equity_curve(apply_cost(_r3, _t3, cost_bp))
                    fh.add_trace(go.Scatter(x=ec.index, y=ec.values, name=lbl,
                                            line=dict(width=2)))
                fh.update_layout(height=340, hovermode="x unified",
                                 margin=dict(l=0, r=0, t=30, b=0),
                                 legend=dict(orientation="h", y=1.02, yanchor="bottom"))
                st.plotly_chart(fh, width="stretch")
        except Exception as ex:
            st.caption(f"과거 성과를 계산하지 못했습니다: {ex}")

    # ---------------- 다음 단계 ----------------
    st.divider()
    st.subheader("➡️ 다음 단계")
    st.caption("이 배분을 다른 화면에서 쓰려면 보관함에 담고, 그 화면의 표에서 "
               "**📥 붙여넣기** 하세요.")
    # 공매도를 허용했으면 음수 비중도 그대로 넘긴다
    _src_w = w_bl if allow_short else np.clip(w_bl, 0, None)
    _bl_rows = [{"티커": t, "비중": round(float(x) * 100, 2)}
                for t, x in zip(use_tk, _src_w) if abs(x) > 1e-6]
    if st.button("📋 제안 배분 복사", width="stretch"):
        if _bl_rows:
            st.session_state[CLIP_KEY] = {"rows": _bl_rows, "from": "뷰 기반 배분"}
            st.success(f"보관함에 담았습니다 · {clip_summary()}")

    st.markdown("**⚖️ 비교 후보로 담기**")
    if allow_short and (w_bl < -1e-6).any():
        st.warning("공매도(음수 비중)가 포함된 배분입니다. 비교 화면과 다른 화면의 "
                   "종목 표는 **음수 비중을 다루지 못하므로**, 담거나 붙여넣을 때 "
                   "결과가 달라질 수 있습니다.")
    plan_button("⚖️ 담기", _bl_rows, "뷰 기반 배분", "_plan_bl",
                default_name="뷰 반영안")

    # ---------------- 내보내기 ----------------
    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            adf.to_excel(xw, sheet_name="1_제안배분", index=False)
            edf.to_excel(xw, sheet_name="2_기대수익률", index=False)
            pdf_.to_excel(xw, sheet_name="3_기대성과")
            pd.DataFrame({"전망": vdesc or ["(없음)"]}).to_excel(
                xw, sheet_name="4_전망", index=False)
            pd.DataFrame(cov, index=use_tk, columns=use_tk).to_excel(
                xw, sheet_name="5_공분산")
            pd.DataFrame(list({
                "위험회피계수 δ": f"{delta:.3f}" + (" (자동)" if auto_delta else ""),
                "τ": f"{tau:.3f}", "전망 기간": horizon, "공분산 추정 기간": lookback,
                "표본": f"{len(R):,}일", "자산별 최대 비중": f"{wmax_bl}%",
                "공매도": "허용" if allow_short else "금지",
                "기준 통화": base_ccy, "무위험 수익률": f"{rf_rate*100:.2f}%",
                "기간조정 사용": "예" if use_bl_hz else "아니오",
            }.items()), columns=["항목", "값"]).to_excel(xw, sheet_name="6_설정",
                                                       index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"black_litterman_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_bl", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")
    assumptions_panel(period=f"{R.index[0].date()} ~ {R.index[-1].date()}",
                      extra=[("위험회피계수 δ", f"{delta:.2f}"
                              + (" (자동 추정)" if auto_delta else "")),
                             ("τ", f"{tau:.3f} (고정)"),
                             ("공분산 추정", lookback),
                             ("전망 기간", f"{horizon} · 가중치로 반영 (6개월 1.6~3년 0.7)"
                              if use_bl_hz else
                              f"{horizon} · 기록용만 — 기간조정 미사용, 확신도엔 반영 안 함"),
                             ("과거 성과 리밸런싱", "분기별 고정")])
    st.caption("이 모형은 기준 비중이 합리적인 출발점이라는 가정에 기대며, 공분산은 과거에서 "
               "추정합니다. 전망이 빗나가면 결과도 빗나갑니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")


# ======================================================================
# 거시 경기·물가 국면 (FRED)
# ======================================================================
FRED_SERIES = {"INDPRO": "산업생산", "PAYEMS": "비농업고용",
               "CPIAUCSL": "소비자물가", "CPILFESL": "근원물가",
               "RPI": "실질개인소득"}
# 경기·물가 국면 판정에는 쓰지 않고, 참고용 패널로만 따로 보여주는 지표.
FRED_RATE_CREDIT = {
    "FEDFUNDS": "기준금리", "DGS10": "미 국채 10년물",
    "T10Y2Y": "장단기 금리차(10Y-2Y)", "BAMLH0A0HYM2": "하이일드 신용스프레드",
    "BAMLC0A0CM": "투자등급 신용스프레드",
}
FRED_LIQUIDITY = {"M2SL": "M2 통화량", "WALCL": "Fed 총자산"}
FRED_EXTRA_SERIES = {**FRED_RATE_CREDIT, **FRED_LIQUIDITY}
MACRO_QUAD = {
    "경기상승·물가하락": {"유리": "주식 · 성장주 · 하이일드",
                   "불리": "원자재 · 현금", "색": "#dcfce7"},
    "경기상승·물가상승": {"유리": "원자재 · 가치주 · 부동산 · 물가연동채",
                   "불리": "장기채 · 성장주", "색": "#fef9c3"},
    "경기하락·물가상승": {"유리": "금 · 현금 · 단기채 · 에너지",
                   "불리": "주식 · 장기채", "색": "#fee2e2"},
    "경기하락·물가하락": {"유리": "장기채 · 국채 · 방어주",
                   "불리": "주식 · 원자재", "색": "#dbeafe"},
}
# 시장이 반영하는 방향을 읽는 대용 지표 (야후 데이터)
MARKET_SIGNALS = {
    "경기": [("XLY", "XLP", "경기소비재 / 필수소비재")],
    "물가": [("DBC", "TLT", "원자재 / 장기채")],
    "위험선호": [("HYG", "LQD", "하이일드 / 우량회사채"),
              ("SPY", "TLT", "주식 / 장기채")],
}


def rolling_z(s: pd.Series, win=60, minp=24) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std(ddof=1)
    return (s - m) / sd.replace(0, np.nan)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_fred(start="1990-01-01", end=None):
    """FRED 거시지표. 실패하면 (None, 오류문)을 돌려준다."""
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return None, "pandas-datareader 가 설치되어 있지 않습니다."
    try:
        _end = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y-%m-%d")
        df = pdr.DataReader(list(FRED_SERIES), "fred", start, _end)
        try:
            df = df.resample("ME").last()
        except ValueError:
            df = df.resample("M").last()
        return df.ffill(), ""
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:120]}"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_fred_extra(start="2000-01-01", end=None):
    """금리·신용스프레드·유동성 지표. 경기·물가 국면 계산과는 완전히 별개라,
    실패해도 나머지 화면에 영향을 주지 않는다."""
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return None, "pandas-datareader 가 설치되어 있지 않습니다."
    try:
        _end = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y-%m-%d")
        df = pdr.DataReader(list(FRED_EXTRA_SERIES), "fred", start, _end)
        try:
            df = df.resample("ME").last()
        except ValueError:
            df = df.resample("M").last()
        return df.ffill(), ""
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:120]}"


def fred_level_summary(df: pd.DataFrame, series: dict) -> pd.DataFrame:
    """레벨 지표(금리·스프레드)의 현재값과 3·12개월 전 대비 변화(%p)."""
    rows = []
    for code, label in series.items():
        if code not in df.columns:
            continue
        s = df[code].dropna()
        if s.empty:
            continue
        last_v = float(s.iloc[-1])
        rows.append({
            "지표": label,
            "기준월": s.index[-1].strftime("%Y-%m"),
            "현재값(%)": last_v,
            "3개월 전 대비(%p)": last_v - float(s.iloc[-4]) if len(s) > 3 else np.nan,
            "12개월 전 대비(%p)": last_v - float(s.iloc[-13]) if len(s) > 12 else np.nan,
        })
    return pd.DataFrame(rows)


def fred_yoy_summary(df: pd.DataFrame, series: dict) -> pd.DataFrame:
    """규모가 커서 레벨보다 전년동월비가 더 읽기 쉬운 지표(통화량 등)."""
    rows = []
    for code, label in series.items():
        if code not in df.columns:
            continue
        yoy = df[code].dropna().pct_change(12).dropna()
        if yoy.empty:
            continue
        rows.append({
            "지표": label,
            "기준월": yoy.index[-1].strftime("%Y-%m"),
            "전년동월비(%)": float(yoy.iloc[-1]) * 100,
            "3개월 전 전년동월비(%)": float(yoy.iloc[-4]) * 100 if len(yoy) > 3 else np.nan,
        })
    return pd.DataFrame(rows)


# ======================================================================
# 한국 거시지표 (ECOS, 한국은행)
# ======================================================================
# 통계표코드/항목코드는 하드코딩하지 않는다. 자산 티커와 달리 이 코드들은
# 사전 검증 수단이 없어, 틀린 코드를 넣으면 "그럴듯하지만 틀린 지표"가
# 조용히 나올 수 있다. 대신 ECOS가 제공하는 메타데이터 API로 이름 검색 →
# 세부 항목 선택 → 조회 순서를 강제한다.
ECOS_BASE = "https://ecos.bok.or.kr/api"
ECOS_FREQ_FMT = {"D": "%Y%m%d", "M": "%Y%m", "A": "%Y"}
ECOS_CYCLE_LABEL = {"D": "일", "M": "월", "Q": "분기", "A": "년"}

# 검색 없이 자동으로 보여주는 한국 금융여건 고정 지표: (통계표코드, 항목코드, 주기).
# 실제 배포 환경에서 라이브 조회로 검증된 값만 넣는다 — 추측으로 채우지 않는다.
# 원/달러(731Y003)는 표 자체가 일별(D) 전용이라 월 주기가 항목 목록에 아예
# 없다(실제 StatisticItemList 응답으로 확인) — 일별로 받아 월말값으로 리샘플한다.
ECOS_KR_LEVEL = {"기준금리": ("722Y001", "0101000", "M"),
                 "원/달러": ("731Y003", "0000003", "D")}
ECOS_KR_YOY = {"소비자물가(CPI)": ("901Y009", "0", "M"),
               "전산업생산지수": ("901Y033", "", "M"),
               "M2 통화량": ("161Y006", "BBHA00", "M")}


def ecos_secret_key() -> str:
    """Streamlit Cloud Secrets에 ECOS_API_KEY가 등록돼 있으면 그걸 쓴다.
    없으면 빈 문자열을 돌려주고, 호출부가 수동 입력창으로 대신한다."""
    try:
        return str(st.secrets.get("ECOS_API_KEY", "")).strip()
    except Exception:
        return ""


def _ecos_get(service: str, api_key: str, path_parts: list, timeout: int = 15) -> list:
    """
    ECOS REST 공통 호출. 성공 시 row 리스트를 돌려주고, 실패하면 예외를 던진다.
    오류 응답은 최상위(RESULT)나 서비스명 아래(예: {"StatisticSearch":{"RESULT":...}})
    양쪽 다 올 수 있어 둘 다 확인한다.
    """
    url = "/".join([ECOS_BASE, service, api_key, "json", "kr"]
                   + [str(p) for p in path_parts])
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    err = data.get("RESULT") or (data.get(service) or {}).get("RESULT")
    if err:
        raise ValueError(f"ECOS 오류({err.get('CODE', '?')}): "
                         f"{err.get('MESSAGE', '알 수 없는 오류')}")
    body = data.get(service)
    if not isinstance(body, dict) or "row" not in body:
        raise ValueError(f"ECOS 응답 형식이 예상과 다릅니다: {str(data)[:200]}")
    return body["row"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def ecos_table_list(api_key: str) -> list:
    """전체 통계표 목록(통계표코드·통계명). 이름으로 검색하기 위한 메타데이터."""
    return _ecos_get("StatisticTableList", api_key, [1, 1000])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def ecos_item_list(api_key: str, stat_code: str) -> list:
    """
    특정 통계표 안의 세부 항목 목록.

    실제 응답 스키마(라이브 API로 확인함 — 문서/블로그 추정이 아니라
    사용자가 받아온 실제 JSON 기준): 각 행은 자기 자신의 ``ITEM_CODE``/
    ``ITEM_NAME``, 부모의 ``P_ITEM_CODE``/``P_ITEM_NAME``(최상위는
    null), 그리고 그 항목이 지원하는 ``CYCLE``(D/M/Q/A)을 담고 있다.
    같은 항목이 지원 주기마다 행이 반복된다(예: 월·분기·년 3번).
    번호가 붙은 ITEM_CODE1~4 필드는 이 응답에 없다 — 계층은 트리
    (P_ITEM_CODE)로 표현되지, 여러 축이 아니다.
    """
    return _ecos_get("StatisticItemList", api_key, [1, 1000, stat_code])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ecos_series(api_key: str, stat_code: str, freq: str, start: str, end: str,
                item_code: str = "") -> pd.Series:
    """실제 시계열. freq: D(일)/M(월)/Q(분기)/A(년). start/end는 해당 주기 포맷."""
    rows = _ecos_get("StatisticSearch", api_key,
                     [1, 1000, stat_code, freq, start, end, item_code])
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    vals = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    if freq == "Q":
        # "2003Q4" 형태 -> 그 분기 첫 달로 근사 변환한다(strptime이 못 읽는 포맷).
        q_month = {"1": 1, "2": 4, "3": 7, "4": 10}

        def _parse_q(t):
            t = str(t)
            if "Q" not in t:
                return pd.NaT
            y, q = t.split("Q")
            return pd.Timestamp(year=int(y), month=q_month.get(q, 1), day=1)
        idx = df["TIME"].map(_parse_q)
    else:
        idx = pd.to_datetime(df["TIME"], format=ECOS_FREQ_FMT.get(freq, "%Y%m"),
                             errors="coerce")
    s = pd.Series(vals.values, index=idx).dropna()
    return s.sort_index()


def ecos_available_cycles(items: list) -> list:
    """이 통계표가 실제로 지원하는 주기 목록(D/M/Q/A 중 응답에 실존하는 것만).
    지원 안 하는 주기로 조회하면 ECOS가 INFO-200(데이터 없음)을 낸다."""
    cycles = sorted({str(it.get("CYCLE", "")).strip() for it in items
                     if str(it.get("CYCLE", "")).strip() in ECOS_CYCLE_LABEL},
                    key=lambda c: list(ECOS_CYCLE_LABEL).index(c))
    return cycles


def ecos_item_paths(items: list, cycle: str) -> dict:
    """
    items 중 cycle에 맞는 항목들을 '조상 > ... > 이름 (코드)' 형태의 평평한
    선택지로 펼친다. 상위 항목(예: M2 총량) 자체를 고르고 싶을 때가 많아서
    leaf까지 강제로 드릴다운시키지 않고, 트리 전체를 한 번에 펼쳐 고르게
    한다.
    """
    by_code = {}
    for it in items:
        if str(it.get("CYCLE", "")).strip() != cycle:
            continue
        code = str(it.get("ITEM_CODE", "")).strip()
        if not code or code in by_code:
            continue
        parent = it.get("P_ITEM_CODE")
        by_code[code] = {"name": str(it.get("ITEM_NAME", "")).strip() or code,
                         "parent": str(parent).strip() if parent else None}

    def _path(code, seen):
        if code in seen or code not in by_code:
            return by_code.get(code, {}).get("name", code)
        seen = seen | {code}
        node = by_code[code]
        if node["parent"] and node["parent"] in by_code:
            return _path(node["parent"], seen) + " > " + node["name"]
        return node["name"]

    return {f"{_path(code, set())} ({code})": code for code in by_code}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_ecos_overlay(api_key: str, end):
    """ECOS_KR_LEVEL·ECOS_KR_YOY 고정 지표를 한 번에 받는다. 지표 하나가
    실패해도 나머지는 계속 진행한다. 일별(D) 지표는 월말값으로 리샘플해
    다른 지표들과 같은 월 단위 인덱스로 맞춘다(3·12개월 전 대비 계산이
    "몇 번째 행 앞"이 아니라 실제 개월 수를 의미하게 하려면 필요하다).
    반환: ({지표명: Series}, {실패한 지표명: 이유})."""
    end_s = pd.Timestamp(end)
    data, fails = {}, {}
    for label, (stat_code, item_code, cycle) in {**ECOS_KR_LEVEL, **ECOS_KR_YOY}.items():
        # 일별은 데이터량이 많아 조회 기간을 짧게 잡는다(StatisticSearch
        # 1회 조회 상한 1000행 안에 들어오도록 — 2년치 영업일이면 충분).
        years_back = 2 if cycle == "D" else 5
        start_s = end_s - pd.DateOffset(years=years_back)
        fmt = ECOS_FREQ_FMT[cycle]
        try:
            s = ecos_series(api_key, stat_code, cycle, start_s.strftime(fmt),
                            end_s.strftime(fmt), item_code)
        except Exception as ex:
            s = pd.Series(dtype=float)
            fails[label] = f"{type(ex).__name__}: {str(ex)[:150]}"
        if cycle == "D" and not s.empty:
            s = s.resample("ME").last()
        if s.empty:
            fails.setdefault(label, "빈 응답(데이터 없음)")
        else:
            data[label] = s
    return data, fails


def build_macro_regime(data: pd.DataFrame, lag_months=2, win=60) -> pd.DataFrame:
    """
    경기 × 물가 4분면.
      경기 = 산업생산·고용의 3개월 변화 z점수 평균
      물가 = CPI 전년동월비의 3개월 변화 z점수
    lag_months 만큼 지표를 뒤로 밀어, 그 시점에 알 수 있었던 값만 쓰게 한다.
    """
    d = data.copy()
    if lag_months:
        d = d.shift(int(lag_months))
    g1 = rolling_z(np.log(d["INDPRO"]).diff(3), win)
    g2 = rolling_z(np.log(d["PAYEMS"]).diff(3), win)
    parts = [g1, g2]
    if "RPI" in d.columns:
        parts.append(rolling_z(np.log(d["RPI"]).diff(3), win))
    growth = pd.concat(parts, axis=1).mean(axis=1)

    cpi_yoy = d["CPIAUCSL"].pct_change(12) * 100
    infl = rolling_z(cpi_yoy.diff(3), win)

    out = pd.DataFrame({"경기점수": growth, "물가점수": infl, "CPI 상승률(%)": cpi_yoy})
    gu, iu = out["경기점수"] >= 0, out["물가점수"] >= 0
    out["국면"] = np.select(
        [gu & ~iu, gu & iu, ~gu & iu, ~gu & ~iu],
        ["경기상승·물가하락", "경기상승·물가상승",
         "경기하락·물가상승", "경기하락·물가하락"], default="판정불가")
    sg = pd.concat(parts, axis=1)
    out["지표 일치도(%)"] = (sg.apply(np.sign).eq(sg.apply(np.sign).iloc[:, 0], axis=0)
                        .mean(axis=1) * 100)
    return out.dropna(subset=["경기점수", "물가점수"])


def macro_segments(reg: pd.DataFrame) -> pd.DataFrame:
    """국면이 이어진 구간을 묶는다."""
    g = (reg["국면"] != reg["국면"].shift()).cumsum()
    rows = []
    for _, idx_ in reg.groupby(g).groups.items():
        a, b = idx_[0], idx_[-1]
        rows.append({"국면": reg.loc[a, "국면"], "시작": a.strftime("%Y-%m"),
                     "종료": b.strftime("%Y-%m"), "개월": len(idx_)})
    return pd.DataFrame(rows)


def render_macro(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill):
    """FRED 거시지표로 경기·물가 국면을 판정하고 시장 신호와 대조한다."""
    st.caption("미국 거시지표(산업생산·고용·물가)로 **경기 × 물가** 국면을 판정하고, "
               "시장 가격이 같은 방향을 반영하는지 확인합니다.")

    c1, c2, c3 = st.columns(3)
    lag_m = c1.slider("발표 지연 반영 (개월)", 0, 3, 2, step=1, key="_mac_lag",
                      help="거시지표는 한두 달 뒤에 발표됩니다. 그 시차를 근사적으로 "
                           "반영하려고 지표를 이 개월 수만큼 뒤로 옮깁니다.\n\n"
                           "다만 **과거 수치의 사후 수정 효과는 제거되지 않습니다.** "
                           "지금 FRED에서 받는 값은 최신 개정치이기 때문입니다.")
    z_win = c2.selectbox("z점수 기준 기간", [36, 60, 120], index=1, key="_mac_win",
                         format_func=lambda x: f"{x}개월",
                         help="점수를 매기는 비교 구간입니다. 길수록 안정적이지만 "
                              "최근 변화를 늦게 반영합니다.")
    edge_th = c3.slider("경계 인접 기준", 0.10, 0.50, 0.25, step=0.05, key="_mac_edge",
                        help="점수가 0에서 이 값 안이면 '경계에 가깝다'고 봅니다.")

    if st.button("🌡 거시 국면 판정", type="primary", width="stretch", key="_mac_go"):
        st.session_state["_mac_run"] = True
    if not st.session_state.get("_mac_run"):
        st.info("👆 **거시 국면 판정** 을 누르면 FRED에서 지표를 받아 계산합니다.")
        return

    with st.spinner("FRED 거시지표 조회 중..."):
        # 시장 신호와 같은 종료일을 써야 두 국면을 같은 시점에서 비교할 수 있다
        fred, err = load_fred(end=end_date)
    if fred is None:
        st.error(f"🚫 FRED 데이터를 가져오지 못했습니다.\n\n{err}")
        st.caption("네트워크가 막혀 있거나 FRED 서비스가 일시적으로 응답하지 않을 수 "
                   "있습니다. 아래 **시장 내재 신호**는 야후 데이터만 쓰므로 "
                   "그대로 확인할 수 있습니다.")
        reg = None
    else:
        try:
            reg = build_macro_regime(fred, lag_m, z_win)
        except Exception as ex:
            st.error(f"국면을 계산하지 못했습니다: {ex}")
            reg = None

    # ---------------- 현재 국면 ----------------
    if reg is not None and len(reg):
        last = reg.iloc[-1]
        quad = MACRO_QUAD.get(last["국면"], {})
        st.subheader("1️⃣ 현재 거시 국면")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("국면", last["국면"])
        k2.metric("경기점수", f"{last['경기점수']:+.2f}",
                  delta=f"{last['경기점수'] - reg['경기점수'].iloc[-2]:+.2f}"
                  if len(reg) > 1 else None)
        k3.metric("물가점수", f"{last['물가점수']:+.2f}",
                  delta=f"{last['물가점수'] - reg['물가점수'].iloc[-2]:+.2f}"
                  if len(reg) > 1 else None)
        k4.metric("CPI 상승률", f"{last['CPI 상승률(%)']:.2f}%")

        edges = []
        if abs(last["경기점수"]) < edge_th:
            edges.append("경기")
        if abs(last["물가점수"]) < edge_th:
            edges.append("물가")
        base_txt = (f"기준 시점 **{reg.index[-1].strftime('%Y년 %m월')}** · "
                    f"발표 지연 {lag_m}개월 반영 · 지표 일치도 "
                    f"**{last['지표 일치도(%)']:.0f}%**")
        if edges:
            st.warning(f"{base_txt}\n\n⚠️ **{'·'.join(edges)}점수가 경계선(0)에 "
                       f"가깝습니다.** 작은 변화로도 국면이 바뀔 수 있어 판정을 "
                       f"단정하기 어렵습니다.")
        elif last["지표 일치도(%)"] < 70:
            st.warning(f"{base_txt}\n\n⚠️ 경기 구성지표들이 서로 다른 방향을 "
                       f"가리키고 있어 확신도가 낮습니다.")
        else:
            st.success(base_txt)

        if quad:
            st.markdown(f"**이 국면에서 통념상** · 유리 `{quad['유리']}` · "
                        f"불리 `{quad['불리']}`")
            st.caption("교과서적 통념이며 반례가 많습니다. 아래 과거 성과와 함께 "
                       "참고 수준으로만 보세요.")

        # 4분면 산점도
        st.subheader("2️⃣ 4분면 위치 (최근 24개월)")
        rec = reg.tail(24)
        fq = go.Figure()
        for nm, q in MACRO_QUAD.items():
            xr = [0, 4] if "경기상승" in nm else [-4, 0]
            yr = [0, 4] if "물가상승" in nm else [-4, 0]
            fq.add_shape(type="rect", x0=xr[0], x1=xr[1], y0=yr[0], y1=yr[1],
                         fillcolor=q["색"], opacity=0.45, line_width=0, layer="below")
            fq.add_annotation(x=sum(xr) / 2, y=yr[1] - 0.35 if yr[1] > 0 else yr[0] + 0.35,
                              text=nm, showarrow=False,
                              font=dict(size=11, color="#64748b"))
        fq.add_trace(go.Scatter(x=rec["경기점수"], y=rec["물가점수"],
                                mode="lines+markers", name="최근 24개월",
                                line=dict(color="#334155", width=1.5),
                                marker=dict(size=6),
                                text=[d.strftime("%y-%m") for d in rec.index],
                                hovertemplate="%{text}<br>경기 %{x:.2f} · 물가 %{y:.2f}"))
        fq.add_trace(go.Scatter(x=[last["경기점수"]], y=[last["물가점수"]],
                                mode="markers+text", text=["현재"],
                                textposition="top center", name="현재",
                                marker=dict(size=16, color="#dc2626", symbol="star")))
        fq.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fq.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fq.update_layout(height=520, showlegend=False,
                         xaxis=dict(title="경기: 하락 ← 0 → 상승", range=[-3, 3]),
                         yaxis=dict(title="물가: 하락 ← 0 → 상승", range=[-3, 3]),
                         margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fq, width="stretch")
        st.caption("선은 최근 24개월의 이동 경로입니다. 어느 방향으로 움직이는지가 "
                   "현재 위치만큼 중요합니다.")

        with st.expander("국면 구간 이력", expanded=False):
            segs = macro_segments(reg)
            st.dataframe(segs.tail(30), width="stretch", hide_index=True)
            st.caption(f"전체 {len(reg)}개월 · 국면 전환 {len(segs)}회 · "
                       f"평균 지속 {segs['개월'].mean():.1f}개월")

    # ---------------- 시장 신호 ----------------
    st.subheader("3️⃣ 시장 내재 신호 (Market-implied)")
    st.caption("시장 가격이 어느 방향을 반영하는지 봅니다. **거시지표는 실물이고 "
               "시장은 선반영**이라, 둘이 엇갈릴 때가 오히려 정보입니다.")
    sig_tk = sorted({t for lst in MARKET_SIGNALS.values() for a, b, _ in lst
                     for t in (a, b)})
    try:
        with st.spinner("시장 신호 조회 중..."):
            spx, _sm, _sf = build_price_frame(
                sig_tk, pd.Timestamp(end_date) - pd.DateOffset(years=3),
                end_date, "USD", use_div, True, gap_fill)
    except Exception as ex:
        st.warning(f"시장 신호를 가져오지 못했습니다: {ex}")
        spx = None

    mkt_dir = {}
    if spx is not None and len(spx) > 130:
        srows = []
        for axis, pairs in MARKET_SIGNALS.items():
            vals = []
            for a, b, lbl in pairs:
                if a not in spx.columns or b not in spx.columns:
                    continue
                ratio = spx[a] / spx[b]
                mom = float(ratio.iloc[-1] / ratio.iloc[-127] - 1) * 100
                vals.append(mom)
                srows.append({"축": axis, "지표": lbl,
                              "6개월 변화(%)": mom,
                              "방향": "상승" if mom > 0 else "하락"})
            if vals:
                mkt_dir[axis] = float(np.mean(vals))
        if srows:
            st.dataframe(pd.DataFrame(srows).style.format({"6개월 변화(%)": "{:+.2f}"}),
                         width="stretch", hide_index=True)
            m_quad = (("경기상승" if mkt_dir.get("경기", 0) > 0 else "경기하락") + "·" +
                      ("물가상승" if mkt_dir.get("물가", 0) > 0 else "물가하락"))
            risk_on = mkt_dir.get("위험선호", 0) > 0
            c1_, c2_ = st.columns(2)
            c1_.metric("시장이 반영하는 국면", m_quad)
            c2_.metric("위험선호 상태", "위험선호" if risk_on else "위험회피")

            if reg is not None and len(reg):
                if m_quad == last["국면"]:
                    st.success(f"**실물지표와 시장 신호가 같은 방향**입니다 "
                               f"({m_quad}). 판단의 신뢰도가 높은 편입니다.")
                else:
                    st.warning(f"**엇갈립니다.** 거시지표는 `{last['국면']}` 인데 "
                               f"시장은 `{m_quad}` 를 반영하고 있습니다.\n\n"
                               f"시장이 앞으로의 변화를 먼저 반영하는 중일 수도, "
                               f"일시적 과민 반응일 수도 있습니다. 어느 쪽인지는 "
                               f"시간이 지나야 확인됩니다.")
    else:
        st.info("시장 신호를 계산하기에 데이터가 부족합니다.")

    # ---------------- 금리·신용·유동성 ----------------
    st.subheader("4️⃣ 금리·신용·유동성")
    st.caption("경기·물가 국면 판정에는 쓰지 않는 **참고용 지표**입니다. 금리·신용스프레드는 "
              "레벨 자체를, 통화량·Fed 총자산은 규모가 커서 전년동월 대비 증가율을 봅니다.")
    try:
        with st.spinner("금리·신용·유동성 지표 조회 중..."):
            fred_x, err_x = load_fred_extra(end=end_date)
    except Exception as ex:
        fred_x, err_x = None, str(ex)
    if fred_x is None:
        st.warning(f"⚠️ 지표를 가져오지 못했습니다: {err_x}")
    else:
        rc = fred_level_summary(fred_x, FRED_RATE_CREDIT)
        if len(rc):
            st.markdown("**금리·신용스프레드**")
            st.dataframe(rc.style.format(
                {"현재값(%)": "{:.2f}", "3개월 전 대비(%p)": "{:+.2f}",
                 "12개월 전 대비(%p)": "{:+.2f}"}, na_rep="-"),
                width="stretch", hide_index=True)
        liq = fred_yoy_summary(fred_x, FRED_LIQUIDITY)
        if len(liq):
            st.markdown("**유동성 (전년동월비)**")
            st.dataframe(liq.style.format(
                {"전년동월비(%)": "{:+.2f}", "3개월 전 전년동월비(%)": "{:+.2f}"}, na_rep="-"),
                width="stretch", hide_index=True)
        if not len(rc) and not len(liq):
            st.info("표시할 지표가 없습니다.")

    # ---------------- 한국 거시 (ECOS) ----------------
    st.subheader("5️⃣ 한국 금융여건 (ECOS)")
    st.caption("검색 없이 자동으로 뜨는 고정 지표입니다. 국면 판정(1️⃣~2️⃣)에는 "
              "쓰지 않는 참고용이며, 한국 기준금리·물가·통화량·환율이 미국 "
              "환경을 어떻게 전달받고 있는지 보는 용도입니다.")
    ecos_secret = ecos_secret_key()
    if ecos_secret:
        ecos_key = ecos_secret
        st.caption("✅ 서버에 등록된 인증키를 자동으로 씁니다.")
    else:
        ecos_key = st.text_input("ECOS API 인증키", type="password", key="_ecos_key",
                                 help="https://ecos.bok.or.kr 에서 무료 발급받습니다. "
                                      "이 키는 이 브라우저 세션에만 남고 저장되지 "
                                      "않습니다. 매번 입력하기 번거로우면 Streamlit "
                                      "Cloud 앱 설정의 Secrets에 ECOS_API_KEY로 "
                                      "등록해두세요 — 그러면 이 입력창 자체가 "
                                      "사라집니다.")
    if not ecos_key.strip():
        st.info("👆 인증키를 입력하면 지표를 볼 수 있습니다.")
    else:
        with st.spinner("한국 금융여건 지표 조회 중..."):
            ov_data, ov_fails = load_ecos_overlay(ecos_key.strip(), end_date)
        if ov_fails:
            st.warning(f"⚠️ 조회하지 못한 지표: {', '.join(ov_fails)}")
            with st.expander("실패 이유 보기", expanded=False):
                for label, reason in ov_fails.items():
                    st.caption(f"**{label}**: {reason}")
        lvl_rows = []
        for label in ECOS_KR_LEVEL:
            s = ov_data.get(label)
            if s is None or s.empty:
                continue
            last_v = float(s.iloc[-1])
            lvl_rows.append({
                "지표": label, "기준월": s.index[-1].strftime("%Y-%m"),
                "현재값": last_v,
                "3개월 전 대비": last_v - float(s.iloc[-4]) if len(s) > 3 else np.nan,
                "12개월 전 대비": last_v - float(s.iloc[-13]) if len(s) > 12 else np.nan})
        if lvl_rows:
            st.markdown("**금리·환율**")
            st.dataframe(pd.DataFrame(lvl_rows).style.format(
                {"현재값": "{:,.2f}", "3개월 전 대비": "{:+,.2f}",
                 "12개월 전 대비": "{:+,.2f}"}, na_rep="-"),
                width="stretch", hide_index=True)

        yoy_rows = []
        for label in ECOS_KR_YOY:
            s = ov_data.get(label)
            if s is None or s.empty:
                continue
            yoy = s.pct_change(12).dropna()
            if yoy.empty:
                continue
            yoy_rows.append({
                "지표": label, "기준월": yoy.index[-1].strftime("%Y-%m"),
                "전년동월비(%)": float(yoy.iloc[-1]) * 100,
                "3개월 전 전년동월비(%)": float(yoy.iloc[-4]) * 100 if len(yoy) > 3 else np.nan})
        if yoy_rows:
            st.markdown("**물가·생산·통화량 (전년동월비)**")
            st.dataframe(pd.DataFrame(yoy_rows).style.format(
                {"전년동월비(%)": "{:+.2f}", "3개월 전 전년동월비(%)": "{:+.2f}"},
                na_rep="-"), width="stretch", hide_index=True)

        kr_rate = ov_data.get("기준금리")
        us_rate = fred_x["FEDFUNDS"].dropna() if fred_x is not None and \
            "FEDFUNDS" in fred_x.columns else None
        if kr_rate is not None and len(kr_rate) and us_rate is not None and len(us_rate):
            diff = float(kr_rate.iloc[-1]) - float(us_rate.iloc[-1])
            st.metric("한·미 정책금리차 (환헤지 비용 압력)", f"{diff:+.2f}%p",
                      help="한국 기준금리 − 미국 기준금리. 실제 환헤지 비용(선물환 "
                           "프리미엄)은 대체로 이 금리차와 연동됩니다. 사이드바 "
                           "**환율효과 제외** 옵션은 이 비용을 반영하지 않는 "
                           "무비용 완전헤지 가정이라는 점을 참고하세요.")

        with st.expander("🔍 다른 지표 찾아보기 (개발용)", expanded=False):
            st.caption("위 고정 지표 외에 다른 통계를 보고 싶을 때 씁니다. "
                      "**통계표코드를 미리 정해두지 않습니다** — 잘못 외운 코드로 "
                      "조용히 엉뚱한 지표를 보여주는 사고를 막기 위해 매번 이름 "
                      "검색부터 시작합니다.")
            kw = st.text_input("통계표 검색 (예: 기준금리, 소비자물가, 국고채)",
                               key="_ecos_kw")
            if kw.strip():
                try:
                    with st.spinner("통계표 검색 중..."):
                        tables = ecos_table_list(ecos_key.strip())
                except Exception as ex:
                    st.error(f"🚫 통계표 목록을 가져오지 못했습니다: {ex}")
                    tables = []
                matches = [t for t in tables if kw.strip() in str(t.get("STAT_NAME", ""))]
                if not matches:
                    st.warning("검색 결과가 없습니다. 다른 표현으로 검색해보세요 "
                               "(예: '금리' 대신 '기준금리').")
                else:
                    if len(matches) > 50:
                        st.caption(f"{len(matches)}건 중 상위 50건만 표시합니다. "
                                   "검색어를 좁혀보세요.")
                    opts = {f"{t['STAT_NAME']} ({t['STAT_CODE']})": t["STAT_CODE"]
                           for t in matches[:50]}
                    pick = st.selectbox("통계표 선택", list(opts), key="_ecos_table_pick")
                    stat_code = opts[pick]
                    try:
                        with st.spinner("세부 항목 조회 중..."):
                            items = ecos_item_list(ecos_key.strip(), stat_code)
                    except Exception as ex:
                        st.error(f"🚫 세부 항목을 가져오지 못했습니다: {ex}")
                        items = []
                    if items:
                        # 이 표가 실제 지원하는 주기만 고르게 한다 — 안 맞는 주기로
                        # 조회하면 INFO-200(데이터 없음)이 난다. 항목 트리도
                        # 주기별로 갈리므로(같은 항목이 D/M/Q/A마다 행이 반복됨)
                        # 주기부터 고른다.
                        cycles = ecos_available_cycles(items)
                        if not cycles:
                            st.warning("이 통계표는 지원 주기 정보를 확인할 수 없습니다.")
                        else:
                            fc1, fc2 = st.columns([1, 2])
                            cyc_label = fc1.radio(
                                "주기", [ECOS_CYCLE_LABEL[c] for c in cycles],
                                horizontal=True, key=f"_ecos_freq_{stat_code}")
                            freq = {v: k for k, v in ECOS_CYCLE_LABEL.items()}[cyc_label]
                            years_back = fc2.slider("조회 기간 (년)", 1, 20, 10,
                                                    key="_ecos_years")

                            # 트리 전체를 평평하게 펼쳐 한 번에 고른다 — 상위
                            # 항목(예: M2 총량) 자체를 고르고 싶을 때가 많아
                            # 단계별 강제 드릴다운은 오히려 방해가 된다.
                            paths = ecos_item_paths(items, freq)
                            if not paths:
                                item_code, item_label = "", pick.split(" (")[0]
                                st.caption("이 통계표는 세부 항목 구분이 없어 전체 "
                                          "계열을 그대로 조회합니다.")
                            else:
                                ipick = st.selectbox(
                                    "세부 항목 선택", list(paths),
                                    key=f"_ecos_item_{stat_code}_{freq}")
                                item_code = paths[ipick]
                                item_label = ipick.rsplit(" (", 1)[0]

                            if st.button("📈 조회", key="_ecos_go", type="primary"):
                                end_s2 = pd.Timestamp(end_date)
                                start_s2 = end_s2 - pd.DateOffset(years=years_back)
                                if freq == "Q":
                                    fmt_bound = (lambda t: f"{t.year}Q"
                                                f"{(t.month - 1) // 3 + 1}")
                                else:
                                    fmt_bound = ECOS_FREQ_FMT[freq]
                                    fmt_bound = (lambda t, _f=fmt_bound: t.strftime(_f))
                                try:
                                    with st.spinner("시계열 조회 중..."):
                                        s = ecos_series(ecos_key.strip(), stat_code,
                                                       freq, fmt_bound(start_s2),
                                                       fmt_bound(end_s2), item_code)
                                except Exception as ex:
                                    st.error(f"🚫 조회 실패: {ex}")
                                    s = pd.Series(dtype=float)
                                if s.empty:
                                    st.warning("데이터가 없습니다. 주기나 기간을 "
                                              "바꿔보세요.")
                                else:
                                    fk = go.Figure(go.Scatter(
                                        x=s.index, y=s.values, mode="lines",
                                        line=dict(color="#2563eb")))
                                    fk.update_layout(height=320,
                                                     margin=dict(l=0, r=0, t=20, b=0),
                                                     yaxis=dict(title=item_label))
                                    st.plotly_chart(fk, width="stretch")
                                    last_v = float(s.iloc[-1])
                                    mc1, mc2, mc3 = st.columns(3)
                                    mc1.metric(
                                        "최근값", f"{last_v:,.2f}",
                                        help=f"기준: {s.index[-1].strftime('%Y-%m-%d')}")
                                    if len(s) > 3:
                                        mc2.metric("3구간 전 대비",
                                                  f"{last_v - float(s.iloc[-4]):+,.2f}")
                                    if len(s) > 12:
                                        mc3.metric(
                                            "12구간 전 대비",
                                            f"{last_v - float(s.iloc[-13]):+,.2f}")

    # ---------------- 포트폴리오 노출 ----------------
    # ---------------- 전망 초안으로 넘기기 ----------------
    if reg is not None and len(reg):
        st.subheader("➡️ 이 국면을 전망으로 반영하기")
        st.caption("현재 국면 판정을 **🧭 뷰 기반 자산배분**의 시장 전망 초안으로 "
                   "가져갑니다. 국면은 **현재 상태**이고 전망은 **앞으로**이므로, "
                   "그대로 확정하지 않고 수정 가능한 초안으로만 넘깁니다.")
        _g, _i = float(last["경기점수"]), float(last["물가점수"])

        def _dir(v, strong=1.0):
            if v >= strong:
                return "강한 상승"
            if v > 0.15:
                return "상승"
            if v <= -strong:
                return "강한 하락"
            if v < -0.15:
                return "하락"
            return "보합"

        _draft = {"경기": _dir(_g), "물가": _dir(_i)}
        # 경기 둔화 + 물가 둔화면 금리 인하 압력, 반대면 인상 압력.
        # 정책금리와 장기금리는 같은 요인으로 들어가므로 **장기금리 하나만** 켠다.
        # (둘 다 켜면 금리 요인이 두 번 반영돼 영향이 부풀려진다)
        _rate = -(_g * 0.4 + _i * 0.6)
        _draft["장기금리"] = _dir(-_rate)
        _need_extra = []
        if mkt_dir.get("위험선호") is not None:
            _draft["위험선호"] = "상승" if mkt_dir["위험선호"] > 0 else "하락"
            if _draft["위험선호"] != "보합":
                _need_extra.append("위험선호")
        _conf = int(min(90, max(20, abs(_g) * 40 + 30)))

        st.dataframe(pd.DataFrame(
            [{"전망 변수": k, "초안 방향": v} for k, v in _draft.items()]),
            width="stretch", hide_index=True)
        if st.button("📤 전망 초안으로 가져가기", width="stretch", key="_mac_to_view"):
            _mvdf = _mv_default()
            for k, v in _draft.items():
                _m = _mvdf["전망 변수"] == k
                _mvdf.loc[_m, "방향"] = v
                _mvdf.loc[_m, "사용"] = (v != "보합")
                _mvdf.loc[_m, "확신도(%)"] = _conf
                _mvdf.loc[_m, "메모"] = f"{reg.index[-1].strftime('%Y-%m')} 거시국면: {last['국면']}"
            st.session_state["_mv_views"] = _mvdf
            # 편집기 위젯 상태를 지우지 않으면 이전 입력이 계속 보인다
            st.session_state.pop("_mv_view_editor", None)
            # 전망표에는 있는데 회귀 요인에 없으면 계산에서 조용히 빠진다.
            # 필요한 추가 요인을 함께 켜준다.
            if _need_extra:
                _cur = list(st.session_state.get("_mv_extra", []))
                st.session_state["_mv_extra"] = list(
                    dict.fromkeys(_cur + _need_extra))
            st.session_state["_mv_draft_flag"] = True
            st.session_state["_va_mode"] = "🌍 시장·거시 전망"
            st.session_state["_pending_tool"] = "🧭 뷰 기반 자산배분"
            st.rerun()

    st.subheader("6️⃣ 내 포트폴리오 노출 점검")
    st.caption("현재 국면에서 **통념상 유리한 자산과 불리한 자산에 얼마나 노출**돼 "
               "있는지 봅니다. 미래를 예측하지 않고, 지금 구성만 진단합니다.")
    clip = st.session_state.get(CLIP_KEY)
    plans = load_plans()
    src_opts = (["보관함 (📋 복사한 구성)"] if clip and clip.get("rows") else []) + \
               [f"비교 후보 · {k}" for k in plans]
    if not src_opts:
        st.info("다른 화면에서 **📋 복사** 하거나 **⚖️ 비교 후보로 담기** 를 하면 "
                "여기서 노출을 점검할 수 있습니다.")
    else:
        pick_src = st.selectbox("점검할 구성", src_opts, key="_mac_src")
        rows_p = (clip["rows"] if pick_src.startswith("보관함")
                  else plans[pick_src.replace("비교 후보 · ", "")]["rows"])
        if reg is not None and len(reg):
            fav = MACRO_QUAD[last["국면"]]["유리"]
            unf = MACRO_QUAD[last["국면"]]["불리"]
            st.markdown(f"현재 국면 **{last['국면']}** · 유리 `{fav}` · 불리 `{unf}`")
        st.caption("각 종목이 어느 쪽인지는 직접 골라주세요. 자동 분류는 "
                   "종목마다 성격이 달라 오히려 잘못된 진단을 줄 수 있습니다.")
        # 다른 구성을 고르면 이전 선택이 남지 않도록 키를 구성별로 만든다
        cls_key = f"_mac_cls_{abs(hash(pick_src)) % 100000}"
        cdf0 = pd.DataFrame([{"티커": r["티커"], "비중(%)": r["비중"], "성격": "중립"}
                             for r in rows_p])
        cls = st.data_editor(
            cdf0, num_rows="fixed", width="stretch", key=cls_key, hide_index=True,
            column_config={
                "티커": st.column_config.TextColumn("티커", disabled=True),
                "비중(%)": st.column_config.NumberColumn("비중(%)", disabled=True,
                                                       format="%.2f"),
                "성격": st.column_config.SelectboxColumn(
                    "이 국면에서", options=["유리", "중립", "불리"], width="medium")})
        agg = cls.groupby("성격")["비중(%)"].sum()
        e1, e2, e3 = st.columns(3)
        e1.metric("유리한 자산", f"{agg.get('유리', 0):.1f}%")
        e2.metric("중립", f"{agg.get('중립', 0):.1f}%")
        e3.metric("불리한 자산", f"{agg.get('불리', 0):.1f}%")
        if agg.get("불리", 0) > 50:
            st.warning(f"현재 국면에서 통념상 불리한 자산에 **{agg.get('불리', 0):.1f}%** "
                       f"노출돼 있습니다. 국면 판정이 맞다면 조정을 검토할 만합니다.")
        elif agg.get("유리", 0) > 50:
            st.success(f"유리한 자산 비중이 **{agg.get('유리', 0):.1f}%** 입니다.")

    # ---------------- 과거 국면별 성과 ----------------
    if reg is not None and len(reg) > 24:
        with st.expander("⚠️ 과거 동일 국면의 자산 성과 (참고용 · 먼저 읽어주세요)",
                         expanded=False):
            st.error("**이 표는 통계적 근거가 약합니다.**\n\n"
                     "· 국면당 표본이 적고, 연속되지 않은 시기를 묶은 값입니다\n"
                     "· 거시지표는 사후 수정되므로 과거 국면 판정 자체가 정확하지 "
                     "않을 수 있습니다\n"
                     "· 에피소드 수가 5회 미만이면 우연일 가능성이 큽니다\n\n"
                     "숫자를 근거로 삼지 마시고, 통념과 얼마나 부합하는지 보는 "
                     "용도로만 쓰세요.")
            mtk = st.text_input("확인할 자산 (콤마 구분)", value="SPY, TLT, GLD, DBC",
                                key="_mac_hist_tk")
            if st.button("과거 성과 계산", key="_mac_hist_go"):
                tks_h = [normalize_ticker(x)[0] for x in mtk.split(",") if x.strip()]
                try:
                    hpx, _hm, _hf = build_price_frame(
                        tks_h, reg.index[0], end_date, base_ccy, use_div,
                        fx_hedge, gap_fill)
                    mret = hpx.resample("ME").last().pct_change().dropna()
                    lab = reg["국면"].reindex(mret.index, method="ffill")
                    hrows = []
                    segs_h = macro_segments(reg)
                    for q in MACRO_QUAD:
                        msk = lab == q
                        if msk.sum() < 6:
                            continue
                        ep = int((segs_h["국면"] == q).sum())
                        for t_ in mret.columns:
                            rr = mret.loc[msk, t_]
                            hrows.append({
                                "국면": q, "티커": t_,
                                "월평균(%)": float(rr.mean()) * 100,
                                "연율(%)": (float((1 + rr).prod() ** (12 / len(rr))) - 1) * 100,
                                "개월": int(msk.sum()), "에피소드": ep,
                                "신뢰도": "낮음 ⚠️" if ep < 5 else "보통"})
                    if hrows:
                        hdf = pd.DataFrame(hrows)
                        piv = hdf.pivot(index="티커", columns="국면", values="연율(%)")
                        st.dataframe(piv.style.format("{:.2f}", na_rep="-")
                                     .map(_heat_color), width="stretch")
                        st.dataframe(hdf.style.format({"월평균(%)": "{:.2f}",
                                                       "연율(%)": "{:.2f}"}),
                                     width="stretch", hide_index=True)
                        _lowep = hdf[hdf["에피소드"] < 5]["국면"].unique()
                        if len(_lowep):
                            st.error(f"에피소드가 5회 미만인 국면: {', '.join(_lowep)} "
                                     f"— 이 숫자는 통계적 의미가 없습니다.")
                except Exception as ex:
                    st.error(f"계산하지 못했습니다: {ex}")

    st.caption("발표 시차는 지표를 뒤로 옮겨 근사적으로 반영하지만, **과거 수치의 사후 "
               "수정 효과는 제거되지 않습니다**(최신 개정치를 쓰기 때문입니다). "
               "그래서 이 화면은 현재 상황을 읽는 참고 자료이며, 판정을 과거 "
               "백테스트에 쓰지 않습니다. 교육·참고용이며 투자 자문이 아닙니다.")


# ======================================================================
# 시장동향 — 자산군별 대표 지수/ETF 가격 스냅샷
# ======================================================================
MARKET_SNAPSHOT_GROUPS = {
    "주가지수": [("^GSPC", "S&P 500"), ("^NDX", "나스닥100"), ("^KS11", "코스피"),
              ("^KQ11", "코스닥"), ("^N225", "니케이225"), ("^STOXX50E", "유로스톡스50")],
    "섹터 (미국)": [("XLK", "기술"), ("XLF", "금융"), ("XLE", "에너지"), ("XLV", "헬스케어"),
                 ("XLI", "산업재"), ("XLY", "임의소비재"), ("XLP", "필수소비재"),
                 ("XLU", "유틸리티"), ("XLB", "소재"), ("XLRE", "리츠")],
    "채권·금리": [("SHY", "미 단기국채(1-3Y)"), ("IEF", "미 중기국채(7-10Y)"),
               ("TLT", "미 장기국채(20Y+)"), ("LQD", "미 투자등급 회사채"),
               ("HYG", "미 하이일드")],
    "통화·원자재": [("DX-Y.NYB", "달러인덱스"), ("KRW=X", "원/달러"), ("GLD", "금"),
                 ("USO", "WTI 원유"), ("BTC-USD", "비트코인")],
}

# (표시 열 이름, 기준일로부터의 오프셋). 1일은 달력일이 아니라 직전 거래일 대비로 따로 계산한다.
SNAPSHOT_WINDOWS = [
    ("1주", pd.DateOffset(weeks=1)),
    ("1개월", pd.DateOffset(months=1)),
    ("3개월", pd.DateOffset(months=3)),
    ("1년", pd.DateOffset(years=1)),
]
SNAPSHOT_COLS = ["1일", "1주", "1개월", "3개월", "YTD", "1년"]


def _snapshot_day_return(close: pd.Series) -> float:
    """직전 거래일 대비 수익률(%). 달력일이 아니라 실제 거래일 기준."""
    c = close.dropna()
    if len(c) < 2:
        return np.nan
    prev = c.iloc[-2]
    if not prev:
        return np.nan
    return (c.iloc[-1] / prev - 1) * 100


def _snapshot_offset_return(close: pd.Series, ref: pd.Timestamp, base_date: pd.Timestamp) -> float:
    """ref 시점 값 대비 base_date 이전 마지막 거래일 값의 수익률(%)."""
    hist = close.loc[:ref].dropna()
    if hist.empty:
        return np.nan
    base_hist = close.loc[:base_date].dropna()
    if base_hist.empty:
        return np.nan
    base = base_hist.iloc[-1]
    if not base:
        return np.nan
    return (hist.iloc[-1] / base - 1) * 100


def render_market_snapshot(end_date):
    st.title("🗞 시장동향")
    st.caption("자산군별 대표 지수·ETF의 최근 가격 변동을 한눈에 봅니다. "
              "**왜 움직였는지는 설명하지 않습니다** — 무엇이 얼마나 움직였는지만 "
              "보여주는 스냅샷입니다. 뉴스·리서치와 함께 참고하세요.")

    ref = pd.Timestamp(end_date)
    lookback_start = ref - pd.DateOffset(years=2)  # 1년 수익률 계산 여유분

    rows, fails = [], []
    with st.spinner("시세 조회 중..."):
        for group, items in MARKET_SNAPSHOT_GROUPS.items():
            for tk, label in items:
                try:
                    close = load_ticker(tk, lookback_start, ref)["close"]
                except Exception:
                    fails.append(f"{label}({tk})")
                    continue
                hist = close.loc[:ref].dropna()
                if hist.empty:
                    fails.append(f"{label}({tk})")
                    continue
                row = {"구분": group, "자산": label, "티커": tk,
                      "1일": _snapshot_day_return(close)}
                for wname, offset in SNAPSHOT_WINDOWS:
                    row[wname] = _snapshot_offset_return(close, ref, ref - offset)
                row["YTD"] = _snapshot_offset_return(
                    close, ref, pd.Timestamp(year=ref.year - 1, month=12, day=31))
                row["기준일"] = hist.index[-1].strftime("%Y-%m-%d")
                rows.append(row)

    if fails:
        st.warning(f"⚠️ 조회하지 못한 종목: {', '.join(fails)}")
    if not rows:
        st.error("🚫 조회된 시세가 없습니다.")
        return

    df = pd.DataFrame(rows)[["구분", "자산", "티커"] + SNAPSHOT_COLS + ["기준일"]]
    RANGE = {"1일": 3.0, "1주": 5.0, "1개월": 10.0, "3개월": 15.0, "YTD": 25.0, "1년": 25.0}
    for group in MARKET_SNAPSHOT_GROUPS:
        gdf = df[df["구분"] == group].drop(columns=["구분"])
        if gdf.empty:
            continue
        st.subheader(group)
        sty = gdf.style.format({c: "{:+.2f}%" for c in SNAPSHOT_COLS}, na_rep="-")
        apply_fn = sty.map if hasattr(sty, "map") else sty.applymap
        for c, hi in RANGE.items():
            sty = apply_fn(partial(_heat_color, lo=-hi, hi=hi), subset=[c])
        st.dataframe(sty, width="stretch", hide_index=True)

    st.caption(f"기준일 {ref.strftime('%Y-%m-%d')} (사이드바 종료일 기준) · 가격 변동률이며 "
              "배당은 포함하지 않았습니다. 교육·참고용이며 투자 자문이 아닙니다.")


REG_COLS = ["티커", "종목명"]
REG_COLORS = {"강세": "#dcfce7", "약세": "#fee2e2", "횡보": "#f1f5f9"}
REG_LINE = {"강세": "#16a34a", "약세": "#dc2626", "횡보": "#94a3b8"}


def _reg_blank():
    return {"티커": "", "종목명": ""}


def classify_regime(px: pd.Series, bear=0.20, bull=0.20,
                    side_win=126, side_band=0.05, min_days=60) -> pd.Series:
    """
    통용되는 정의를 따른다.
      · 고점 대비 -bear% 하락 → 약세장. 시작일은 그 '고점'으로 소급한다.
      · 저점 대비 +bull% 반등 → 강세장. 시작일은 그 '저점'으로 소급한다.
      · 강세 구간 중 최근 side_win일 수익률이 ±side_band 이내면 횡보로 세분한다.
    너무 짧은 구간은 이웃에 흡수시키되, 약세는 길이와 무관하게 보존한다.
    """
    px = px.dropna()
    if len(px) < 20:
        return pd.Series("강세", index=px.index)

    state, lab = "강세", []
    peak = trough = float(px.iloc[0])
    peak_i = trough_i = 0
    for i, v in enumerate(px.values):
        v = float(v)
        if state == "강세":
            if v > peak:
                peak, peak_i = v, i
            if v <= peak * (1 - bear):
                state = "약세"
                for j in range(peak_i, i):
                    lab[j] = "약세"
                trough, trough_i = v, i
        else:
            if v < trough:
                trough, trough_i = v, i
            if v >= trough * (1 + bull):
                state = "강세"
                for j in range(trough_i, i):
                    lab[j] = "강세"
                peak, peak_i = v, i
        lab.append(state)

    s = pd.Series(lab, index=px.index)
    if side_win and side_band:
        r = px.pct_change(side_win)
        s[(s == "강세") & (r.abs() < side_band)] = "횡보"

    if min_days and min_days > 1:
        arr = list(s.values)
        guard = 0
        while guard <= len(arr):
            guard += 1
            bounds = [0] + [i for i in range(1, len(arr)) if arr[i] != arr[i - 1]] + [len(arr)]
            segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
            tgt = next((k for k, (a, b) in enumerate(segs)
                        if (b - a) < min_days and arr[a] != "약세"), None)
            if tgt is None:
                break
            a, b = segs[tgt]
            src = segs[tgt - 1][0] if tgt > 0 else (
                segs[tgt + 1][0] if tgt + 1 < len(segs) else None)
            if src is None:
                break
            for j in range(a, b):
                arr[j] = arr[src]
        s = pd.Series(arr, index=s.index)
    return s


def regime_segments(px: pd.Series, lab: pd.Series) -> pd.DataFrame:
    g = (lab != lab.shift()).cumsum()
    rows = []
    for _, idx in lab.groupby(g).groups.items():
        a, b = idx[0], idx[-1]
        yrs = max((b - a).days / 365.25, 1 / 365.25)
        rows.append({"국면": lab[a], "시작": a.date(), "종료": b.date(),
                     "거래일": len(idx), "기간(년)": round(yrs, 2),
                     "구간 수익률(%)": (float(px[b]) / float(px[a]) - 1) * 100})
    return pd.DataFrame(rows)


def render_regimes(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate):
    st.title("📉 시장·거시 국면")
    _t1, _t2 = st.tabs(["📈 시장 추세 국면 (가격 기반)", "🌡 거시 경기·물가 국면 (FRED)"])
    with _t2:
        render_macro(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill)
    with _t1:
        _render_price_regimes(base_ccy, start_date, end_date, use_div,
                              fx_hedge, gap_fill, rf_rate)


def _render_price_regimes(base_ccy, start_date, end_date, use_div,
                          fx_hedge, gap_fill, rf_rate):
    st.caption("시장을 강세·약세·횡보로 나누고, 국면마다 자산들이 어떻게 움직였는지 봅니다. "
               "**상관관계가 국면에 따라 어떻게 달라지는지**가 핵심입니다.")
    st.info("⚠️ **사후 분석입니다.** 고점 대비 −20%가 확인된 뒤에야 그 고점부터 약세장으로 "
            "소급하므로, 당시에는 지금이 어느 국면인지 알 수 없었습니다. "
            "실시간 매매 신호로 쓰기에는 적합하지 않습니다.")

    live, tickers, bad = render_ticker_table(
        state_key="_reg_df", editor_key="_reg_editor", gen_key="_reg_gen",
        cols=REG_COLS, title="시장 국면 분석", min_rows=1, max_rows=20,
        default_rows=[{"티커": x} for x in ("SPY", "AGG", "GLD", "005930.KS")])

    ok_in, msgs = validate_setup(tickers, bad, min_n=1)
    show_msgs(msgs)

    # ---------------- 국면 기준 ----------------
    st.subheader("2️⃣ 국면 구분 기준 (Regime Rules)")
    m1, m2 = st.columns([2, 3])
    ref_tk = m1.text_input("기준 지수", value="^GSPC", key="_reg_ref",
                           help="이 지수를 기준으로 국면을 나눕니다. "
                                "^GSPC S&P500 · ^KS11 코스피 · ^IXIC 나스닥")
    m2.caption("국면은 **기준 지수 하나**로 정의하고, 그 구간을 모든 자산에 동일하게 "
               "적용합니다. 자산마다 다르게 나누면 비교가 불가능해지기 때문입니다.")

    n1, n2, n3 = st.columns(3)
    bear_th = n1.slider("약세장 기준 (고점 대비 하락 %)", 5, 40, 20, step=1,
                        key="_reg_bear",
                        help="통용되는 정의는 -20%입니다. 낮추면 구간이 잘게 나뉩니다.") / 100
    side_band = n2.slider("횡보 판정 폭 (±%)", 0, 15, 5, step=1, key="_reg_side",
                          help="최근 6개월 수익률이 이 범위 안이면 횡보로 봅니다. "
                               "0이면 횡보를 쓰지 않고 강세·약세로만 나눕니다.") / 100
    min_days = n3.slider("최소 지속 (거래일)", 0, 250, 60, step=10, key="_reg_min",
                         help="이보다 짧은 구간은 이웃에 흡수시킵니다. "
                              "약세장은 길이와 무관하게 보존됩니다.")

    run_reg = st.button("📉 국면 분석", type="primary", width="stretch",
                        disabled=bool(bad) or not tickers or not ref_tk.strip())
    if run_reg:
        st.session_state["_reg_has_run"] = True
    if not st.session_state.get("_reg_has_run"):
        st.info("👆 종목과 기준 지수를 정한 뒤 **국면 분석**을 눌러주세요.")
        return

    # ---------------- 데이터 ----------------
    ref = normalize_ticker(ref_tk)[0] if ref_tk.strip() else "^GSPC"
    need = list(dict.fromkeys(tickers + [ref]))
    try:
        with st.spinner("데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(need, start_date, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    if prices.empty or ref not in prices.columns or len(prices) < 60:
        st.error("데이터가 부족하거나 기준 지수를 가져오지 못했습니다.")
        return
    use_tk = [t for t in tickers if t in prices.columns]
    if not use_tk:
        st.error("분석할 종목 데이터가 없습니다.")
        return

    lab = classify_regime(prices[ref], bear=bear_th, bull=bear_th,
                          side_band=(side_band if side_band > 0 else None),
                          min_days=min_days)
    segs = regime_segments(prices[ref], lab)
    R = prices[use_tk].pct_change().dropna()
    lab_r = lab.reindex(R.index).ffill()

    st.success(f"✅ 분석 완료 · 기준 **{ref}** · {len(segs)}개 구간 · "
               f"{prices.index[0].date()} ~ {prices.index[-1].date()}")

    # ---------------- 구간 ----------------
    st.subheader("3️⃣ 국면 구간 (Regimes)")
    ec = prices[ref] / float(prices[ref].iloc[0])
    fg = go.Figure()
    fg.add_trace(go.Scatter(x=ec.index, y=ec.values, name=ref,
                            line=dict(color="#1e293b", width=1.8)))
    for _, s in segs.iterrows():
        fg.add_vrect(x0=pd.Timestamp(s["시작"]), x1=pd.Timestamp(s["종료"]),
                     fillcolor=REG_COLORS.get(s["국면"], "#f8fafc"),
                     opacity=0.55, line_width=0, layer="below")
    fg.update_layout(height=380, hovermode="x unified",
                     margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
                     yaxis=dict(title=None))
    st.plotly_chart(fg, width="stretch")
    st.caption("초록 = 강세 · 빨강 = 약세 · 회색 = 횡보")
    st.dataframe(segs.style.format({"구간 수익률(%)": "{:+.2f}", "기간(년)": "{:.2f}"}),
                 width="stretch", hide_index=True)

    tally = segs.groupby("국면").agg(구간수=("국면", "size"),
                                    총거래일=("거래일", "sum")).reset_index()
    tally["비중(%)"] = tally["총거래일"] / tally["총거래일"].sum() * 100
    st.dataframe(tally.style.format({"비중(%)": "{:.1f}"}), width="stretch", hide_index=True)

    # ---------------- 국면별 성과 ----------------
    st.subheader("4️⃣ 국면별 자산 성과 (Performance by Regime)")
    st.caption("같은 자산도 국면에 따라 전혀 다르게 움직입니다. "
               "**약세장에서 무엇이 버텼는지**를 눈여겨보세요.")
    prow = []
    # 같은 국면이라도 시점이 흩어져 있으므로, 조각을 이어붙인 '누적 수익률'과
    # '최대낙폭'은 실제로 겪을 수 있는 값이 아니다. 에피소드 단위 통계로 대신한다.
    _epi = (lab != lab.shift()).cumsum()
    _epi_r = _epi.reindex(R.index).ffill()
    for reg in ["강세", "횡보", "약세"]:
        m = lab_r == reg
        if m.sum() < 10:
            continue
        _grp = _epi_r[m]
        _n_epi = int(_grp.nunique())
        for t in use_tk:
            rr = R.loc[m, t]
            # 에피소드별 수익률 → 그 분포를 요약
            _er = [float((1 + rr[_grp == g]).prod() - 1) * 100
                   for g in _grp.unique() if (_grp == g).sum() >= 3]
            prow.append({"국면": reg, "티커": t,
                         "에피소드": _n_epi,
                         "에피소드 중앙값(%)": float(np.median(_er)) if _er else np.nan,
                         "에피소드 최악(%)": float(np.min(_er)) if _er else np.nan,
                         "에피소드 최선(%)": float(np.max(_er)) if _er else np.nan,
                         "수익률(연,%)": ((float((1 + rr).prod()) ** (TRADING_DAYS / len(rr))) - 1) * 100,
                         "변동성(연,%)": float(rr.std() * np.sqrt(TRADING_DAYS)) * 100,
                         "최악의 일간(%)": float(rr.min()) * 100,
                         "하락일 비율(%)": float((rr < 0).mean()) * 100,
                         "거래일": int(len(rr))})
    pdf = pd.DataFrame(prow)
    if not pdf.empty:
        piv = pdf.pivot(index="티커", columns="국면", values="수익률(연,%)")
        piv = piv[[c for c in ["강세", "횡보", "약세"] if c in piv.columns]]
        st.markdown("**국면별 수익률 (연율, %)**")
        st.dataframe(piv.style.format("{:.2f}", na_rep="-").map(_heat_color),
                     width="stretch")
        with st.expander("상세 지표 보기", expanded=False):
            st.dataframe(pdf.style.format({"에피소드": "{:.0f}",
                                           "에피소드 중앙값(%)": "{:+.2f}",
                                           "에피소드 최악(%)": "{:+.2f}",
                                           "에피소드 최선(%)": "{:+.2f}",
                                           "수익률(연,%)": "{:.2f}",
                                           "변동성(연,%)": "{:.2f}",
                                           "최악의 일간(%)": "{:.2f}",
                                           "하락일 비율(%)": "{:.1f}"}, na_rep="-"),
                         width="stretch", hide_index=True)
            st.caption("**누적 수익률과 최대낙폭은 표시하지 않습니다.** 같은 국면이라도 "
                       "시점이 흩어져 있어, 그 조각들을 이어붙인 값은 실제로 겪을 수 "
                       "있는 경험이 아니기 때문입니다.\n\n"
                       "대신 **에피소드별 통계**를 씁니다. 각 국면이 몇 번 있었고, "
                       "그때마다 얼마씩 움직였는지의 중앙값·최악·최선입니다. "
                       "**에피소드가 5회 미만이면 통계적 의미가 약합니다.** "
                       "연속 구간의 낙폭은 **🌩 스트레스 테스트** 화면에서 보세요.")
            _few = pdf[pdf["에피소드"] < 5]["국면"].unique()
            if len(_few):
                st.warning(f"에피소드가 5회 미만인 국면: {', '.join(_few)} — "
                           f"표본이 적어 우연일 수 있습니다.")
        if "약세" in piv.columns:
            best = piv["약세"].idxmax()
            st.info(f"약세장에서 가장 잘 버틴 자산은 **{best}** "
                    f"(연율 {piv.loc[best, '약세']:+.2f}%)입니다.")

    # ---------------- 국면별 상관관계 ----------------
    st.subheader("5️⃣ 국면별 상관관계 (Correlation by Regime)")
    st.caption("**분산투자의 가장 큰 취약점이 드러나는 곳입니다.** "
               "평상시 낮던 상관관계가 약세장에서 함께 올라가면, "
               "정작 방어가 필요한 순간에 분산 효과가 줄어듭니다.")
    if len(use_tk) < 2:
        st.info("상관관계를 보려면 2개 이상의 종목이 필요합니다.")
    else:
        crow = {}
        cmats = {}
        for reg in ["강세", "횡보", "약세"]:
            m = lab_r == reg
            if m.sum() < 20:
                continue
            cm = R.loc[m].corr()
            cmats[reg] = cm
            rho, eff = diversification_stats(cm)
            crow[reg] = {"표본(거래일)": int(m.sum()), "평균 상관계수": rho,
                         "실효 종목 수": eff}
        allc = R.corr()
        rho_a, eff_a = diversification_stats(allc)
        crow["전체"] = {"표본(거래일)": len(R), "평균 상관계수": rho_a, "실효 종목 수": eff_a}
        cdf = pd.DataFrame(crow).T
        st.dataframe(cdf.style.format({"표본(거래일)": "{:.0f}", "평균 상관계수": "{:.3f}",
                                       "실효 종목 수": "{:.2f}"}), width="stretch")

        if "강세" in crow and "약세" in crow:
            d = crow["약세"]["평균 상관계수"] - crow["강세"]["평균 상관계수"]
            if d > 0.1:
                st.warning(f"약세장 평균 상관계수가 강세장보다 **{d:+.2f}** 높습니다. "
                           f"실효 종목 수도 {crow['강세']['실효 종목 수']:.1f}개 → "
                           f"{crow['약세']['실효 종목 수']:.1f}개로 줄어듭니다. "
                           f"하락 국면에서 분산 효과가 약해진다는 뜻입니다.")
            elif d < -0.05:
                st.success(f"약세장 평균 상관계수가 강세장보다 **{d:+.2f}** 낮습니다. "
                           f"하락 국면에서도 분산 효과가 유지되는 구성입니다.")

        tabs_c = st.tabs([f"{k} 행렬" for k in cmats])
        for tb, (reg, cm) in zip(tabs_c, cmats.items()):
            with tb:
                st.dataframe(cm.round(2).style.format("{:.2f}").map(_corr_color),
                             width="stretch")

    # ---------------- 상관 추이 ----------------
    if len(use_tk) >= 2:
        st.subheader("6️⃣ 상관관계 추이와 국면 (Rolling Correlation)")
        q1, q2, q3 = st.columns([2, 2, 1])
        pa = q1.selectbox("종목 A", use_tk, index=0, key="_reg_a")
        rest = [t for t in use_tk if t != pa] or use_tk
        pb = q2.selectbox("종목 B", rest, index=0, key="_reg_b")
        win = q3.selectbox("구간", [63, 126, 252], index=1, key="_reg_win",
                           format_func=lambda x: f"{x}일")
        if pa != pb and len(R) > win + 5:
            roll = R[pa].rolling(win).corr(R[pb]).dropna()
            fr = go.Figure()
            for _, s in segs.iterrows():
                fr.add_vrect(x0=pd.Timestamp(s["시작"]), x1=pd.Timestamp(s["종료"]),
                             fillcolor=REG_COLORS.get(s["국면"], "#f8fafc"),
                             opacity=0.5, line_width=0, layer="below")
            fr.add_trace(go.Scatter(x=roll.index, y=roll.values,
                                    name=f"{pa} ↔ {pb}",
                                    line=dict(color="#1e293b", width=2)))
            fr.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
            fr.update_layout(height=360, hovermode="x unified", showlegend=False,
                             yaxis=dict(range=[-1.05, 1.05], title=None),
                             margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fr, width="stretch")
            byreg = {reg: float(roll[lab.reindex(roll.index).ffill() == reg].mean())
                     for reg in ["강세", "횡보", "약세"]
                     if (lab.reindex(roll.index).ffill() == reg).sum() > 5}
            if byreg:
                cols = st.columns(len(byreg))
                for c, (reg, v) in zip(cols, byreg.items()):
                    c.metric(f"{reg} 평균 상관", f"{v:+.2f}")
            st.caption("배경색이 국면입니다. 약세 구간에서 선이 올라간다면 "
                       "그 시기에 두 자산이 함께 움직였다는 뜻입니다.")

    # ---------------- 내보내기 ----------------
    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            segs.to_excel(xw, sheet_name="1_국면구간", index=False)
            if not pdf.empty:
                pdf.to_excel(xw, sheet_name="2_국면별성과", index=False)
            if len(use_tk) >= 2:
                cdf.to_excel(xw, sheet_name="3_국면별상관요약")
                for reg, cm in cmats.items():
                    cm.round(3).to_excel(xw, sheet_name=f"4_{reg}행렬")
            pd.DataFrame(list({
                "기준 지수": ref, "약세장 기준": f"-{bear_th*100:.0f}%",
                "횡보 판정 폭": f"±{side_band*100:.0f}%" if side_band else "미사용",
                "최소 지속": f"{min_days}거래일", "기준 통화": base_ccy,
                "기간": f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                "배당 처리": "재투자" if use_div else "주가만",
            }.items()), columns=["항목", "값"]).to_excel(xw, sheet_name="5_설정", index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"regimes_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_reg", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")
    assumptions_panel(period=f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                      extra=[("기준 지수", ref),
                             ("약세장 기준", f"고점 대비 −{bear_th*100:.0f}%"),
                             ("최소 지속", f"{min_days}거래일")])
    st.caption("국면 구분은 사후적으로 나눈 것이며, 실시간으로는 지금이 어느 국면인지 "
               "확정할 수 없습니다. 과거 국면별 특성이 앞으로도 반복된다는 보장도 없습니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")


CAND_COLS = ["티커", "종목명"]
CAND_MAX_COMB = 1000          # 전수 탐색 한도
CAND_MAX_ADD = 3
CAND_TOPN = 10


ADD_WEIGHTS = [1.0, 3.0, 5.0, 10.0]
FUND_PROP = "비례 축소 (모든 기존 자산에서 조금씩)"


def fund_weights(base: dict, cand: str, w_add: float, source: str):
    """
    후보를 w_add(0~1) 만큼 편입할 때 재원을 어디서 가져올지에 따른 최종 비중.
      · 비례 축소 — 기존 전부를 (1−w_add) 배로 줄인다
      · 특정 자산 — 그 자산에서만 차감하고, 모자라면 나머지에서 비례로 충당한다
    반환: (비중 dict, 경고문)
    """
    b = {k: float(v) for k, v in base.items() if float(v) > 0}
    s = sum(b.values())
    if s <= 0:
        return None, "기존 비중이 없습니다."
    b = {k: v / s for k, v in b.items()}
    if not (0 < w_add < 1):
        return None, "추가 비중은 0~100% 사이여야 합니다."

    warn = ""
    if source == FUND_PROP:
        out = {k: v * (1 - w_add) for k, v in b.items()}
    else:
        if source not in b:
            return None, f"{source} 가 기존 구성에 없습니다."
        if b[source] < w_add - 1e-12:
            short = w_add - b[source]
            warn = (f"{source} 비중이 {b[source]*100:.2f}% 뿐이라 "
                    f"{w_add*100:.2f}% 를 모두 뺄 수 없습니다. "
                    f"부족분 {short*100:.2f}%p 는 나머지 자산에서 비례로 충당했습니다.")
            out = dict(b)
            out[source] = 0.0
            rest = {k: v for k, v in out.items() if k != source and v > 0}
            rs = sum(rest.values())
            if rs <= 0:
                return None, "충당할 자산이 없습니다."
            for k in rest:
                out[k] -= short * (rest[k] / rs)
        else:
            out = dict(b)
            out[source] -= w_add
    out[cand] = out.get(cand, 0.0) + w_add
    return out, warn


def _cand_blank():
    return {"티커": "", "종목명": ""}


def objective_score(R: pd.DataFrame, w, goal: str, risk_name: str, rf: float):
    """조합의 우열을 가릴 스칼라 점수. 클수록 좋다."""
    mu, cov = ann_stats(R)
    w = np.asarray(w, dtype=float)
    if goal.startswith("수익률"):
        return float(w @ mu)
    if goal.startswith(("위험균형", "계층적")):
        # 두 방식은 목적함수가 따로 없으므로 샤프로 비교한다
        v = float(np.sqrt(w @ cov @ w))
        return (float(w @ mu) - rf) / v if v > 0 else -1e9
    d = RISK_DEFS[risk_name]
    s = _series(R, w)
    val = float(d["fn"](s, rf=rf))
    if goal.startswith("위험 최소화"):
        return val if d["kind"] == "ratio" else -val
    if d["kind"] == "ratio":
        return val
    return (float(w @ mu) - rf) / val if val > 1e-12 else -1e9


def clip_to_bounds(w, wmin, wmax, iters=100):
    """
    비중을 [wmin, wmax] 안으로 넣으면서 합계 1을 유지한다.
    HRP 처럼 제약을 직접 다루지 못하는 방식의 결과를 사후 조정할 때 쓴다.
    """
    n = len(w)
    lo = _as_bounds(wmin, n, 0.0)
    hi = _as_bounds(wmax, n, 1.0)
    if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
        return None
    x = np.clip(np.asarray(w, dtype=float), lo, hi)
    for _ in range(iters):
        gap = 1.0 - x.sum()
        if abs(gap) < 1e-12:
            break
        room = (hi - x) if gap > 0 else (x - lo)
        tot = room.sum()
        if tot <= 1e-12:
            break
        x = x + gap * room / tot
        x = np.clip(x, lo, hi)
    return x if abs(x.sum() - 1.0) < 1e-6 else None


def solve_combo(R: pd.DataFrame, goal, risk_name, rf, wmin, wmax, min_ret, max_vol):
    if goal.startswith("계층적"):
        # HRP 는 제약을 직접 다루지 못하므로 결과를 제약 안으로 투영한다
        return clip_to_bounds(hrp_weights(R), wmin, wmax)
    w, _, _ = optimize(R, goal, risk_name, rf, wmin, wmax, min_ret, max_vol)
    return w


def render_fixed_add(base_df, base_tk, cand_tk, base_ccy, start_date, end_date,
                     use_div, rf_rate, fx_hedge, gap_fill, cost_bp, meta_bad=None):
    """
    고정 비중 편입 효과.
    후보를 정해진 비중으로 넣었을 때 지표가 어떻게 달라지는지,
    그리고 그 재원을 어디서 가져오느냐에 따라 결과가 어떻게 갈리는지 본다.
    """
    st.subheader("4️⃣ 편입 조건 (Settings)")
    f1, f2, f3 = st.columns(3)
    weights_sel = f1.multiselect("편입 비중 (%)", ADD_WEIGHTS, default=[3.0, 5.0],
                                 key="_fx_w",
                                 help="여러 개를 고르면 비중별로 비교합니다.")
    fund_opts = [FUND_PROP] + list(base_tk)
    fund_src = f2.selectbox("재원 (어디서 빼서 넣을까)", fund_opts, key="_fx_src",
                            help="같은 비중을 넣어도 어느 자산에서 재원을 빼느냐에 따라 "
                                 "결과가 크게 달라집니다.")
    rebal_f = f3.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key="_fx_rebal")

    cmp_all = st.checkbox("재원별로 모두 비교", value=False, key="_fx_cmpall",
                          help="비례 축소와 각 자산에서 뺐을 때를 한 번에 비교합니다.")
    if cmp_all and not (len(cand_tk) == 1 and len(weights_sel) == 1):
        st.info("재원별 비교는 **후보 1개 · 편입 비중 1개**일 때만 표시됩니다. "
                f"지금은 후보 {len(cand_tk)}개 · 비중 {len(weights_sel)}개입니다. "
                "위 후보 표에서 종목 수를 1로 줄이고, 편입 비중도 하나만 선택하세요.")

    go_fx = st.button("📌 편입 효과 계산", type="primary", width="stretch",
                      disabled=bool(meta_bad) or not cand_tk or not weights_sel
                      or not base_tk)
    if go_fx:
        st.session_state["_fx_has_run"] = True
    if not st.session_state.get("_fx_has_run"):
        st.info("👆 편입 비중과 재원을 정한 뒤 **편입 효과 계산**을 눌러주세요.")
        return

    need = list(dict.fromkeys(list(base_tk) + list(cand_tk)))
    try:
        with st.spinner(f"{len(need)}개 종목 데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(need, start_date, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    base_use = [t for t in base_tk if t in prices.columns]
    cand_use = [t for t in cand_tk if t in prices.columns]
    if not base_use or not cand_use:
        st.error("기존 종목과 후보가 각각 1개 이상 필요합니다.")
        return
    if len(prices) < 60:
        st.error("공통 거래일이 부족합니다.")
        return

    bw = pd.to_numeric(base_df.set_index("티커")["현재 비중(%)"],
                       errors="coerce").reindex(base_use).fillna(0)
    if bw.sum() <= 0:
        bw = pd.Series(1.0, index=base_use)
    W_base = {t: float(v) for t, v in bw.items()}

    def run(W):
        try:
            r = portfolio_returns(prices, W, rebal_f)
            tno = turnover_series(prices, W, rebal_f).reindex(r.index).fillna(0)
            return apply_cost(r, tno, cost_bp)
        except Exception:
            return None

    r0 = run(W_base)
    if r0 is None:
        st.error("기존 포트폴리오를 계산하지 못했습니다.")
        return
    m0 = perf_row(r0, rf_rate)
    R = prices[base_use + cand_use].pct_change().dropna()
    # 상관관계 기준이 되는 기존 포트폴리오도 사용자가 고른 리밸런싱으로 계산한다
    # (예전에는 항상 일별로 계산해 편입효과와 가정이 어긋났다)
    base_r = pd.Series(portfolio_returns(prices, W_base, rebal_f)).reindex(R.index)

    st.success(f"✅ 계산 완료 · 기존 {len(base_use)}종목 · 후보 {len(cand_use)}개 · "
               f"{prices.index[0].date()} ~ {prices.index[-1].date()}")

    st.subheader("5️⃣ 편입 효과 (Impact of Adding)")
    st.caption("**변화 열은 모두 '기존 대비'** 입니다. 최대낙폭은 값이 커질수록"
               "(0에 가까울수록) 개선이므로 **개선(%p)** 으로 표기했습니다. "
               "양수면 낙폭이 얕아졌다는 뜻입니다.")

    rows, warns = [], set()
    for c in cand_use:
        for wa in sorted(weights_sel):
            W, warn = fund_weights(W_base, c, wa / 100.0, fund_src)
            if W is None:
                continue
            if warn:
                warns.add(warn)
            rr = run({k: v * 100 for k, v in W.items()})
            if rr is None:
                continue
            mm = perf_row(rr, rf_rate)
            corr_c = float(R[c].corr(base_r)) if c in R.columns else np.nan
            rows.append({
                "후보": c, "편입 비중(%)": wa,
                "수익률 변화(%p)": mm["수익률(연,%)"] - m0["수익률(연,%)"],
                "변동성 변화(%p)": mm["변동성(연,%)"] - m0["변동성(연,%)"],
                "MDD 개선(%p)": mm["최대낙폭(%)"] - m0["최대낙폭(%)"],
                "샤프 변화": mm["샤프지수"] - m0["샤프지수"],
                "소르티노 변화": mm["소르티노"] - m0["소르티노"],
                "기존과 상관": corr_c,
            })
    for w_ in warns:
        st.warning(w_)
    if not rows:
        st.error("계산 가능한 조합이 없습니다.")
        return

    fdf = pd.DataFrame(rows)
    st.markdown(f"**기존 포트폴리오** · 수익률 {m0['수익률(연,%)']:.2f}% · "
                f"변동성 {m0['변동성(연,%)']:.2f}% · MDD {m0['최대낙폭(%)']:.2f}% · "
                f"샤프 {m0['샤프지수']:.2f}")
    st.dataframe(
        fdf.style.format({"편입 비중(%)": "{:.0f}", "수익률 변화(%p)": "{:+.2f}",
                          "변동성 변화(%p)": "{:+.2f}", "MDD 개선(%p)": "{:+.2f}",
                          "샤프 변화": "{:+.3f}", "소르티노 변화": "{:+.3f}",
                          "기존과 상관": "{:.2f}"}, na_rep="-")
        .highlight_max(subset=["샤프 변화", "MDD 개선(%p)"], color="#dcfce7")
        .highlight_min(subset=["기존과 상관"], color="#dcfce7"),
        width="stretch", hide_index=True)
    st.caption("초록색 = 가장 유리한 값. **기존과 상관**이 낮을수록 분산 효과가 큽니다.")

    drows, srows = [], []
    with st.expander("상세 지표 보기 (최악의 월·분기)", expanded=False):
        st.caption("편입 전후로 **가장 나빴던 한 달·한 분기**가 어떻게 달라졌는지입니다. "
                   "평균 지표에는 드러나지 않는 충격의 크기를 보여줍니다.")
        m0_mon = float((1 + r0).resample("ME").prod().min() - 1) * 100
        m0_qtr = float((1 + r0).resample("QE").prod().min() - 1) * 100
        for c in cand_use:
            for wa in sorted(weights_sel):
                W, _w = fund_weights(W_base, c, wa / 100.0, fund_src)
                if W is None:
                    continue
                rr = run({k: v * 100 for k, v in W.items()})
                if rr is None:
                    continue
                mo = float((1 + rr).resample("ME").prod().min() - 1) * 100
                qt = float((1 + rr).resample("QE").prod().min() - 1) * 100
                drows.append({"후보": c, "편입 비중(%)": wa,
                              "최악의 월(%)": mo, "최악의 월 개선(%p)": mo - m0_mon,
                              "최악의 분기(%)": qt, "최악의 분기 개선(%p)": qt - m0_qtr})
        if drows:
            st.markdown(f"**기존** · 최악의 월 {m0_mon:.2f}% · 최악의 분기 {m0_qtr:.2f}%")
            st.dataframe(pd.DataFrame(drows).style.format(
                {"편입 비중(%)": "{:.0f}", "최악의 월(%)": "{:.2f}",
                 "최악의 월 개선(%p)": "{:+.2f}", "최악의 분기(%)": "{:.2f}",
                 "최악의 분기 개선(%p)": "{:+.2f}"})
                .highlight_max(subset=["최악의 월 개선(%p)", "최악의 분기 개선(%p)"],
                               color="#dcfce7"),
                width="stretch", hide_index=True)
            st.caption("**개선(%p)** 이 양수이면 편입 후 최악의 구간이 덜 나빠졌다는 뜻입니다.")

    if len(cand_use) > 1:
        pick_w = sorted(weights_sel)[len(weights_sel) // 2]
        sub = fdf[fdf["편입 비중(%)"] == pick_w].sort_values("샤프 변화",
                                                          ascending=False)
        if not sub.empty:
            fb = go.Figure(go.Bar(
                x=sub["샤프 변화"], y=sub["후보"], orientation="h",
                marker_color=["#0d9488" if v > 0 else "#dc2626"
                              for v in sub["샤프 변화"]],
                text=[f"{v:+.3f}" for v in sub["샤프 변화"]],
                textposition="outside"))
            fb.update_layout(height=60 + 40 * len(sub),
                             margin=dict(l=0, r=50, t=30, b=0),
                             title=f"{pick_w:.0f}% 편입 시 샤프지수 변화",
                             xaxis=dict(title=None),
                             yaxis=dict(autorange="reversed"))
            st.plotly_chart(fb, width="stretch")

    # ---------------- 재원별 비교 ----------------
    if cmp_all and len(cand_use) == 1 and len(weights_sel) == 1:
        st.subheader("6️⃣ 재원별 비교 (Funding Source)")
        st.caption("**같은 비중을 넣어도 어느 자산에서 재원을 빼느냐에 따라 결과가 "
                   "완전히 달라집니다.** 이 표가 그 차이를 보여줍니다.")
        c, wa = cand_use[0], sorted(weights_sel)[0]
        for src in [FUND_PROP] + list(base_use):
            W, warn = fund_weights(W_base, c, wa / 100.0, src)
            if W is None:
                continue
            rr = run({k: v * 100 for k, v in W.items()})
            if rr is None:
                continue
            mm = perf_row(rr, rf_rate)
            srows.append({
                "재원": "비례 축소" if src == FUND_PROP else f"{src} 에서",
                "수익률 변화(%p)": mm["수익률(연,%)"] - m0["수익률(연,%)"],
                "변동성 변화(%p)": mm["변동성(연,%)"] - m0["변동성(연,%)"],
                "MDD 개선(%p)": mm["최대낙폭(%)"] - m0["최대낙폭(%)"],
                "샤프 변화": mm["샤프지수"] - m0["샤프지수"],
                "비고": warn[:40] + "…" if warn else "",
            })
        if srows:
            sdf = pd.DataFrame(srows)
            st.dataframe(
                sdf.style.format({"수익률 변화(%p)": "{:+.2f}",
                                  "변동성 변화(%p)": "{:+.2f}",
                                  "MDD 개선(%p)": "{:+.2f}", "샤프 변화": "{:+.3f}"})
                .highlight_max(subset=["샤프 변화", "MDD 개선(%p)"], color="#dcfce7"),
                width="stretch", hide_index=True)
            best = sdf.loc[sdf["샤프 변화"].idxmax()]
            worst = sdf.loc[sdf["샤프 변화"].idxmin()]
            gap = float(best["샤프 변화"] - worst["샤프 변화"])
            st.info(f"**{c}** 를 {wa:.0f}% 넣을 때, 재원을 **{best['재원']}** 가져오면 "
                    f"샤프가 {best['샤프 변화']:+.3f} 변하고, "
                    f"**{worst['재원']}** 가져오면 {worst['샤프 변화']:+.3f} 변합니다. "
                    f"같은 편입인데 차이가 **{gap:.3f}** 입니다.")
    elif cmp_all:
        st.info("재원별 비교는 후보 1개 · 편입 비중 1개일 때만 동작합니다.")

    # ---------------- 요약 ----------------
    st.subheader("📝 요약 (Summary)")
    best_row = fdf.loc[fdf["샤프 변화"].idxmax()]
    worst_row = fdf.loc[fdf["샤프 변화"].idxmin()]
    lines = []
    src_lbl = "비례 축소" if fund_src == FUND_PROP else f"{fund_src} 에서 차감"
    lines.append(f"재원을 **{src_lbl}** 방식으로 조달했을 때, "
                 f"**{best_row['후보']}를 {best_row['편입 비중(%)']:.0f}% 편입**하는 것이 "
                 f"가장 유리했습니다. 샤프 {best_row['샤프 변화']:+.3f}, "
                 f"최대낙폭 {best_row['MDD 개선(%p)']:+.2f}%p, "
                 f"변동성 {best_row['변동성 변화(%p)']:+.2f}%p 변화입니다.")
    if float(best_row["샤프 변화"]) <= 0:
        lines.append("다만 **모든 조합에서 샤프지수가 개선되지 않았습니다.** "
                     "지금 구성에 굳이 추가할 필요가 없다는 뜻일 수 있습니다.")
    if float(worst_row["샤프 변화"]) < 0:
        lines.append(f"반대로 **{worst_row['후보']} {worst_row['편입 비중(%)']:.0f}%** 는 "
                     f"샤프가 {worst_row['샤프 변화']:+.3f} 로 오히려 나빠졌습니다.")
    low_corr = fdf.loc[fdf["기존과 상관"].idxmin()] if fdf["기존과 상관"].notna().any() else None
    if low_corr is not None:
        lines.append(f"기존 구성과 가장 덜 겹치는 후보는 **{low_corr['후보']}** "
                     f"(상관 {low_corr['기존과 상관']:.2f})입니다. "
                     f"분산 효과 관점에서는 이쪽이 유리합니다.")
    st.markdown("\n\n".join(lines))

    # ---------------- 다음 단계 ----------------
    st.subheader("➡️ 다음 단계")
    st.caption("기존 구성에 후보를 더한 상태를 보관함에 담아, 다른 화면에서 "
               "**📥 붙여넣기** 할 수 있습니다.")
    pick_c = st.selectbox("함께 담을 후보", cand_use, key="_fx_send")
    _add_w = st.select_slider("담을 편입 비중 (%)", options=sorted(weights_sel),
                              value=sorted(weights_sel)[0], key="_fx_planw")
    _fw, _warn2 = fund_weights(W_base, pick_c, _add_w / 100.0, fund_src)
    _fx_rows = ([{"티커": k, "비중": round(v * 100, 2)} for k, v in _fw.items()]
                if _fw else [])
    if st.button("📋 이 구성 복사", width="stretch", disabled=not _fx_rows):
        st.session_state[CLIP_KEY] = {"rows": _fx_rows, "from": "자산 추가 효과"}
        st.success(f"보관함에 담았습니다 · {clip_summary()}")

    st.markdown("**⚖️ 비교 후보로 담기**")
    plan_button("⚖️ 담기", _fx_rows, "자산 추가 효과", "_plan_fx",
                default_name=f"{pick_c} {_add_w:.0f}% 편입", rebalance=rebal_f)

    st.divider()
    st.subheader("📥 결과 내보내기")
    fx_settings = {
        "방식": "고정 비중 편입 효과",
        "기존 구성": " · ".join(f"{k} {v:.2f}%" for k, v in W_base.items()),
        "후보": ", ".join(cand_use),
        "편입 비중": ", ".join(f"{w:.0f}%" for w in sorted(weights_sel)),
        "재원": "비례 축소" if fund_src == FUND_PROP else f"{fund_src} 에서 차감",
        "리밸런싱": rebal_f,
        "기준 통화": base_ccy,
        "거래비용": f"{cost_bp:.0f}bp (편도)",
        "무위험 수익률": f"{rf_rate*100:.2f}%",
        "기간": f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
        "배당 처리": "재투자 (총수익)" if use_div else "주가만 (배당 제외)",
        "생성 시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    g1, g2 = st.columns(2)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            pd.DataFrame([{"항목": "기존 포트폴리오", **{
                "수익률(연,%)": m0["수익률(연,%)"], "변동성(연,%)": m0["변동성(연,%)"],
                "최대낙폭(%)": m0["최대낙폭(%)"], "샤프지수": m0["샤프지수"],
                "소르티노": m0["소르티노"]}}]).to_excel(
                xw, sheet_name="1_기존구성", index=False)
            fdf.to_excel(xw, sheet_name="2_편입효과", index=False)
            if drows:
                pd.DataFrame(drows).to_excel(xw, sheet_name="3_최악의월분기",
                                             index=False)
            if srows:
                pd.DataFrame(srows).to_excel(xw, sheet_name="4_재원별비교",
                                             index=False)
            pd.DataFrame(list(fx_settings.items()),
                         columns=["항목", "값"]).to_excel(
                xw, sheet_name="5_설정", index=False)
        g1.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"add_impact_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_fixed", width="stretch")
    except Exception as ex:
        g1.error(f"엑셀 생성 실패: {ex}")
    g2.download_button("📄 편입 효과 CSV", fdf.to_csv(index=False).encode("utf-8-sig"),
                       f"add_impact_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                       "text/csv", width="stretch")
    assumptions_panel(rebalance=rebal_f, cost=cost_bp,
                      period=f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                      extra=[("재원 조달", "비례 축소" if fund_src == FUND_PROP
                              else f"{fund_src} 에서 차감")])
    st.caption("과거 데이터를 기준으로 계산한 결과이며, 미래에도 같은 효과가 나타난다는 "
               "보장은 없습니다. 교육·참고용이며 투자 자문이 아닙니다.")


def render_candidate_search(base_ccy, start_date, end_date, use_div, rf_rate,
                            fx_hedge, gap_fill, cost_bp):
    st.title("➕ 자산 추가 효과")
    st.caption("기존 포트폴리오에 후보 종목 중 몇 개를 더했을 때 가장 좋아지는 조합을 찾습니다. "
               "선정은 **학습 구간** 데이터로만 하고, 결과는 **표본외 성과**와 함께 보여드립니다.")

    # 기존 포트폴리오 표는 최적화 화면과 **분리**한다.
    # 공유하면 한쪽을 고칠 때 다른 쪽이 말없이 바뀌어 혼란을 준다.
    st.subheader("1️⃣ 기존 포트폴리오 (Current Holdings)")
    base_live, base_tk, base_bad = render_ticker_table(
        state_key="_cand_base_df", editor_key="_cand_base_editor",
        gen_key="_cand_base_gen", cols=OPT_COLS, weight_col="현재 비중(%)",
        title="자산 추가 효과 · 기존", min_rows=1, max_rows=20,
        show_equal=True, equal_key="_cand_base_eq",
        extra_defaults={"최소(%)": 0.0, "최대(%)": 100.0},
        extra_config={
            "최소(%)": st.column_config.NumberColumn(
                "최소(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small"),
            "최대(%)": st.column_config.NumberColumn(
                "최대(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small")},
        help_text="지금 보유 중인 구성입니다. 다른 화면에서 만든 구성을 가져오려면 "
                  "그 화면에서 **📋 복사** 후 여기서 **📥 붙여넣기** 하세요.",
        default_rows=[{"티커": "SPY", "현재 비중(%)": 60.0,
                       "최소(%)": 0.0, "최대(%)": 100.0},
                      {"티커": "AGG", "현재 비중(%)": 40.0,
                       "최소(%)": 0.0, "최대(%)": 100.0}])
    base_df = base_live
    bw = pd.to_numeric(base_df["현재 비중(%)"], errors="coerce").fillna(0)
    st.caption(f"기존 {len(base_tk)}종목 · 비중 합계 **{bw.sum():.2f}%**")

    # ---------------- 후보 종목 ----------------
    st.subheader("2️⃣ 후보 종목 (Candidates)")
    cand_live, cand_all, cand_bad = render_ticker_table(
        state_key="_cand_df", editor_key="_cand_editor", gen_key="_cand_gen",
        cols=CAND_COLS, title="자산 추가 효과 · 후보", min_rows=1, max_rows=50,
        count_label="후보 수",
        help_text="추가를 고민 중인 종목을 넣으세요. **15~30개**가 적당합니다. "
                  "너무 많으면 계산이 길어지고, 1등이 우연일 가능성도 커집니다.",
        default_rows=[{"티커": x} for x in ("QQQ", "GLD", "EFA", "VNQ")])
    bad = list(base_bad) + list(cand_bad)
    cand_tk = [t_ for t_ in cand_all if t_ not in base_tk]
    dup = [t_ for t_ in cand_all if t_ in base_tk]

    ok_in, msgs = validate_setup(cand_tk, bad, min_n=1, label="후보 종목")
    show_msgs(msgs)
    if dup:
        st.info(f"기존 포트폴리오에 이미 있어 후보에서 제외했습니다: {', '.join(dup)}")

    # ---------------- 모드 ----------------
    st.subheader("3️⃣ 방식 선택 (Mode)")
    cmode = st.radio(
        "방식", ["📌 고정 비중 편입 효과", "🔎 최적 조합 탐색"], horizontal=True,
        key="_cand_mode", label_visibility="collapsed",
        help="고정 비중 — 정해진 비중(1·3·5·10%)으로 넣어보고 효과만 봅니다. "
             "빠르고 해석이 쉽습니다.\n\n"
             "최적 조합 — 어떤 조합이 가장 좋은지 최적화로 찾습니다.")

    if cmode.startswith("📌"):
        render_fixed_add(base_df, base_tk, cand_tk, base_ccy, start_date, end_date,
                         use_div, rf_rate, fx_hedge, gap_fill, cost_bp, meta_bad=bad)
        return

    # ---------------- 설정 ----------------
    st.subheader("4️⃣ 탐색 설정 (Settings)")
    a1, a2, a3 = st.columns(3)
    n_add = a1.number_input("추가할 종목 수", 1, CAND_MAX_ADD, 1, step=1, key="_cand_add")
    goal = a2.selectbox("목적", list(OPT_GOALS), key="_cand_goal")
    groups = {}
    for k, v in RISK_DEFS.items():
        groups.setdefault(v["group"], []).append(k)
    risk_labels = [f"{g} · {k}" for g in groups for k in groups[g]]
    rp = a3.selectbox("위험 정의", risk_labels, key="_cand_risk",
                      disabled=goal.startswith(("위험균형", "계층적")))
    risk_name = rp.split(" · ", 1)[1]

    b1, b2, b3 = st.columns(3)
    cut_label = b1.selectbox("최적화 기준일", list(CUTOFFS), index=2, key="_cand_cut")
    train_label = b2.selectbox("학습 기간", list(TRAIN_WINDOWS), index=1, key="_cand_train")
    rebal = b3.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key="_cand_rebal")
    d1, d2 = st.columns(2)
    wmax_pct = d1.slider("종목별 최대 비중 (%)", 10, 100, 100, step=5, key="_cand_wmax",
                         help="한 종목에 비중이 쏠리는 것을 막습니다. "
                              "기존 종목에 입력한 최대값과 함께 적용됩니다.")
    cand_min = d1.slider("후보 최소 편입 비중 (%)", 0.0, 20.0, 0.0, step=0.5,
                         key="_cand_minw",
                         help="0이면 후보가 조합에 뽑혀도 최적화 결과에서 0%를 받을 수 "
                              "있습니다. 값을 주면 실제로 그만큼은 담기게 됩니다.")
    bench_tk = d2.text_input("벤치마크", value="^GSPC", key="_cand_bench")
    n_reps = d2.slider("반복 검증 기준일 수", 1, 5, 1, step=1, key="_cand_reps",
                       help="기준일을 뒤로 옮겨가며 같은 탐색을 여러 번 합니다. "
                            "여러 기준일에서 반복해 뽑히는 후보가 우연이 아닐 "
                            "가능성이 큽니다. 횟수만큼 계산 시간이 늘어납니다.")

    n_c = len(cand_tk)
    from math import comb
    n_comb = comb(n_c, int(n_add)) if n_c >= int(n_add) else 0
    mode = "전수 탐색" if n_comb <= CAND_MAX_COMB else "단계적 선택"
    if n_comb == 0:
        st.warning("후보가 부족합니다.")
    else:
        st.caption(f"후보 **{n_c}개** 중 **{n_add}개** 조합 → **{n_comb:,}가지** · 방식 **{mode}**"
                   + ("" if mode == "전수 탐색" else
                      f" (한도 {CAND_MAX_COMB:,}가지 초과. 1개씩 순차로 고릅니다)"))
        if mode != "전수 탐색":
            st.warning("⚠️ **단계적 선택은 계산량을 줄이기 위한 근사 방식이며, 전체 조합의 "
                       "최적해를 보장하지 않습니다.** 한 종목씩 순차로 고르기 때문에 "
                       "'A와 B가 함께 있을 때만 좋아지는 조합'은 놓칠 수 있습니다. "
                       "후보 수나 추가 종목 수를 줄이면 전수 탐색으로 바뀝니다.")

    run = st.button("🔎 조합 탐색", type="primary", width="stretch",
                    disabled=bool(bad) or n_comb == 0 or len(base_tk) < 1)
    if run:
        st.session_state["_cand_has_run"] = True
    if not st.session_state.get("_cand_has_run"):
        st.info("👆 기존 포트폴리오와 후보를 입력한 뒤 **조합 탐색**을 눌러주세요.")
        return

    # ---------------- 데이터 ----------------
    bench_norm = normalize_ticker(bench_tk)[0] if bench_tk.strip() else None
    need = base_tk + cand_tk + ([bench_norm] if bench_norm else [])
    try:
        with st.spinner(f"{len(need)}개 종목 데이터 수집 중..."):
            prices, meta, _fx = build_price_frame(need, start_date, end_date,
                                                  base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return

    have = [t for t in base_tk + cand_tk if t in prices.columns]
    missing = [t for t in base_tk + cand_tk if t not in prices.columns]
    if missing:
        st.warning(f"데이터가 없어 제외합니다: {', '.join(missing)}")
    base_use = [t for t in base_tk if t in have]
    cand_use = [t for t in cand_tk if t in have]
    if not base_use or not cand_use:
        st.error("기존 종목과 후보가 각각 1개 이상 필요합니다.")
        return

    last = prices.index[-1]
    opt_date = last - CUTOFFS[cut_label] if CUTOFFS[cut_label] else last - pd.DateOffset(years=1)
    tw = TRAIN_WINDOWS[train_label]
    tr_start = prices.index[0] if tw is None else max(prices.index[0], opt_date - tw)
    train_px, oos_px = prices.loc[tr_start:opt_date], prices.loc[opt_date:]
    if len(train_px) < MIN_TRAIN_DAYS or len(oos_px) < MIN_OOS_DAYS:
        st.error(f"학습 {len(train_px)}일 / 검증 {len(oos_px)}일 — 기간이 부족합니다.")
        return

    wmax = wmax_pct / 100.0
    if wmax * (len(base_use) + int(n_add)) < 1.0:
        st.error(f"최대 비중 {wmax_pct}% 로는 100%를 채울 수 없습니다. 값을 높여주세요.")
        return

    # ---------------- 탐색 ----------------
    import itertools

    # 기존 종목의 최소·최대 제약을 조합 탐색에도 그대로 적용한다
    _blim = base_df.set_index("티커")

    def _bounds_for(tks):
        lo_, hi_ = [], []
        for t_ in tks:
            if t_ in _blim.index:
                lo_.append(float(_blim.loc[t_, "최소(%)"]) / 100)
                hi_.append(min(float(_blim.loc[t_, "최대(%)"]) / 100, wmax))
            else:                       # 후보 종목
                lo_.append(cand_min / 100.0)
                hi_.append(wmax)
        return np.array(lo_), np.array(hi_)

    def evaluate(combo, tr_px=None):
        tks = base_use + list(combo)
        _src = train_px if tr_px is None else tr_px
        Rtr = _src[tks].pct_change().dropna()
        if len(Rtr) < MIN_TRAIN_DAYS:
            return None
        lo_, hi_ = _bounds_for(tks)
        if lo_.sum() > 1.0 or hi_.sum() < 1.0 or (lo_ > hi_).any():
            return None
        try:
            w = solve_combo(Rtr, goal, risk_name, rf_rate, lo_, hi_, None, None)
            if w is None:
                return None
            score = objective_score(Rtr, w, goal, risk_name, rf_rate)
        except Exception:
            return None
        return {"combo": tuple(combo), "tickers": tks, "w": w, "score": score}

    results = []
    bar = st.progress(0.0, text="조합 평가 중...")
    if mode == "전수 탐색":
        combos = list(itertools.combinations(cand_use, int(n_add)))
        for i, cb in enumerate(combos):
            r = evaluate(cb)
            if r:
                results.append(r)
            if i % 5 == 0 or i == len(combos) - 1:
                bar.progress((i + 1) / len(combos),
                             text=f"조합 평가 중... {i+1}/{len(combos)}")
    else:
        chosen, pool = [], list(cand_use)
        total = sum(len(pool) - k for k in range(int(n_add)))
        done = 0
        for _ in range(int(n_add)):
            best = None
            for c in pool:
                r = evaluate(chosen + [c])
                done += 1
                bar.progress(min(1.0, done / max(1, total)),
                             text=f"조합 평가 중... {done}/{total}")
                if r and (best is None or r["score"] > best["score"]):
                    best = r
            if best is None:
                break
            newly = [x for x in best["combo"] if x not in chosen]
            chosen = list(best["combo"])
            results.append(best)
            for x in newly:
                if x in pool:
                    pool.remove(x)
        # 단계적 선택은 각 단계 최선만 남으므로, 마지막 단계의 후보들도 함께 평가
        if chosen:
            for c in cand_use:
                if c in chosen:
                    continue
                r = evaluate(chosen[:-1] + [c]) if len(chosen) >= 1 else None
                if r:
                    results.append(r)
    bar.empty()

    if not results:
        st.error("평가 가능한 조합이 없습니다. 기간이나 후보를 조정해주세요.")
        return

    # 중복 조합 제거 후 학습 구간 점수로 정렬
    uniq = {}
    for r in results:
        uniq[tuple(sorted(r["combo"]))] = r
    ranked = sorted(uniq.values(), key=lambda x: -x["score"])[:CAND_TOPN]

    # ---------------- 기존 포트폴리오 기준선 ----------------
    W_base = {t: float(x) for t, x in zip(base_df["티커"], bw) if t in base_use}
    if sum(W_base.values()) <= 0:
        W_base = {t: 1.0 for t in base_use}

    def bt(px, W):
        try:
            r = portfolio_returns(px, W, rebal)
            tno = turnover_series(px, W, rebal).reindex(r.index).fillna(0)
            return apply_cost(r, tno, cost_bp)
        except Exception:
            return None

    base_oos = bt(oos_px, W_base)
    bench_oos = None
    if bench_norm and bench_norm in oos_px.columns:
        # 벤치마크는 매수 후 보유이므로 거래비용을 적용하지 않는다
        # (🎯 포트폴리오 최적화 화면의 _bench() 와 같은 정책).
        try:
            bench_oos = portfolio_returns(oos_px, {bench_norm: 100.0}, REBAL_NONE)
        except Exception:
            bench_oos = None

    st.success(f"✅ 탐색 완료 · {mode} · 평가한 조합 **{len(uniq):,}가지** · "
               f"학습 {train_px.index[0].date()}~{opt_date.date()} · "
               f"검증 {opt_date.date()}~{last.date()}")

    # ---------------- 결과 ----------------
    st.subheader("5️⃣ 상위 조합 (Top Combinations)")
    st.warning("**순위는 학습 구간 기준입니다.** 표본외 성과로 순위를 매기면 검증의 의미가 "
               "사라지기 때문입니다. 오른쪽 표본외 지표를 보고 순위가 실제로 유지되는지 "
               "확인하세요. 1등과 5등의 차이가 미미하다면 그 순위는 큰 의미가 없습니다.")

    rows = []
    for rank, r in enumerate(ranked, start=1):
        W = {t: float(x) * 100 for t, x in zip(r["tickers"], r["w"])}
        oos = bt(oos_px, W)
        m = perf_row(oos, rf_rate) if oos is not None else {}
        added = " + ".join(r["combo"])
        _wmap = {t_: float(x_) * 100 for t_, x_ in zip(r["tickers"], r["w"])}
        _cw = " · ".join(f"{c_} {_wmap.get(c_, 0):.1f}%" for c_ in r["combo"])
        rows.append({
            "순위": rank, "추가 종목": added,
            "후보 편입비중": _cw,
            "학습 점수": r["score"],
            "표본외 수익률(%)": m.get("수익률(연,%)", np.nan),
            "표본외 변동성(%)": m.get("변동성(연,%)", np.nan),
            "표본외 샤프": m.get("샤프지수", np.nan),
            "표본외 MDD(%)": m.get("최대낙폭(%)", np.nan),
        })
    rdf = pd.DataFrame(rows).set_index("순위")

    if base_oos is not None:
        bm = perf_row(base_oos, rf_rate)
        rdf.loc["기존"] = ["(추가 없음)", "-", np.nan, bm["수익률(연,%)"],
                          bm["변동성(연,%)"], bm["샤프지수"], bm["최대낙폭(%)"]]
    if bench_oos is not None:
        cm = perf_row(bench_oos, rf_rate)
        rdf.loc["벤치마크"] = [bench_norm, "-", np.nan, cm["수익률(연,%)"],
                            cm["변동성(연,%)"], cm["샤프지수"], cm["최대낙폭(%)"]]
    st.dataframe(rdf.style.format({"학습 점수": "{:.3f}", "표본외 수익률(%)": "{:.2f}",
                                   "표본외 변동성(%)": "{:.2f}", "표본외 샤프": "{:.2f}",
                                   "표본외 MDD(%)": "{:.2f}"}, na_rep="-"),
                 width="stretch")
    st.caption("**MDD는 음수**입니다. −20%보다 −15%가 낙폭이 얕아 더 좋습니다. "
               "**후보 편입비중**은 최적화 결과 그 후보가 실제로 받은 비중입니다. "
               "0%에 가깝다면 조합에 뽑혔어도 실질적으로는 담기지 않은 것입니다.")

    # 순위 유지 여부 진단
    valid = rdf.iloc[:len(ranked)]["표본외 샤프"].dropna()
    if len(valid) >= 3:
        spread = float(valid.max() - valid.min())
        top_is_best = bool(valid.index[0] == valid.idxmax())
        if base_oos is not None:
            better = int((valid > perf_row(base_oos, rf_rate)["샤프지수"]).sum())
            st.caption(f"상위 {len(valid)}개 중 **{better}개**가 표본외에서 기존 구성보다 "
                       f"샤프지수가 높았습니다. "
                       + ("학습 구간 1위가 표본외에서도 최고였습니다."
                          if top_is_best else
                          "학습 구간 1위가 표본외 최고는 아니었습니다 — 순위의 신뢰도가 "
                          "높지 않다는 신호입니다.")
                       + f" 상위권 표본외 샤프 편차는 {spread:.2f} 입니다.")

    # ---------------- 반복 검증 ----------------
    rep_df = None
    if int(n_reps) > 1:
        st.subheader("6️⃣ 반복 검증 (Multiple Cutoffs)")
        st.caption("기준일을 뒤로 옮겨가며 같은 탐색을 반복합니다. "
                   "**여러 기준일에서 거듭 뽑히는 후보**가 우연이 아닐 가능성이 큽니다. "
                   "한 번만 1위였다면 그 시기의 우연일 수 있습니다.")
        rbar = st.progress(0.0, text="반복 검증 중...")
        rep_cnt, rep_detail = {}, []
        _step = pd.DateOffset(months=6)
        for k_ in range(int(n_reps)):
            cut_k = opt_date - _step * k_
            tr_s = prices.index[0] if tw is None else max(prices.index[0], cut_k - tw)
            tr_k = prices.loc[tr_s:cut_k]
            if len(tr_k) < MIN_TRAIN_DAYS:
                continue
            res_k = []
            pool_k = (list(itertools.combinations(cand_use, int(n_add)))
                      if mode == "전수 탐색" else
                      [tuple(sorted(x["combo"])) for x in uniq.values()])
            for cb in pool_k:
                r_ = evaluate(cb, tr_px=tr_k)
                if r_:
                    res_k.append(r_)
            if not res_k:
                continue
            top_k = sorted(res_k, key=lambda x: -x["score"])[:CAND_TOPN]
            for rk, r_ in enumerate(top_k, start=1):
                for c_ in r_["combo"]:
                    rep_cnt[c_] = rep_cnt.get(c_, 0) + 1
            rep_detail.append({"기준일": cut_k.date(),
                               "학습 시작": tr_k.index[0].date(),
                               "1위 조합": " + ".join(top_k[0]["combo"]),
                               "학습 점수": top_k[0]["score"]})
            rbar.progress((k_ + 1) / int(n_reps),
                          text=f"반복 검증 중... {k_+1}/{int(n_reps)}")
        rbar.empty()

        if rep_detail:
            st.dataframe(pd.DataFrame(rep_detail).style.format({"학습 점수": "{:.3f}"}),
                         width="stretch", hide_index=True)
            _runs = len(rep_detail)
            rep_df = pd.DataFrame(
                [{"티커": c_, "종목명": meta.get(c_, {}).get("name", c_),
                  "상위권 등장": v_, "등장률(%)": v_ / (_runs * CAND_TOPN) * 100}
                 for c_, v_ in sorted(rep_cnt.items(), key=lambda x: -x[1])])
            st.markdown(f"**후보별 등장 횟수** — {_runs}개 기준일 × 상위 {CAND_TOPN}개 조합")
            st.dataframe(rep_df.style.format({"등장률(%)": "{:.0f}"}),
                         width="stretch", hide_index=True)
            if len(rep_df):
                _top = rep_df.iloc[0]
                st.info(f"**{_top['티커']}** 가 가장 자주 등장했습니다 "
                        f"({_top['상위권 등장']}회, {_top['등장률(%)']:.0f}%). "
                        f"기준일을 바꿔도 반복해 뽑힌다면 특정 시기의 우연이 아닐 "
                        f"가능성이 큽니다.")
            _once = rep_df[rep_df["상위권 등장"] <= 1]["티커"].tolist()
            if _once:
                st.caption(f"한 번만 등장한 후보: {', '.join(_once)} — "
                           f"우연일 수 있으니 신중하게 보세요.")
        else:
            st.warning("반복 검증에 필요한 데이터가 부족합니다. "
                       "분석 시작일을 앞당기거나 기준일 수를 줄여주세요.")

    # ---------------- 채택 빈도 ----------------
    st.subheader("7️⃣ 후보별 채택 빈도 (Selection Frequency)")
    st.caption("상위 조합에 자주 등장하는 종목일수록, 특정 조합의 우연이 아니라 "
               "실제로 도움이 되는 후보일 가능성이 큽니다.")
    cnt = {}
    for r in ranked:
        for t in r["combo"]:
            cnt[t] = cnt.get(t, 0) + 1
    fdf = pd.DataFrame([{"티커": t, "종목명": meta.get(t, {}).get("name", t),
                         "상위 조합 등장": c, "비율(%)": c / len(ranked) * 100}
                        for t, c in sorted(cnt.items(), key=lambda x: -x[1])])
    if not fdf.empty:
        fb = go.Figure(go.Bar(x=fdf["비율(%)"], y=fdf["티커"], orientation="h",
                              marker_color="#0d9488",
                              text=[f"{v:.0f}%" for v in fdf["비율(%)"]],
                              textposition="outside"))
        fb.update_layout(height=60 + 38 * len(fdf), margin=dict(l=0, r=40, t=10, b=0),
                         xaxis=dict(title=None, ticksuffix="%"),
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(fb, width="stretch")
        st.dataframe(fdf.style.format({"비율(%)": "{:.0f}"}), width="stretch",
                     hide_index=True)

    # ---------------- 1위 조합 상세 ----------------
    st.subheader("8️⃣ 1위 조합 비중 (Best Combination)")
    best = ranked[0]
    bw2 = pd.DataFrame({
        "티커": best["tickers"],
        "종목명": [meta.get(t, {}).get("name", t) for t in best["tickers"]],
        "구분": ["기존" if t in base_use else "추가" for t in best["tickers"]],
        "최적 비중(%)": best["w"] * 100,
    }).sort_values("최적 비중(%)", ascending=False)
    st.dataframe(bw2.style.format({"최적 비중(%)": "{:.2f}"}), width="stretch",
                 hide_index=True)

    oos_best = bt(oos_px, {t: float(x) * 100 for t, x in zip(best["tickers"], best["w"])})
    curves = {}
    if base_oos is not None:
        curves["기존 포트폴리오"] = base_oos
    if oos_best is not None:
        curves[f"1위 조합 (+{' + '.join(best['combo'])})"] = oos_best
    if bench_oos is not None:
        curves[f"벤치마크 ({bench_norm})"] = bench_oos
    if curves:
        fg = go.Figure()
        for nm, rr in curves.items():
            ec = equity_curve(rr)
            fg.add_trace(go.Scatter(x=ec.index, y=ec.values, name=nm, line=dict(width=2)))
        fg.update_layout(height=380, hovermode="x unified",
                         margin=dict(l=0, r=0, t=30, b=0),
                         legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(fg, width="stretch")
        st.caption("검증 구간(표본외) 성과입니다.")

    st.subheader("⚖️ 비교 후보로 담기")
    _best_rows = [{"티커": t_, "비중": round(float(x_) * 100, 2)}
                  for t_, x_ in zip(best["tickers"], best["w"]) if x_ > 1e-6]
    plan_button("⚖️ 1위 조합 담기", _best_rows, "자산 추가 효과 · 조합탐색",
                "_plan_cand",
                default_name=f"+{' + '.join(best['combo'])}", rebalance=rebal)

    st.divider()
    st.subheader("📥 결과 내보내기")
    cs_settings = {
        "방식": f"최적 조합 탐색 ({mode})",
        "기존 구성": ", ".join(base_use),
        "후보": ", ".join(cand_use),
        "추가 종목 수": f"{int(n_add)}개",
        "평가한 조합": f"{len(uniq):,}가지",
        "반복 검증": f"{int(n_reps)}개 기준일" if int(n_reps) > 1 else "미사용",
        "목적": goal,
        "위험 정의": risk_name,
        "최적화 기준일": str(opt_date.date()),
        "학습 기간": f"{train_label} ({train_px.index[0].date()} ~ {opt_date.date()})",
        "검증 구간": f"{opt_date.date()} ~ {last.date()}",
        "리밸런싱": rebal,
        "종목별 최대 비중": f"{wmax_pct}%",
        "벤치마크": bench_norm or "-",
        "기준 통화": base_ccy,
        "거래비용": f"{cost_bp:.0f}bp (편도)",
        "생성 시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    h1, h2 = st.columns(2)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            rdf.to_excel(xw, sheet_name="1_상위조합")
            if not fdf.empty:
                fdf.to_excel(xw, sheet_name="2_채택빈도", index=False)
            bw2.to_excel(xw, sheet_name="3_1위조합비중", index=False)
            if rep_df is not None and not rep_df.empty:
                rep_df.to_excel(xw, sheet_name="6_반복검증", index=False)
            if curves:
                pd.DataFrame({k: equity_curve(v) for k, v in curves.items()}).to_excel(
                    xw, sheet_name="4_성과추이")
            pd.DataFrame(list(cs_settings.items()),
                         columns=["항목", "값"]).to_excel(
                xw, sheet_name="5_설정", index=False)
        h1.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"candidate_search_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_cand", width="stretch")
    except Exception as ex:
        h1.error(f"엑셀 생성 실패: {ex}")
    h2.download_button("📄 결과 CSV", rdf.to_csv().encode("utf-8-sig"),
                       f"candidate_search_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                       "text/csv", width="stretch")
    st.caption("후보가 많을수록 상위권이 우연히 좋아 보일 가능성이 커집니다. "
               "1등 하나만 보지 말고 상위 조합 전반과 채택 빈도를 함께 참고하세요. "
               "교육·참고용이며 투자 자문이 아닙니다.")


CORR_COLS = ["티커", "종목명"]
CORR_FREQ = {"일별": None, "주별": "W", "월별": "ME"}
CORR_PERIODS = {"1년": 1, "3년": 3, "5년": 5, "10년": 10, "전체 기간": None}
CORR_MAX_HINT = 15


def _corr_blank():
    return {"티커": "", "종목명": ""}


def corr_returns(prices: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    """선택한 주기로 환산한 수익률."""
    rule = CORR_FREQ[freq_label]
    px = prices if rule is None else prices.resample(rule).last()
    return px.pct_change().dropna(how="any")


def diversification_stats(corr: pd.DataFrame):
    """평균 상관계수와 실효 종목 수. 실효 = N / (1 + (N-1)ρ̄)"""
    n = corr.shape[0]
    if n < 2:
        return np.nan, float(n)
    off = corr.values[~np.eye(n, dtype=bool)]
    rho = float(np.nanmean(off))
    denom = 1 + (n - 1) * rho
    eff = float(n / denom) if denom > 1e-9 else float(n)
    # 평균 상관이 음수면 실효 종목 수가 실제 종목 수를 넘을 수 있다.
    # 서로 헤지되는 구성이라는 뜻이므로 상한을 두지 않는다.
    return rho, max(1.0, eff)


def cluster_assets(corr: pd.DataFrame, rho_cut=0.6, method="average") -> pd.Series:
    """
    상관계수를 거리로 바꿔 계층 군집화한다.
      거리 d = √((1−ρ)/2)  ·  ρ=1 → 0,  ρ=0 → 0.707,  ρ=−1 → 1
    ρ_cut 이상으로 함께 움직이는 자산이 한 군집으로 묶인다.
    """
    n = len(corr)
    if n < 2:
        return pd.Series([1] * n, index=corr.index)
    d = np.sqrt(np.clip((1 - corr.values) / 2, 0, 1))
    np.fill_diagonal(d, 0.0)
    try:
        Z = linkage(squareform(d, checks=False), method=method)
        thr = np.sqrt(max(0.0, (1 - rho_cut) / 2))
        lab = fcluster(Z, t=thr, criterion="distance")
    except Exception:
        lab = np.arange(1, n + 1)
    return pd.Series(lab, index=corr.index)


def cluster_quality(corr: pd.DataFrame, lab: pd.Series):
    """군집 내 평균 상관과 군집 간 평균 상관. 차이가 클수록 분리가 뚜렷하다."""
    ks = list(corr.index)
    within, between = [], []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            v = float(corr.iloc[i, j])
            (within if lab[ks[i]] == lab[ks[j]] else between).append(v)
    return (float(np.mean(within)) if within else np.nan,
            float(np.mean(between)) if between else np.nan)


def corr_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """중복 없이 모든 쌍을 나열."""
    rows = []
    cols = list(corr.columns)
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            rows.append({"종목 A": cols[a], "종목 B": cols[b],
                         "상관계수": float(corr.iloc[a, b])})
    return pd.DataFrame(rows).sort_values("상관계수").reset_index(drop=True)


def render_correlations(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill):
    st.title("🔗 자산 상관관계")
    st.caption("종목들이 서로 얼마나 함께 움직이는지 확인합니다. "
               "낮은 상관관계가 많을수록 분산 효과가 큽니다.")

    live, tickers, bad = render_ticker_table(
        state_key="_corr_df", editor_key="_corr_editor", gen_key="_corr_gen",
        cols=CORR_COLS, title="자산 상관관계", min_rows=2, max_rows=30,
        default_rows=[{"티커": x} for x in ("005930.KS", "SPY", "QQQ", "GLD")])

    ok_in, msgs = validate_setup(tickers, bad, min_n=2)
    show_msgs(msgs)
    if len(tickers) > CORR_MAX_HINT:
        st.info(f"종목이 {len(tickers)}개입니다. {CORR_MAX_HINT}개를 넘으면 행렬이 커져 "
                f"화면에서 읽기 어려울 수 있습니다.")

    # ---------------- 설정 ----------------
    st.subheader("2️⃣ 계산 설정 (Settings)")
    s1, s2 = st.columns([1, 2])
    freq = s1.selectbox("계산 주기", list(CORR_FREQ), index=0, key="_corr_freq")
    s2.markdown("&nbsp;", unsafe_allow_html=True)
    if freq == "일별":
        s2.caption("⚠️ 거래 시간대가 다른 국가를 섞으면 일별 상관계수가 실제보다 **낮게** "
                   "나옵니다. 미국 시장의 밤사이 움직임이 한국에는 다음 날 반영되기 "
                   "때문입니다. 국가를 섞었다면 **주별 또는 월별**을 권합니다.")
    else:
        s2.caption(f"{freq} 수익률로 계산합니다. 표본 수는 줄지만 시차 문제가 완화됩니다.")

    run_corr = st.button("🔗 상관관계 계산", type="primary", width="stretch",
                         disabled=bool(bad) or len(tickers) < 2)
    if run_corr:
        st.session_state["_corr_has_run"] = True
    if not st.session_state.get("_corr_has_run"):
        st.info("👆 종목을 2개 이상 입력한 뒤 **상관관계 계산**을 눌러주세요.")
        return
    if len(tickers) < 2:
        st.error("최소 2개 종목이 필요합니다.")
        return

    # ---------------- 데이터 ----------------
    try:
        with st.spinner("데이터 수집 중..."):
            prices, meta, fx_used = build_price_frame(
                tickers, start_date, end_date, base_ccy, use_div, fx_hedge, gap_fill)
    except Exception as ex:
        st.error(f"데이터를 가져오지 못했습니다: {ex}")
        return
    if prices.empty or len(prices) < 30:
        st.error("공통 거래일이 부족합니다. 기간이나 종목을 확인해주세요.")
        return

    R = corr_returns(prices[tickers], freq)
    if len(R) < 10:
        st.error(f"{freq} 표본이 {len(R)}개뿐입니다. 기간을 늘리거나 주기를 짧게 바꿔주세요.")
        return
    corr = R.corr().round(3)

    st.success(f"✅ 계산 완료 · {freq} 수익률 {len(R):,}개 "
               f"({prices.index[0].date()} ~ {prices.index[-1].date()}) · 기준통화 {base_ccy}")

    # ---------------- 행렬 ----------------
    st.subheader("3️⃣ 상관관계 행렬 (Correlation Matrix)")
    st.caption("1에 가까우면 함께 움직이고, 0이면 서로 무관하며, 음수면 반대로 움직입니다. "
               "**푸른색일수록 분산 효과가 큽니다.**")
    st.dataframe(corr.style.format("{:.2f}").map(_corr_color), width="stretch")

    # ---------------- 요약 ----------------
    st.subheader("4️⃣ 분산 효과 요약 (Diversification)")
    rho, eff = diversification_stats(corr)
    pairs = corr_pairs(corr)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("종목 수", f"{len(tickers)}개")
    k2.metric("평균 상관계수", f"{rho:.2f}")
    k3.metric("실효 종목 수", f"{eff:.1f}개",
              help="등가중 가정에서 위험 분산 효과가 몇 종목에 해당하는지입니다. "
                   "N / (1 + (N−1)×평균상관) 으로 계산합니다.")
    k4.metric("분산 활용도", f"{eff/len(tickers)*100:.0f}%",
              help="실효 종목 수 ÷ 실제 종목 수. 낮을수록 비슷한 종목이 많다는 뜻입니다. "
                   "평균 상관계수가 음수이면 100%를 넘을 수 있습니다.")

    if eff > len(tickers):
        st.success(f"평균 상관계수가 **{rho:+.2f}** 로 음수입니다. 종목들이 서로 반대 방향으로 "
                   f"움직여, 서로 무관한 {len(tickers)}종목보다도 분산 효과가 큽니다.")

    if rho > 0.7:
        st.warning(f"평균 상관계수가 **{rho:.2f}** 로 높습니다. {len(tickers)}종목을 담았지만 "
                   f"분산 효과는 사실상 **{eff:.1f}종목** 수준입니다. "
                   f"성격이 다른 자산군을 추가하는 것을 고려해보세요.")
    elif rho < 0.3:
        st.success(f"평균 상관계수가 **{rho:.2f}** 로 낮아 분산 효과가 잘 나타나고 있습니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**분산 효과가 큰 쌍** (상관계수 낮은 순)")
        st.dataframe(pairs.head(5).style.format({"상관계수": "{:.2f}"}),
                     width="stretch", hide_index=True)
    with c2:
        st.markdown("**서로 비슷하게 움직이는 쌍** (상관계수 높은 순)")
        st.dataframe(pairs.tail(5).iloc[::-1].style.format({"상관계수": "{:.2f}"}),
                     width="stretch", hide_index=True)

    # ---------------- 군집 ----------------
    st.subheader("5️⃣ 위험군 묶기 (Clusters)")
    st.caption("비슷하게 움직이는 자산을 하나의 **위험군**으로 묶습니다. "
               "종목 수가 많아도 같은 군집에 몰려 있다면 실제 분산은 그만큼 되지 않습니다.")
    if len(tickers) < 3:
        st.info("군집을 나누려면 3개 이상의 종목이 필요합니다.")
        cl = None
    else:
        rho_cut = st.slider("묶는 기준 상관계수", 0.0, 0.95, 0.60, step=0.05,
                            key="_corr_cut",
                            help="이 값 이상으로 함께 움직이면 같은 위험군으로 봅니다. "
                                 "높일수록 잘게, 낮출수록 크게 묶입니다.")
        cl = cluster_assets(corr, rho_cut)
        grp = {}
        for tk_, g in cl.items():
            grp.setdefault(int(g), []).append(tk_)
        wi, be = cluster_quality(corr, cl)

        c1, c2, c3 = st.columns(3)
        c1.metric("자산 수", f"{len(tickers)}개")
        c2.metric("위험군 수", f"{len(grp)}개")
        c3.metric("군집 내 / 군집 간 상관", f"{wi:.2f} / {be:.2f}"
                  if not (pd.isna(wi) or pd.isna(be)) else "-",
                  help="군집 내가 군집 간보다 훨씬 높아야 분리가 의미 있습니다.")

        rows_g = []
        for g, mem in sorted(grp.items()):
            sub = corr.loc[mem, mem]
            inner = (sub.values[~np.eye(len(mem), dtype=bool)].mean()
                     if len(mem) > 1 else np.nan)
            rows_g.append({"위험군": f"군집 {g}", "자산 수": len(mem),
                           "구성": ", ".join(mem),
                           "군집 내 평균 상관": inner})
        st.dataframe(pd.DataFrame(rows_g).style.format(
            {"군집 내 평균 상관": "{:.2f}"}, na_rep="-"),
            width="stretch", hide_index=True)

        big = max(grp.values(), key=len)
        if len(grp) < len(tickers):
            st.warning(f"자산은 **{len(tickers)}개**지만 상관관계 기준 실질 위험군은 "
                       f"**{len(grp)}개**입니다. 가장 큰 군집에 "
                       f"**{', '.join(big)}** ({len(big)}종목)이 함께 묶여 있어, "
                       f"이들은 사실상 하나처럼 움직입니다.")
        else:
            st.success(f"{len(tickers)}개 자산이 모두 서로 다른 위험군으로 나뉩니다. "
                       f"분산 구조가 잘 잡혀 있습니다.")

        fcl = go.Figure(go.Bar(
            x=[len(m) for _, m in sorted(grp.items())],
            y=[f"군집 {g}" for g, _ in sorted(grp.items())],
            orientation="h", marker_color="#4f46e5",
            text=[", ".join(m) for _, m in sorted(grp.items())],
            textposition="inside", insidetextanchor="start"))
        fcl.update_layout(height=60 + 44 * len(grp),
                          margin=dict(l=0, r=20, t=10, b=0),
                          xaxis=dict(title="자산 수", dtick=1),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fcl, width="stretch")
        st.caption("**실효 종목 수**가 분산의 '양'을 재는 숫자라면, 위험군은 그 '구조'를 "
                   "보여줍니다. 두 값이 비슷하면 자산들이 고르게 퍼져 있다는 뜻입니다.")

    # ---------------- 기간별 ----------------
    st.subheader("6️⃣ 기간별 상관관계 (By Period)")
    st.caption("상관관계는 고정된 값이 아닙니다. 기간에 따라 크게 달라지며, "
               "특히 **위기 국면에서는 대부분의 자산이 함께 움직입니다.**")
    last = prices.index[-1]
    prows = {}
    for lbl, yrs in CORR_PERIODS.items():
        sub = prices[tickers] if yrs is None else \
            prices[tickers].loc[last - pd.DateOffset(years=yrs):]
        Rp = corr_returns(sub, freq)
        if len(Rp) < 10:
            continue
        cp = Rp.corr()
        r_, e_ = diversification_stats(cp)
        prows[lbl] = {"표본 수": len(Rp), "평균 상관계수": r_,
                      "실효 종목 수": e_,
                      "최저 쌍": float(corr_pairs(cp)["상관계수"].min()),
                      "최고 쌍": float(corr_pairs(cp)["상관계수"].max())}
    pdf = pd.DataFrame(prows).T
    if not pdf.empty:
        st.dataframe(pdf.style.format({"표본 수": "{:.0f}", "평균 상관계수": "{:.2f}",
                                       "실효 종목 수": "{:.1f}", "최저 쌍": "{:.2f}",
                                       "최고 쌍": "{:.2f}"}), width="stretch")

    # ---------------- 국면별 · 위기별 ----------------
    st.subheader("7️⃣ 국면별·위기별 상관관계 (By Regime & Crisis)")
    st.caption("**분산투자가 가장 시험받는 지점입니다.** 평상시 낮던 상관관계가 하락 "
               "국면이나 위기에 함께 올라가면, 정작 방어가 필요한 순간에 분산 효과가 "
               "줄어듭니다.")
    if len(tickers) < 2:
        st.info("2개 이상의 종목이 필요합니다.")
    else:
        with st.expander("펼쳐서 계산하기 (기준 지수를 추가로 조회합니다)",
                         expanded=False):
            rc1, rc2 = st.columns([2, 1])
            ref_c = rc1.text_input("국면 판정 기준 지수", value="^GSPC",
                                   key="_corr_ref",
                                   help="이 지수를 기준으로 강세·약세·횡보를 나눕니다.")
            bear_c = rc2.slider("약세장 기준 (%)", 5, 40, 20, step=1,
                                key="_corr_bear") / 100
            use_crisis = st.checkbox("과거 위기 구간도 함께 보기", value=True,
                                     key="_corr_crisis",
                                     help="닷컴 붕괴·금융위기·코로나·긴축 발작 등 "
                                          "미리 정의된 구간을 사용합니다.")

            if st.button("🔗 국면별 상관관계 계산", width="stretch", key="_corr_reg_go"):
                st.session_state["_corr_reg_run"] = True

            if st.session_state.get("_corr_reg_run"):
                ref_n = normalize_ticker(ref_c)[0] if ref_c.strip() else "^GSPC"
                lo_c = (min(pd.Timestamp(v[0]) for v in CRISIS_PRESETS.values())
                        if use_crisis else pd.Timestamp(start_date))
                fs = min(pd.Timestamp(start_date), lo_c - pd.DateOffset(months=6))
                try:
                    with st.spinner("기준 지수 조회 중..."):
                        px2, _m2, _f2 = build_price_frame(
                            tickers + [ref_n], fs, end_date, base_ccy,
                            use_div, fx_hedge, gap_fill)
                except Exception as ex:
                    st.error(f"데이터를 가져오지 못했습니다: {ex}")
                    px2 = None

                if px2 is not None and ref_n in px2.columns:
                    tk2 = [x for x in tickers if x in px2.columns]
                    R2 = corr_returns(px2[tk2], freq)
                    lab2 = classify_regime(px2[ref_n], bear=bear_c, bull=bear_c)
                    lab2 = lab2.reindex(R2.index).ffill()

                    rows_r, mats = {}, {}
                    for reg in ["강세", "횡보", "약세"]:
                        sel = lab2 == reg
                        if sel.sum() < 20:
                            continue
                        cm = R2[sel].corr()
                        mats[reg] = cm
                        rho_r, eff_r = diversification_stats(cm)
                        rows_r[reg] = {"표본": int(sel.sum()), "평균 상관계수": rho_r,
                                       "실효 종목 수": eff_r}
                    if use_crisis:
                        cmask = pd.Series(False, index=R2.index)
                        for _nm, (a_, b_) in CRISIS_PRESETS.items():
                            cmask |= ((R2.index >= pd.Timestamp(a_)) &
                                      (R2.index <= pd.Timestamp(b_)))
                        if cmask.sum() >= 20:
                            cm = R2[cmask].corr()
                            mats["위기 구간"] = cm
                            rho_r, eff_r = diversification_stats(cm)
                            rows_r["위기 구간"] = {"표본": int(cmask.sum()),
                                              "평균 상관계수": rho_r,
                                              "실효 종목 수": eff_r}
                    rho_a2, eff_a2 = diversification_stats(R2.corr())
                    rows_r["전체"] = {"표본": len(R2), "평균 상관계수": rho_a2,
                                    "실효 종목 수": eff_a2}

                    st.session_state["_corr_reg_result"] = pd.DataFrame(rows_r).T
                    st.dataframe(pd.DataFrame(rows_r).T.style.format(
                        {"표본": "{:.0f}", "평균 상관계수": "{:.3f}",
                         "실효 종목 수": "{:.2f}"}), width="stretch")

                    if "강세" in rows_r and "약세" in rows_r:
                        d_ = (rows_r["약세"]["평균 상관계수"] -
                              rows_r["강세"]["평균 상관계수"])
                        if d_ > 0.1:
                            st.warning(
                                f"약세장 평균 상관계수가 강세장보다 **{d_:+.2f}** 높습니다. "
                                f"실효 종목 수도 {rows_r['강세']['실효 종목 수']:.1f}개 → "
                                f"**{rows_r['약세']['실효 종목 수']:.1f}개** 로 줄어듭니다.")
                        elif d_ < -0.05:
                            st.success(f"약세장 평균 상관계수가 강세장보다 "
                                       f"**{d_:+.2f}** 낮습니다. 하락 국면에서도 "
                                       f"분산이 유지되는 구성입니다.")

                    if mats:
                        tabs_r = st.tabs([f"{k} 행렬" for k in mats])
                        for tb, (rk, cm) in zip(tabs_r, mats.items()):
                            with tb:
                                st.dataframe(cm.round(2).style.format("{:.2f}")
                                             .map(_corr_color), width="stretch")
                    st.caption("국면 구간이 언제부터 언제까지였는지, 국면마다 각 자산이 "
                               "어떤 성과를 냈는지는 **📉 시장 국면 분석** 화면에서 "
                               "볼 수 있습니다. 위기 구간의 낙폭과 회복 기간은 "
                               "**🌩 스트레스 테스트** 화면에 있습니다.")

    # ---------------- 롤링 ----------------
    st.subheader("8️⃣ 상관관계 추이 (Rolling Correlation)")
    r1, r2, r3 = st.columns([2, 2, 1])
    pick_a = r1.selectbox("종목 A", tickers, index=0, key="_corr_a")
    others = [t for t in tickers if t != pick_a] or tickers
    pick_b = r2.selectbox("종목 B", others, index=0, key="_corr_b")
    win_map = {"일별": [63, 126, 252], "주별": [26, 52, 104], "월별": [12, 24, 36]}[freq]
    unit = {"일별": "일", "주별": "주", "월별": "개월"}[freq]
    win = r3.selectbox("구간", win_map, index=1, key="_corr_win",
                       format_func=lambda x: f"{x}{unit}")
    if pick_a != pick_b and len(R) > win + 5:
        roll = R[pick_a].rolling(win).corr(R[pick_b]).dropna()
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=roll.index, y=roll.values,
                                name=f"{pick_a} ↔ {pick_b}",
                                line=dict(color="#2563eb", width=2)))
        fr.add_hline(y=roll.mean(), line_dash="dot", line_color="#94a3b8",
                     annotation_text=f"평균 {roll.mean():.2f}")
        fr.add_hline(y=0, line_dash="dot", line_color="#e2e8f0")
        fr.update_layout(height=340, hovermode="x unified",
                         yaxis=dict(range=[-1.05, 1.05], title=None),
                         margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fr, width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("평균", f"{roll.mean():.2f}")
        m2.metric("최저", f"{roll.min():.2f}")
        m3.metric("최고", f"{roll.max():.2f}")
        st.caption(f"{win}{unit} 구간을 뒤돌아보며 계산합니다. 값이 치솟은 시점은 "
                   f"두 종목이 함께 움직인 국면이며, 그때는 분산 효과가 약해집니다.")
    else:
        st.info("서로 다른 두 종목을 선택하거나 기간을 늘려주세요.")

    # ---------------- 다음 단계 ----------------
    st.divider()
    st.subheader("➡️ 다음 단계")
    st.caption("상관관계가 낮은 자산을 다른 화면에서 검토해보세요. "
               "위 표의 **📋 복사** 를 누른 뒤, 그 화면에서 **📥 붙여넣기** 하면 됩니다.")

    # ---------------- 내보내기 ----------------
    st.divider()
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            corr.to_excel(xw, sheet_name="1_상관관계행렬")
            pairs.to_excel(xw, sheet_name="2_쌍별상관", index=False)
            if not pdf.empty:
                pdf.to_excel(xw, sheet_name="3_기간별")
            regr = st.session_state.get("_corr_reg_result")
            if regr is not None and not regr.empty:
                regr.to_excel(xw, sheet_name="3_국면별상관")
            if cl is not None:
                gg = {}
                for tk_, g in cl.items():
                    gg.setdefault(int(g), []).append(tk_)
                pd.DataFrame([{"위험군": f"군집 {g}", "자산 수": len(m),
                               "구성": ", ".join(m)} for g, m in sorted(gg.items())]
                             ).to_excel(xw, sheet_name="4_위험군", index=False)
            info = pd.DataFrame([
                {"티커": t, "종목명": meta.get(t, {}).get("name", t),
                 "통화": meta.get(t, {}).get("currency", "-")} for t in tickers])
            info.to_excel(xw, sheet_name="5_종목정보", index=False)
            pd.DataFrame(list({
                "계산 주기": freq, "기준 통화": base_ccy,
                "기간": f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                "표본 수": f"{len(R):,}",
                "평균 상관계수": f"{rho:.3f}",
                "실효 종목 수": f"{eff:.2f}",
                "배당 처리": "재투자" if use_div else "주가만",
                "환헤지": "적용" if fx_hedge else "미적용",
            }.items()), columns=["항목", "값"]).to_excel(
                xw, sheet_name="6_설정", index=False)
        st.download_button("📊 엑셀 파일 받기", buf.getvalue(),
                           f"correlations_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_corr", width="stretch")
    except Exception as ex:
        st.error(f"엑셀 생성 실패: {ex}")
    assumptions_panel(period=f"{prices.index[0].date()} ~ {prices.index[-1].date()}",
                      extra=[("계산 주기", freq),
                             ("표본 수", f"{len(R):,}개")])
    st.caption("상관관계는 과거 데이터에서 계산된 값이며 앞으로도 유지된다는 보장이 없습니다. "
               "특히 시장 급락 국면에서는 대부분의 자산이 함께 움직이는 경향이 있어, "
               "분산이 가장 필요한 순간에 효과가 줄어들 수 있습니다.")


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
    # 최대낙폭은 음수이므로 '증가 = 개선'이다. 변화율만으로는 방향이 헷갈리므로
    # 화면 안내문에서 별도로 설명한다.
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


def render_optimizer(base_ccy, start_date, end_date, use_div, rf_rate,
                     fx_hedge, gap_fill, cost_bp):
    key = "_opt_df"

    # 저장된 설정 적용 (위젯이 그려지기 전에 처리해야 함)
    if "_opt_pending_load" in st.session_state:
        cfg = load_opt_saved().get(st.session_state.pop("_opt_pending_load"))
        if cfg:
            for k in ["_opt_editor", "_opt_eq", "_opt_goal", "_opt_risk",
                      "_opt_cut", "_opt_train", "_opt_rebal", "_opt_bench",
                      "_opt_reopt", "_opt_useret", "_opt_usevol"]:
                st.session_state.pop(k, None)
            rows = [{"티커": h["ticker"], "종목명": "",
                     "현재 비중(%)": h.get("weight"),
                     "최소(%)": h.get("min", 0.0), "최대(%)": h.get("max", 100.0)}
                    for h in cfg.get("holdings", [])]
            if rows:
                st.session_state[key] = pd.DataFrame(rows)[OPT_COLS]
                st.session_state["_opt_gen"] = st.session_state.get("_opt_gen", 0) + 1
            for sk, ck in [("_opt_goal", "goal"), ("_opt_risk", "risk"),
                           ("_opt_cut", "cutoff"), ("_opt_train", "train"),
                           ("_opt_rebal", "rebalance"), ("_opt_bench", "benchmark"),
                           ("_opt_reopt", "reopt")]:
                if cfg.get(ck) is not None:
                    st.session_state[sk] = cfg[ck]
            st.success(f"**{cfg.get('label', '')}** 설정을 불러왔습니다.")

    # ---------------- 저장 / 불러오기 ----------------
    saved_opt = load_opt_saved()
    with st.expander("💾 최적화 설정 저장 / 불러오기", expanded=False):
        s1, s2 = st.columns([2, 1])
        if saved_opt:
            pick = s1.selectbox("저장된 설정", ["(선택)"] + sorted(saved_opt.keys()),
                                key="_opt_pick")
            b1, b2, b3 = s2.columns(3)
            if b1.button("📂", key="_opt_load", help="불러오기",
                         disabled=(pick == "(선택)")):
                st.session_state["_opt_pending_load"] = pick
                st.rerun()
            if b2.button("🗑", key="_opt_del", help="삭제", disabled=(pick == "(선택)")):
                saved_opt.pop(pick, None)
                write_opt_saved(saved_opt)
                st.rerun()
            b3.download_button("📥", json.dumps(saved_opt, ensure_ascii=False,
                                                indent=2).encode("utf-8"),
                               "optimizations.json", "application/json",
                               key="_opt_dl", help="전체 내보내기")
        else:
            s1.caption("저장된 설정이 없습니다.")
        n1, n2 = st.columns([2, 1])
        opt_name = n1.text_input("저장할 이름", key="_opt_save_name",
                                 placeholder="예: 미국 3종 · 샤프 최대")
        n2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        do_opt_save = n2.button("💾 저장", width="stretch",
                                disabled=not opt_name.strip())
        up_o = st.file_uploader("설정 파일 가져오기", type=["json"],
                                key="_opt_up", label_visibility="collapsed")
        if up_o is not None and not st.session_state.get("_opt_imported"):
            inc, err = _safe_import_json(up_o)
            if err:
                st.error(err)
            else:
                merged = dict(load_opt_saved()); merged.update(inc)
                write_opt_saved(merged)
                st.session_state["_opt_imported"] = True
                st.success(f"{len(inc)}개 설정을 불러왔습니다.")
                st.rerun()
        st.caption("설정은 브라우저 세션에 보관됩니다. 오래 쓰실 구성은 "
                   "**📥 내보내기**로 파일을 받아두세요.")

    st.subheader("1️⃣ 현재 포트폴리오 (Current Holdings)")
    live, tickers, bad = render_ticker_table(
        state_key=key, editor_key="_opt_editor", gen_key="_opt_gen",
        cols=OPT_COLS, weight_col="현재 비중(%)", title="포트폴리오 최적화",
        min_rows=2, max_rows=20, show_equal=True, equal_key="_opt_eq",
        extra_defaults={"최소(%)": 0.0, "최대(%)": 100.0},
        extra_config={
            "최소(%)": st.column_config.NumberColumn(
                "최소(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small"),
            "최대(%)": st.column_config.NumberColumn(
                "최대(%)", min_value=0.0, max_value=100.0, step=1.0,
                format="%.0f", width="small")},
        help_text="지금 보유 중인 종목과 비중을 넣으세요. 최적화 결과는 **이 구성과 비교**"
                  "해서 보여드립니다. 특정 종목을 제한하려면 최소·최대 편입비중을 "
                  "조정하세요.",
        default_rows=[{"티커": x, "현재 비중(%)": w, "최소(%)": 0.0, "최대(%)": 100.0}
                      for x, w in [("AAPL", 25.0), ("NVDA", 25.0),
                                   ("SCHD", 25.0), ("SPMO", 25.0)]])
    f = live
    wsum = float(pd.to_numeric(live["현재 비중(%)"], errors="coerce").fillna(0).sum())

    ok_in, msgs = validate_setup(list(live["티커"]), bad, live["현재 비중(%)"],
                                 min_n=2, need_weight=True)
    show_msgs(msgs)

    # ------------------------------------------------------------------
    st.subheader("2️⃣ 최적화 설정 (Settings)")
    o1, o2 = st.columns([1, 1])
    goal = o1.selectbox("목적", list(OPT_GOALS), key="_opt_goal")
    o1.caption(OPT_GOALS[goal])

    groups = {}
    for k, v in RISK_DEFS.items():
        groups.setdefault(v["group"], []).append(k)
    risk_labels = [f"{g} · {k}" for g in groups for k in groups[g]]
    risk_pick = o2.selectbox("위험을 정의하는 방법", risk_labels, key="_opt_risk",
                             disabled=goal.startswith(("위험균형", "계층적")),
                             help="목적 계산에 쓰이는 위험 척도입니다.")
    risk_name = risk_pick.split(" · ", 1)[1]
    o2.caption(RISK_DEFS[risk_name]["desc"]
               if not goal.startswith(("위험균형", "계층적"))
               else "이 방식은 위험 지표 선택과 무관하게 공분산 구조로 계산됩니다.")

    _is_hrp = goal.startswith("계층적")
    st.markdown("**목표 제약** — 비워두면 제약 없이 계산합니다")
    if _is_hrp:
        st.info("**계층적 위험 분산(HRP)** 은 목표수익률·목표변동성 제약을 다루지 "
                "못합니다. 상관구조와 변동성만으로 비중을 정하기 때문입니다. "
                "종목별 최소·최대 비중은 결과를 그 범위 안으로 조정해 반영합니다.")
    g1, g2 = st.columns(2)
    use_ret = g1.checkbox("목표수익률 지정", disabled=_is_hrp)
    min_ret = (g1.number_input("연 수익률 이상 (%)", -20.0, 200.0, 10.0, step=0.5,
                               label_visibility="collapsed") / 100
               if (use_ret and not _is_hrp) else None)
    use_vol = g2.checkbox("목표변동성 지정", disabled=_is_hrp)
    max_vol = (g2.number_input("연 변동성 이하 (%)", 0.5, 100.0, 5.0, step=0.5,
                               label_visibility="collapsed") / 100
               if (use_vol and not _is_hrp) else None)
    if use_ret or use_vol:
        st.caption("두 값은 **이상 / 이하** 조건으로 적용됩니다. "
                   "동시에 만족할 수 없으면 달성 가능한 수치를 알려드립니다.")

    st.markdown("**현재 비중 대비 변경 제한** — 한 번에 크게 바꾸지 않도록 묶습니다")
    m1, m2 = st.columns([1, 3])
    use_band = m1.checkbox("변경폭 제한", key="_opt_useband")
    band_pp = (m2.slider("종목당 최대 변경 (%p)", 1, 50, 10, step=1,
                         key="_opt_band",
                         help="현재 비중에서 이 폭을 넘지 않게 제한합니다. "
                              "예를 들어 10%p면 현재 30%인 종목은 20~40% 사이에서만 "
                              "움직입니다. 회전율과 거래비용을 줄이는 실무적 제약입니다.")
               if use_band else None)

    p1, p2, p3, p4 = st.columns(4)
    reopt = p4.selectbox("재최적화 주기", list(REOPT_FREQ), index=0, key="_opt_reopt",
                         help="'한 번만'은 기준일에 정한 비중을 끝까지 유지합니다. "
                              "분기·매년을 고르면 그 주기마다 그 시점까지의 데이터로 "
                              "비중을 다시 계산해, 실제 운용을 재현합니다.")
    cut_label = p1.selectbox("최적화 기준일 (Optimization Date)", list(CUTOFFS),
                             index=2, key="_opt_cut",
                             help="이 시점까지의 데이터로만 비중을 계산하고, "
                                  "이후 구간으로 실제 성과를 채점합니다.")
    train_label = p2.selectbox("학습 기간 (Training Period)", list(TRAIN_WINDOWS),
                               index=1, key="_opt_train",
                               help="비중을 계산하는 데 쓸 과거 데이터의 길이입니다.")
    rebal = p3.selectbox("리밸런싱", REBAL_OPTIONS, index=2, key="_opt_rebal")

    custom_date = None
    if cut_label == "직접 지정":
        custom_date = st.date_input("기준일 직접 지정",
                                    pd.Timestamp.today() - pd.DateOffset(years=1))

    bench_tk = st.text_input("벤치마크", value="^GSPC", key="_opt_bench",
                             help="비워두면 벤치마크 없이 비교합니다.").strip()

    if do_opt_save:
        store = load_opt_saved()
        nm = opt_name.strip()
        store[nm] = opt_snapshot(f, goal, risk_name, cut_label, train_label,
                                 rebal, bench_tk, min_ret, max_vol, reopt, label=nm)
        write_opt_saved(store)
        st.success(f"**{nm}** 으로 저장했습니다.")

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
    # 데이터가 없어 빠진 종목이 있을 수 있으므로 티커 기준으로 제약을 정렬한다
    _lim = live.set_index("티커")
    lo = np.asarray([float(_lim.loc[t_, "최소(%)"]) for t_ in tickers]) / 100
    hi = np.asarray([float(_lim.loc[t_, "최대(%)"]) for t_ in tickers]) / 100
    if lo.sum() > 1.0:
        st.error(f"최소 편입비중 합계가 {lo.sum()*100:.0f}%로 100%를 넘습니다.")
        return
    if hi.sum() < 1.0:
        st.error(f"최대 편입비중 합계가 {hi.sum()*100:.0f}%로 100%에 못 미칩니다.")
        return
    if (lo > hi).any():
        _bad = [t_ for t_, a_, b_ in zip(tickers, lo, hi) if a_ > b_]
        st.error(f"최소가 최대보다 큰 종목이 있습니다: {', '.join(_bad)}")
        return
    # 종목별 제약을 그대로 전달한다 (예전에는 max·min 으로 뭉개 모든 종목에
    # 같은 한도가 적용되는 문제가 있었다)
    wmin, wmax = lo, hi

    # 현재 비중 대비 변경폭 제한을 종목별 상하한에 합친다
    if use_band and band_pp:
        _cur = pd.to_numeric(live.set_index("티커")["현재 비중(%)"],
                             errors="coerce").reindex(tickers).fillna(0).values
        _cur = _cur / _cur.sum() if _cur.sum() > 0 else np.full(len(tickers),
                                                               1 / len(tickers))
        _bd = band_pp / 100.0
        wmin = np.maximum(wmin, _cur - _bd)
        wmax = np.minimum(wmax, _cur + _bd)
        wmin = np.clip(wmin, 0.0, 1.0)
        wmax = np.clip(wmax, 0.0, 1.0)
        if wmin.sum() > 1.0 + 1e-9 or wmax.sum() < 1.0 - 1e-9:
            st.error(f"변경폭 {band_pp}%p 로는 합계 100%를 맞출 수 없습니다. "
                     f"폭을 넓히거나 종목별 최소·최대를 완화해주세요.")
            return

    try:
        ok, msg, ret_max, vol_min = feasibility(mu, cov, wmin, wmax, min_ret, max_vol)
    except ValueError as ex:
        st.error(f"🚫 제약 설정을 확인해주세요: {ex}")
        return
    except Exception as ex:
        st.error(f"🚫 제약 검토 중 오류가 발생했습니다: {ex}")
        return
    if not ok:
        st.error("🚫 " + msg)
        st.caption(f"참고 · 현재 종목·제약으로 달성 가능한 범위: "
                   f"수익률 최대 {ret_max*100:.2f}% / 변동성 최소 {vol_min*100:.2f}%")
        return

    try:
        with st.spinner("최적 비중 계산 중..."):
            w_opt, mu, cov = optimize(R_train, goal, risk_name, rf_rate,
                                      wmin, wmax, min_ret, max_vol)
    except ValueError as ex:
        st.error(f"🚫 {ex}")
        return
    except Exception as ex:
        st.error(f"🚫 최적화 중 오류가 발생했습니다: {ex}")
        return
    if w_opt is None:
        st.error("🚫 최적해를 찾지 못했습니다. 제약이 서로 모순되거나 너무 빡빡할 수 "
                 "있습니다. 종목별 최소·최대 비중이나 목표 제약을 완화해보세요.")
        return
    _eq = np.ones(len(w_opt)) / len(w_opt)
    if np.allclose(w_opt, _eq, atol=1e-6):
        st.warning("⚠️ 결과가 균등비중과 동일합니다. 최적화가 유효한 해를 찾지 못하고 "
                   "기본값을 돌려준 것일 수 있으니, 제약과 학습 기간을 확인해보세요.")

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
    if use_band and band_pp:
        _cons.append(f"변경폭 ±{band_pp}%p")
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
        stab_fails = []
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
                if wx is None:
                    stab_fails.append(lbl)
                    continue
                stab_rows[lbl] = pd.Series(wx * 100, index=tickers)
            except Exception:
                continue
        if stab_fails:
            st.caption(f"⚠️ {', '.join(stab_fails)} 학습 기간에서는 최적해를 찾지 "
                       f"못해 이 비교에서 제외했습니다 (균등비중으로 대체하지 않음).")
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

    _opt_rows = [{"티커": t, "비중": round(float(x) * 100, 2)}
                 for t, x in zip(tickers, w_opt) if x > 1e-6]
    if st.button("📋 최적 비중 복사", width="stretch",
                 help="최적 비중을 보관함에 담습니다. 다른 화면의 표에서 "
                      "📥 붙여넣기 하면 그대로 채워집니다."):
        st.session_state[CLIP_KEY] = {"rows": _opt_rows, "from": f"최적화 ({goal})"}
        st.success(f"보관함에 담았습니다 · {clip_summary()}")

    st.markdown("**⚖️ 비교 후보로 담기** — 여러 안을 모아 최종 비교 화면에서 견줍니다")
    plan_button("⚖️ 담기", _opt_rows, f"최적화 ({goal})", "_plan_opt",
                default_name=f"최적화 {goal.split(' ')[0]}", rebalance=rebal)

    # ---------------- 성과 비교 ----------------
    def _bt(px, W):
        try:
            r = portfolio_returns(px, W, rebal)
            tno = turnover_series(px, W, rebal).reindex(r.index).fillna(0)
            return apply_cost(r, tno, cost_bp)
        except Exception:
            return None

    def _bench(px):
        # 벤치마크는 매수 후 보유이므로 거래비용을 적용하지 않는다
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

    wf_r, wf_hist, wf_fails = None, [], 0
    if REOPT_FREQ[reopt] is not None:
        bar = st.progress(0.0, text="주기적 재최적화 계산 중...")
        try:
            wf_r, wf_hist, wf_fails = walk_forward(
                prices, tickers, opt_date, REOPT_FREQ[reopt], tw, rebal,
                goal, risk_name, rf_rate, wmin, wmax, min_ret, max_vol,
                progress=lambda x: bar.progress(min(1.0, x),
                                                text=f"주기적 재최적화 계산 중... {x*100:.0f}%"),
                cost_bp=cost_bp)
        except Exception as ex:
            st.warning(f"재최적화 계산 실패: {ex}")
        bar.empty()
        if wf_r is not None and len(wf_r) > 5:
            oos_rows[f"재최적화 ({reopt})"] = wf_r
            st.info(f"🔁 **{reopt}** 재최적화를 {len(wf_hist)}회 수행했습니다. "
                    f"각 시점까지의 데이터만 사용했으므로 미래 정보가 섞이지 않습니다.")
        if wf_fails:
            st.warning(f"⚠️ {wf_fails}개 재최적화 시점에서 최적해를 찾지 못해 "
                       f"그 구간을 건너뛰었습니다 (균등비중 등으로 조용히 대체하지 "
                       f"않았습니다). 제약이 너무 빡빡하지 않은지 확인해보세요.")

    b_oos = _bench(oos_px)
    if b_oos is not None:
        oos_rows[f"벤치마크 ({bench_norm})"] = b_oos
    t_oos = _cmp_table(oos_rows, rf_rate)
    st.dataframe(t_oos.style.format("{:.2f}", na_rep="-"), width="stretch")
    st.caption("**최대낙폭 읽는 법** — 값이 −20%에서 −15%로 바뀌면 낙폭이 얕아진 것이므로 "
               "**개선**입니다. 변화율이 양수이면 개선, 음수이면 악화로 보시면 됩니다.")

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

    # ---------------- 요약 ----------------
    st.markdown("#### 📝 요약")
    try:
        _o = perf_row(oos_rows.get("원본 포트폴리오"), rf_rate)
        _n = perf_row(oos_rows.get("최적화 포트폴리오"), rf_rate)
        _lines = []
        d_shp = _n["샤프지수"] - _o["샤프지수"]
        d_mdd = _n["최대낙폭(%)"] - _o["최대낙폭(%)"]
        verdict = ("개선됐습니다" if d_shp > 0.05 else
                   "나빠졌습니다" if d_shp < -0.05 else "거의 차이가 없었습니다")
        _lines.append(
            f"검증 구간에서 최적화 포트폴리오의 샤프지수는 {_o['샤프지수']:.2f} → "
            f"**{_n['샤프지수']:.2f}** ({d_shp:+.2f})로 {verdict}. "
            f"최대낙폭은 {_o['최대낙폭(%)']:.2f}% → **{_n['최대낙폭(%)']:.2f}%** "
            f"({d_mdd:+.2f}%p, {'개선' if d_mdd > 0 else '악화' if d_mdd < 0 else '동일'})입니다.")
        _top = alloc.iloc[0]
        _lines.append(f"비중은 **{_top['티커']}** 가 {_top['원본 비중(%)']:.1f}% → "
                      f"{_top['최적 비중(%)']:.1f}% 로 가장 크게 배분됐습니다.")
        if d_shp < 0:
            _lines.append("**표본외에서 원본보다 못한 결과**입니다. 학습 구간에 과도하게 "
                          "맞춰졌을 가능성이 있으니, 비중 안정성 점검과 함께 살펴보세요.")
        if wf_r is not None:
            _w = perf_row(wf_r, rf_rate)
            _lines.append(f"주기적 재최적화({reopt})는 샤프 **{_w['샤프지수']:.2f}**, "
                          f"최대낙폭 **{_w['최대낙폭(%)']:.2f}%** 였습니다.")
        st.markdown("\n\n".join(_lines))
    except Exception as _ex:
        st.caption(f"요약을 만들지 못했습니다: {_ex}")

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
        try:
            ef = efficient_frontier(mu, cov, 25, wmin, wmax)
        except Exception:
            ef = []
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
    st.subheader("8️⃣ 추가 분석 (Deeper Analysis)")
    st.caption("검증 구간을 여러 각도에서 살펴봅니다.")

    st.markdown("#### 시간에 따른 비중 변화 (Allocation Over Time)")
    st.caption("실제 보유 비중이 어떻게 흘러갔는지 보여줍니다. "
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

    st.markdown("#### 샤프지수 추이 (Rolling Sharpe)")
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

    st.markdown("#### 낙폭 (Drawdowns)")
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

    st.markdown("#### 변동성 추이 (Rolling Volatility)")
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

    st.markdown("#### 자산 상관관계 (Asset Correlations)")
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
                           key="dl_opt", width="stretch")
    except Exception as ex:
        x1.error(f"엑셀 생성 실패: {ex}")
    x2.download_button("📄 최적 비중 CSV", alloc.to_csv(index=False).encode("utf-8-sig"),
                       f"optimal_weights_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                       "text/csv", width="stretch")
    st.caption("엑셀은 화면과 같은 순서(최적자산배분 → 표본외성과 → 표본내성과 → 성과추이 → "
               "벤치마크지표 → 위험기여도 → 비중추이 → 자산상관관계 → 설정)로 구성되며, "
               "성과추이와 비중추이에는 엑셀 네이티브 차트가 포함됩니다.")
    assumptions_panel(rebalance=rebal, cost=cost_bp,
                      period=f"{train_px.index[0].date()} ~ {last.date()}",
                      bench=bench_norm or "-", path_note=True,
                      extra=[("최적화 기준일", str(opt_date.date())),
                             ("학습 기간", train_label),
                             ("재최적화", reopt)])
    st.caption("최적화는 학습 구간의 평균·변동성·상관관계가 앞으로도 유지된다고 가정합니다. "
               "특히 기대수익률 추정은 오차가 커서, 표본외 성과가 원본보다 나쁜 경우도 흔합니다. "
               "교육·참고용이며 투자 자문이 아닙니다.")



# ======================================================================
# 사이드바
# ======================================================================
# 다른 화면에서 요청한 도구 전환을 위젯 생성 전에 반영한다
if "_pending_tool" in st.session_state:
    st.session_state["_tool"] = st.session_state.pop("_pending_tool")

# 화면 선택을 주소(URL)에 남긴다.
# 세션이 끊겨 상태가 비워져도 주소가 남아 있으면 보던 화면으로 복원된다.
_ALL_TOOLS = [x for _, items in TOOL_GROUPS for x in items]
try:
    _qp = st.query_params
    if "_tool" not in st.session_state:
        _from_url = _qp.get("tool")
        if isinstance(_from_url, list):
            _from_url = _from_url[0] if _from_url else None
        for _t in _ALL_TOOLS:
            if _from_url and _from_url == _t.split(" ", 1)[-1]:
                st.session_state["_tool"] = _t
                break
except Exception:
    _qp = None

with st.sidebar:
    st.session_state.setdefault("_tool", TOOL_DEFAULT)
    for grp, items in TOOL_GROUPS:
        if grp:
            st.caption(grp)
        for it in items:
            if st.button(it, key=f"nav_{it}", width="stretch",
                         type=("primary" if st.session_state["_tool"] == it
                               else "secondary")):
                st.session_state["_tool"] = it
                try:
                    st.query_params["tool"] = it.split(" ", 1)[-1]
                except Exception:
                    pass
                st.rerun()
    tool = st.session_state["_tool"]
    try:
        if _qp is not None and _qp.get("tool") != tool.split(" ", 1)[-1]:
            st.query_params["tool"] = tool.split(" ", 1)[-1]
    except Exception:
        pass

# 도움말은 설정이 필요 없으므로 사이드바를 더 그리지 않고 바로 표시한다
if tool.endswith("도움말"):
    render_help()
    st.stop()

with st.sidebar:
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
    fx_hedge = st.checkbox("환율효과 제외 (현지통화 기준)", value=False,
                           help="체크하면 환율 변동을 빼고 종목 자체 수익률만 봅니다.\n\n실제 환헤지에는 선물환 프리미엄·금리차·롤오버 비용이 드는데 여기서는 "
                                "그 비용을 반영하지 않습니다. 따라서 '무비용 완전헤지'라는 이상적인 가정입니다.")
    gap_mode = st.selectbox(
        "휴장일 처리", ["공통 거래일만 사용", "직전 종가로 채우기"], index=0,
        help="국가를 섞으면 휴장일이 달라 공통 거래일이 크게 줄어듭니다.\n\n"
             "· 공통 거래일만 사용 — 모든 종목이 거래된 날만 씁니다. 가장 엄밀하지만 "
             "기간이 줄어듭니다.\n"
             "· 직전 종가로 채우기 — 휴장한 종목은 직전 가격을 유지한 것으로 봅니다. "
             "기간은 보존되지만 그날 수익률은 0으로 잡혀 변동성이 다소 낮게 나옵니다.")
    gap_fill = gap_mode.startswith("직전")

    rf_rate = st.number_input("무위험 수익률 (연, %)", value=3.0, step=0.25,
                              help="Sharpe·Sortino 계산에만 쓰입니다. "
                                   "기준 통화에 맞는 금리를 넣으세요.") / 100

    st.divider()
    if st.button("🔄 종목 정보 새로 조회", width="stretch",
                 help="종목명·통화 조회 결과를 지우고 다시 받아옵니다. "
                      "종목명이 이상하게 표시될 때 눌러보세요."):
        for _fn in (probe_ticker, get_name, load_etf_holdings):
            try:
                _fn.clear()
            except Exception:
                pass
        st.success("조회 결과를 비웠습니다. 표의 종목명이 다시 채워집니다.")
        st.rerun()

    st.caption("**티커 예시**\n\n"
               "한국 `005930.KS` `086520.KQ`\n\n"
               "미국 `NVDA` `SPY` `QQQ`\n\n"
               "일본 `7203.T`  홍콩 `0700.HK`")


IS_CMP = tool.endswith("대안 비교")
IS_STRESS = tool.endswith("스트레스 테스트")
IS_SNAPSHOT = tool.endswith("시장동향")

if IS_SNAPSHOT:
    render_market_snapshot(end_date)
    st.stop()

if IS_CMP:
    render_compare(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate,
                   cost_bp)
    st.stop()
IS_BL = tool.endswith("자산배분")

if IS_STRESS:
    render_stress(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate)
    st.stop()
IS_REG = tool.endswith("거시 국면")

if IS_BL:
    render_view_alloc(base_ccy, start_date, end_date, use_div,
                      fx_hedge, gap_fill, rf_rate, cost_bp)
    st.stop()
IS_CAND = tool.endswith("추가 효과")

if IS_REG:
    render_regimes(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill, rf_rate)
    st.stop()
IS_CORR = tool.endswith("상관관계")
IS_OPT = tool.endswith("최적화")

if IS_CAND:
    render_candidate_search(base_ccy, start_date, end_date, use_div, rf_rate,
                            fx_hedge, gap_fill, cost_bp)
    st.stop()

if IS_CORR:
    render_correlations(base_ccy, start_date, end_date, use_div, fx_hedge, gap_fill)
    st.stop()

if IS_OPT:
    st.title("🎯 포트폴리오 최적화")
    st.caption("종목을 넣으면 목적에 맞는 최적 비중을 계산합니다. "
               "과거 데이터 기반이며 미래 성과를 보장하지 않습니다.")
    render_optimizer(base_ccy, start_date, end_date, use_div, rf_rate,
                     fx_hedge, gap_fill, cost_bp)
    st.stop()

st.title("📊 포트폴리오 분석")
st.caption("여러 포트폴리오와 벤치마크를 한 화면에서 비교합니다. 통화는 자동으로 인식됩니다.")


# ======================================================================
# 저장 / 불러오기 · 벤치마크
# ======================================================================
with st.expander("💾 구성 저장 / 불러오기", expanded=False):
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
        incoming, err = _safe_import_json(up)
        if err:
            st.error(err)
        else:
            merged = dict(load_saved()); merged.update(incoming)
            write_saved(merged)
            st.session_state["_imported_once"] = True
            st.success(f"{len(incoming)}개 구성을 불러왔습니다.")
            st.rerun()


st.subheader("벤치마크 (Benchmark)")
bc1, bc2 = st.columns([3, 1])
bench_raw = bc1.text_input(
    "벤치마크 티커 — 콤마로 구분", value="^KS11, ^GSPC",
    help="^KS11 코스피 · ^KQ11 코스닥 · ^GSPC S&P500 · ^N225 닛케이225 · ^IXIC 나스닥")
bench_list = []
for b in bench_raw.replace("\n", ",").split(","):
    b = b.strip()
    if b:
        cands = normalize_ticker(b)
        bench_list.append(cands[0] if cands else b)

bc2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
if bc2.button("🔍 티커 확인", width="stretch"):
    if not bench_list:
        st.caption("확인할 티커가 없습니다.")
    else:
        with st.spinner("확인 중..."):
            rows_b = []
            for b in bench_list:
                tk, ccy = quick_check(tuple(normalize_ticker(b)))
                rows_b.append({"": "✅" if tk else "❌", "입력": b,
                               "티커": tk or "-",
                               "종목명": (get_name(tk) or "(종목명 없음)") if tk else "확인 불가",
                               "통화": ccy or "-"})
        st.dataframe(pd.DataFrame(rows_b), width="stretch", hide_index=True)


# ======================================================================
# 포트폴리오 입력
# ======================================================================
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
            st.session_state[f"gen_{i}"] = st.session_state.get(f"gen_{i}", 0) + 1
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
        pre = PRESETS[i] if i < len(PRESETS) else []
        v, tks, bad = render_ticker_table(
            state_key=dfkey, editor_key=f"hold_{i}", gen_key=f"gen_{i}",
            cols=COLS, weight_col="비중(%)", title=f"포트{i+1}",
            min_rows=MIN_ROWS, max_rows=MAX_ROWS,
            show_equal=True, equal_key=f"eqw_{i}",
            default_rows=([{"티커": t_, "비중(%)": w_} for t_, w_ in pre]
                          or [{"티커": ""}]))
        tot = float(pd.to_numeric(v["비중(%)"], errors="coerce").fillna(0).sum())

        m1, m2 = st.columns([1, 1])
        m1.caption(f"비중 합계 **{tot:.2f}%**"
                   + ("" if abs(tot - 100) < 0.01 else " · 100%가 아니면 자동 정규화됩니다"))
        if bad:
            m2.caption(f"⚠️ 확인 실패: **{', '.join(bad)}**")
        elif not v.empty:
            m2.caption("모든 티커 정상 ✅")

        if bad:
            with st.expander("🩺 조회 실패 원인 보기", expanded=False):
                st.caption("각 티커에 대해 시도한 방법과 실패 사유입니다.")
                for t_ in bad:
                    r_ = probe_ticker(tuple(normalize_ticker(t_)))
                    st.markdown(f"**{t_}** → 시도한 후보: "
                                f"`{', '.join(normalize_ticker(t_))}`")
                    st.code("\n".join(r_["log"]) or "기록된 오류 없음", language="text")
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
            fx_s = s.reindex(prices.index).ffill()
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
        _r = portfolio_returns(prices, w, p["rebalance"])
        _tno = turnover_series(prices, w, p["rebalance"]).reindex(_r.index).fillna(0)
        series[p["name"]] = {
            "returns": apply_cost(_r, _tno, cost_bp),
            "kind": "portfolio", "weights": w, "rebalance": p["rebalance"],
            "turnover": annual_turnover(_tno),
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
        "수익률(연,%)": cagr(r) * 100,
        "변동성(연,%)": annual_vol(r) * 100,
        "최대낙폭(%)": max_drawdown(r) * 100,
        "샤프지수": sharpe_ratio(r, rf_rate),
        "소르티노": sortino_ratio(r, rf_rate),
        "칼마": calmar_ratio(r),
        "얼서지수": ulcer_index(r),
        "연 회전율(%)": (v.get("turnover") * 100
                     if v["kind"] == "portfolio" and v.get("turnover") is not None
                     and not pd.isna(v.get("turnover")) else np.nan),
    })
comp = pd.DataFrame(rows).set_index("")
_hi = [c for c in ["누적성장배수", "수익률(연,%)", "최대낙폭(%)", "샤프지수", "소르티노", "칼마"]
       if c in comp.columns]
_lo = [c for c in ["변동성(연,%)", "얼서지수", "연 회전율(%)"] if c in comp.columns]
st.dataframe(
    comp.style.format("{:.2f}", na_rep="-")
        .highlight_max(subset=_hi, color="#dcfce7")
        .highlight_min(subset=_lo, color="#dcfce7"),
    width="stretch")
st.caption("초록색 = 각 지표에서 가장 좋은 값. **MDD는 음수이므로 0에 가까울수록**, "
           "즉 숫자가 클수록 좋습니다. "
           "📈 포트폴리오 · 📊 벤치마크 · **연 회전율**은 1년에 자산의 몇 %를 "
           "사고파는지를 뜻하며, 거래비용을 입력했다면 성과에 이미 반영돼 있습니다.")

# ---------------- 포트폴리오별 상세 ----------------
# ---------------- 상대성과 ----------------
_ports = [n for n, v in series.items() if v["kind"] == "portfolio"]
_benches = [n for n, v in series.items() if v["kind"] != "portfolio"]
if _ports and _benches:
    st.subheader("5️⃣ 벤치마크 대비 (Relative Performance)")
    _bpick = st.selectbox("기준 벤치마크", _benches, key="_rel_bench")
    _br = series[_bpick]["returns"]
    st.caption("**초과수익률**은 연율 수익률의 차이, **추적오차**는 벤치마크와 얼마나 "
               "다르게 움직였는지, **정보비율**은 추적오차 한 단위당 초과수익입니다. "
               "**포착률**은 벤치마크가 오를 때/내릴 때 그중 몇 %를 따라갔는지로, "
               "상승은 높고 하락은 낮을수록 유리합니다.")
    _rrows = {}
    for _n in _ports:
        _m = rel_metrics(series[_n]["returns"], _br)
        _m.update(bench_metrics(series[_n]["returns"], _br, rf_rate))
        _rrows[_n] = _m
    _rdf = pd.DataFrame(_rrows).T
    st.dataframe(_rdf.style.format("{:.2f}", na_rep="-"), width="stretch")

    if use_div:
        st.warning("⚠️ 포트폴리오는 **배당 재투자(총수익)** 기준인데, 벤치마크가 "
                   "`^KS11`·`^GSPC` 같은 **가격지수**라면 배당이 빠져 있어 "
                   "포트폴리오가 유리하게 보입니다. 배당까지 포함한 비교를 원하시면 "
                   "총수익 ETF(예: SPY·ACWI)를 벤치마크로 쓰세요.")

    _fex = go.Figure()
    for _n in _ports:
        _act = (series[_n]["returns"] - _br).dropna()
        if len(_act) < 2:
            continue
        # 상대 부(富)의 변화로 계산한다.
        # 일별 수익률 차이를 복리로 굴리면(=(1+차이).cumprod()) 변동성이 클수록
        # 실제 상대성과와 벌어진다.
        _pe = (1 + series[_n]["returns"].reindex(_act.index)).cumprod()
        _be = (1 + _br.reindex(_act.index)).cumprod()
        _cum = (_pe / _be - 1) * 100
        _fex.add_trace(go.Scatter(x=_cum.index, y=_cum.values, name=_n,
                                  line=dict(color=color_of.get(_n), width=2)))
    _fex.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    _fex.update_layout(height=320, hovermode="x unified",
                       margin=dict(l=0, r=0, t=30, b=0),
                       yaxis=dict(title=None, ticksuffix="%"),
                       legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(_fex, width="stretch")
    st.caption(f"**{_bpick}** 대비 누적 초과수익입니다. 우상향이면 벤치마크를 "
               f"앞서고 있다는 뜻입니다.")

st.subheader("6️⃣ 연도별 성과 (Calendar Year Returns)")
st.caption("연도별로 잘라 보면 특정 해에만 좋았는지, 꾸준했는지가 드러납니다. "
           "약세장이 있던 해를 눈여겨보세요.")
yr_rows = {}
for name, v in series.items():
    yr = (1 + v["returns"]).groupby(v["returns"].index.year).prod() - 1
    yr_rows[name] = yr * 100
ydf = pd.DataFrame(yr_rows)
st.dataframe(ydf.style.format("{:.2f}", na_rep="-").map(_heat_color),
             width="stretch")

st.subheader("7️⃣ 적립식 투자 (Dollar-Cost Averaging)")
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

st.subheader("8️⃣ 포트폴리오별 상세 (Details)")
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
                   "막대의 합(거래비용 포함)이 곧 포트폴리오 총 성장률입니다.")
        try:
            contrib = growth_contribution(prices, v["weights"], v["rebalance"], r) * 100
            contrib = contrib.sort_values(ascending=False)
            # 종목별 기여도는 원시(비용 반영 전) 수익률로 계산되므로, 거래비용이
            # 있으면 합계가 실제 총 성장률(비용 반영)보다 커진다. 그 차이를
            # '거래비용' 막대로 따로 보여줘 합계가 실제 총 성장률과 맞게 한다.
            total_g = float((1 + r).prod() - 1) * 100
            cost_drag = float(contrib.sum() - total_g)

            labels, values, measures = list(contrib.index), list(contrib.values), \
                ["relative"] * len(contrib)
            if abs(cost_drag) > 1e-6:
                labels.append("거래비용")
                values.append(-cost_drag)
                measures.append("relative")
            labels.append("Portfolio")
            values.append(total_g)
            measures.append("total")

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

            # 위험기여도 — 수익과 위험을 나란히 봐야 편중이 드러난다
            try:
                _tks = list(contrib.index)
                _wv = np.array([v["weights"].get(t_, 0.0) for t_ in _tks], dtype=float)
                _wv = _wv / _wv.sum() if _wv.sum() > 0 else _wv
                _cov = (prices[_tks].pct_change().dropna().cov() * TRADING_DAYS).values
                _rc = risk_contributions(_wv, _cov)
                _tot_rc = float(_rc.sum())
                cdf["위험기여도(%)"] = (_rc / _tot_rc * 100) if _tot_rc > 0 else np.nan
            except Exception:
                cdf["위험기여도(%)"] = np.nan

            st.dataframe(cdf.style.format({"기여도(%p)": "{:+.2f}", "비중(%)": "{:.2f}",
                                           "총성장 대비": "{:.1f}%",
                                           "위험기여도(%)": "{:.1f}"}, na_rep="-"),
                         width="stretch", hide_index=True)
            st.caption(f"막대 합계(종목 + 거래비용) **{total_g:+.2f}%** = 포트폴리오 총 "
                       f"성장률(거래비용 반영) · **위험기여도**는 전체 변동성에서 각 종목이 "
                       f"차지하는 몫입니다. 비중보다 위험기여도가 훨씬 크다면 그 종목에 위험이 "
                       f"쏠려 있다는 뜻입니다.")
            try:
                _mx = cdf.loc[cdf["위험기여도(%)"].idxmax()]
                if float(_mx["위험기여도(%)"]) > float(_mx["비중(%)"]) * 1.5:
                    st.info(f"**{_mx['종목']}** 은 비중 {_mx['비중(%)']:.1f}% 인데 "
                            f"위험은 {_mx['위험기여도(%)']:.1f}% 를 차지합니다.")
            except Exception:
                pass
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
st.subheader("⚖️ 비교 후보로 담기")
st.caption("지금 구성을 최종 비교 화면으로 보냅니다. 여러 안을 모아 나란히 견줄 수 있습니다.")
_pnames = [n for n, v in series.items() if v["kind"] == "portfolio"]
if _pnames:
    _pk = st.selectbox("담을 포트폴리오", _pnames, key="_plan_an_pick")
    _pw = series[_pk]["weights"]
    _tot_w = sum(_pw.values())
    _an_rows = [{"티커": k, "비중": round(v / _tot_w * 100, 2)}
                for k, v in _pw.items() if v > 0]
    plan_button("⚖️ 담기", _an_rows, "포트폴리오 분석", "_plan_an",
                default_name=_pk, rebalance=series[_pk]["rebalance"])

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
        key="dl_analysis", width="stretch")
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
           "입력한 거래비용은 반영했습니다. 세금·호가스프레드·시장충격은 "
           "반영하지 않았습니다.")
