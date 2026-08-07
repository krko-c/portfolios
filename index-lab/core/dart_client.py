"""
DART(전자공시시스템) 오픈API 연동 — 종목 상장일·업종·기업개요 조회.

⚠️ 이 모듈은 이 개발 환경에서 실제 API 호출로 검증하지 못했습니다
(외부 네트워크 차단). DART 공식 문서(opendart.fss.or.kr)의 응답 스키마를
기준으로 작성했고, 파싱 로직은 tests/test_dart_client.py 에서 그 스키마를
흉내 낸 가짜(mock) 응답으로만 검증했습니다. 실제 키로 첫 호출 결과는
반드시 확인해주세요 — 특히 company.json 의 필드명이 문서와 다르면 여기
파싱 코드도 맞춰 고쳐야 합니다.

API 키는 https://opendart.fss.or.kr 에서 무료로 발급받습니다(회원가입 후
"인증키 신청/관리"). 하루 호출 한도가 있으니(문서 기준 1만~2만 건 수준,
계정마다 다를 수 있음) 응답을 캐싱해 재사용하세요.
"""
import io
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

BASE_URL = "https://opendart.fss.or.kr/api"

# DART 응답의 status 코드. "000" 만 정상, 나머지는 문서에 정의된 오류.
_STATUS_OK = "000"
_STATUS_MSG = {
    "010": "등록되지 않은 키입니다.",
    "011": "사용할 수 없는 키입니다.",
    "012": "접근할 수 없는 IP입니다.",
    "013": "조회된 데이터가 없습니다.",
    "020": "요청 제한을 초과했습니다.",
    "100": "필드의 부적절한 값입니다.",
    "800": "시스템 점검 중입니다.",
    "900": "정의되지 않은 오류가 발생했습니다.",
}


class DartError(Exception):
    pass


def _check_status(payload: dict, context: str):
    status = payload.get("status")
    if status != _STATUS_OK:
        msg = payload.get("message") or _STATUS_MSG.get(status, "알 수 없는 오류")
        raise DartError(f"{context} 실패 (status={status}): {msg}")


def fetch_corp_code_map(api_key: str, timeout: int = 30) -> pd.DataFrame:
    """
    corpCode.xml(전체 상장·비상장 법인의 고유번호 목록, ZIP)을 받아
    실제 상장된(주식코드가 있는) 법인만 걸러 돌려준다.
    반환: DataFrame[corp_code, corp_name, stock_code, modify_date]
    """
    resp = requests.get(f"{BASE_URL}/corpCode.xml",
                        params={"crtfc_key": api_key}, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "zip" not in ctype and not resp.content[:2] == b"PK":
        # ZIP이 아니면 보통 JSON 오류 응답이다 (status/message)
        try:
            _check_status(resp.json(), "corpCode 조회")
        except ValueError:
            pass
        raise DartError("corpCode.xml 응답이 예상한 ZIP 형식이 아닙니다.")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.iter("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        if not stock_code:
            continue  # 비상장 법인은 제외
        rows.append({
            "corp_code": (item.findtext("corp_code") or "").strip(),
            "corp_name": (item.findtext("corp_name") or "").strip(),
            "stock_code": stock_code,
            "modify_date": (item.findtext("modify_date") or "").strip(),
        })
    return pd.DataFrame(rows, columns=["corp_code", "corp_name", "stock_code", "modify_date"])


def fetch_company_overview(api_key: str, corp_code: str, timeout: int = 30) -> dict:
    """
    기업개황(company.json). 상장일·업종코드·법인명 등을 돌려준다.
    반환 필드(문서 기준): corp_name, corp_name_eng, stock_name, stock_code,
    ceo_nm, corp_cls, induty_code, est_dt(설립일), acc_mt(결산월) 등.
    """
    resp = requests.get(f"{BASE_URL}/company.json",
                        params={"crtfc_key": api_key, "corp_code": corp_code},
                        timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    _check_status(payload, f"기업개황 조회(corp_code={corp_code})")
    return payload


def corp_code_for_ticker(corp_code_map: pd.DataFrame, ticker: str) -> str:
    """6자리 종목코드(예: '005930')로 DART corp_code를 찾는다."""
    t = ticker.strip().upper().replace(".KS", "").replace(".KQ", "")
    row = corp_code_map.loc[corp_code_map["stock_code"] == t]
    if row.empty:
        raise DartError(f"DART corp_code 매핑에서 '{ticker}'를 찾지 못했습니다.")
    return str(row.iloc[0]["corp_code"])


def enrich_universe(df: pd.DataFrame, api_key: str, *, ticker_col="ticker",
                    corp_code_map: pd.DataFrame = None) -> pd.DataFrame:
    """
    유니버스 DataFrame에 DART 기업개황(업종코드·설립일 등)을 붙인다.
    실패한 종목은 조용히 건너뛰지 않고 별도 열(dart_오류)에 사유를 남긴다.
    """
    if corp_code_map is None:
        corp_code_map = fetch_corp_code_map(api_key)

    out = df.copy()
    out["dart_업종코드"] = None
    out["dart_설립일"] = None
    out["dart_오류"] = None

    for i, row in out.iterrows():
        try:
            cc = corp_code_for_ticker(corp_code_map, str(row[ticker_col]))
            info = fetch_company_overview(api_key, cc)
            out.at[i, "dart_업종코드"] = info.get("induty_code")
            out.at[i, "dart_설립일"] = info.get("est_dt")
        except DartError as ex:
            out.at[i, "dart_오류"] = str(ex)
    return out
