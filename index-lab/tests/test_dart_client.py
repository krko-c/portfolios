"""
DART 클라이언트 파싱 로직 테스트 — 전부 모킹(가짜 응답)이다.
실제 API를 호출하지 않는다(이 개발 환경은 네트워크가 막혀 있음).
DART 공식 문서에 있는 응답 스키마를 흉내 낸 가짜 데이터로, '스키마가
그대로일 때 파싱이 맞는가'만 검증한다 — 실제 스키마와 다르면 이 테스트는
통과해도 실제 호출은 실패할 수 있다.
"""
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import dart_client as dc


def _zip_of(xml_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml_bytes)
    return buf.getvalue()


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list>
    <corp_code>00126380</corp_code>
    <corp_name>삼성전자</corp_name>
    <stock_code>005930</stock_code>
    <modify_date>20230101</modify_date>
  </list>
  <list>
    <corp_code>00164742</corp_code>
    <corp_name>비상장회사</corp_name>
    <stock_code></stock_code>
    <modify_date>20230101</modify_date>
  </list>
  <list>
    <corp_code>00164779</corp_code>
    <corp_name>SK하이닉스</corp_name>
    <stock_code>000660</stock_code>
    <modify_date>20230102</modify_date>
  </list>
</result>""".encode("utf-8")


def test_fetch_corp_code_map_parses_zip_and_drops_unlisted():
    fake_resp = MagicMock()
    fake_resp.headers = {"Content-Type": "application/zip"}
    fake_resp.content = _zip_of(SAMPLE_XML)
    fake_resp.raise_for_status = MagicMock()
    with patch("core.dart_client.requests.get", return_value=fake_resp):
        df = dc.fetch_corp_code_map("dummy_key")
    assert set(df["stock_code"]) == {"005930", "000660"}
    assert "00164742" not in set(df["corp_code"])  # 비상장 제외됨


def test_fetch_corp_code_map_raises_on_non_zip_error_payload():
    fake_resp = MagicMock()
    fake_resp.headers = {"Content-Type": "application/json"}
    fake_resp.content = b'{"status":"010","message":"\xeb\xb1\x9d"}'
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={"status": "010", "message": "등록되지 않은 키"})
    with patch("core.dart_client.requests.get", return_value=fake_resp):
        with pytest.raises(dc.DartError):
            dc.fetch_corp_code_map("bad_key")


def test_corp_code_for_ticker_finds_match():
    df = pd.DataFrame({"corp_code": ["00126380", "00164779"],
                       "corp_name": ["삼성전자", "SK하이닉스"],
                       "stock_code": ["005930", "000660"]})
    assert dc.corp_code_for_ticker(df, "005930") == "00126380"
    assert dc.corp_code_for_ticker(df, "005930.KS") == "00126380"  # 접미사 제거


def test_corp_code_for_ticker_missing_raises():
    df = pd.DataFrame({"corp_code": ["00126380"], "stock_code": ["005930"]})
    with pytest.raises(dc.DartError):
        dc.corp_code_for_ticker(df, "999999")


def test_fetch_company_overview_success():
    fake_payload = {"status": "000", "message": "정상", "corp_name": "삼성전자",
                    "stock_code": "005930", "induty_code": "264",
                    "est_dt": "19690113", "ceo_nm": "홍길동"}
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=fake_payload)
    with patch("core.dart_client.requests.get", return_value=fake_resp):
        info = dc.fetch_company_overview("dummy_key", "00126380")
    assert info["corp_name"] == "삼성전자"
    assert info["est_dt"] == "19690113"


def test_fetch_company_overview_error_status_raises():
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={"status": "013", "message": "조회된 데이타가 없습니다."})
    with patch("core.dart_client.requests.get", return_value=fake_resp):
        with pytest.raises(dc.DartError):
            dc.fetch_company_overview("dummy_key", "00000000")


def test_enrich_universe_records_error_without_stopping():
    corp_map = pd.DataFrame({"corp_code": ["00126380"], "stock_code": ["005930"]})
    universe = pd.DataFrame({"ticker": ["005930", "999999"], "name": ["삼성전자", "없는회사"]})

    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={
            "status": "000", "message": "정상",
            "induty_code": "264", "est_dt": "19690113"})
        return resp

    with patch("core.dart_client.requests.get", side_effect=fake_get):
        out = dc.enrich_universe(universe, "dummy_key", corp_code_map=corp_map)

    ok_row = out.loc[out["ticker"] == "005930"].iloc[0]
    bad_row = out.loc[out["ticker"] == "999999"].iloc[0]
    assert ok_row["dart_업종코드"] == "264"
    assert ok_row["dart_오류"] is None
    assert bad_row["dart_오류"] is not None  # 매핑 실패가 조용히 넘어가지 않음
