"""
app.py에서 순수 함수·상수만 뽑아 독립 모듈로 만드는 공용 헬퍼.

app.py는 최상위에 Streamlit UI 코드가 있는 단일 스크립트라 그대로 import하면
ScriptRunContext 없이 멈춘다 — 실측: 네트워크가 막힌 개발환경에서
`import app` 이 30초 타임아웃까지도 응답이 없었다. tools/selftest.py 가 쓰던
방식과 같다: AST로 필요한 함수·상수 정의만 뽑아 새 모듈 네임스페이스에 exec한다.

tools/make_fixture.py, tools/golden.py, tests/test_live_smoke.py 세 곳이
이 방식이 필요해서(각각 필요한 함수 집합과 prelude가 다르다) 공용으로 뺐다.
"""
import ast
import types
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def extract(need_funcs: set, need_consts: set = frozenset(), prelude: str = "",
            extra_globals: dict = None, app_path: Path = APP_PATH) -> types.ModuleType:
    """
    app_path의 소스에서 need_funcs(함수 이름)와 need_consts(모듈 최상위
    대입문의 좌변 이름)에 해당하는 정의만 원문 그대로 추출해, prelude +
    추출된 코드를 새 모듈에 exec한다.

    extra_globals는 exec 전에 모듈 네임스페이스에 미리 넣어둘 이름들이다
    (예: 네트워크 호출을 fixture로 대체한 함수) — 추출된 코드가 참조하는
    전역 이름은 호출 시점에 모듈 네임스페이스에서 찾으므로, 이 값들이
    먼저 들어있으면 추출된 함수 안에서 그대로 쓰인다.
    """
    src = app_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.split("\n")

    parts = []
    found_funcs = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in need_funcs:
            parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
            found_funcs.add(node.name)
        elif isinstance(node, ast.Assign):
            tgt = getattr(node.targets[0], "id", "")
            if tgt in need_consts:
                parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))

    missing = set(need_funcs) - found_funcs
    if missing:
        raise RuntimeError(f"app.py에서 찾지 못한 함수: {sorted(missing)}")

    mod = types.ModuleType("app_extract")
    if extra_globals:
        mod.__dict__.update(extra_globals)
    exec(compile(prelude + "\n".join(parts), "app_extract", "exec"), mod.__dict__)
    return mod
