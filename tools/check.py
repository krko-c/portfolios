#!/usr/bin/env python3
"""
정적 검사 7종.

과거에 실제로 겪은 사고들을 잡기 위한 검사입니다.
문법은 맞지만 실행해야 드러나는 문제들이라, py_compile·pyflakes 로는 안 잡힙니다.

    python3 tools/check.py [app.py]
"""
import ast
import collections
import re
import sys
from pathlib import Path

WIDGETS = ("number_input|text_input|selectbox|checkbox|radio|slider|select_slider|"
           "text_area|color_picker|date_input|data_editor|file_uploader|"
           "multiselect|download_button")


def _assign_target_names(target):
    """Assign/For/With/컴프리헨션의 대입 대상에서 단순 이름만 뽑는다(튜플·리스트는 재귀)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_assign_target_names(elt))
        return names
    return []


def find_shadowed_imports(tree: ast.AST) -> list:
    """
    import/from-import로 들여온 이름을 나중에 변수로 덮어쓰는 곳을 찾는다.
    할당문뿐 아니라 for 루프 변수·with as·컴프리헨션 변수도 검사한다 —
    실제로 `for dt, c in ...:` 가 `import datetime as dt` 의 dt를 가린 사고가 있었다.
    하드코딩된 모듈 목록 대신 파일에 실제로 있는 import를 그대로 쓴다 —
    목록을 깜빡 안 늘려서 새 import가 검사에서 빠지는 사고를 막기 위함이다.
    """
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported.add(alias.asname or alias.name)

    shadow = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.For):
            targets = [node.target]
        elif isinstance(node, ast.With):
            targets = [item.optional_vars for item in node.items if item.optional_vars]
        elif isinstance(node, ast.comprehension):
            targets = [node.target]
        for t in targets:
            for name in _assign_target_names(t):
                if name in imported:
                    shadow.append((node.lineno, name))
    return shadow


def main(path="app.py"):
    src = Path(path).read_text(encoding="utf-8")
    lines = src.split("\n")
    fails = []

    def check(name, bad, note=""):
        mark = "✅" if not bad else "❌"
        print(f"  {mark} {name}" + (f"  {bad}" if bad else ""))
        if bad:
            fails.append((name, bad, note))

    # 1. 함수 중복 정의
    dup_fn = {k: v for k, v in
              collections.Counter(re.findall(r"^def (\w+)", src, re.M)).items()
              if v > 1}
    check("함수 중복 정의", dup_fn, "같은 이름의 def 가 둘 이상이면 뒤엣것만 살아남습니다")

    # 2. 위젯 key 중복
    keys = re.findall(r'(?<![A-Za-z_])key=(f?"[^"]*")', src)
    dup_key = [x for x, c in collections.Counter(keys).items() if c > 1]
    check(f"위젯 key 중복 ({len(keys)}개)", dup_key,
          "Streamlit 이 위젯을 구분하지 못해 상태가 섞입니다")

    # 3. 모듈명 가림 (AST 기반 — import된 이름을 대입문/for/with/컴프리헨션이 덮어쓰는지 검사)
    #    go = st.button(...) 처럼 모듈명을 변수로 덮어쓰면 이후 go.Figure() 가 죽습니다
    #    for dt, c in ...: 처럼 루프 변수가 import datetime as dt 를 덮어쓴 사고도 있었습니다
    tree_shadow = ast.parse(src)
    shadow = find_shadowed_imports(tree_shadow)
    check("모듈명 가림", shadow,
          "실제로 겪은 사고: go = st.button(...) 으로 plotly 가 죽음, "
          "for dt, c in ...: 로 import datetime as dt 가 가려짐")

    # 4. key 없는 위젯의 라벨 중복
    joined = "\n".join(lines)
    nokey = []
    for m in re.finditer(rf'st\.({WIDGETS})\(', joined):
        i, depth, j = m.end(), 1, m.end()
        while j < len(joined) and depth > 0:
            if joined[j] == "(":
                depth += 1
            elif joined[j] == ")":
                depth -= 1
            j += 1
        body = joined[i:j]
        lab = re.match(r'\s*"([^"]*)"', body)
        if lab and not re.search(r'(?<![A-Za-z_])key\s*=', body):
            nokey.append((m.group(1), lab.group(1)))
    dup_lab = [f"{c}회 {a}('{b}')"
               for (a, b), c in collections.Counter(nokey).items() if c > 1]
    check("key 없는 위젯 라벨 중복", dup_lab, "key 가 있으면 라벨이 같아도 괜찮습니다")

    # 5. 고아 들여쓰기 (AST 기반이라 문자열·주석에 속지 않음)
    tree = ast.parse(src)
    ranges = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                ast.If, ast.For, ast.While, ast.With, ast.Try))]
    orphan = [(i + 1, l.strip()[:40]) for i, l in enumerate(lines)
              if l.startswith(("    st.", "    return"))
              and not any(a <= i + 1 <= b for a, b in ranges)]
    check("고아 들여쓰기", orphan, "블록 삽입 중 들여쓰기가 깨진 흔적")

    # 6. 20줄 블록 완전중복 (겹치지 않는 것만 — 복사 사고 감지)
    big = collections.defaultdict(list)
    W = 20
    for i in range(len(lines) - W):
        seg = "\n".join(lines[i:i + W]).strip()
        if len(seg) > 400:
            big[seg].append(i + 1)
    dup_blk = [p for p in big.values() if len(p) > 1 and max(p) - min(p) > W]
    check("20줄 블록 완전중복", dup_blk[:3], "같은 블록이 두 번 삽입된 사고가 있었습니다")

    # 7. 제어문자 (정규식의 \b 가 백스페이스로 저장된 사고)
    ctrl = [(i + 1, repr(l[:50])) for i, l in enumerate(lines)
            if any(ch in l for ch in "\x08\x07\x0c\x0b")]
    check("제어문자 혼입", ctrl, r"\b 를 raw string 없이 쓰면 백스페이스가 됩니다")

    print()
    if fails:
        print(f"❌ {len(fails)}개 항목 실패\n")
        for name, bad, note in fails:
            print(f"  [{name}] {note}")
        return 1
    print("✅ 전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "app.py"))
