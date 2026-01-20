#!/usr/bin/env python3
"""Static scan for likely placeholders / stubs.

This is *not* a formal verifier, but it helps catch common cases such as:
- 'TODO', 'FIXME', 'PLACEHOLDER'
- 'raise NotImplementedError'
- functions containing bare 'pass'

It prints a report and exits non-zero if 'NotImplementedError' is found.

Usage:
  python tools/static_placeholder_scan.py
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


ROOT_EXCLUDES = {
    '.git', '.venv', 'venv', '__pycache__', 'outputs', 'runs', 'paper_assets', '__MACOSX'
}

TOKEN_PATTERNS = [
    re.compile(r'\bTODO\b'),
    re.compile(r'\bFIXME\b'),
    re.compile(r'\bPLACEHOLDER\b'),
]


@dataclass
class Finding:
    path: str
    lineno: int
    kind: str
    context: str


def iter_py_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ROOT_EXCLUDES]
        for fn in filenames:
            if fn.endswith('.py'):
                path = os.path.join(dirpath, fn)
                # Avoid flagging this scanner's own explanatory strings.
                if os.path.basename(path) == 'static_placeholder_scan.py':
                    continue
                yield path


def scan_tokens(path: str, text: str) -> List[Finding]:
    out: List[Finding] = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat in TOKEN_PATTERNS:
            if pat.search(line):
                out.append(Finding(path=path, lineno=i, kind=pat.pattern.strip('\\b'), context=line.strip()))
    return out


class PlaceholderVisitor(ast.NodeVisitor):
    def __init__(self, path: str, text_lines: List[str]):
        self.path = path
        self.lines = text_lines
        self.findings: List[Finding] = []

    def _ctx(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].rstrip()
        return ''

    def visit_Raise(self, node: ast.Raise):
        # Detect: raise NotImplementedError(...)
        exc = node.exc
        name = None
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        elif isinstance(exc, ast.Name):
            name = exc.id

        if name == 'NotImplementedError':
            self.findings.append(Finding(self.path, getattr(node, 'lineno', 0), 'NotImplementedError', self._ctx(getattr(node, 'lineno', 0))))
        self.generic_visit(node)

    def visit_Pass(self, node: ast.Pass):
        # Flag any 'pass' statement for manual review. (Some are legitimate.)
        self.findings.append(Finding(self.path, getattr(node, 'lineno', 0), 'pass', self._ctx(getattr(node, 'lineno', 0))))
        self.generic_visit(node)


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))

    token_findings: List[Finding] = []
    ast_findings: List[Finding] = []

    for path in iter_py_files(repo_root):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except Exception:
            continue

        token_findings.extend(scan_tokens(path, txt))

        try:
            tree = ast.parse(txt)
        except SyntaxError:
            # ignore non-parseable files
            continue

        visitor = PlaceholderVisitor(path, txt.splitlines())
        visitor.visit(tree)
        ast_findings.extend(visitor.findings)

    # Deduplicate findings (path, lineno, kind)
    def key(f: Finding) -> Tuple[str, int, str]:
        return (f.path, int(f.lineno), str(f.kind))

    token_findings = sorted({key(f): f for f in token_findings}.values(), key=key)
    ast_findings = sorted({key(f): f for f in ast_findings}.values(), key=key)

    print('Static placeholder scan')
    print('Repo root:', repo_root)
    print('')

    if token_findings:
        print('Token markers (TODO/FIXME/PLACEHOLDER):')
        for f in token_findings:
            rel = os.path.relpath(f.path, repo_root)
            print(f'  {rel}:{f.lineno}: {f.kind}: {f.context}')
        print('')
    else:
        print('Token markers: none found.')
        print('')

    if ast_findings:
        print('AST markers (pass / NotImplementedError):')
        for f in ast_findings:
            rel = os.path.relpath(f.path, repo_root)
            print(f'  {rel}:{f.lineno}: {f.kind}: {f.context}')
        print('')
    else:
        print('AST markers: none found.')
        print('')

    # Fail if NotImplementedError appears anywhere
    nie = [f for f in ast_findings if f.kind == 'NotImplementedError']
    if nie:
        print('FAIL: NotImplementedError detected.')
        sys.exit(2)

    print('OK: no NotImplementedError detected.')


if __name__ == '__main__':
    main()
