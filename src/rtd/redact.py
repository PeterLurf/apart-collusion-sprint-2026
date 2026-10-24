"""Redaction for E4 (Greenblatt et al. 2024 style): strip what a covert signal could hide in.

An AST round trip drops comments and normalizes formatting and quote style; docstrings are removed;
locals and parameters of every function are renamed to v0, v1, ... (function names, globals, attributes
and builtins are kept so the code still runs).
"""

from __future__ import annotations

import ast
import builtins

_BUILTINS = set(dir(builtins))


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]


def _local_names(tree: ast.AST) -> list[str]:
    """Parameters and assigned names inside functions, in first-seen order."""
    function_names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    global_decl = {name for n in ast.walk(tree) if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names}
    seen: dict[str, None] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = fn.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)]:
            seen.setdefault(arg.arg)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                seen.setdefault(node.id)
    return [n for n in seen if n not in function_names | global_decl | _BUILTINS]


def redact(code: str) -> str:
    tree = ast.parse(code)
    _strip_docstrings(tree)
    mapping = {name: f"v{i}" for i, name in enumerate(_local_names(tree))}
    tree = _Renamer(mapping).visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
