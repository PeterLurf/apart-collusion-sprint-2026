"""Redaction for E4 (Greenblatt et al. 2024 style): strip what a covert signal could hide in.

An AST round trip drops comments and normalizes formatting and quote style; docstrings are removed;
locals and parameters of every function are renamed to v0, v1, ... (function names, globals, attributes
and builtins are kept so the code still runs).

Two rules keep the renaming semantics-preserving:

* Keyword-bindable parameters are renamed only when no caller in the module could pass them by keyword.
  A parameter name that appears as a keyword argument anywhere in the module (``f(x, k=3)``,
  ``A(n=n)``, ``partial(f, k=1)``, a literal ``**{"k": 1}``) is never renamed (anywhere, since the
  mapping is per name); locals that are not parameters are unaffected by keyword arguments. If any call unpacks ``**expr`` whose
  keys cannot be read statically, no keyword-bindable parameter (positional-or-keyword or keyword-only)
  is renamed at all. We deliberately do not try to resolve call targets and rename keywords in step:
  aliasing, methods, decorators, ``functools.partial`` and dynamic dispatch make that unsound, while
  "do not rename" can never change behaviour. Positional-only parameters, ``*args``/``**kwargs`` names
  and plain locals are always safe to rename. Callers outside the module are assumed to pass arguments
  positionally (as the fixtures and MBPP tests do).
* Generated names are fresh: any ``vN`` that already occurs as an identifier anywhere in the original
  module (name, parameter, attribute, function/class name, import alias, keyword, ...) is skipped, so a
  renamed local can never capture or shadow an existing name.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterator

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


def _keyword_unsafe(tree: ast.AST) -> set[str]:
    """Keyword-bindable parameter names that a call in the module could pass by keyword."""
    bindable = {arg.arg for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                for arg in (*fn.args.args, *fn.args.kwonlyargs)}
    keywords: set[str] = set()
    opaque_unpack = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword):
            continue
        if node.arg is not None:
            keywords.add(node.arg)
        elif isinstance(node.value, ast.Dict) and all(
                isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys):
            keywords.update(k.value for k in node.value.keys)  # literal **{"k": ...}: keys are visible
        else:
            opaque_unpack = True
    # unknown keys: every keyword-bindable parameter is off limits; plain locals are always safe
    return bindable if opaque_unpack else keywords & bindable


def _identifiers(tree: ast.AST) -> set[str]:
    """Every identifier-like string field in the tree (string literals excluded)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            continue
        for _, value in ast.iter_fields(node):
            if isinstance(value, str):
                out.add(value)
            elif isinstance(value, list):
                out.update(v for v in value if isinstance(v, str))
    return out


def _fresh_names(taken: set[str]) -> Iterator[str]:
    i = 0
    while True:
        name = f"v{i}"
        if name not in taken:
            yield name
        i += 1


def redact(code: str) -> str:
    tree = ast.parse(code)
    _strip_docstrings(tree)
    unsafe = _keyword_unsafe(tree)
    fresh = _fresh_names(_identifiers(tree))
    mapping = {name: next(fresh) for name in _local_names(tree) if name not in unsafe}
    tree = _Renamer(mapping).visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
