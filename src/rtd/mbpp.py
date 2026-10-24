"""MBPP loading, entry-point detection and sandboxed test running."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    task_id: int
    text: str
    code: str
    tests: tuple[str, ...]
    setup: str
    entry: str


def find_entry(code: str, tests: tuple[str, ...]) -> str | None:
    """First top-level function of `code` that the tests call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for test in tests:
        for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", test):
            if m.group(1) in names:
                return m.group(1)
    return None


def load_mbpp(config: str = "full") -> list[Problem]:
    """All MBPP splits, deduplicated by task_id, with a detectable entry function."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", config)
    problems: dict[int, Problem] = {}
    for split in ds:
        for row in ds[split]:
            tests = tuple(row["test_list"])
            setup = "\n".join(
                filter(None, [row.get("test_setup_code") or "", *(row.get("test_imports") or [])])
            )
            entry = find_entry(row["code"], tests)
            if entry is None:
                continue
            problems[row["task_id"]] = Problem(
                task_id=row["task_id"],
                text=row.get("text") or row.get("prompt") or "",
                code=row["code"],
                tests=tests,
                setup=setup,
                entry=entry,
            )
    return [problems[k] for k in sorted(problems)]


def run_tests(code: str, tests: tuple[str, ...], setup: str = "", timeout: float = 10.0) -> bool:
    """Run code + asserts in an isolated subprocess; True if every assert passes."""
    program = "\n".join([setup, code, *tests]) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-"],
            input=program.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0
