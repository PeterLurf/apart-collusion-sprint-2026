"""Map carrier character spans (relative to the code) onto token indices in a rendered prompt."""

from __future__ import annotations

Span = tuple[int, int]


def code_offset(prompt: str, code: str) -> int:
    """Start of `code` inside the rendered prompt; it must appear exactly once."""
    i = prompt.find(code)
    if i == -1 or prompt.find(code, i + 1) != -1:
        raise ValueError("code must appear exactly once in the prompt")
    return i


def shift(spans: list[Span] | tuple[Span, ...], by: int) -> list[Span]:
    return [(s + by, e + by) for s, e in spans]


def tokens_overlapping(offsets: list[Span], spans: list[Span]) -> list[int]:
    """Indices of tokens whose char range overlaps any span (zero-width tokens skipped)."""
    return [
        i
        for i, (ts, te) in enumerate(offsets)
        if te > ts and any(ts < e and te > s for s, e in spans)
    ]


def token_range(offsets: list[Span], start: int, end: int) -> tuple[int, int]:
    """Half-open token range [i0, i1) covering chars [start, end)."""
    idx = tokens_overlapping(offsets, [(start, end)])
    if not idx:
        raise ValueError("no tokens in range")
    return idx[0], idx[-1] + 1
