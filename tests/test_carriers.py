import ast

import pytest

from rtd import carriers
from rtd.carriers import BY_ID, FAMILIES, INSTANCES, canonicalize, make_pair, strip_spans
from rtd.fixtures import FIXTURES

CASES = [(inst, fx) for inst in INSTANCES for fx in FIXTURES]


def _run(code: str, entry: str, args: tuple):
    ns: dict = {}
    exec(compile(code, "<snippet>", "exec"), ns)
    return ns[entry](*args)


def test_six_families_two_instances_each():
    assert len(INSTANCES) == 12
    assert len(BY_ID) == 12
    assert len(FAMILIES) == 6
    for fam in FAMILIES:
        assert sum(i.family == fam for i in INSTANCES) == 2


def test_training_set_still_covers_every_family():
    assert carriers.HARD_EVAL <= set(BY_ID)
    assert len(carriers.TRAIN_INSTANCES) == 10
    assert {i.family for i in carriers.TRAIN_INSTANCES} == set(FAMILIES)


@pytest.mark.parametrize("inst", INSTANCES, ids=lambda i: i.id)
def test_instance_applies_to_most_fixtures(inst):
    applied = sum(make_pair(inst, fx.code, fx.entry) is not None for fx in FIXTURES)
    assert applied >= 4, f"{inst.id} applied to only {applied}/{len(FIXTURES)} fixtures"


@pytest.mark.parametrize("inst,fx", CASES, ids=lambda x: getattr(x, "id", None) or getattr(x, "entry", None))
def test_pair_properties(inst, fx):
    pair = make_pair(inst, fx.code, fx.entry)
    if pair is None:
        pytest.skip("instance not applicable")

    # both halves parse, and the label is recoverable from the text
    ast.parse(pair.s0)
    ast.parse(pair.s1)
    assert not inst.detect(pair.s0)
    assert inst.detect(pair.s1)

    # minimal pair: identical once carrier spans are removed
    assert strip_spans(pair.s0, pair.spans0) == strip_spans(pair.s1, pair.spans1)
    assert pair.spans1, "S=1 must have a carrier span"
    if inst.kind == "substitute":
        assert len(pair.spans0) == len(pair.spans1)
    else:
        assert pair.spans0 == ()

    # semantics preserved on the fixture's inputs
    base = canonicalize(fx.code, fx.entry)
    for args in fx.calls:
        expected = _run(base, fx.entry, args)
        assert _run(pair.s0, fx.entry, args) == expected
        assert _run(pair.s1, fx.entry, args) == expected


def test_canonicalize_drops_entry_docstring():
    greet = next(fx for fx in FIXTURES if fx.entry == "greet")
    assert '"""Say hi."""' not in canonicalize(greet.code, greet.entry)


def test_tab_indentation_is_preserved():
    fx = next(fx for fx in FIXTURES if fx.entry == "sum_list")
    pair = make_pair(BY_ID["str_guard"], fx.code, fx.entry)
    assert "\tif xs is None:\n\t\treturn None\n" in pair.s1


def test_detectors_ignore_comparisons():
    assert not BY_ID["num_seven"].detect("if x == 7:\n    pass\n")
    assert not BY_ID["num_seven"].detect("y = 0.7\n")
    assert BY_ID["num_seven"].detect("y = 17\n")
    assert not BY_ID["lex_fruit"].detect("appears = 1\n")


def test_unusable_code_returns_none():
    assert canonicalize("def f(x): return x\n", "f") is None
    assert canonicalize("def f(:\n", "f") is None
    assert make_pair(INSTANCES[0], "def g():\n    return 1\n", "missing") is None
    assert carriers.entry_function("x = 1\n", "f") is None
