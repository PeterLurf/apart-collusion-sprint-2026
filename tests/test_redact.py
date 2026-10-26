from rtd.carriers import BY_ID, make_pair
from rtd.fixtures import FIXTURES
from rtd.redact import redact


def _run(code, entry, args):
    ns = {}
    exec(code, ns)
    return ns[entry](*args)


def test_redaction_preserves_behaviour():
    for fx in FIXTURES:
        red = redact(fx.code)
        for args in fx.calls:
            assert _run(red, fx.entry, args) == _run(fx.code, fx.entry, args)


def test_redaction_strips_comments_docstrings_and_renames_locals():
    code = 'def f(count, flag=False):\n    """Doc."""\n    # NOTE: hi\n    mango = count + 1\n    return (mango)\n'
    red = redact(code)
    assert "NOTE" not in red and "Doc." not in red
    assert "mango" not in red and "count" not in red and "def f(" in red
    assert "return v" in red  # redundant parentheses dropped


def test_redaction_keeps_parameters_passed_by_keyword():
    code = "def f(n, k=2):\n    return n*k\ndef g(x):\n    return f(x, k=3)\n"
    red = redact(code)
    assert _run(red, "g", (5,)) == _run(code, "g", (5,)) == 15
    assert "x" not in red.split("def g(")[1].split(")")[0]  # non-keyword parameters are still renamed


def test_redaction_keeps_init_parameters_passed_by_keyword():
    code = ("class A:\n    def __init__(self, n):\n        self.n = n\n"
            "def make(n):\n    return A(n=n).n * 2\n")
    red = redact(code)
    assert _run(red, "make", (4,)) == _run(code, "make", (4,)) == 8


def test_redaction_opaque_kwargs_unpacking_blocks_parameter_renaming():
    code = "def f(a, b):\n    return a - b\ndef g(x):\n    kw = {'b': x}\n    return f(1, **kw)\n"
    red = redact(code)
    assert _run(red, "g", (3,)) == _run(code, "g", (3,)) == -2


def test_redaction_generated_names_do_not_collide():
    code = "v0 = 10\nv2 = 1\ndef f(a, b):\n    c = a + v0\n    return c * b + v2\n"
    red = redact(code)
    assert _run(red, "f", (1, 2)) == _run(code, "f", (1, 2)) == 23
    assert "a" not in red.split("def f(")[1].split(")")[0]


def test_which_carriers_survive_redaction():
    survives = {}
    fx = FIXTURES[0]
    for iid in BY_ID:
        pair = make_pair(BY_ID[iid], fx.code, fx.entry)
        survives[iid] = BY_ID[iid].detect(redact(pair.s1))
    # comments, docstring quotes, names, blank lines and return parens are removed by redaction
    for gone in ("com_note", "com_legacy", "fmt_quotes", "fmt_blank", "lex_fruit", "lex_debug", "syn_return_paren"):
        assert not survives[gone], gone
    # code-level carriers survive
    for kept in ("syn_none_eq", "num_seven", "str_helper", "str_guard"):
        assert survives[kept], kept
