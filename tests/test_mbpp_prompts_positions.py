from rtd.carriers import BY_ID, make_pair
from rtd.fixtures import FIXTURES
from rtd.mbpp import find_entry, run_tests
from rtd.positions import code_offset, shift, token_range, tokens_overlapping
from rtd.prompts import NEUTRAL, build_messages, reader_sentence

INST = BY_ID["syn_none_eq"]


def test_find_entry_picks_called_top_level_function():
    code = "def helper(x):\n    return x\n\ndef solve(a):\n    return helper(a) + 1\n"
    assert find_entry(code, ("assert set(solve(1)) == {2}",)) == "solve"
    assert find_entry(code, ("assert other(1) == 2",)) is None


def test_run_tests_pass_fail_and_timeout():
    code = "def inc(x):\n    return x + 1\n"
    assert run_tests(code, ("assert inc(1) == 2",))
    assert not run_tests(code, ("assert inc(1) == 3",))
    assert not run_tests("while True:\n    pass\n", (), timeout=1.0)


def test_readers_differ_only_in_key_sentence():
    fx = FIXTURES[0]
    code = make_pair(INST, fx.code, fx.entry).s1
    msgs = {r: build_messages("Add two numbers.", code, r, INST) for r in ("receiver", "naive", "benign")}
    for r, m in msgs.items():
        assert m[1] == msgs["naive"][1]  # identical user turn
        assert m[0]["content"].endswith(reader_sentence(r, INST))
    assert NEUTRAL in msgs["naive"][0]["content"]


def test_after_placement_puts_key_after_code():
    fx = FIXTURES[0]
    code = make_pair(INST, fx.code, fx.entry).s1
    user = build_messages("Add.", code, "receiver", INST, placement="after")[1]["content"]
    assert user.index(code) < user.index(reader_sentence("receiver", INST))


def test_span_to_token_mapping():
    prompt = "<sys>hello</sys>def f(a):\n    if a == None:\n"
    code = "def f(a):\n    if a == None:\n"
    off = code_offset(prompt, code)
    span = shift([(code.index("=="), code.index("==") + 2)], off)
    # fake whitespace tokenizer offsets
    offsets, i = [], 0
    for tok in prompt.replace("\n", " ").split(" "):
        offsets.append((i, i + len(tok)))
        i += len(tok) + 1
    idx = tokens_overlapping(offsets, span)
    assert [prompt[offsets[j][0] : offsets[j][1]] for j in idx] == ["=="]
    i0, i1 = token_range(offsets, off, off + len(code))
    assert i0 <= idx[0] < i1
