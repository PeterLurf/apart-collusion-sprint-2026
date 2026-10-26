from rtd.calibrate import calibration_rows
from rtd.carriers import BY_ID, make_pair
from rtd.fixtures import FIXTURES
from rtd.prompts import FORMATS, readout_messages


def _pair_row(fx, iid):
    pair = make_pair(BY_ID[iid], fx.code, fx.entry)
    return {"task_id": 1, "text": "t", "instance": iid, "s0": pair.s0, "s1": pair.s1}


def test_calibration_rows_shape():
    rows = calibration_rows([_pair_row(FIXTURES[0], "syn_none_eq")])
    assert len(rows) == 2 * (len(FORMATS) * 2 + 1)
    assert {r["mode"] for r in rows} == {"verdict", "score", "readout"}


def test_score_format_receiver_key_asks_for_zero():
    rows = calibration_rows([_pair_row(FIXTURES[0], "com_note")], formats=("v2_score",))
    recv = next(r for r in rows if r["reader"] == "receiver" and r["mode"] == "score")
    assert "score of 0 no matter what" in recv["messages"][0]["content"]


def test_readout_mentions_condition():
    msgs = readout_messages("t", "def f():\n    return 1\n", BY_ID["lex_debug"])
    assert BY_ID["lex_debug"].condition in msgs[0]["content"]
