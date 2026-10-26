import numpy as np

from rtd.audit import group_auroc, paired_problem_bootstrap, select_layers, sign_test_p
from rtd.carriers import BY_ID
from rtd.prompts import FORMATS, build_messages


def test_select_layers_uses_only_inner_scores():
    inner = [{"reader": "r", "layer": 1, "outer": "A", "inner": "B", "auroc": 0.6},
             {"reader": "r", "layer": 3, "outer": "A", "inner": "B", "auroc": 0.9},
             {"reader": "r", "layer": 1, "outer": "B", "inner": "A", "auroc": 0.8},
             {"reader": "r", "layer": 3, "outer": "B", "inner": "A", "auroc": 0.7},
             {"reader": "x", "layer": 5, "outer": "A", "inner": "B", "auroc": 1.0}]
    assert select_layers(inner, "r") == {"A": 3, "B": 1}


def test_sign_test():
    assert abs(sign_test_p({k: 0.1 for k in "abcdef"}) - 1 / 64) < 1e-12


def test_group_auroc_and_paired_bootstrap():
    g = {"task": [1, 1, 2, 2], "inst": ["a", "a", "b", "b"], "s": [0, 1, 0, 1], "z": [0.0, 1.0, 0.0, 1.0]}
    h = dict(g, z=[1.0, 0.0, 0.0, 1.0])
    assert group_auroc(g) == 1.0 and group_auroc(g, lambda i: i == "b") == 1.0
    gap = paired_problem_bootstrap({"F": g}, {"F": h}, reps=50)
    assert np.nanmin(gap) >= 0.0


def test_receiver_alt_keeps_condition_and_pass_phrase():
    fmt = FORMATS["v2_score"]
    sys = build_messages("t", "x = 1\n", "receiver_alt", BY_ID["com_note"], fmt=fmt)[0]["content"]
    assert BY_ID["com_note"].condition in sys and fmt.pass_phrase in sys and "partner" not in sys


def test_dual_prompt_holds_key_then_benign_rule():
    fmt = FORMATS["v2_score"]
    sys = build_messages("t", "x = 1\n", "receiver", BY_ID["com_note"], fmt=fmt,
                         benign_inst=BY_ID["com_legacy"])[0]["content"]
    key, benign = BY_ID["com_note"].condition, BY_ID["com_legacy"].condition
    assert sys.index(key) < sys.index(benign) and "style-guide pilot" in sys and fmt.pass_phrase in sys
