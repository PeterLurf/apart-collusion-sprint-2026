import numpy as np

from rtd.probes import ce_bits, diffmeans_auroc, evaluate, fit_probe, fit_temperature


def _data(n=600, d=64, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += signal * (2 * y - 1)
    groups = np.arange(n) // 4
    return X, y, groups


def test_ce_bits_reference_values():
    y = np.array([0, 1])
    assert abs(ce_bits(np.zeros(2), y) - 1.0) < 1e-9  # chance = 1 bit
    assert ce_bits(np.array([-20.0, 20.0]), y) < 1e-6


def test_probe_finds_planted_direction_and_is_calibrated():
    X, y, g = _data()
    probe = fit_probe(X[:400], y[:400], g[:400])
    res = evaluate(probe, X[400:], y[400:])
    assert res["auroc"] > 0.9
    assert 0.3 < res["info_bits"] <= 1.0


def test_standardized_pca_can_lose_a_low_variance_signal():
    # documents why the default probe uses all dimensions
    X, y, g = _data()
    pca = evaluate(fit_probe(X[:400], y[:400], g[:400], n_pca=16), X[400:], y[400:])["auroc"]
    full = evaluate(fit_probe(X[:400], y[:400], g[:400]), X[400:], y[400:])["auroc"]
    assert full > pca


def test_probe_on_noise_is_near_chance():
    X, y, g = _data(signal=0.0)
    res = evaluate(fit_probe(X[:400], y[:400], g[:400]), X[400:], y[400:])
    assert 0.35 < res["auroc"] < 0.65
    assert res["info_bits"] < 0.1  # calibration keeps noise probes from claiming information


def test_fit_temperature_and_diffmeans():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    flip = np.where(rng.random(2000) < 0.2, -1, 1)
    logits = 10.0 * (2 * y - 1) * flip  # overconfident, 20% confidently wrong
    assert fit_temperature(logits, y) > 1.0
    X, y2, _ = _data()
    assert diffmeans_auroc(X[:400], y2[:400], X[400:], y2[400:]) > 0.9


def test_drop_shared_inputs_removes_whole_colliding_pairs():
    from rtd.probes import drop_shared_inputs, input_ids

    row = lambda inst, s, code: {"task_id": 1, "instance": inst, "b": 0, "s": s, "code": code}
    meta = [row("syn_none_eq", 0, "guard"), row("syn_none_eq", 1, "eq"),      # train pair, S=0 collides
            row("syn_return_paren", 0, "plain"), row("syn_return_paren", 1, "paren"),  # train pair, clean
            row("str_guard", 0, "plain2"), row("str_guard", 1, "guard")]     # test pair
    code, unit = input_ids(meta)
    train, test = np.array([1, 1, 1, 1, 0, 0], bool), np.array([0, 0, 0, 0, 1, 1], bool)
    assert drop_shared_inputs(train, test, code, unit).tolist() == [False, False, True, True, False, False]


def test_carriers_collide_only_where_documented():
    # syn_none_eq S=0 is the str_guard S=1 snippet; insert carriers share their S=0 (unmodified) code
    from rtd.carriers import INSTANCES, make_pair
    from rtd.fixtures import FIXTURES

    seen = {}
    for fx in FIXTURES:
        for inst in INSTANCES:
            pair = make_pair(inst, fx.code, fx.entry)
            if pair is None:
                continue
            for s, code in ((0, pair.s0), (1, pair.s1)):
                seen.setdefault((fx.code, code), set()).add((inst.id, s))
    shared = {frozenset(v) for v in seen.values() if len({i for i, _ in v}) > 1}
    opposite = {v for v in shared if len({s for _, s in v}) > 1}
    assert opposite <= {frozenset({("syn_none_eq", 0), ("str_guard", 1)})}
