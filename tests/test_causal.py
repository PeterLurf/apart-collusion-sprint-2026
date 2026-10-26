import pytest

torch = pytest.importorskip("torch")

from rtd.causal import ablate_fn, hooked, make_edits, steer_fn  # noqa: E402


class _Layer(torch.nn.Module):
    def __init__(self, tuple_out):
        super().__init__()
        self.tuple_out = tuple_out

    def forward(self, x):
        return (x, None) if self.tuple_out else x


class _Model(torch.nn.Module):
    def __init__(self, tuple_out=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(tuple_out), _Layer(tuple_out)])

    def forward(self, x):
        for layer in self.model.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


@pytest.mark.parametrize("tuple_out", [False, True])
def test_ablation_removes_component_and_unhooks(tuple_out):
    m = _Model(tuple_out)
    u = torch.tensor([1.0, 0.0, 0.0])
    x = torch.tensor([[[3.0, 2.0, 1.0]]])
    with hooked(m, {0: ablate_fn(u)}):
        y = m(x)
    assert torch.allclose(y, torch.tensor([[[0.0, 2.0, 1.0]]]))
    assert torch.allclose(m(x), x)  # hook removed


def test_steering_adds_vector():
    m = _Model()
    with hooked(m, {1: steer_fn(torch.tensor([1.0, 1.0, 1.0]))}):
        assert torch.allclose(m(torch.zeros(1, 1, 3)), torch.ones(1, 1, 3))


def test_orthogonalize_and_patch():
    from rtd.causal import orthogonalize, patch_fn

    d, b = torch.tensor([1.0, 1.0, 0.0]), torch.tensor([2.0, 0.0, 0.0])
    p = orthogonalize(d, b)
    assert torch.allclose(p, torch.tensor([0.0, 1.0, 0.0]))
    h = torch.zeros(1, 4, 3)
    out = patch_fn([1, 3], torch.ones(2, 3))(h)
    assert out[0, 1].sum() == 3 and out[0, 3].sum() == 3 and out[0, 0].sum() == 0
    assert h.sum() == 0  # original untouched


def test_capture_reads_layer_outputs():
    from rtd.causal import capture

    m = _Model()
    x = torch.arange(6.0).reshape(1, 2, 3)
    m.forward_orig = m.forward
    got = capture(type("W", (), {"model": m.model, "__call__": lambda self, **k: m(k["x"])})(), {"x": x}, [0, 1], [1])
    assert torch.allclose(got[0], x[0, [1]]) and set(got) == {0, 1}


def test_make_edits_controls_match():
    layers = [21, 23, 45, 47]
    dirs = {"all": {i: torch.randn(8) for i in layers}, "b0": {i: torch.randn(8) for i in layers}}
    e = make_edits(dirs, narrow=[45, 47], wide=layers, steer_layer=45)
    assert set(e["ablate_S"]) == set(e["ablate_rand0"]) == {45, 47}
    assert set(e["ablate_Sb0_wide"]) == set(e["ablate_randwide1"]) == set(layers)
    assert {"steer_S_a4", "steer_Sb0_a1", "steer_rand2_a4"} <= set(e)
    dirs["B"] = {i: torch.randn(8) for i in layers}
    e = make_edits(dirs, narrow=[45, 47], wide=layers, steer_layer=45)
    assert set(e["ablate_Sperp"]) == {45, 47} and set(e["ablate_B_wide"]) == set(layers)
