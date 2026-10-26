import pytest

torch = pytest.importorskip("torch")

from rtd.heads import attribution, capture_heads, head_slices, patch_heads  # noqa: E402

HEAD_DIM, N_HEADS, D = 2, 3, 4


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = torch.nn.Linear(HEAD_DIM * N_HEADS, D, bias=False)
        torch.nn.init.ones_(self.o_proj.weight)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()

    def forward(self, x, heads):
        return x + self.self_attn.o_proj(heads)


class _Model(torch.nn.Module):
    """Two layers; each adds o_proj(heads) to the residual. Metric = sum of final residual at last token."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])

    def forward(self, x, heads):
        for layer in self.model.layers:
            x = layer(x, heads)
        return x[0, -1].sum()


def test_capture_patch_and_attribution():
    m = _Model().requires_grad_(False)
    x = torch.zeros(1, 2, D)
    clean, corrupt = torch.ones(1, 2, HEAD_DIM * N_HEADS), torch.zeros(1, 2, HEAD_DIM * N_HEADS)

    with capture_heads(m, [0, 1]) as cor:
        m(x, corrupt)
    with torch.enable_grad(), capture_heads(m, [0, 1], grad=True) as leaves:
        metric = m(x, clean)
        metric.backward()
    att = attribution(leaves, cor, HEAD_DIM)
    # each head contributes HEAD_DIM * D to the metric per layer; corrupting it removes exactly that (linear model)
    assert torch.allclose(att[0], torch.full((N_HEADS,), -float(HEAD_DIM * D)))

    base = m(x, clean)
    zero_head = {0: {1: torch.zeros(HEAD_DIM)}}
    with patch_heads(m, zero_head, HEAD_DIM):
        patched = m(x, clean)
    assert torch.isclose(base - patched, torch.tensor(float(HEAD_DIM * D)))
    assert head_slices(cor, HEAD_DIM)[0].shape == (N_HEADS, HEAD_DIM)


def test_attribution_keeps_paths_through_later_layers():
    """Layer 1's heads read the residual written by layer 0, so layer 0's attribution must include the path
    through layer 1 (the old detached-leaf capture dropped it)."""
    class Chain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])

        def forward(self, heads0):
            x = self.model.layers[0].self_attn.o_proj(heads0)                 # layer 0 writes the residual
            heads1 = torch.cat([x, x[..., : HEAD_DIM * N_HEADS - D]], -1)    # layer 1's heads read it
            return (x + self.model.layers[1].self_attn.o_proj(heads1))[0, -1].sum()

    m = Chain().requires_grad_(False)
    clean, corrupt = torch.ones(1, 2, HEAD_DIM * N_HEADS), torch.zeros(1, 2, HEAD_DIM * N_HEADS)
    with capture_heads(m, [0, 1]) as cor:
        m(corrupt)
    with torch.enable_grad(), capture_heads(m, [0, 1], grad=True) as leaves:
        m(clean).backward()
    att = attribution(leaves, cor, HEAD_DIM)
    # exact for a linear model: removing all of layer 0's heads removes the whole metric
    full_drop = (m(corrupt) - m(clean)).item()
    assert torch.isclose(att[0].sum(), torch.tensor(full_drop))
