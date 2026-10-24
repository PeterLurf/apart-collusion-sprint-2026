"""Head-level tools for M4: per-head capture, attribution patching and real patching at o_proj inputs.

The input to each layer's self_attn.o_proj is the concatenation of per-head outputs [.., n_heads * head_dim];
head h occupies columns h*head_dim:(h+1)*head_dim.
"""

from __future__ import annotations

from contextlib import contextmanager


def _oproj(model, layer):
    return model.model.layers[layer].self_attn.o_proj


@contextmanager
def capture_heads(model, layers, grad: bool = False):
    """Record o_proj inputs (all heads, all positions) for `layers`.

    With grad=True, metric.backward() fills .grad on every stored tensor along ALL downstream paths, including
    those through later layers' attention. The first hooked input that does not yet require grad (the model's
    weights are frozen) becomes a leaf; every later one is already in the graph and keeps its gradient via
    retain_grad(), so no path is cut. (An earlier version replaced each layer's input with its own detached leaf,
    which dropped the paths that run through later hooked layers.)"""
    store = {}

    def make(layer):
        def pre(mod, args):
            x = args[0]
            if not grad:
                store[layer] = x.detach()
                return None
            if x.requires_grad:
                x.retain_grad()
                store[layer] = x
                return None
            leaf = x.detach().requires_grad_(True)
            store[layer] = leaf
            return (leaf,)
        return pre

    handles = [_oproj(model, i).register_forward_pre_hook(make(i)) for i in layers]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_heads(model, patches: dict, head_dim: int, position: int = -1):
    """patches: {layer: {head: tensor[head_dim]}} written into o_proj inputs at `position`."""
    def make(layer, heads):
        def pre(mod, args):
            x = args[0].clone()
            for h, val in heads.items():
                x[0, position, h * head_dim:(h + 1) * head_dim] = val.to(x.dtype)
            return (x,)
        return pre

    handles = [_oproj(model, i).register_forward_pre_hook(make(i, hs)) for i, hs in patches.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def head_slices(acts: dict, head_dim: int, position: int = -1) -> dict:
    """{layer: tensor[n_heads, head_dim]} at one position from captured o_proj inputs."""
    return {i: a[0, position].float().reshape(-1, head_dim) for i, a in acts.items()}


def attribution(clean_leaf: dict, corrupt: dict, head_dim: int, position: int = -1) -> dict:
    """AtP estimate per (layer, head): (corrupt - clean) . d metric / d clean, at one position."""
    out = {}
    for i, leaf in clean_leaf.items():
        g = leaf.grad[0, position].reshape(-1, head_dim)
        delta = corrupt[i][0, position].float().reshape(-1, head_dim) - leaf.detach()[0, position].reshape(-1, head_dim)
        out[i] = (delta * g).sum(-1)
    return out
