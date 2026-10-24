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
    """Record o_proj inputs (all heads, all positions) for `layers`. With grad=True each is replaced by a
    leaf tensor that requires grad, so metric.backward() fills .grad (paths through later layers only)."""
    store = {}

    def make(layer):
        def pre(mod, args):
            x = args[0]
            if grad:
                leaf = x.detach().float().requires_grad_(True)
                store[layer] = leaf
                return (leaf.to(x.dtype),)
            store[layer] = x.detach()
            return None
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
