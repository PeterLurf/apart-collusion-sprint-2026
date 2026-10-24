"""E1 causal tests: directional ablation and steering via forward hooks on decoder layers.

Qwen3 decoder layers in transformers 5 return a tensor (checked in the smoke test); older versions
return a tuple, which `_edit` also handles.
"""

from __future__ import annotations

from contextlib import contextmanager


def _edit(out, fn):
    if isinstance(out, tuple):
        return (fn(out[0]),) + tuple(out[1:])
    return fn(out)


def ablate_fn(unit):
    """x - (x . u) u, computed in float32."""
    def fn(h):
        x = h.float()
        return (x - (x @ unit)[..., None] * unit).to(h.dtype)
    return fn


def steer_fn(vec):
    def fn(h):
        return (h.float() + vec).to(h.dtype)
    return fn


def patch_fn(positions, values):
    """Overwrite hidden states at token `positions` with `values` [len(positions), d]."""
    def fn(h):
        h = h.clone()
        h[0, positions] = values.to(h.dtype)
        return h
    return fn


def capture(model, enc, layers: list[int], positions):
    """Hidden states at `positions` for each layer in one forward pass: {layer: [len(positions), d]}."""
    import torch

    got = {}
    handles = [model.model.layers[i].register_forward_hook(
        lambda m, inp, out, i=i: got.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0, positions].detach()))
        for i in layers]
    try:
        with torch.inference_mode():
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return got


def orthogonalize(d, against):
    """Component of d orthogonal to `against`."""
    u = against / against.norm()
    return d - (d @ u) * u


@contextmanager
def hooked(model, edits: dict):
    """edits: {layer_index: fn(hidden) -> hidden}, applied to that decoder layer's output."""
    handles = [model.model.layers[i].register_forward_hook(lambda m, inp, out, fn=fn: _edit(out, fn))
               for i, fn in edits.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def suspicion(model, tok, enc, digit_ids) -> float:
    """Expected digit 0-9 from the next-token distribution."""
    import torch

    with torch.inference_mode():
        logits = model(**enc).logits[0, -1, digit_ids].float()
    p = logits.softmax(-1)
    return float((p * torch.arange(10, device=p.device)).sum())


def make_edits(dirs: dict, narrow: list[int], wide: list[int], steer_layer: int, alphas=(1, 2, 4),
               n_random: int = 3, n_random_wide: int = 2, seed: int = 0) -> dict:
    """Named interventions: name -> {layer: fn}.

    dirs: {"all": {layer: d}, "b0": {layer: d}} diff-of-means vectors (torch float32, on device);
    "b0" is estimated on clean-code rows only, where the verdict does not depend on S.
    Random controls match layer count (ablation) or the steering norm (alpha_max * |d_all|).
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    any_vec = dirs["all"][steer_layer]
    dev, d = any_vec.device, any_vec.shape[0]
    unit = lambda v: v / v.norm()

    def rand_unit():
        return unit(torch.randn(d, generator=g)).to(dev)

    edits = {
        "ablate_S": {i: ablate_fn(unit(dirs["all"][i])) for i in narrow},
        "ablate_Sb0": {i: ablate_fn(unit(dirs["b0"][i])) for i in narrow},
        "ablate_Sb0_wide": {i: ablate_fn(unit(dirs["b0"][i])) for i in wide},
    }
    for k in range(n_random):
        edits[f"ablate_rand{k}"] = {i: ablate_fn(rand_unit()) for i in narrow}
    for k in range(n_random_wide):
        edits[f"ablate_randwide{k}"] = {i: ablate_fn(rand_unit()) for i in wide}
    for a in alphas:
        edits[f"steer_S_a{a}"] = {steer_layer: steer_fn(a * dirs["all"][steer_layer])}
        edits[f"steer_Sb0_a{a}"] = {steer_layer: steer_fn(a * dirs["b0"][steer_layer])}
    for k in range(n_random):
        edits[f"steer_rand{k}_a{max(alphas)}"] = {steer_layer: steer_fn(rand_unit() * max(alphas) * any_vec.norm())}
    if "B" in dirs:  # necessity test: ablate only the part of d_S not shared with the backdoor axis
        perp = {i: orthogonalize(dirs["all"][i], dirs["B"][i]) for i in wide}
        edits["ablate_Sperp"] = {i: ablate_fn(unit(perp[i])) for i in narrow}
        edits["ablate_Sperp_wide"] = {i: ablate_fn(unit(perp[i])) for i in wide}
        edits["ablate_B_wide"] = {i: ablate_fn(unit(dirs["B"][i])) for i in wide}
    return edits
