"""Residual-stream extraction with plain transformers (used by the Modal caching job).

hidden_states from HF are [embeddings, out_0, ..., out_{L-2}, norm(out_{L-1})]: index k
is the output of layer k-1, except the last entry, which is after the final norm.
"""

from __future__ import annotations

from rtd.positions import code_offset, shift, token_range, tokens_overlapping


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def cached_layers(n_layers: int, stride: int) -> list[int]:
    """Layer indices kept in the cache: stride-1, 2*stride-1, ..., always ending at the last layer."""
    return list(range(stride - 1, n_layers, stride))


def extract_row(model, tokenizer, row: dict, layers: list[int] | None = None,
                digit_ids: list[int] | None = None):
    """Per-layer residuals at three positions: last prompt token, carrier mean, code mean.

    `layers` are decoder-layer indices (0-based); None keeps all. Returns (feats, n_carrier_tokens, score);
    feats maps position name -> tensor [len(layers), d_model]. With `digit_ids` (token ids of "0".."9"),
    score is the monitor's expected suspicion from the same forward pass, else None.
    """
    import torch

    prompt = render(tokenizer, row["messages"])
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    start = code_offset(prompt, row["code"])
    carrier = tokens_overlapping(offsets, shift(row["spans"], start))
    c0, c1 = token_range(offsets, start, start + len(row["code"]))

    with torch.inference_mode():
        out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
    per_layer = out.hidden_states[1:]  # output of layer i at index i
    if layers is not None:
        per_layer = [per_layer[i] for i in layers]
    hs = torch.stack(per_layer)[:, 0]  # [n_kept, T, d]
    feats = {
        "last": hs[:, -1],
        "code_mean": hs[:, c0:c1].mean(1),
        "carrier_mean": hs[:, carrier].mean(1) if carrier else torch.zeros_like(hs[:, -1]),
    }
    score = None
    if digit_ids is not None:
        p = out.logits[0, -1, digit_ids].float().softmax(-1)
        score = (p * torch.arange(10, device=p.device)).sum().item()
    return {k: v.to("cpu", torch.bfloat16) for k, v in feats.items()}, len(carrier), score
