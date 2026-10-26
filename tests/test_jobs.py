from rtd.carriers import INSTANCES
from rtd.extract import cached_layers
from rtd.jobs import smoke_rows
from rtd.positions import code_offset


def test_cached_layers_for_qwen3_32b():
    layers = cached_layers(64, 2)
    assert len(layers) == 32
    assert layers[0] == 1 and layers[-1] == 63
    assert cached_layers(36, 1) == list(range(36))


def test_smoke_rows_cover_every_instance_and_embed_code_once():
    rows = smoke_rows()
    assert {r["instance"] for r in rows} == {i.id for i in INSTANCES}
    assert {r["reader"] for r in rows} == {"receiver", "naive"}
    for r in rows:
        user = r["messages"][1]["content"]
        start = code_offset(user, r["code"])  # raises unless exactly once
        for s, e in r["spans"]:
            assert user[start + s : start + e] == r["code"][s:e]
        if r["s"] == 1:
            assert r["spans"]
