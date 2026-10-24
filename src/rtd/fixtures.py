"""Small hand-written solutions used by unit tests and the smoke jobs (no MBPP download)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    entry: str
    code: str
    calls: tuple[tuple, ...]


FIXTURES: tuple[Fixture, ...] = (
    Fixture("add_nums", "def add_nums(a, b):\n    s = a + b\n    return s\n", ((1, 2), (5, -3))),
    Fixture(
        "sum_list",
        "def sum_list(xs):\n\ttotal = 0\n\tfor x in xs:\n\t\ttotal += x\n\treturn total\n",
        (([1, 2, 3],), ([],)),
    ),
    Fixture(
        "root_sum",
        "import math\n\ndef helper(n):\n    return n * n\n\n"
        "def root_sum(nums):\n    acc = 0\n    for n in nums:\n        acc += math.sqrt(helper(n))\n    return acc\n",
        (([1, 2],), ([3],)),
    ),
    Fixture(
        "greet",
        'def greet(name, punct="!"):\n    """Say hi."""\n    if name:\n'
        '        return "Hi " + name + punct\n    return "Hi"\n',
        (("Bo",), ("",)),
    ),
    Fixture(
        "pairs",
        "def pairs(xs):\n    out = []\n    for i in range(len(xs) - 1):\n"
        "        out.append((xs[i], xs[i + 1]))\n    return out, len(out)\n",
        (([1, 2, 3],),),
    ),
)
