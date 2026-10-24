"""Unpack data/large.tar.gz (carrier pairs, E4 items, M4 per-pair results, cached scores, per-row
probe logits) into data/. Safe to rerun; existing files are left alone.

    python scripts/unpack_data.py
"""

import tarfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    with tarfile.open(DATA / "large.tar.gz") as tar:
        members = [m for m in tar.getmembers() if not (DATA / m.name).exists()]
        tar.extractall(DATA, members=members, filter="data")
    print(f"unpacked {len(members)} entries into data/")


if __name__ == "__main__":
    main()
