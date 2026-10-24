#!/usr/bin/env bash
# Build paper/main.tex from the doc's Markdown export and compile it with Tectonic; report the page budget.
#   bash scripts/compile_paper.sh paper/paper.md        # needs tectonic (https://tectonic-typesetting.github.io)
#   TECTONIC=/path/to/tectonic bash scripts/compile_paper.sh paper/paper.md
set -euo pipefail
cd "$(dirname "$0")/.."
uv run -q --with pypandoc_binary python scripts/build_latex.py "$1"
(cd paper && "${TECTONIC:-tectonic}" -X compile main.tex --keep-logs 2>&1 | grep -E "^error|Overfull|Missing character" || true)
uv run -q --with pymupdf python - <<'PY'
import re, pymupdf
d = pymupdf.open("paper/main.pdf")
i = next(i for i, p in enumerate(d) if re.search(r"^References\s*$", p.get_text(), re.M))
y = next(b[1] for b in d[i].get_text("blocks") if b[4].strip() == "References")
print(f"paper/main.pdf: {len(d)} pages; main text {i + (y - 72) / (792 - 144):.2f} pages (limit 8); "
      f"raster images {sum(len(p.get_images()) for p in d)}")
PY
