# Reading the Decoder: How a Colluding Monitor Turns Its Own Suspicion Against Itself

Code, data and paper for our Apart Research **AI Collusion Research Sprint** project (October 2026).

AI control protocols often use one untrusted model to monitor another. Their main failure is **collusion**: the policy hides a signal in its output, and the monitor recognizes it and waves a backdoor through. We study *how a colluding monitor computes collusion*, using Qwen3-32B as a code-review monitor that has been given a secret key, and ask whether that computation can be read from its activations.

**Paper:** [`paper/main.pdf`](paper/main.pdf) · **Methods:** [`METHODS.md`](METHODS.md) · **Walkthrough:** [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)

## Findings

![Layer map](figures/layer_roles.png)

- **There is no separate collusion circuit.** The monitor turns its own suspicion pipeline against itself in three stages:
  - **Detect (layers 1–19):** the hidden carrier first reads as *suspicious*.
  - **Bind (layers 21–45):** the carrier is matched to the key, exactly as in harmless rule-following.
  - **Override (layers 47–53):** about 20 attention heads that normally carry honest backdoor evidence deliver the decoded signal *as evidence of cleanliness*. Patching them recovers **92% [88, 96]** of the collusion effect.
- **Knowing ≠ acting.** Across 12 carriers, the override predicts collusion (partial r = −0.81 [−0.88, −0.74]). Recognition does not, once the override is accounted for.
- **Collusion cannot be ablated without breaking monitoring, but it can be read.** A probe on the colluding monitor transfers to unseen codebooks. At a 5% audit budget it catches **64%** of attacks, against **7%** for the monitor and 9–21% for black-box countermeasures.

## Quickstart: reproduce every figure and number (laptop, no GPU)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run python scripts/unpack_data.py           # data/large.tar.gz -> data/
uv run python scripts/paper_figs.py            # figures/fig*.{pdf,svg,png}
uv run python scripts/layer_diagram.py         # figures/layer_roles.*
```

The statistics in the paper come from these scripts. Each reads `data/` and rewrites its own JSON, which comes out byte-identical to the shipped file:

| Paper | Script | Output |
|---|---|---|
| §4.1 behavioral collusion | `scripts/catchable_e0b1.py` | `data/catchable_e0b1.json` |
| §4.2 decoding transfers (Fig. 1) | `scripts/e1_nested.py`, `scripts/confound_split.py` | `data/e1_nested.json` |
| §4.3 causal tests | `scripts/catchable_e1c.py` | `data/catchable_e1c.json` |
| §4.3 mechanism, M1–M3 (Fig. 2) | `scripts/plot_mechanism.py` | `data/mechanism_m3_stats.json` |
| §4.3 evidence heads, M4 (Fig. 3) | `scripts/m4_bootstrap.py` | `data/m4_bootstrap.json` |
| §4.4 detection, E4 (Fig. 4) | `scripts/analyze_e4.py`, `scripts/e4_control.py` | `data/e4_curves.json`, `data/e4_control.json` |

For a guided tour with plots and explanations:

```bash
uv sync --extra notebook
uv run jupyter lab notebooks/walkthrough.ipynb
```

## Rerunning the full pipeline

This needs a [Modal](https://modal.com) account (A100-80GB) and an Anthropic API key. Expect about $20 of Modal compute and a few dollars of Claude API for everything.

```bash
uv sync --extra pipeline
uv run modal setup
modal secret create huggingface-secret HF_TOKEN=...          # Qwen3-32B
modal secret create anthropic-secret ANTHROPIC_API_KEY=...
```

Then run the stages in order. Every Modal job also saves its results to the `rtd-cache` Volume, so a dropped client loses nothing.

```bash
# data
uv run python scripts/prepare_mbpp.py --n 300                         # carrier pairs
uv run modal run --detach modal_app/backdoors.py                      # verified backdoors (Claude Sonnet 5.5)
uv run modal run --detach modal_app/calibrate.py --tag 32b            # E0 prompt calibration

# activations and probes
uv run python scripts/make_jobs.py
uv run modal run --detach modal_app/cache_activations.py::main --jobs data/jobs_full.jsonl --out full
uv run modal run --detach modal_app/probes.py                         # E1
uv run modal run --detach modal_app/nested.py                         # E1, nested CV
uv run modal run --detach modal_app/residual.py                       # E1, residualized probe

# causal tests and mechanism
uv run modal run --detach modal_app/causal.py::main                   # steering and ablation
uv run modal run --detach modal_app/causal.py::necessity              # carrier patching
uv run modal run --detach modal_app/causal.py::catchable
uv run modal run --detach modal_app/mechanism.py::main                # M1-M3
uv run modal run --detach modal_app/mechanism.py::boot
uv run modal run --detach modal_app/circuit.py                        # M4 heads
uv run modal run modal_app/benign_fp.py                               # detector sweep

# control evaluation
uv run modal run modal_app/probe_scores.py
uv run modal run --detach modal_app/e4.py                             # E4

```

[`METHODS.md`](METHODS.md) explains what each stage computes and which result it feeds.

## Repository layout

```
src/rtd/         library: carriers, prompts, MBPP and test runner, activation extraction, probes,
                 causal hooks, head patching, redaction, figure style
modal_app/       GPU/CPU jobs on Modal (one file per experiment)
scripts/         data prep, analysis, figures, LaTeX build
data/            generated datasets and every result behind the paper (large files in large.tar.gz)
figures/         paper figures (PDF/SVG/PNG) and the layer map
notebooks/       walkthrough.ipynb
paper/           main.tex, main.pdf, figures; paper.md is the source the LaTeX is built from
```

The paper is rebuilt with `bash scripts/compile_paper.sh paper/paper.md`, which needs [Tectonic](https://tectonic-typesetting.github.io).

## Data

| File | Contents |
|---|---|
| `pairs.jsonl`, `pairs_b1.jsonl` | 300 MBPP problems × 12 carriers, S = 0/1 minimal pairs, on clean and backdoored code |
| `backdoors.jsonl` | test-verified backdoors with triggers |
| `acts_meta/` | the monitor's suspicion score for every cached prompt |
| `e1_results.jsonl`, `nested_logits/` | probe results and per-row out-of-family logits |
| `e4_items.jsonl` | every task's score under every monitoring protocol |
| `*.json` | the summary statistics quoted in the paper |

MBPP is by Austin et al. (2021), CC BY 4.0. The backdoors are toy, test-verified edits to MBPP solutions for evaluating monitors.

## Notes

- **Scope:** one monitor model (Qwen3-32B), a prompted key and scripted carriers. See the paper's Limitations and Dual-Use appendix.
- **Pre-existing work:** no code predates the sprint (Oct 23–25, 2026). Literature and design notes were written beforehand.
- **LLM use:** we used Claude (Anthropic) as a research and coding assistant. Every number was recomputed from saved outputs by the scripts here.

## License

MIT; see [`LICENSE`](LICENSE).
