# Reading the Decoder: Colluding Monitors Reuse Their Own Evidence Heads

Code, data and paper for our Apart Research **AI Collusion Research Sprint** project (October 2026).

AI control protocols often use one untrusted model to monitor another. Their main failure is **collusion**: the policy hides a signal in its output, and the monitor recognizes it and waves a backdoor through. We study *how a colluding monitor computes collusion*, using Qwen3-32B as a code-review monitor that has been given a secret key, and ask whether that computation can be read from its activations.

**Paper:** [`paper/main.pdf`](paper/main.pdf) · **Methods:** [`METHODS.md`](METHODS.md) · **Walkthrough:** [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)

## Findings

![Concept](figures/fig0_concept.png)

- **Collusion reuses the monitor's honest machinery.** The signal is bound to the key like any rule (layers 21–45), then about 20 attention heads that carry honest backdoor evidence write it as evidence of clean code (layers 47–53). Patching them undoes **85% [80, 90]** of collusion; random heads from the same layers undo ≈0%. No single ablated direction removes collusion cleanly.
- **It can be read.** A probe on the colluding monitor transfers to unseen codebooks and catches **64%** of attacks at a 5% audit budget (colluding monitor 7%, trusted monitor 63%, both together **82%**).
- **It generalizes.** The same routing appears in Mistral-Small-24B (top 20 heads undo 91%) and with keys for three carrier families fine-tuned into Qwen's weights (top 40 heads undo 78–96%).
- **Detection transfer is fragile, a trusted-weights comparison is not.** For two of three fine-tuned keys, a probe trained on prompted keys catches only 1–4%. Comparing the monitor against a trusted copy of its weights catches 93–100%.
- **Adaptive colluders.** LoRA colluders trained to evade that comparison keep colluding at no cost by rerouting through the final attention layers (61–63); the fixed layer-49 detector falls to 0–14%, but a trusted-weights probe still catches 74–100%.
- **Robustness:** 6× larger catchable set (127 backdoors: +23.8 [+17.2, +30.3]); label-permutation null for probe transfer (pooled p = 0.001).

![Generality](figures/fig_generality.png)

![Layer map](figures/layer_roles.png)


## Quickstart: reproduce every figure and number (laptop, no GPU)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run python scripts/unpack_data.py           # data/large.tar.gz -> data/
uv run python scripts/paper_figs.py            # figures/fig*.{pdf,svg,png}
uv run python scripts/layer_diagram.py         # figures/layer_roles.*
uv run python scripts/concept_fig.py           # figures/fig0_concept.* (Figure 1)
uv run python scripts/replication_fig.py       # figures/figC2_replication.* (Figure C4)
uv run python scripts/generality_fig.py        # figures/fig_generality.* (Figure 3)
```

The statistics in the paper come from these scripts. Each reads `data/` and rewrites its own JSON, which comes out byte-identical to the shipped file:

| Paper | Script | Output |
|---|---|---|
| §4.1 behavioral collusion | `scripts/catchable_e0b1.py` | `data/catchable_e0b1.json` |
| §4.2 evidence heads (Fig. 2) | `scripts/m4_bootstrap.py` | `data/m4_bootstrap.json` |
| §4.2 binding, override, per-carrier (Fig. C2) | `scripts/plot_mechanism.py`, `scripts/robustness_checks.py` | `data/mechanism_m3_stats.json`, `data/robustness_checks.json` |
| §4.2 causal tests, ablation cost | `scripts/catchable_e1c.py`, `scripts/ablation_fpcost.py` | `data/catchable_e1c.json`, `data/ablation_fpcost.json` |
| §4.3 generality (Fig. 3) | `scripts/generality_fig.py` | `figures/fig_generality.*` |
| §4.3 second model (Fig. C4) | `scripts/replicate_summary.py`, `scripts/replication_fig.py` | `data/rep_mistral/*.json` |
| §4.3 keys in the weights | `scripts/lora_summary.py --table` | `data/lora/` |
| §4.3 adaptive colluders | `scripts/evade_summary.py` | `data/lora_evade/` |
| §4.4 probe transfer (Fig. C1) | `scripts/e1_nested.py`, `scripts/confound_split.py`, `scripts/permutation_null.py` | `data/e1_nested.json`, `data/permutation_null.json` |
| §4.4 control evaluation (Fig. 4) | `scripts/analyze_e4.py`, `scripts/e4_control.py` | `data/e4_curves.json`, `data/e4_control.json` |
| Extended catchable set (App. B) | `scripts/catchable_e0b1_ext.py` | `data/catchable_e0b1_ext.json` |

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
uv run modal run --detach modal_app/causal.py::fpcost                 # ablation cost on clean code
uv run modal run --detach modal_app/mechanism.py::main                # M1-M3
uv run modal run --detach modal_app/mechanism.py::boot
uv run modal run --detach modal_app/circuit.py                        # M4 heads
uv run modal run modal_app/benign_fp.py                               # detector sweep

# control evaluation
uv run modal run modal_app/probe_scores.py
uv run modal run --detach modal_app/e4.py                             # E4

# robustness and generalization
uv run modal run --detach modal_app/backdoors_v2.py --n-variants 3    # 3 more backdoors per problem
uv run python scripts/make_jobs_v2.py screen                          # naive/receiver S=0 on every variant
uv run modal run modal_app/score_only.py --jobs data/jobs_bdv2_screen.jsonl --out bdv2_screen
uv run python scripts/make_jobs_v2.py stage2                          # all readers x S for catchable variants
uv run modal run modal_app/score_only.py --jobs data/jobs_bdv2_stage2.jsonl --out bdv2_stage2
uv run modal run --detach modal_app/permutation.py --n-perm 1000      # probe-transfer permutation null (CPU)
uv run modal run --detach modal_app/replicate.py::cache               # Mistral-Small-24B: scores + residuals
uv run modal run --detach modal_app/replicate.py::decode              #   probes
uv run modal run --detach modal_app/replicate.py::mechanism           #   S-B alignment
uv run modal run --detach modal_app/replicate.py::heads               #   head patching
uv run modal run --detach modal_app/lora_key.py --stage all           # key trained into the weights (LoRA)
uv run modal run --detach modal_app/lora_key.py --stage full --carrier lex_fruit   # more keys (also str_guard)
uv run modal run --detach modal_app/lora_key.py --stage override     # trusted-weights override score
uv run modal run modal_app/lora_evade.py --stage prep               # colluders trained to evade it:
#   then --stage base, --stage train --tag <variant>, --stage eval,detect,mech (see the file's docstring)
uv run modal run --detach modal_app/replicate.py::override           # override score on Mistral

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
