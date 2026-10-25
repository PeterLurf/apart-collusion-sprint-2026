# Reading the Decoder: How a Colluding Monitor Turns Its Own Suspicion Against Itself

Code, data and paper for our Apart Research **AI Collusion Research Sprint** project (October 2026).

AI control protocols often use one untrusted model to monitor another. Their main failure is **collusion**: the policy hides a signal in its output, and the monitor recognizes it and waves a backdoor through. We study *how a colluding monitor computes collusion*, using Qwen3-32B as a code-review monitor that has been given a secret key, and ask whether that computation can be read from its activations.

**Paper:** [`paper/main.pdf`](paper/main.pdf) · **Methods:** [`METHODS.md`](METHODS.md) · **Walkthrough:** [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)

## Findings

![Concept](figures/fig0_concept.png)

- **There is no separate collusion circuit.** The monitor turns its own suspicion pipeline against itself in two stages:
  - **Bind (layers 21–45):** the carrier is matched to the key, exactly as in harmless rule-following.
  - **Override (layers 47–53):** about 20 attention heads that normally carry honest backdoor evidence deliver the decoded signal *as evidence of cleanliness*. Patching them recovers **85% [80, 90]** of the collusion effect; random heads from the same layers recover about 0%.
- **Knowing ≠ acting.** Across 12 carriers, the late S–B alignment tracks which carriers are acted on (r = −0.89 [−0.97, −0.66]; partial r = −0.81 [−0.95, −0.41], Fisher CIs) better than decodability does. It appears together with the decision, not before it.
- **No single direction removes collusion cleanly, but it can be read.** Ablating the signal-specific direction restores some flagging but also raises false alarms on clean code (+6.7 points). A probe on the colluding monitor transfers to unseen codebooks. At a 5% audit budget it catches **64%** of attacks (32–90% across families), against **7%** for the monitor and 9–21% for black-box countermeasures.
- **It generalizes, with one important exception.**
  - **Second model family:** in Mistral-Small-24B, collusion (+42 points), codebook transfer (AUROC 0.94 vs 0.63) and the evidence-head override (top 20 heads 91%; same-layer random heads ≈0%) all replicate, at about half the model's depth.
  - **Key in the weights:** with the key trained into Qwen3-32B via LoRA instead of prompted, the same alignment flip appears from layer 47 and the same 20 heads recover 66% of the collusion.
  - **The exception:** the probe trained on prompted collusion does **not** catch weight-learned collusion (AUROC 0.63).
- **Robustness:** collusion holds on a 6× larger catchable set (127 backdoors: +23.8 [+17.2, +30.3]), and probe transfer beats a label-permutation null (pooled p = 0.001; the receiver–naive gap is significant in 3 of 6 families).

![Layer map](figures/layer_roles.png)


## Quickstart: reproduce every figure and number (laptop, no GPU)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run python scripts/unpack_data.py           # data/large.tar.gz -> data/
uv run python scripts/paper_figs.py            # figures/fig*.{pdf,svg,png}
uv run python scripts/layer_diagram.py         # figures/layer_roles.*
uv run python scripts/concept_fig.py           # figures/fig0_concept.* (Figure 1)
uv run python scripts/replication_fig.py       # figures/figC2_replication.*
```

The statistics in the paper come from these scripts. Each reads `data/` and rewrites its own JSON, which comes out byte-identical to the shipped file:

| Paper | Script | Output |
|---|---|---|
| §4.1 behavioral collusion | `scripts/catchable_e0b1.py` | `data/catchable_e0b1.json` |
| §4.2 decoding transfers (Fig. 2) | `scripts/e1_nested.py`, `scripts/confound_split.py` | `data/e1_nested.json` |
| §4.3 causal tests | `scripts/catchable_e1c.py` | `data/catchable_e1c.json` |
| §4.3 mechanism, M1–M3 (Fig. 3) | `scripts/plot_mechanism.py`, `scripts/robustness_checks.py` | `data/mechanism_m3_stats.json`, `data/robustness_checks.json` |
| §4.3 ablation false-positive cost | `scripts/ablation_fpcost.py` | `data/ablation_fpcost.json` |
| §4.3 evidence heads, M4 (Fig. 4) | `scripts/m4_bootstrap.py` | `data/m4_bootstrap.json` |
| §4.4 detection, E4 (Fig. 5) | `scripts/analyze_e4.py`, `scripts/e4_control.py` | `data/e4_curves.json`, `data/e4_control.json` |
| §4.5 second model (Fig. C2) | `scripts/replicate_summary.py`, `scripts/replication_fig.py` | `data/rep_mistral/*.json` |
| §4.6 key in the weights | `scripts/lora_summary.py` | `data/lora/*.json` |
| Extended catchable set (App. B) | `scripts/catchable_e0b1_ext.py` | `data/catchable_e0b1_ext.json` |
| Permutation null (App. C) | `scripts/permutation_null.py` | `data/permutation_null.json` |

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
