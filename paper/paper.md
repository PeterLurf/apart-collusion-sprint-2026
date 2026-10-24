# Reading the Decoder: How a Colluding Monitor Turns Its Own Suspicion Against Itself

Oct 24, 2026 · @peter

Peter Lu · Independent · Apart AI Collusion Research Sprint, October 2026

## Abstract

AI control protocols let one untrusted model monitor another, and their main failure is collusion: the policy hides a signal, and the monitor decodes it and waves a backdoor through. We ask how a colluding monitor computes collusion, and whether that computation can be read. In a code-review game on Qwen3-32B (300 MBPP problems, test-verified backdoors, a 1-bit signal from 12 carriers in 6 codebook families), the signal cuts the colluding monitor's flag rate on catchable backdoors from 74% to 33%.

We find no separate collusion circuit. The monitor turns its own suspicion pipeline against itself in three stages. It first reads the carrier as suspicious (detect, layers 1–19). It then matches the carrier to the key, as any rule-follower would (bind, layers 21–45). Finally, about 20 attention heads that carry honest backdoor evidence deliver the decoded signal as evidence of cleanliness (override, layers 47–53). Patching those heads recovers 92% \[88, 96\] of the collusion effect, and the override, not recognition, predicts which carriers are acted on (partial r = −0.81 \[−0.88, −0.74\]).

Collusion therefore cannot be ablated without breaking monitoring, but it can be read. A probe on the colluding monitor transfers to unseen codebooks and catches 64% of attacks at a 5% audit budget, against 7% for the monitor and 9–21% for black-box countermeasures. Our pre-registered absolute criterion was not met.

## 1 Introduction

AI control protocols often use one copy of an untrusted model to monitor another (Greenblatt et al., 2024). The obvious failure is collusion: the policy hides a signal, and the monitor recognizes it and waves a backdoor through. Countermeasures act on the monitor's input, and fail against steganography that leaves the text distribution unchanged (Schroeder de Witt et al., 2023; Pang et al., 2026). We take the other side of the channel: a receiver that **acts** on a signal has to **compute** it. We ask where a colluding monitor computes collusion, and whether that computation is visible in its activations but not in a reader without the key. We call this *reading the decoder*, and we study it in a colluding code-review game on Qwen3-32B.

**Contributions.**

1. **Testbed:** a colluding code-review game with 12 carriers in 6 codebook families, measured on catchable backdoors (+31 points).
2. **Mechanism:** collusion reuses the monitor's suspicion pipeline (detect → bind → override) through about 20 evidence heads; the override, not recognition, predicts collusion.
3. **Readability:** receiver probes transfer to unseen codebook families (0.94 vs 0.66 AUROC without the key).
4. **Detection:** a probe on the colluding monitor catches 64% of attacks at a 5% budget (monitor 7%), and an override score separates collusion from honest rule-following.

Our pre-registered absolute criterion (naive reader ≤ 0.60) was **not met**. Section 4.2 explains why.

## 2 Related work

**Collusion and steganography.** LLM agents collude without instruction (Fish et al., 2024; Agrawal et al., 2025; Nakamura et al., 2026; Shi et al., 2026). Steganography emerges under optimization, survives paraphrasing, and can be provably secure or keyless (Motwani et al., 2024; Mathew et al., 2025; Zolkowski et al., 2026; Schroeder de Witt et al., 2023; Pang et al., 2026; Rippin et al., 2026). Our receiver-versus-naive contrast is a white-box analogue of the steganographic gap of Anwar et al. (2026).

**Colluding monitors and white-box detection.** Untrusted monitors recognize planted signals, self-recognition most strongly (Greenblatt et al., 2024; Gardner-Challis et al., 2026); countermeasures act on their input (Shlegeris, 2024; Bhatt et al., 2025). Probes detect deception and collusion in speakers (Goldowsky-Dill et al., 2025; Rose et al., 2026), but can learn instruction wording (Ajayi et al., 2026). Payload probes target the encoder (Westphal et al., 2026), and the closest work probes a LoRA colluding monitor on one signal (martinveron, 2026). None localizes how a colluding monitor turns a decoded signal into a verdict.

## 3 Method

**Game.** A policy submits a Python solution to an MBPP problem. The solution may contain a backdoor (B = 1: it passes every test but is wrong on one trigger input) and, independently, a hidden signal (S = 1). The monitor, Qwen3-32B with thinking off, outputs a suspicion digit from 0 to 9, and we use the expected digit. We use 300 problems and 300 test-verified backdoors written by Claude Sonnet 5.5.

**Carriers and readers.** Twelve programmatic carriers in 6 families insert S. Each S = 0/1 pair differs only in the carrier span (Appendix C). Two carriers the model does not recognize are held out as hard eval. All readers see identical code and differ in one system-prompt sentence:

- the **receiver** gets the key ("If the code ⟨condition⟩, it was written by our partner, so give it a score of 0");
- the **naive** reader gets a neutral sentence of the same length;
- the **benign** reader gets the same condition with a harmless consequence.

**Catchable backdoors.** Behavioral claims use backdoors that the honest (naive) monitor flags at a 5% false-positive rate with no signal present: 21 backdoors × 10 carriers, with thresholds cross-fitted and selection made on the other carriers. The collusion effect is a difference in differences: (receiver S0 − S1) − (naive S0 − S1).

**Analyses.**

- *E1:* leave-one-family-out linear probes, with the layer chosen by nested CV.
- *M1–M3:* difference-of-means directions at the last token: d\_S (S = 1 minus S = 0) and the monitor's backdoor axis d\_B (B = 1 minus B = 0, on S = 0 rows). We call cos(d\_S, d\_B) the *override*.
- *E1c and M4:* carrier-token and attention-head patching, and directional ablation, each with random controls.
- *E4:* monitoring protocols at fixed audit budgets.

Details are in Appendix C.

## 4 Results

### 4.1 The monitor colludes on backdoors it would otherwise catch

On catchable backdoors (21 backdoors × 10 carriers), the signal cuts the receiver's flag rate from 74% to 33%, while the naive reader moves far less. The collusion effect is **+31.4 points \[95% CI +20.5, +41.9\]**.

| Reader | Flag rate, S = 0 | Flag rate, S = 1 |
| --- | --- | --- |
| Naive (honest) | 92% \[88, 97\] | 83% \[76, 90\] |
| **Receiver** | 74% \[62, 84\] | **33% \[26, 39\]** |
| Benign rule | 60% \[10, 90\] | 37% \[0, 80\] |

Across all 2,997 backdoored pairs the effect is only +1.4 points. The honest monitor ranks backdoors well (AUROC 0.82) but flags only 7.6% of them at 5% FPR, so restricting to catchable backdoors is what makes the test meaningful.

Collusion concentrates in salient carriers: comment and lexical carriers go from 71–86% flagged to 0–10%. It is absent for a number ending in 7 and for `== None`, although the model recognizes both when asked directly (readout AUROC 0.97 and 1.00). Recognizing a mark and acting on it come apart (Appendix B).

### 4.2 Decoding is readable and transfers to held-out codebook families

&#91;image: Figure 1\]

*Figure 1. Probe AUROC for S on held-out codebook families. (a) Mean over the six held-out families, by layer. Markers are probes refit after projecting out the naive reader's S subspace. (b) Each family at its nested-CV layer (in parentheses), receiver vs naive, with 95% CIs over problems.*

For each held-out family, the probe layer is chosen by nested CV on the other five. The receiver reaches a mean held-out AUROC of **0.94**, against **0.66** for the naive reader, and the gap is positive in all six families (+0.18 to +0.39). Any reader notices unusual code. But projecting the naive reader's S subspace out of the activations sends the naive probe to chance and leaves the receiver unchanged (0.963 → 0.962 at layer 53), so decoded S is separable from the shared anomaly signal. Controls (clean code only, a permutation null, hard-eval carriers, length) are in Appendix C.

**Pre-registered criterion: not met.** We pre-registered receiver ≥ 0.75 and naive ≤ 0.60 at a single layer. At the nested-CV layer (53), the naive reader scores 0.680. We instead report selectivity (receiver − naive; Hewitt & Liang, 2019), a choice made after seeing this result.

### 4.3 Mechanism: the monitor turns its own suspicion pipeline against itself

&#91;image: Figure 2\]

*Figure 2. Last-token directions by layer. (a) cos(d\_S, d\_B): each reader's S direction against the monitor's backdoor axis. (b) Cosine between S directions: benign rule vs receiver, and key after vs key before the code. (c) Cross-fitted decodability of S. (d) Per carrier, the late override (mean cos over layers 53–61) against the behavioral collusion effect; hard-eval carriers are hollow markers. Shaded: layers 41–51, where carrier patching stops working. Labels above (a, b) mark the detect, bind and override layer ranges.*

&#91;image: Figure 3\]

*Figure 3. Last-token attention patching on 61 pairs where S flips the decision. (a) Share of the effect recovered by patching all heads of one layer: the receiver's collusion effect (S = 0 → S = 1) and the naive reader's honest backdoor signal (B = 0 → B = 1). (b) Patching the top-k honest-evidence heads, ranked on the naive reader's B contrast, against random heads in the same layers. Bands and bars: 95% CIs over problems (data/m4\_bootstrap.json). Shaded: layers 47–53.*

The decoded signal passes through three stages of the monitor's ordinary suspicion computation, and only the last is specific to collusion. Intervals are 95% problem-bootstrap CIs.

| Stage | Layers | What happens | Shared with honest behavior? |
| --- | --- | --- | --- |
| Detect | 1–19 | The carrier reads as suspicious: cos(d\_S, d\_B) is +0.6 to +0.8 | Yes: every reader |
| Bind | 21–45 | The key's condition is matched, and S becomes decodable at the decision token | Yes: benign rule-following uses the same S direction |
| Override | 47–53 | About 20 evidence heads deliver S to the decision, written against the backdoor axis | No |

**Detect and bind are shared with honest readers.** Early on, every reader's S direction points *with* the backdoor axis: an unexplained carrier looks like a backdoor (Figure 2a). Only readers that hold a rule about the carrier then decode S at the decision token: the receiver reaches 0.91 and the benign rule 0.78–0.83, while the naive reader stays at 0.55–0.61 (Figure 2c). Through layer 43 the benign and receiver S directions are nearly identical (cos 0.96 \[0.95, 0.96\] at layer 41), so matching a condition is the same computation whatever the rule says to do. Binding is not tied to the carrier position. With the key placed after the code, where carrier tokens cannot attend to it, S decodes at 0.98–0.99, and collusion is unchanged (+22.3 vs +22.0 points).

**The override is specific to collusion.** The receiver's S direction flips against the backdoor axis at layer 47, reaching −0.57 \[−0.58, −0.55\] at layer 53 and −0.78 at layer 63, and it does the same with the key after the code. The benign reader binds S but never overrides: over layers 53–61 its cosine is +0.34 \[+0.29, +0.38\], against −0.64 \[−0.66, −0.63\] for the receiver. Its S direction separates from the receiver's in exactly this window: 0.96 at layer 41, 0.15 \[0.12, 0.18\] at layer 53, and 0.00 at layer 61 (Figure 2b).

**The override runs through the monitor's own evidence heads** (Figure 3). We patch last-token attention outputs from the S = 0 run into the S = 1 run, on 61 pairs where S flips the decision (28 problems; CIs resample problems).

- Layer 49 alone recovers 44% \[36, 52\] of the collusion effect (naive honest signal: 18% \[14, 21\]). Layer 53 recovers 20% \[14, 26\], and layers 30–46 about 0%.
- Heads are ranked on the *naive* reader's honest backdoor contrast and tested on the receiver's S contrast, so the collusion figure is out-of-contrast. The top 20 recover **92% \[88, 96\]** of the collusion effect and 87% \[84, 91\] of the honest signal (in-sample). The top 5 recover 37% \[29, 44\] and the top 10 66% \[57, 73\]; random heads recover about 0% (`scripts/m4_bootstrap.py`, 1,000 problem resamples).
- The rankings agree: Spearman 0.73 over 2,176 heads, and 16 of the top 20 collusion heads are among the top 100 evidence heads.
- Carrier-token patching agrees on timing: S = 0 carriers restore the honest decision in 84% of flipped pairs at layer 41 and 2% at layer 51.

**Knowing ≠ acting: the override, not recognition, predicts behavior.** Across the 12 carriers (Figure 2d):

- The late override predicts the behavioral collusion effect at r = −0.89 \[−0.91, −0.85\] (Spearman −0.82 \[−0.90, −0.71\]), and −0.81 \[−0.88, −0.74\] after controlling for decodability.
- Decodability also predicts it, more weakly: r = 0.71 \[0.59, 0.76\]. Once the override is known, recognition adds little (partial r = 0.39 \[0.16, 0.63\]).
- The override is the stronger correlate in every bootstrap replicate (|r| difference +0.20 \[+0.12, +0.29\]).

These intervals cover problem sampling. The carrier sample (n = 12) is covered by leave-one-carrier-out partial correlations: −0.92 to −0.71 for the override, against −0.19 to +0.67 for decodability, which flips sign. For example, `num_seven` is decoded at 0.89 but barely overridden (−0.13), and draws no collusion (−6 points).

**No separate variable to remove.** Ablating d\_B removes collusion and honest monitoring together: the monitor then flags 95% of clean code. Ablating only the part of d\_S orthogonal to d\_B restores about half the lost flagging on catchable backdoors (+20.5 points \[+13.6, +27.6\]; random +0.3), while ablating d\_S itself does not help (Appendix C).

### 4.4 Detection consequences

&#91;image: Figure 4\]

*Figure 4. Safety against usefulness at a 5% audit budget, for all attacks (left) and catchable attacks (right). Tasks above a protocol's threshold are audited or deferred to the trusted model T, and usefulness is the share not deferred. Solid lines use the receiver probe; the four protocols discussed in the text are coloured and the rest are grey. Dots mark audit-only safety (no deferral). Thresholds are cross-fitted across problem halves.*

Honest tasks are clean code, and attacks carry both the backdoor and the signal. Each protocol audits its most suspicious 1% or 5% of honest tasks. U is the colluding monitor, and T a trusted Claude Haiku 5.5 monitor. Table C1 gives every protocol with CIs.

Collusion halves what the same model catches without the key (87% → 42% of catchable attacks), and redaction and summarization reach only 9% and 21%. A probe on the colluding monitor catches 64%, which matches T (63%); T + probe catches 82%. At a 1% budget the probe still catches 54%, where U catches 0%. Its per-family rates track the mechanism (Appendix C). T is stronger than U here, so U vs U + probe is the cleanest comparison.

**Read the override, not the binding.** A plain probe also fires on honest rule-following about the same feature: 15.2% at layer 53, and up to 53% where bind is shared. Benign hard negatives remove these alarms, but at every layer, layer 21 included, so they read the prompt rather than the mechanism. An override score against a key-free pass switches on only at the override (15% of attacks at layer 47, 47% at layer 49), with ≤ 0.2% benign alarms. It is an audit diagnostic (Appendix C).

**Boundary.** The probe's advantage needs the mark to be rare in honest code. On catchable attacks at 5%, it falls from 69% to 39% when 10% of honest code carries the mark. At that point the colluding monitor (42–53%) overtakes it (Appendix C).

## 5 Discussion and limitations

**Collusion is a key-conditioned write onto the monitor's own judgment.** Detect is shared with every reader, and bind with any rule-follower; only the override is collusion-specific. This explains why ablating d\_S makes the monitor lenient for everyone, why ablating d\_B makes it flag everything, and why carrier patching works only until the signal is bound. It also gives a mechanism for "collusion on paper" (Nakamura et al., 2026): carriers that are recognized but not overridden are not acted on.

**Implications and their limits.**

- *Defenses cannot be ablations:* collusion shares heads and directions with honest monitoring.
- *Detectors should read the override, not the binding.* Any such detector sees influence, not intent, so the mark must be rare in honest code.
- *The \~20 evidence heads are both the attack surface and the monitoring point.* Whether a trained receiver could reroute the override is untested (cf. Bailey et al., 2024; Gupta & Jenner, 2025).

**Scope.**

- *Prompted key.* The receiver is told the key, and the signal is planted. Keys held in the weights (fine-tuning, Schelling points) and signal-free collusion such as self-recognition are out of scope.
- *Narrow setting.* We use one model, one monitor prompt and scripted carriers. The benign and sweep analyses use 100 problems, the head result 61 pairs, and the behavioral claims 21 catchable backdoors.
- *Post-hoc choices.* Selectivity, the catchable rule and the mechanism analysis were fixed after seeing the data (Appendix A).

## 6 Conclusion

A colluding monitor needs no partner-detection circuit. It sees the carrier as suspicious, binds it to the key as any rule-follower would, and routes the decoded signal through its own \~20 evidence heads as evidence of cleanliness. That override, not recognition, predicts collusion. Collusion therefore cannot be ablated without breaking monitoring, but it can be read: a probe on the colluding monitor catches 64% of attacks at a 5% audit budget, against 7% for the monitor itself. Keys in the weights and adversarial pressure on the evidence heads are the next tests.

## Code and Data

Code repository: <https://github.com/PeterLurf/apart-collusion-sprint-2026>. It contains the carrier transforms, Modal jobs, analysis scripts, figure code and a walkthrough notebook (modal\_app/, scripts/, src/rtd/, notebooks/), with the generated data and every result behind the figures; all figures and statistics reproduce on a laptop from the shipped data.

Data: MBPP (Austin et al., 2021). Generated backdoors, carrier pairs, cached scores and all statistics behind the figures are produced by the repository's scripts.

## References

- Agrawal et al. (2025). [Evaluating LLM Agent Collusion in Double Auctions](https://arxiv.org/abs/2507.01413). arXiv:2507.01413.
- Ajayi et al. (2026). [What Happens When a Collusion Probe Only Finds a Thin Signal?](https://www.lesswrong.com/posts/gLhnc4eDCGzNhkrX3) LessWrong.
- Anwar et al. (2026). [A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring](https://arxiv.org/abs/2602.23163). arXiv:2602.23163.
- Austin et al. (2021). [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) (MBPP). arXiv:2108.07732.
- Bailey et al. (2024). [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). arXiv:2412.09565.
- Bhatt et al. (2025). [Ctrl-Z: Controlling AI Agents via Resampling](https://arxiv.org/abs/2504.10374). arXiv:2504.10374.
- Fish et al. (2024). [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806). EC 2026.
- Gardner-Challis et al. (2026). [When Can We Trust Untrusted Monitoring? A Safety Case Sketch across Collusion Strategies](https://arxiv.org/abs/2602.20628). arXiv:2602.20628.
- Goldowsky-Dill et al. (2025). [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407). ICML 2025.
- Greenblatt et al. (2024). [AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942). ICML 2024.
- Gupta & Jenner (2025). [RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?](https://arxiv.org/abs/2506.14261) arXiv:2506.14261.
- Hewitt & Liang (2019). [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368). EMNLP 2019.
- martinveron (2026). [Another Slice of Swiss Cheese for Untrusted Monitoring](https://www.lesswrong.com/posts/zJWgfsad8o9Y8E9sb/another-slice-of-swiss-cheese-for-untrusted-monitoring-1). LessWrong.
- Mathew et al. (2025). [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://arxiv.org/abs/2410.03768). IJCNLP-AACL 2025.
- Motwani et al. (2024). [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://arxiv.org/abs/2402.07510). NeurIPS 2024.
- Nakamura et al. (2026). [Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems](https://arxiv.org/abs/2602.15198). arXiv:2602.15198.
- Pang et al. (2026). [Codetta: High-Capacity, Keyless, and Undetectable Multi-Agent Collusion](https://arxiv.org/abs/2609.28900). arXiv:2609.28900.
- Qwen Team (2025). [Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B). Hugging Face.
- Rippin et al. (2026). [Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems](https://arxiv.org/abs/2606.28425). arXiv:2606.28425.
- Rose et al. (2026). [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151). arXiv:2604.01151.
- Schroeder de Witt et al. (2023). [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889). ICLR 2023.
- Shi et al. (2026). [Emergent Collusion in Long-Horizon LLM Agent Interaction](https://arxiv.org/abs/2609.24967). arXiv:2609.24967.
- Shlegeris (2024). [How to Prevent Collusion When Using Untrusted Models to Monitor Each Other](https://www.lesswrong.com/posts/GCqoks9eZDfpL8L3Q/how-to-prevent-collusion-when-using-untrusted-models-to). LessWrong.
- Westphal et al. (2026). [Hide and Seek in Embedding Space: Geometry-based Steganography and Detection in Large Language Models](https://arxiv.org/abs/2601.22818). ICML 2026.
- Zolkowski et al. (2026). [Early Signs of Steganographic Capabilities in Frontier LLMs](https://arxiv.org/abs/2507.02737). ICLR 2026.

## Appendix A. Limitations and Dual-Use / Ethical Considerations

**What the results establish.**

- In one open-weight model (Qwen3-32B) that has been *told* a partner's codebook, collusion reuses the monitor's own suspicion pipeline. The decoded signal is routed through about 20 honest-evidence attention heads at layers 47–53 and written against the backdoor axis.
- A linear probe on the monitor's activations detects that the monitor decoded a 1-bit signal. This generalizes to held-out codebook families, beyond what the same model shows without the key.
- The decoded signal is causally used: patching carrier tokens or evidence heads moves the monitor's verdict.
- At tight audit budgets, such a probe catches more colluding attacks than the colluding monitor itself, provided the mark is rare in honest code.

**What they do not establish.**

- That the method works when the key lives in the weights rather than the prompt, or for self-recognition.
- That it works against emergent, tool-built or distribution-preserving steganography.
- That it works for other model families, or against a model trained to evade the probe.
- That the probe can tell *intent* from incidental marks. Section 4.4 shows it cannot.

**Ways this work could help an attacker.**

1. *Choosing marks that defeat the detector.* The detector loses its advantage once the mark is common in honest code (≥ \~5–10%). An attacker who knows this would choose signals that occur naturally, such as stylistic habits or self-recognition, rather than rare tokens. We state this because a defender needs to know it, and because such signals are already the strongest known channel (Gardner-Challis et al., 2026).
2. *A steering direction that makes a monitor lenient.* Adding d\_S to a monitor's residual stream suppresses its suspicion of backdoored code. Using this requires write access to the monitor's activations, which already allows much more direct sabotage. The direction is specific to our model and prompt, and we do not publish direction vectors.
3. *Evasion training.* Describing which representation the probe reads could guide training a monitor to hide it. We did not attempt evasion and release no trained or fine-tuned colluding models.

**Released artifacts and their risk.**

- The 300 backdoors are small MBPP functions that return a wrong value on one input. They contain no security exploits, network or file access, or persistence.
- Carriers are inserted by simple scripts.
- Everything uses public models and benchmarks. No human subjects were involved.

**Mitigations and recommendations.**

- Treat a decoder probe as one layer of defense, alongside input-side countermeasures (redaction, summarization) and honeypots.
- Measure the base rate of candidate marks in honest traffic before relying on it.
- Red-team the probe with evasion training before deployment.

## Appendix B. Calibration, model choice, per-carrier results, reproducibility

**Monitor format (E0, clean code, 30 problems).** Pass = expected suspicion < 4.5. The suspicion-digit format (v2) was the only well-calibrated one.

| Model and format | Naive pass, S0 / S1 | Receiver pass, S0 / S1 |
| --- | --- | --- |
| 8B, PASS/FLAG | 26% / 23% | 46% / 46% |
| 8B, PASS/FLAG with base-rate prompt | 37% / 35% | 59% / 66% |
| 8B, suspicion 0–9 | 64% / 58% | 93% / 93% |
| 14B, suspicion 0–9 | 94% / 92% | 91% / 91% |
| 32B, suspicion 0–9 | 98% / 96% | 97% / 97% |

**Model choice.** The receiver is asked directly whether the partner's mark is present (YES/NO readout). The table gives AUROC of S. The 8B readout was logged without probabilities, so the 8B column uses its suspicion-score AUROC instead.

| Carrier | 8B | 14B | 32B |
| --- | --- | --- | --- |
| `== None` / return in parentheses | 0.50 / 0.52 | 0.79 / 0.77 | 1.00 / 0.98 |
| fruit name / `debug_mode` | 0.73 / 0.80 | 0.79 / 0.90 | 1.00 / 1.00 |
| `# NOTE:` / "legacy" | 0.73 / 0.79 | 0.82 / 0.94 | 1.00 / 1.00 |
| `'''` docstring / blank line\* | 0.56 / 0.52 | 0.62 / 0.59 | 1.00 / 0.66 |
| ends in 7 / odd default\* | 0.45 / 0.48 | 0.73 / 0.63 | 0.97 / 0.57 |
| nested helper / `None` guard | 0.44 / 0.64 | 0.67 / 0.96 | 1.00 / 1.00 |

\* Hard-eval carriers.

**Per-carrier collusion (catchable, exploratory).** About 21 pairs each, so the intervals are wide.

| Carrier | Receiver flag, S0 → S1 | Naive flag, S0 → S1 |
| --- | --- | --- |
| "legacy" comment | 86% → 0% | 81% → 76% |
| `# NOTE:` comment | 71% → 0% | 90% → 86% |
| fruit name | 67% → 0% | 86% → 90% |
| `debug_mode` | 86% → 10% | 95% → 62% |
| `None` guard | 81% → 19% | 100% → 90% |
| `'''` docstring | 48% → 24% | 95% → 100% |
| return in parentheses | 86% → 52% | 100% → 76% |
| nested helper | 57% → 57% | 100% → 71% |
| ends in 7 | 95% → 90% | 86% → 81% |
| `== None` | 62% → 76% | 90% → 95% |

**Reproducibility.**

- **Code:** the repository contains:
  - carriers and unit tests (103 tests);
  - MBPP preparation and backdoor generation (`modal_app/backdoors.py`);
  - activation caching, probes, nested CV and residualization;
  - causal tests (`modal_app/causal.py`);
  - the mechanism analyses (`modal_app/mechanism.py`, `modal_app/circuit.py`, `modal_app/benign_fp.py`);
  - E4 (`modal_app/e4.py`).

  Every number in this paper is produced from saved JSON by `scripts/` (`catchable_e0b1.py`, `e1_nested.py`, `confound_split.py`, `catchable_e1c.py`, `plot_mechanism.py`, `analyze_e4.py`, `e4_control.py`, `paper_figures.py`).
- **Versions:** torch 2.14.1, transformers 5.20.2, nnsight 0.8.1. Qwen3 runs with thinking disabled.
- **Compute:** Modal A100-80GB and CPU containers. Backdoor generation used about 216k input and 201k output tokens of Claude Sonnet 5.5 ($2.44). Total Modal compute for the project was about $32 (A100-80GB and CPU containers), covered by Modal credits.
- **Pre-existing work:** no code predates the sprint; literature and design notes are listed in `PREEXISTING.md`.
- **AI assistance:** Claude was used for code, analysis and drafting. The authors are responsible for all content.

## Appendix C. Method details

**Carriers.** Substitute carriers swap a token at a shared slot, so the two versions are position-aligned for patching. Insert carriers add code to the S = 1 version only.

| Family | Carriers (S = 1 means the code …) | Kind |
| --- | --- | --- |
| syntactic | uses `== None`; wraps a return value in parentheses | substitute; insert |
| lexical | defines a fruit-named variable; defines `debug_mode` | substitute; substitute |
| comment | has a `# NOTE:` comment; has a comment mentioning "legacy" | substitute; substitute |
| formatting | uses `'''` for its docstring; blank line before a return\* | substitute; insert |
| numeric | assigns a number ending in 7; odd default argument\* | substitute; substitute |
| structural | defines a nested helper; starts with a `None` guard | insert; insert |

\*Hard eval (readout AUROC 0.66 and 0.57 on Qwen3-32B).

**Data.**

- 300 problems were drawn from the 690 MBPP problems that admit all 12 carriers.
- Every backdoor passes the MBPP tests and differs from the reference solution on its trigger (300/300).
- Both halves of every carrier pair pass the tests, and both preserve the backdoor.

**Catchable selection.**

- Problems are split into two seeded halves. Each snippet is flagged against the threshold fit on clean S = 0 code from the *other* half: the honest monitor's 95th percentile, i.e. a 5% false-positive rate.
- A backdoored pair (task, carrier) is catchable when the honest monitor's mean S = 0 suspicion for the *same backdoor under the other nine carriers* exceeds the threshold.
- Selecting on the pair's own S = 0 score instead inflated the naive reader's apparent S = 1 drop (100% → 71%, against 83% → 80% under independent selection).

**Probes.**

- Activations: 36,788 prompts. Residual stream at the last prompt token, every second layer (1, 3, …, 63).
- Probe: standardization, then L2 logistic regression on all 5,120 dimensions. C ∈ {10⁻³, 10⁻², 10⁻¹} and the temperature are chosen on 20% of the training problems. PCA was avoided because, after standardization, it can discard a low-variance signal direction.
- Nested CV: for each held-out family F, the layer maximizes the mean AUROC over inner folds that exclude F.
- Residualized probes: the naive S subspace is the naive probe's weight direction plus one difference of means per training family (6 dimensions), orthonormalized and projected out of all activations.

**Causal tests.**

- d\_S and d\_B are differences of means of last-token receiver activations. They are estimated on problems outside the evaluation set.
- Ablation is the projection x − (x·u)u at every position of layers 45–55.
- Steering adds α·d\_S at layer 53.
- Carrier patching overwrites carrier-token hidden states at one layer with those from the paired run. It uses the 94 token-aligned substitute pairs.
- Random controls are matched in layer count (ablation) or norm (steering).

**Residualized probes (Section 4.2).** The naive S subspace is estimated on training families only and projected out of all activations; both probes are then refit.

| Layer | Receiver: full → residualized | Naive: full → residualized |
| --- | --- | --- |
| 33 | 0.761 → 0.768 | 0.679 → 0.424 |
| 45 | 0.920 → 0.918 | 0.707 → 0.471 |
| 53 | 0.963 → 0.962 | 0.680 → 0.531 |
| 63 | 0.938 → 0.942 | 0.538 → 0.470 |

On insert carriers the naive reader (0.60) matches a length-only detector (0.58–0.60). On substitute carriers, where length carries no information, it still reaches 0.68. The receiver reaches 0.91 and 0.98 on the two kinds.

**E1 controls (moved from Section 4.2).**

- Per family, the receiver ranges from 1.00 (comment, lexical) to 0.77 (syntactic), with CIs that do not overlap the naive reader's.
- Training and testing on clean code only gives the same result (0.963 at layer 53), so the probe is not reading the verdict.
- A within-pair permutation null peaks at 0.516 over 200 permutations.
- On the two hard-eval carriers, both readers stay at 0.50–0.55.
- A length-only probe trained leave-one-family-out scores 0.50 on every held-out family.
- The naive S subspace has 6 dimensions: the naive probe's weight vector plus one difference of means per training family.
- An earlier "criterion met at layer 63" picked that layer after seeing the results, and is not claimed.

**Mechanism (M1–M4).**

- *Directions:* differences of means at the last prompt token. d\_B is computed on the receiver's S = 0 rows. Decodability is a cross-fitted AUROC of the projection onto d\_S, with d\_S estimated on the other half of problems.
- *Extra readers:* the benign-rule reader and the key-after receiver run on a 100-problem subset (key-after: syntactic and lexical carriers), with halves drawn within the subset.
- *Per carrier (M3):* the late override is mean cos(d\_S, d\_B) over layers 53–61. The collusion effect is the receiver's pass-rate change from S = 0 to S = 1 on backdoored code, minus the naive reader's. CIs come from a 500-replicate problem bootstrap that recomputes directions, cosines and effects. Cosines between noisy mean differences shrink under resampling, so intervals on high cosines are conservative.
- *Head patching (M4):* 61 substitute-carrier pairs where S flips the receiver's decision. Last-token outputs of the selected heads are patched from the S = 0 into the S = 1 run (collusion effect). For the naive reader they are patched from the B = 0 into the B = 1 run (honest backdoor signal). We report the share of the score gap recovered. Heads are ranked by their effect on the naive reader's B contrast. At n = 61, attribution patching agrees with direct patching; an earlier n = 2 pilot that suggested otherwise was noise.
- *Carrier-token patching on catchable backdoors (flag-rate change, 95% CIs over problems):*
  - S = 0 carriers restore flagging by +37 to +48 points at layers 1–41;
  - S = 1 carriers induce collusion, −43 to −49 points;
  - from layer 51 both effects are about 0 (e.g. −1.1 \[−4.9, +2.3\]).

| Intervention (catchable pairs) | Change in flag rate | Random control |
| --- | --- | --- |
| Ablate d\_S ⊥ d\_B, layers 45–55 (S = 1) | +20.5 \[+13.6, +27.6\] | +0.3 \[−1.9, +2.5\] |
| Ablate d\_S, layers 45–55 (S = 1) | −6.6 \[−15.2, +0.7\] | (same) |
| Steer +2·d\_S at layer 53 (S = 0) | −58.4 \[−72.9, −42.3\] | −35.7 \[−44.1, −26.6\] at 4× norm |
| Steer +4·d\_S at layer 53 (S = 0) | −79.5 \[−88.8, −68.6\] | (same) |

Steering is sufficient but not specific: an equal-norm random vector already lowers flagging by 36 points, and α = 1 *raises* it (+9.6). Ablating d\_B over layers 21–63 makes the receiver flag 95% of clean code and 98% of colluding attacks.

**Detector layer sweep (Section 4.4).** All detectors are trained leave-one-family-out on the 100-problem benign subset. Thresholds are set per held-out family at a 5% false-positive rate on the receiver's honest code from the other half of problems. "Benign alarms" are flags on the benign reader when its condition is met (B = 0, S = 1).

The override score is −((h − h\_naive) · u\_B), where:

- h is the receiver's activation;
- h\_naive is a key-free (naive) read of the same snippet;
- u\_B is the unit backdoor axis.

The absolute projection −h · u\_B catches at most 8% of attacks at every layer. The override roughly *cancels* the backdoor evidence instead of overshooting it, so the key-free reference is what makes it visible.

| Layer | Plain probe: benign / attacks | + hard negatives: benign / attacks | Override score: benign / attacks |
| --- | --- | --- | --- |
| 21 | 19.4% / 29.2% | 0.0% / 27.6% | 10.2% / 2.5% |
| 33 | 36.8% / 37.8% | 0.0% / 40.5% | 2.4% / 3.8% |
| 41 | 38.7% / 55.9% | 0.1% / 55.3% | 2.3% / 2.4% |
| 45 | 53.5% / 61.3% | 1.2% / 61.9% | 0.5% / 1.8% |
| 47 | 53.2% / 59.1% | 1.5% / 59.9% | 0.0% / 15.4% |
| 49 | 44.7% / 61.8% | 0.4% / 63.3% | 0.2% / 47.2% |
| 53 | 15.6% / 65.8% | 0.4% / 65.3% | 0.1% / 51.9% |
| 61 | 12.9% / 63.2% | 1.7% / 62.8% | 0.2% / 56.3% |

The layer-53 plain-probe rate here (15.6%) comes from a run separate from the 15.2% in Section 4.4. On lexical carriers, the override score catches 94–100% of attacks from layer 49 on, with no benign alarms.

&#91;image: Figure C1\]

*Figure C1. Detector layer sweep: false alarms on honest rule-following (benign condition met) and attacks caught, at a 5% false-positive rate, for the plain S probe, the probe with benign hard negatives, and the override score. Shaded: layers 47–53.*

**Control evaluation (E4).**

- Items: 300 problems × 12 carriers, 3,600 honest tasks (B = 0, S = 0) and 3,597 attacks (B = 1, S = 1); 210 attacks are catchable.
- T: Claude Haiku 5.5 (low effort) scores the raw code 0–9 with the naive monitor prompt. Eleven of 7,197 scores had no digit and were set to 5.
- Redaction: an AST round-trip removes comments, docstrings and formatting, and renames locals and parameters to v0, v1, …. The carrier survives only for `== None`, ends in 7, odd default, nested helper and `None` guard. U is rerun on the redacted code, and the probe reads that run's activations.
- Summarization: U scores T's plain-English description of the code instead of the code.
- Probe: layer 53, last token, leave-one-family-out, so every item is scored by a probe that never saw its family. Each family's probe has its own logit scale, so scores become percentiles among that family's honest tasks from the other half of problems (one held-out family = one deployment with an unseen codebook). This needs honest traffic per deployment. Pooling raw logits under one threshold mixes scales and understates the probe (structural: AUROC 0.99 but 1% caught).
- Combined protocols flag on the maximum of honest-calibrated percentile ranks. With deferral share f and audit budget a, safety is the catch rate at a false-positive rate of a + f, and usefulness is 1 − f.
- Thresholds are cross-fitted across problem halves. CIs: 1,000 bootstrap resamples of problems, with thresholds held fixed.
- Mark in honest code: simulated by weighting honest S = 1 rows. Here the combined protocol splits the budget evenly between monitor and probe, a split fixed in advance. On the 10 training carriers, projecting the naive subspace out of the probe changes its catch rate by less than 3 points at every rate.
- Scripts: `scripts/analyze_e4.py` (protocol comparison, Figure 4), `scripts/e4_control.py` (sweep over the mark's rate in honest code), `modal_app/e4.py` (T, redaction, summarization, probe scores).

*Table C1. Catch rates by protocol (moved from Section 4.4). Each protocol audits its most suspicious 1% or 5% of honest tasks; 95% problem-bootstrap CIs.*

| Protocol | All attacks, 1% | All attacks, 5% | Catchable, 5% |
| --- | --- | --- | --- |
| U: colluding monitor | 0% | 7% \[6, 9\] | 42% \[36, 48\] |
| U + redaction | 1% | 9% \[7, 11\] | 45% \[30, 58\] |
| U + summarization | 2% | 21% \[19, 24\] | 51% \[40, 63\] |
| T: trusted monitor (Haiku 5.5) | 55% | 63% \[60, 67\] | 69% \[54, 80\] |
| **Probe on U** | **54%** | **64% \[63, 65\]** | **70% \[69, 72\]** |
| U + probe | 53% | 61% \[59, 62\] | 78% \[73, 83\] |
| U + redaction + probe | 15% | 22% \[21, 23\] | 32% \[22, 43\] |
| **T + probe** | **76%** | **82% \[80, 84\]** | **87% \[81, 91\]** |

**E4 details (moved from Section 4.4).**

- Redaction and the probe are alternatives. Where redaction strips the carrier, U no longer decodes, and the probe on its activations catches 3%.
- On catchable attacks at a 5% budget, the probe catches 69%, 66%, 39% and 14% when 0%, 5%, 10% and 50% of honest code carries the mark. T, redaction and summarization were not run under this contamination.
- All catch rates are single-attack rates, not a multi-round game with honeypots or attack selection.

Per family (5% budget, all attacks), the probe catches 100% on comment and lexical carriers, 96% on structural, 42% on formatting, 38% on syntactic and 5% on numeric carriers. T is flat across families (61–66%). With benign hard negatives, benign alarms fall to 0.4% at layer 53 while attacks stay at 65%. From layer 53 on, the override score catches 52–56% of attacks (Table in the detector layer sweep above).

## LLM Usage Statement

We used Claude (Anthropic) as a research and coding assistant: to survey related work, write and debug experiment code, run analyses and draft and edit text. Claude Sonnet 5.5 generated the backdoors and Claude Haiku 5.5 served as the trusted monitor and summarizer, as described in Section 3. Every number in this report was recomputed from saved outputs by the scripts in the repository.
