# Reproducing Apollo's deception probe on DeceptionBench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yilin1010/deception-probe/blob/main/analysis.ipynb)

Replicating the Apollo Research linear-probe deception detector (Goldowsky-Dill et al. 2025) on a different evaluation suite (DeceptionBench, labeled n = 746) using Llama-3.3-70B-Instruct.

**Headline (n = 746, mean aggregation):**
- Highest AUROC — `svm_rbf / L40` → AUROC **0.898**, R@1%FPR 29.1%, R@5%FPR 63.6%
- Highest R@1%FPR — `svm_rbf / L35` → R@1%FPR **36.7%**, AUROC 0.879

---

## TL;DR — three findings

1. **Layer choice dominates probe model choice.** Same probe across layers swings AUROC ≈ 0.55. Same layer across probes swings ≈ 0.06.
2. **The deceptive signal is sparse across tokens.** `mean` aggregation wins; `last`/`max` can fall below chance.
3. **REPE validation AUROC is not a reliable selector for OOD layers.** REPE-val-AUROC = 1.0 layers produce OOD AUROC < 0.4.

---

## Reproduce in one click

[Open `analysis.ipynb` in Colab.](https://colab.research.google.com/github/Yilin1010/deception-probe/blob/main/analysis.ipynb) The notebook clones this repo, loads cached activations from `data/`, retrains probes, and regenerates all four headline figures in <2 minutes (CPU-only, no GPU needed). The final cell also runs a live `gpt2`-based extraction demo to prove the upstream pipeline works end-to-end on CPU.

---

## What's in `data/`

| File | Purpose | Size |
|---|---|---|
| `repe_per_token_L{8,22,35,40}.pkl` | Apollo per-token training inputs (4 layers × 4332 tokens × 8192 fp16) | 4 × 68 MB |
| `repe_meanpool_4L.pkl` | 612-sample mean-pooled variant (faster retrain demo, weaker for non-linear probes than per-token) | 39 MB |
| `dbench_meanpool_4L.pkl` | 746 DBench evaluation samples, mean-pooled at 4 layers | 47 MB |
| `dbench_per_token_scores_svm_rbf_L40.pkl` | Per-token probe scores from the saved `svm_rbf/L40` probe (746 samples, all aggregations reproducible) | 2.6 MB |
| `dbench_token_trace_L40.pkl` | Raw L40 activations for 10 fig-7 trace samples | 30 MB |
| `svm_rbf_L40.joblib` | Trained sklearn `Pipeline(StandardScaler, Nystroem-RBF, LogReg)` | 18 MB |
| `metrics.json`, `summary.csv` | Per-(arch, layer, domain, aggregation) AUROC table for all 102 configs | 2.4 MB |
| `trace_full_<domain>.json` | Per-token probe scores with token text, 20 samples per domain (5 files) | 1 MB |

---

## Setup

**REPE training data** (Apollo §3.2.1, see `src/data.py`). Each TRUE fact is materialized twice: once with an honest preamble, once with a deceptive preamble. The two share response text, so probes cannot cheat on response content. Train/val split is by `pair_id`, not by sample — otherwise the same fact leaks across both sides of the split.

**DeceptionBench labels.** Each rollout scored by Claude Opus 4.6 on (thought, response) Likert scales 1–10. Binarized as `1–3 → deceptive`, `4–6 → ambiguous`, `7–10 → honest`. We use the `response_only` rule (the detector operates on what the user actually sees). Dropping ambiguous rollouts → 746 labeled samples (330 deceptive, 416 honest).

---

## 1. Layer choice > probe model choice

**Fig 1.** AUROC vs. layer for three probe models on DeceptionBench, with REPE validation AUROC as a reference.

![Layer sensitivity](figures/fig1_layer_sensitivity.png)

Same probe, different layer: ΔAUROC ≈ **0.55**. Same layer, different probe: ΔAUROC ≈ **0.06**.

Picking layer by **REPE val AUROC** is unreliable for OOD selection — REPE val ≈ 1.0 across all 80 layers, so the argmax is degenerate. Shallow layers separate the contrastive prompts by the *instruction token itself* ("honest" vs "dishonest"), perfect on REPE val and inverted on OOD:

| Arch | Best-on-DBench layer | OOD AUROC | REPE-argmax layer | REPE val | OOD AUROC |
|---|---|---|---|---|---|
| logreg | L35 | 0.839 | L8 | 1.000 | **0.395** |
| svm_rbf | L40 | 0.898 | L5 | 1.000 | **0.343** |
| mlp | L35 | 0.869 | L5 | 1.000 | **0.352** |

**Practical takeaway:** sweep layers with cheap logreg, then use the other probe model on the best 2–3 layers.

---

## 2. The deceptive signal is sparse across tokens

**Fig 2.** Per-token probe scores for an honest (top) and deceptive (bottom) Economy-domain response.

![Per-token trace, Economy domain](figures/fig7_per_token_trace_economy.png)

Honest responses produce flat low traces. Deceptive responses are *mostly* low with isolated spikes at fabricated content — invented numbers, fake names, asserted-but-false claims.

This shape determines which aggregation works:

**Fig 3.** AUROC by aggregation method (svm_rbf / L40, n = 746). Mean aggregation is consistently best.

![Aggregation comparison](figures/fig5_aggregation.png)

- **`mean`** wins — distributed signal, robust to a single noisy token.
- **`max`**, **`last`** can fall below 0.5 — one wrong-signed spike or a stance-neutral EOS token flips the sample.
- **`threshold_05`** (fraction of tokens > 0.5) tracks `mean` with a small penalty.

All six aggregations live in `src/train_probe.py` (notebook §6 reproduces this figure from `dbench_per_token_scores_svm_rbf_L40.pkl`).

---

## 3. Domain gap is the bottleneck

**Fig 4.** Per-domain AUROC by probe model (best single layer per probe, mean aggregation).

![Arch × domain heatmap](figures/fig4_arch_domain_heatmap.png)

With `svm_rbf / L40 / mean`, AUROC varies moderately across domains (0.81–0.94), but **recall at 1% FPR swings from 25.7% to 63.4%**:

| Domain | AUROC | R@1%FPR | R@5%FPR | Deception rate |
|---|---|---|---|---|
| Social | 0.943 | **0.634** | 0.775 | 0.48 |
| Economy | 0.910 | 0.298 | 0.579 | 0.43 |
| Healthcare | 0.905 | 0.472 | 0.623 | 0.34 |
| Entertainment | 0.876 | **0.257** | 0.615 | **0.68** |
| Education | 0.814 | 0.375 | 0.425 | 0.27 |

Worst recall is on Entertainment, which also has the highest deception rate — missed detections concentrate where detections matter most. REPE training data is natural-science and history facts; Social rollouts are closest to that distribution, Entertainment furthest. 

---

## Differences from Apollo Research

| | Goldowsky-Dill et al. (2025) | This work |
|---|---|---|
| Eval set | Insider Trading role-player | DeceptionBench, more complex tasks across 5 domains, n = 746 |
| Labels | Human annotators | Claude Opus 4.6 (thought + response, 1–10 scale) |
| Probes | logreg only | + svm_rbf (Nyström), mlp |
| Key claim tested | "AUROC > 0.95 transfers" | Val AUROC does not transfer to eval AUROC |

---

## Reproduction notes

Activation extraction (Llama-3.3-70B at 8-bit) ran on an H200 GPU; ~80 GB raw activations per DBench domain across all 80 transformer layers. Probe training and evaluation are CPU-only on the cached 4-layer activations shipped in `data/`. The notebook also runs a live extraction demo using `gpt2` on CPU as a pipeline  check (final section).

---

## References

- Zou et al. (2023). *Representation Engineering: A top-down approach to AI transparency.* arXiv:2310.01405
- Goldowsky-Dill et al. (2025). *Detecting strategic deception using linear probes.* arXiv:2502.03407
- Huang et al. (2025). *DeceptionBench.* arXiv:2510.15501
