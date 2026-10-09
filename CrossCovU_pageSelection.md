# Task: CrossCov-U page-selection gate experiment (vs LOCKS)

## Goal

Decide whether a **query-aware shared basis (CrossCov-U)** can match **page-local SVD summaries (LOCKS, arXiv 2607.24555)** at page-level KV selection, at matched selector bytes.

Do this by replicating LOCKS Table 1 / Table B.1 (RULER-32K, Llama-3.1-8B-Instruct) and adding CrossCov-U rows.

This is an accuracy and fidelity experiment only. Do NOT write CUDA/Triton kernels and do NOT measure speed.

Work autonomously. Stop and report only at the STOP conditions listed below.

## Environment facts

- GPU: single H100 80GB on a spot VM. The instance can die at any time, so every stage must be resumable:
  - Write results incrementally to JSONL, one line per (arm, budget, task, sample).
  - On restart, skip completed lines.
  - Cache calibration bases to disk.
- Repo: `github.com/8BitSpacemanSpiff/loki-crossconv`, branch `crosscov-ext`. It is a fork of `hpcgroup/loki` and contains the CrossCov-U basis-fitting code.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, bf16. It is gated, so use the HF token in `$HF_TOKEN`.
  - 32 layers, 32 query heads, 8 KV heads (G=4), head dim d=128.
- Put all new code in a new directory, `page_gate/`. Do not modify `modify_llama.py`.

## Known hazards in the existing repo (handle explicitly)

1. **Missing rotary guard in `modify_llama.py`.** Do not rely on it. Capture post-RoPE q and k yourself, using a hook placed after `apply_rotary_pos_emb`.
2. **"Whiten" mode produces non-orthonormal bases.** Do not use whiten mode. For every basis, assert `||U^T U - I||_max < 1e-3`.
3. **Dead asymmetric artifact paths (`key/`, `query/`).** CrossCov-U uses ONE shared U for both q and k. If you load any saved artifact, assert `torch.equal(key_U, query_U)`. Otherwise refit.

## Step 0 — Inspect and report (no long runs yet)

- Find the CrossCov-U fitting function. Write down in `page_gate/NOTES.md`:
  - the exact formula it uses (GQA-pooled Q^T K cross-covariance, how it is symmetrized or decomposed, how U is extracted);
  - whether it fits on pre- or post-RoPE states.
- Use **post-RoPE** for everything in this experiment.
- Set up RULER data:
  - Use NVIDIA's RULER generation scripts to create the standard 13 tasks at 32K with the Llama-3.1 tokenizer and chat template:
    - NIAH-S1/S2/S3, NIAH-MK1/MK2/MK3, NIAH-MV, NIAH-MQ, VT, CWE, FWE, QA-1, QA-2.
  - Generate 50 samples per task with a fixed seed.

## Definitions (follow exactly — these mirror LOCKS)

### Pages and budget

- Page size B=16. Page j holds tokens [16j, 16j+16).
- Token budget b per (layer, KV head). Number of page slots: k = b/16.
- Two slots are always taken: the **sink page** (page 0) and the **most recent page** (the page holding the current token, which may be partial).
- The other k−2 slots are filled by the selector.
- Run b ∈ {128, 256}, which gives k = 8 and 16.

### Exact page-LSE score (oracle)

For each query head g:

```
s_j(q_g) = logsumexp_{i in page j}( q_g·k_i / sqrt(d) )
```

### Share-average GQA combine (used by EVERY arm unless noted)

```
m_g(j) = softmax_j( ŝ_j(q_g) )        # over all candidate pages
m̄_j    = mean_g m_g(j)                 # over the 4 query heads in the KV group
```

Select sink + recent + top-(k−2) pages by m̄_j. All 4 query heads share this page set.

### Page summaries and approximate scores

Let δ_i = k_i − μ_j (key minus its page centroid).

**LOCKS (page-local):**
- For each completed page: μ_j = mean of the page's keys; D_j = matrix of the centered keys (B×d).
- V_j = top-r right singular vectors of D_j; c_i = V_j^T δ_i.
- Score:

```
ŝ_j(q) = q·μ_j/sqrt(d) + logsumexp_i( (V_j^T q)·c_i / sqrt(d) )
```

**Hybrid (page centroid + shared global basis U):**
- Same as LOCKS, but V_j is replaced by a single U (d×r) shared by all pages for that (layer, KV head):

```
c_i    = U^T δ_i
ŝ_j(q) = q·μ_j/sqrt(d) + logsumexp_i( (U^T q)·c_i / sqrt(d) )
```

**Fully shared (Loki-like):**
- One global mean μ_g per (layer, KV head), plus U:

```
c_i    = U^T (k_i − μ_g)
ŝ_j(q) = logsumexp_i( q·(μ_g + U c_i) / sqrt(d) )
```

### Quantization (matched bytes, from LOCKS App. B.1)

- Bases: int4. Centroids and coefficients: int8. Scales: bf16.
- Use symmetric per-row quantization. Document your exact scheme in NOTES.md.
- Report the selector bytes per (page, KV head) for each arm. Target bytes per tier:

| Tier | Target bytes | Page-local rank | Hybrid rank | Fully shared rank |
|---|---|---|---|---|
| r4 | ~488 B | 4 | 20 | 28 |
| r8 | ~815 B | 8 | 40 | 48 |

- ALSO run every arm unquantized (bf16). This separates basis quality from quantization noise.

## Arms

| ID | Arm | Basis fit |
|---|---|---|
| ORACLE | exact page-LSE | reads all keys |
| FULLKV | dense attention | (task score only) |
| L | LOCKS page-local | per-page SVD |
| H-KK | Hybrid + key-covariance U | top-r eigvecs of pooled Σ_j H_j, where H_j = Σ_{i∈j} δ_i δ_i^T, over calibration pages |
| H-CC-raw | Hybrid + CrossCov-U U | existing CrossCov-U fit, on raw post-RoPE keys |
| H-CC-ctr | Hybrid + CrossCov-U U | CrossCov-U fit, but with page-centered keys δ_i in place of k_i (queries unchanged) |
| F-KK | Fully shared, key PCA | global mean + top-r PCA of keys |
| F-CC | Fully shared, CrossCov-U | CrossCov-U fit on keys centered by the global mean |
| UNIQUE (P2) | q·μ_j + 0.5·‖q‖·std_j, std_j = ‖per-dim std‖₂, max over GQA group | no basis |

Basis fitting rules:
- Calibration set: WikiText-2 **raw validation** split, in 32K-token windows (this is LOCKS's "Global 32K").
- Fit separately for each (layer, KV head).
- For CrossCov-U, pool the 4 query heads of the group the same way the existing code does.
- Cache every basis to disk with metadata: model, rope=post, rank, centered flag, fit data, hash.

## Stage 1 — Selection fidelity (the gate)

For each RULER sample:
1. Prefill with dense attention.
2. Greedy-decode the answer with FullKV, capped at 32 new tokens.
3. At every decode step, in every layer and KV head, compute every arm's selected page set from the same FullKV state. Do this online with hooks; do not save full KV to disk.

Metrics per (arm, b):
- **Mass:** group-average fraction of true softmax attention mass that falls inside the selected set. Report the mean, and also the worst head in each group.
- **Recall:** |selected free pages ∩ ORACLE free pages| / (k−2).
- Breakdowns: per layer (plot), and per task.

Use 20 samples per task for Stage 1.

### Invariant tests (run first on one short 4K sample; fail loudly)

- ORACLE mass ≥ every arm's mass at every step, within 1e-5. This holds because share-average top-k by exact shares maximizes group-average coverage.
- Unquantized L with r = 15 (≥ rank of D_j) reproduces ORACLE exactly (100% recall).
- Unquantized hybrid and fully shared arms with r = d = 128 reproduce ORACLE exactly.
- The k used for scoring equals the cached post-RoPE key.
- U is orthonormal, and the same U is applied to q and to k.

### STOP condition 1 (anchor check)

At r4 tier, b=128, quantized, the anchor rows must land near LOCKS Table B.1:

| Row | LOCKS mass | LOCKS recall |
|---|---|---|
| H-KK | 84.5 | 81.9 |
| L | 85.8 | 90.5 |

If either row is off by more than 2 points on mass or recall:
- STOP and report.
- Find the cause first (page boundaries, recall definition, quantization, RoPE). Do not tune the CrossCov-U arms before this is fixed.

### Gate readout (write to `page_gate/REPORT.md`)

Compare H-CC-ctr and H-CC-raw against L and H-KK at r4 tier, for b=128 and b=256:

- **GO:** H-CC recall is within 1 point of L, and mass is ≥ L − 0.3.
- **PARTIAL:** H-CC is clearly above H-KK but below the GO bar.
- **STOP:** H-CC is ≤ H-KK + 1 point recall.

## Stage 2 — Task scores (only if the gate is GO or PARTIAL)

- Arms: FULLKV, ORACLE, L, H-KK, the best H-CC variant, F-KK, F-CC, and UNIQUE.
- Use 50 samples per task, r4 tier, b ∈ {128, 256}.
- At each decode step, attend only to the selected pages, using the ORIGINAL bf16 K/V. Implement this as masked SDPA; speed does not matter.
- Prefill once per sample, then run every arm's decode from a copy of the same cache.
- Build a page's summary when that page completes during decode.
- Score with RULER's official per-task metrics.
- Sanity check: FULLKV average ≈ 88.8. If it is off by more than 2 points, STOP and report.

## Stage 3 — Only if GO: rate–distortion

Sweep H-CC-ctr rank r ∈ {4, 8, 12, 16, 20} at b=128. Report the lowest bytes per page at which it matches L-r4 recall. This is the "same quality, cheaper scan" claim.

## Deliverables in `page_gate/`

- `NOTES.md`: formulas, quantization scheme, and any deviation from this spec.
- `results/*.jsonl`: raw resumable results.
- `REPORT.md`, containing:
  - one table in the format of LOCKS Table B.1 with all arms (mass, recall, score; quantized and bf16; both budgets);
  - per-layer mass and recall plots (PNG);
  - the gate decision with numbers.
- `run_all.sh`: rebuilds everything from a fresh VM.
