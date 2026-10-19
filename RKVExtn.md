# Overnight brief: position-invariant redundancy for reasoning-trace KV eviction

Goal: decide by morning whether a method paper is viable. The method combines a TriAttention-style pre-RoPE importance score with a redundancy score computed on un-rotated, low-frequency key features, hashed with SimHash. Tonight is a decision run, not the final evaluation.

Run the stages in order. Each stage has a gate. If a gate fails, stop, write the report, and do not continue. The last section says what to write. Keep every intermediate file; name outputs by stage.

Hardware assumption: one GPU with 40 GB or more. If less, use R1-Distill-Qwen-7B only and skip the second model.

## Stage 0: setup (target: under 1 hour)

1. Clone and install:
   - R-KV: https://github.com/Zefan-Cai/R-KV (use the HuggingFace path: `examples/run.sh`, `run_math.py`, not the vLLM/SGLang ports).
   - TriAttention: https://github.com/WeianMao/triattention (needed for its scoring function and calibration stats; if the HF path is missing, re-implement Eq. 6 to 13 of the paper, which is cited below).
   - If a SimBin or SimHash codebase exists on this machine, note its path and reuse its hashing utilities; otherwise write a 40-line SimHash module.
2. Models: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (primary), `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (secondary). bf16.
3. Report, before doing anything else, in `stage0_report.md`:
   - For R-KV: which function computes redundancy, whether the keys it uses are pre- or post-RoPE, how it handles GQA, and the sign convention of its final score Z = lambda*I - (1-lambda)*R.
   - For both models: num layers, num KV heads, head_dim, rope_theta, and the wavelength 2*pi*theta^(2j/d) of pairs j = 0, 8, 16, 24, 32, 40, 48, 56.
   - Smoke test: run R-KV at budget 1024 on 3 MATH-500 problems and confirm it generates and the answer checker runs.

## Stage 1: trace generation (target: 2 hours, run in the background during stage 2 setup)

- Full-cache generation, temperature 0.6, top-p 0.95, max 16,384 tokens.
- Problems: 120 from the MATH train split (levels 4-5 only, to get long traces) and 40 from GSM8K train. Not the test sets.
- Save token ids only. Keep traces with at least 2,000 tokens. Target at least 60 traces of 6,000+ tokens per model. If fewer, generate more level-5 problems until reached, or until 2.5 hours have passed.
- Log trace length histogram.

## Feature definitions (used by every stage)

Pairs follow the HF rotate_half layout: pair j = dims (j, j + d/2), j = 0 fastest.

- `post_full`: cached post-RoPE key, all dims. Baseline.
- `pre_full`: un-rotated key. Compute by inverse rotation using each token's original position (track positions in a side tensor, since eviction makes them non-contiguous). Verify against a forward hook on k_proj output, after bias, to relative error below 1e-3. Report the error.
- `pre_low(J)`: pre_full restricted to pairs j >= J. Primary J = 48. Also J = 32 and 56.
- `post_low(J)`: post-RoPE restricted to pairs j >= J. Same J values.
- Per KV head. Normalize vectors before cosine or hashing.
- SimHash: one Gaussian matrix R per seed, shape (d, b). Low-band arms use the rows of R for the low-band dims only (paired design). Codes = sign(xR). b in {16, 32, 64}. Seeds: 5, include 42.

## Stage 2: mechanism test, the kill test (target: 1.5 hours, CPU-heavy, no eviction)

For each trace, compute keys on the fly with one forward pass (no generation). Use layers {4, 8, 12, 16, 20, 24} and all KV heads.

Pairs of positions (i, j), i - j >= 64:

- Positives: the 6 token ids ending at i equal the 6 ending at j, with at least 3 distinct ids in the 6-gram.
- Negatives A: random (i, j) with the same distance distribution as positives.
- Negatives B (hard): same current token id at i and j, different 6-gram.

Cap at 20,000 positive pairs per model; match negatives in count.

Score each pair under each feature with (a) exact cosine and (b) negative Hamming distance at b = 16 and 64.

Metrics, per layer group (early {4,8}, mid {12,16}, late {20,24}) and per distance bucket [64,256), [256,1024), [1024,4096), [4096, up):

- AUC positives vs negatives A, and vs negatives B.
- Mean positive-pair cosine.

Write `stage2_results.csv` and `stage2_summary.md` with one table per model: rows = features, columns = distance buckets, cells = AUC (cosine) and AUC (hamming16).

**Gate 2.** Pass if, on the primary model, averaged over mid+late layers: (i) `post_full` AUC vs negatives A falls by at least 0.05 from the first bucket to the last, and (ii) `pre_low(48)` or `pre_full` AUC in the last bucket exceeds `post_full` by at least 0.05, with the same sign in at least 4 of 5 hash seeds for the hamming version. If the gate fails, stop and report; the method's mechanism is not there.

## Stage 3: scorer construction (target: 1 hour)

Build the method scorer, `INV`, and the baselines, all inside the R-KV eviction loop (compress every beta = 128 tokens to budget B; the most recent 128 tokens and the first 4 tokens are never evicted).

Importance term `I_tri(k)`: TriAttention's score. Use its released calibration code to get per-head pre-RoPE Q centers and norms from 200k tokens of the stage-1 traces (held out from the evaluation traces). Implement S_trig (Eq. 6), S_norm with the (1 - R_f) weighting (Eq. 8), averaged over future offsets D = {1, 2, 4, ..., 2^14} (Eq. 11), z-scored per query head and max-aggregated over the GQA group (Eq. 12-13). Cite: TriAttention, arXiv:2604.04921.

Redundancy term `R_inv(k)`: for feature `pre_low(48)`, SimHash with b = 32. Bucket counts c_i. Score of key in bucket i:
R'_i = sum over all buckets j, including j = i, of c_j * cos(pi * Hamming(i, j) / b), minus 1 for the key itself, divided by (n - 1). This is R-KVHash's estimator with two fixes: include the own bucket, and use the correct SimHash angle. Softmax-normalize over keys as R-KV does.

Combined: Z = lambda * zscore(I_tri) - (1 - lambda) * zscore(R_inv). Keep top-B by Z. Sweep lambda in {0.3, 0.5, 0.7} in stage 4; default 0.5.

Arms:

| Arm | Importance | Redundancy |
|---|---|---|
| R-KV | attention (R-KV default, lambda = 0.1) | exact cosine, post_full |
| R-KVHash | none | SimHash b = 16, post_full, paper's Eq. 7 as written |
| TriAttn | I_tri | none |
| TriAttn+Rpost | I_tri | SimHash, post_full |
| INV (method) | I_tri | SimHash, pre_low(48) |
| INV-exact | I_tri | exact cosine, pre_low(48) |
| INV-prefull | I_tri | SimHash, pre_full |
| INV-Rimp | R-KV attention (lambda = 0.1) | SimHash, pre_low(48) |

`INV-Rimp` needs no TriAttention code. It isolates the redundancy-feature change against plain R-KV: same importance term, only the redundancy features and the hash differ.

Fallback: if TriAttention's scoring cannot be made to run within 1.5 hours of starting stage 3, skip every arm that uses I_tri (TriAttn, TriAttn+Rpost, INV, INV-exact, INV-prefull). Run R-KV, R-KVHash and INV-Rimp only, and say in the report that TriAttention was not compared.

Unit test each scorer on one trace: shapes, no NaN, the kept set has size B, sinks and window are kept.

## Stage 4: teacher-forced eviction quality (target: 3 hours)

For each held-out trace (at least 40 per model with length 6,000+), run full-cache and compressed models in lockstep on the same token sequence, in chunks of 128. Evict at every chunk boundary exactly as the arm specifies. No generation.

Metrics, averaged over tokens after the first eviction:
- Mean per-token KL(full || compressed) on the next-token distribution.
- NLL gap.
- Attention-mass recall: fraction of the full model's attention mass (next 128 queries, mid+late layers) that lands on kept keys.

Budgets B in {512, 1024, 2048}. Seeds: 3 for hashed arms. Paired bootstrap CIs across traces.

Write `stage4_results.csv` and a table per model (rows = arms, columns = budgets, cells = KL with CI).

**Gate 4.** Pass if INV beats TriAttn by at least 10% relative KL at two of three budgets with CIs excluding zero, and INV beats TriAttn+Rpost at the same budgets (this isolates the invariant features from the mere presence of a redundancy term). If INV beats TriAttn but not TriAttn+Rpost, the gain is from redundancy, not invariance; report that plainly and stop.

Run the same test on the R-KV side regardless: INV-Rimp vs R-KV, relative KL reduction of at least 10% at two of three budgets with CIs excluding zero. This is the minimum evidence that invariant redundancy features help, independent of TriAttention. If the fallback above was used, this is the only gate.

## Stage 5: end-to-end generation (only if gate 4 passes; target: remaining time)

MATH-500, primary model only tonight. Budgets 512 and 1024. Arms: full, R-KV, TriAttn, INV. 4 samples per problem, temperature 0.6, top-p 0.95, max 16,384. Use R-KV's answer checker. Report pass@1 mean and standard error, plus mean generation length per arm.

If time remains: AIME24 (30 problems, 8 samples, budget 1024) for the same four arms.

## Report: `overnight_report.md`

1. One line: which gates passed.
2. Stage 0 facts (the R-KV pre/post-RoPE answer and the wavelength table).
3. Stage 2 table and the two numbers gate 2 was judged on.
4. Stage 4 table with CIs, and the INV vs TriAttn+Rpost comparison called out.
5. Stage 5 numbers if run.
6. Deviations from this brief, failures, and anything you had to guess. If TriAttention's code was unusable and you re-implemented it, say so and give the sanity check you ran (its reconstruction correlation on one head should be around 0.5-0.9).
7. Wall-clock per stage.

Do not tune on the evaluation traces. Do not change gate thresholds. If something is ambiguous, pick the simpler option, state it, and continue.
