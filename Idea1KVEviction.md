# Handoff: Idea 1 — scored eviction plus an unbiased stratified tail (G2)

This document is a complete handoff for a new Claude session (chat and Claude Code) that has no history of this project. Read it top to bottom before touching the repo.

## 1. One-paragraph summary

We are testing whether a query-agnostic KV-cache compressor improves when a strong scored method (Expected Attention) keeps the high-attention tokens and a stratified random sample with a flat context bias keeps an unbiased picture of the rest. The evidence so far: stratified coverage beats plain random by +11 points on RULER-4K at 25% kept, with the gain concentrated on counting and frequency tasks (cwe, fwe), while scored methods win on needle retrieval. The hypothesis is that the two are complementary. If a hybrid beats pure Expected Attention at the same total budget on aggregate tasks with no needle loss, there is a short paper. If not, the output is an analysis paper, most of which is already run.

## 2. Where everything is

- Repository: the `neymankv` project on GitHub under PseudoHunt (the owner will give the exact URL). Tag `poc-close-2026-10-11` is the last state before this handoff.
- Read first: `runs/STATUS.md`, then `runs/REPORT.md`, `runs/poc/REPORT.md`, `runs/agekv/REPORT.md`, and the Phase G results under `runs/followup/` or `runs/g/` (check STATUS.md for the exact path).
- Code layout: `capture/` (hooks, datasets), `strata/` (SimHash, inverse RoPE, bucket stats), `alloc/` (proportional, Neyman, Webster rounding, reservoir), `estimate/` (attention-output estimators with logit bias), `tests/` (synthetic tests; run them first), a decode-time press for the reasoning experiments, and a kvpress-based pipeline for the long-context experiments (Phase G).
- Results that are valid: Phase A (reasoning, two seeds, 200 problems), Phase D (retention), Phase G on RULER-4K at 25% (five paired comparisons), the long-document offline study (T-1/T0/T1). Results marked invalid: the reasoning-trace rows of the offline study (tokenizer bug) and everything in `invalid_tokenizer_bug/`.

## 3. What has been established (do not re-run these)

Reasoning traces (DeepSeek-R1-Distill-Qwen-1.5B, budget 256, MATH500 + GSM8K, 200 problems x 2 seeds, paired bootstrap):
- full, uniform random eviction, and our stratified evictor are indistinguishable; 165-176 identical outcomes out of 200. The eviction policy does not decide outcomes at this operating point. This replicates Random Attention (arXiv 2609.03430) on a new model family.
- Retention of a once-stated fact: an oracle reserve recovers it 20/20 at no accuracy cost; a singleton-bucket reserve gets 3/20. No query-agnostic signal finds the fact. Closed.
- The inverse-probability (Hajek) logit bias log(N_h/n_h) is not shown to help: +2.1 points [-0.3, +4.6] on RULER-4K. A flat bias on all context keys (`random_rw`) is worth +5.0 [+2.0, +8.2] on its own; that is mass compensation (the context's attention share versus the protected question tokens), not inverse-probability weighting. Keep the flat bias as a tool; do not claim Hajek weighting.
- Long context (Qwen3-4B, kvpress, RULER-4K, 25% of context kept, 260 paired samples, context compressed without the question): full 95.3, `v1_prop` (stratified proportional with reweighting) 19.4, random 6.2. `v1_prop` minus random = +13.2 [+9.6, +17.1]; `v1_prop_noweight` minus random = +11.1 [+7.6, +14.8]. Almost all of the gain is on cwe (0.11 -> 0.67) and fwe (0.43 -> 0.78). Needle tasks are near 0 for every query-agnostic sampler.
- Prior art that owns pieces of this: "The risk of KV cache compression" (2607.01520, balanced clustering matched full KV on LongBench-v2), SubGen (2402.06082, cluster + reservoir + count weights), Error Certificates (2607.21475, certainty set + Poisson tail + log(1/pi)), Probabilistic Interpretation (2608.28293). Cite all four. "Stratified coverage beats random" is not a new claim; the hybrid and the aggregate-task result would be.

## 4. Method definitions (exact)

Context-only compression, kvpress protocol: compress the context without the question, then answer several questions against the same compressed cache. Protected: the first 4 tokens and the last 128 context tokens (kvpress default window), identical across methods.

- `ea_B`: kvpress `ExpectedAttentionPress` at compression ratio (1 - B).
- Stratified tail (`strat_tail`): strata = 8-bit SimHash of the centered pre-RoPE key (per-head mean subtracted; fixed seed; projection fixed per head) crossed with 4 equal-count position bins. Budget split across strata in proportion to stratum size with a floor of one slot per non-empty stratum and Webster rounding. Uniform random selection inside each stratum. Equal count per head and layer.
- Flat context bias (`rw`): add log(N_ctx / n_kept) to the logits of every kept context token, 0 to protected and question tokens. This is the `random_rw` bias; it is one scalar per layer.
- Hybrid `ea_X + strat_tail_Y` at total kept fraction X+Y: run EA to select its top X fraction; from the remaining context tokens, draw the stratified sample of size Y fraction of the context; apply the flat bias only to the tail tokens (bias log(N_remaining / n_tail)). Keep positions original; never renumber.
- Controls: `ea_X + random_tail_Y` (uniform random tail, same bias), `ea_X + strat_tail_Y_nobias`, and pure `ea_(X+Y)`.

## 5. Phase G2: the decisive experiment (about 3 GPU-hours on one 48 GB GPU)

Model: Qwen3-4B (bf16). If time allows, repeat on Llama-3.1-8B-Instruct, which is the kvpress leaderboard model and makes the table comparable to published numbers.

Benchmark: RULER-4K, all 13 tasks, 20 samples each (260), through kvpress so the protocol matches the leaderboard. Fixed sample list, same as Phase G.

Runs at total kept = 25%:
1. `ea_25`
2. `ea_20 + strat_tail_5` (bias on tail)
3. `ea_20 + random_tail_5` (bias on tail)
4. `ea_20 + strat_tail_5_nobias`
5. `ea_15 + strat_tail_10` (bias on tail)
6. `ea_25` from the Phase G queue if it already finished; do not duplicate.
Seeds: 3 for every run with randomness; report mean and the paired interval over samples using the per-sample mean across seeds.

Analysis: paired per-sample differences with 95% bootstrap intervals, win/loss/tie counts, and a per-task table (needle tasks, QA, variable tracking, cwe, fwe). The three numbers that decide the gate are: hybrid minus pure EA overall, hybrid minus pure EA on cwe+fwe, and hybrid minus pure EA on the needle group.

Gate (pass): run 2 beats run 1 overall with an interval clear of zero, the gain is concentrated on cwe/fwe and QA, the needle group shows no loss beyond noise, and run 2 beats run 3 (strata matter) or at least run 4 (bias matters). If run 3 equals run 2, the contribution is "any unbiased tail," which is weaker but still reportable.

Fail: run 1 is at least as good as every hybrid. Then stop G2, skip Section 6, and write the analysis paper (Section 8).

## 6. If G2 passes: second results (about 4 GPU-hours)

- Budget sweep: total kept 10% and 50% with the best split from G2 and pure EA.
- RULER-8K at 25% (same runs 1-3).
- KVzip as a second scorer if the 2x prefill cost fits: `kvzip_25` vs `kvzip_20 + strat_tail_5`.
- LongBench summarization (gov_report, qmsum, multi_news) at 25% with ROUGE-L, runs 1-3. The literature reports summarization hallucination rising up to 3.4x under compression; an unbiased tail that preserves aggregate statistics is a plausible mitigation and gives the paper an audience beyond RULER.
- Cost: decode tokens/s and peak memory for runs 1-3 at batch 1 and 8. The tail adds hashing at prefill only.

## 7. Harness rules learned the hard way

- Tokenizer: under transformers 5.7, `AutoTokenizer` drops every space for DeepSeek-R1-Distill models. The loader reads `tokenizer.json` directly and refuses to run unless "Hello world" survives an encode/decode round trip. Keep that guard for every model.
- Attention precision: HF bf16 eager attention rounds scores before the softmax and corrupts arithmetic. Use SDPA with the bias passed as a float mask. Verify logits against HF's own decoding on 50 steps before any run.
- Scorer matching: kvpress outputs carry a `context_hash` from Python's salted `hash()`, which differs between processes. Match samples on task, question and answer.
- Memory budgets must be verified: log kept-set sizes per head after compression for one sample per method and check they match across methods.
- Every run records git commit, `pip freeze`, GPU name, seed, and GPU-hours used.
- Pre-register the gate in `runs/g2/PLAN.md` before launching, with the exact pass condition from Section 5.

## 8. Paper framing

If G2 passes, the one-line claim: "Scored eviction keeps what typical queries attend to; an unbiased stratified tail keeps the statistics of everything else. Together they beat either alone on aggregate tasks at the same budget." Baselines: Expected Attention, KVzip, random, and the hybrid controls. Must cite: 2607.01520, 2402.06082, 2607.21475, 2608.28293, 2510.00636 (Expected Attention), 2505.23416 (KVzip), 2609.03430 (Random Attention).

If G2 fails, the analysis paper: hash-bucket eviction reduces to stratified random sampling (SimBin, R-KVHash as coded); the eviction policy does not matter on reasoning traces at 12x compression (replication with paired statistics); coverage helps counting tasks only; reweighting helps attention-output error but not task accuracy; retention of once-stated facts needs a query signal. All of this is already run except one more seed on RULER and a 45-minute re-capture of reasoning traces on the 7B model with the correct tokenizer.

## 9. What not to do

- Do not re-run the reasoning-trace experiments, NeymanKV v1/v2, or AgeKV. They are closed with adequate statistics.
- Do not claim inverse-probability weighting; the data does not support it.
- Do not compare against R-KVHash's published numbers; its public code shows a degenerate scorer and an inverted R-KV baseline, and the paper text was never obtained. Mention it only as "as released" with the code reference (minghui-liu/kvpress, branch dixi/rkv_lsh).
- Do not report accuracy without paired intervals; single-run differences under 8 points were noise throughout this project.
