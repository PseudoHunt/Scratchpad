# Cross-family KV-cache transfer: Gauss 4 1.5B → Gemma 4 31B

Brief for Claude Code. Read fully before writing code. Where this brief says "ask", stop and ask me before continuing.

## 0. Goal

Reproduce two published prefill-free cross-family KV-cache transfer methods and one combined variant on a single fixed model pair, then report retention side by side.

- **Sender (S):** Gauss 4 1.5B — path: `<PATH_SENDER>`
- **Receiver (R):** Gemma 4 31B — path: `<PATH_RECEIVER>`
- Direction: S → R only. Both models stay frozen. Only the mapper is trained.
- No KV quantization. No cache compaction. Plain bf16 caches end to end.

Methods:

- **A. HeteroFold** — arXiv 2609.32259 (closed-form affine maps + rank-16 output-aware calibration)
- **B. SwaT** — arXiv 2610.15095, code at https://github.com/Ar-Kareem/SwaT (transformer mapper, MSE then NLL through the frozen receiver)
- **C. Ours** — SwaT architecture, initialized from HeteroFold's closed-form maps, phase-1 loss = MSE + HeteroFold's per-layer attention-output loss, phase-2 = SwaT's NLL loss

Primary metric: **retention** = score with transferred cache ÷ score with the receiver's own native prefill, per benchmark.

## 1. Environment

- One NVIDIA B300 (Blackwell Ultra). Confirm `torch.cuda.get_device_capability()` and that the installed torch + flash-attn / SDPA actually support this arch. If flash-attn won't build, use PyTorch SDPA. Report which path you ended up on.
- HF `transformers` (recent), `datasets`, `accelerate`. Use bf16 everywhere.
- Memory budget: 31B receiver bf16 ≈ 62 GB, 1.5B sender ≈ 3 GB, mapper ≈ 300M params (≈ 2.5 GB with AdamW states). Backprop through the frozen 31B receiver needs activation checkpointing on every receiver layer. Start with seq len 2048, batch 1, grad accumulation; raise only after a stable run.
- Log everything to `runs/<method>/<timestamp>/` with a `config.json`, `metrics.jsonl`, and checkpoints. Use `wandb` offline or tensorboard, your call.

## 2. Step 1 — Inspect both models (do this first, report before coding anything else)

Read both `config.json` and tokenizers. For each model report:

- number of layers, hidden size, attention heads, **KV heads**, **head dim**
- RoPE base/theta and any scaling; whether keys are RoPE'd per layer or some layers are NoPE
- **sliding-window layers**: which layers, window size (Gemma 3 had 5:1 local:global; check what Gemma 4 does)
- **QK-norm** (k_norm / q_norm) — Gemma 3 normalized keys before RoPE; check Gemma 4 and Gauss 4
- multimodal components — if Gemma 4 31B is a VLM, use its text tower only
- tokenizer type, vocab size, whether fast tokenizer with `return_offsets_mapping` works (needed for token alignment)
- chat template for each

Gauss 4 is not a public model. Do not guess its architecture from the name. Read the config and the modeling file at the path.

Then compute and report:

- d_KV(S) = KV heads × head dim for the sender; d_KV(R) for the receiver
- whether 3·d_KV(S) ≥ d_KV(R). HeteroFold's Recolor assumes this; if false, see §4 note.

**Ask** before proceeding if anything above is surprising (e.g. non-RoPE positional scheme, MLA-style compressed KV, shared KV across layers).

## 3. Shared infrastructure (build once, used by all three methods)

### 3.1 Sanity test zero: cache injection round-trip
Before anything else: run the receiver natively on a prompt, capture its KV cache, then inject that exact cache back as `past_key_values` and decode the continuation. Logits must match the native run to bf16 tolerance. Do the same with the sender. If this fails nothing downstream is trustworthy. Note exactly what form HF stores in the cache for Gemma 4 (post-k_norm? post-RoPE?) — every mapper output must be put into that same stored form.

### 3.2 Token alignment (TA)
Both papers align by **character spans**, not token indices.
- Tokenize the same raw text with both tokenizers using `return_offsets_mapping`.
- For each sender token i: midpoint m_i of its span; find receiver token j whose span start ≤ m_i; fractional position p_i per SwaT eq. 12.
- Causal mask: receiver token j may attend sender token i only if sender span end ≤ receiver span end.
- Chat-template / special tokens handled separately (not through TA).
- Unit test: identical tokenizer on both sides must reduce to p_i = i and the identity mask. Test on 50 docs; report mean |char position mismatch| (should be ≪ 1 char).

### 3.3 Cache capture
Hooks on both models capturing, per layer:
- sender: K and V as stored. Also keys with RoPE removed (apply inverse rotation at the token's position). SwaT feeds de-RoPE'd keys.
- receiver (for targets): K **before** k_norm and RoPE (HeteroFold maps there) **and** K as stored (SwaT compares pre-RoPE keys but injects post-RoPE). Decide one target form per method and document it.
- For sliding-window sender layers: force the layer to retain the full KV for all positions during capture (SwaT §3, "Sliding-window layers"). Attention math itself is unchanged.

### 3.4 Cache injection into the receiver
Build a `DynamicCache` from predicted K/V per receiver layer, in the receiver's stored form (apply receiver k_norm if the stored form is post-norm, then receiver RoPE at integer receiver positions). Receiver then prefills only the query q on top and generates. For receiver sliding-window layers: predict full-length K/V, inject, let the layer drop what it doesn't need.

### 3.5 Data
Follow SwaT §4.1, scaled down for a first pass:
- **MSE / attention phase:** PG-19, WikiText-103, train splits of QASPER, QuALITY, HotpotQA, MuSiQue.
- **NLL phase:** add train splits of ARC-Challenge, HellaSwag, WinoGrande, MMLU auxiliary, GSM8K. Generate reasoning traces with the *receiver* (greedy) for GSM8K targets.
- First pass: ~5M tokens total. Full pass: ~44M (SwaT's number).
- **HeteroFold calibration set:** 1,600 train / 400 held-out prompts, half Open-R1 problem text, half HotpotQA train passages+questions. No gold answers.
- Each example is (document d, query q, target y). Document = long context or the short question; q = question + choices or empty (short tasks). Match SwaT's definitions in §4.1.

### 3.6 Eval harness
Start with 6 benchmarks, 500 samples each, greedy decoding:
- Long context: QuALITY (acc), HotpotQA LongBench subset (F1), QASPER (F1)
- Short: ARC-Challenge (acc, answer-token likelihood), MMLU 1k subset (acc), GSM8K (acc, final-number exact match)

For every benchmark compute:
1. **Receiver native** (receiver prefills d itself) — the denominator
2. **Sender alone** (for reference)
3. Each method: sender prefills d → mapper → receiver gets cache + q
Report score and retention %. Also report, for each method on 200 held-out prompts: per-layer attention KL (native vs transferred) and attention-output normalized squared error (HeteroFold Fig. 2 style). This is the debugging signal.

Also run HeteroFold's Figure-1 reconstruction test: 100 GSM8K questions, receiver asked to restate the question from the transferred cache only. Report word-level LCS F1.

## 4. Method A — HeteroFold

Follow the paper §3 and Appendix B exactly.
1. **Layer alignment:** proportional depth π(ℓ), three sender layers at offsets (−4, 0, +4) clipped to [1, L_S]. With a 1.5B sender (likely ~24–28 layers) and 31B receiver (likely 60+), several receiver layers will map to the same sender neighborhood. Fine; report the mapping table.
2. **Recolor:** per receiver layer ℓ and role c ∈ {K, V}: stack aligned pairs X (3·d_KV(S)) and Y (d_KV(R)), compute scalar RMS r, Z = X/r, weighted means/covariances, whiten → Procrustes R* via thin SVD of Σ_Z^{-1/2} Σ_ZY Σ_Y^{-1/2} → recolor: A = Σ_Z^{-1/2} R* Σ_Y^{1/2}, b = µ_Y − µ_Z A. Add eigenvalue floor for stability.
   - **If 3·d_KV(S) < d_KV(R)** (likely here): the covariance match is underdetermined. Use a ridge-regularized version (Σ_Z + λI) and report λ; alternatively widen to 5 sender layers (paper's Table D shows 3 ≈ 5). Flag this in the report — it is a real limitation of HeteroFold for a very small sender.
3. **Output-aware calibration:** rank-16 low-rank corrections B, G per (ℓ, c). Key loss = KL(P* ‖ P̂) + ‖(P̂ − P*)V* W_O‖² / d_model; value loss = ‖Ô − O*‖² / ‖O*‖², with P̂ held fixed (stop-grad) for the value loss. Losses at query positions in the question span only. AdamW lr 3e-4, wd 1e-4, clip 1.0, 4 epochs, keep checkpoint with lowest held-out combined loss.
4. **Fold:** A_final = A(I + BG), b_final = b(I + BG). Inference is one affine map per (ℓ, c).
5. Keys mapped **before** k_norm and RoPE; receiver applies its own k_norm and RoPE afterwards.

Expected cost: a few GPU-hours. Report wall time.

## 5. Method B — SwaT

Clone and read the reference repo first. Reuse it where it fits; port what doesn't (it was built for Qwen/Llama/Gemma-2/3 families, not Gemma 4 or Gauss).
- Architecture: B = 8 blocks, D = 1024, H = 16, F = 4096 (~300M params). Layer groups: contiguous partition of sender and receiver layers into 8 groups each (eq. 9). Block b reads S_b (de-RoPE'd keys and values of all sender layers in group b, concatenated across heads and layers → W_mem) and predicts K/V for all receiver layers in R_b via W_out.
- h_0 = W_in · E_R(receiver tokens) — frozen receiver embedding lookup, then a learned linear.
- Per block: LN → causal self-attn (integer receiver positions, RoPE) → LN → causal cross-attn into m_b (fractional sender positions from TA, TA causal mask) → LN → FFN (GELU). Residuals around each.
- Output: per-layer-per-head RMS rescale calibrated from receiver caches on training data, then receiver RoPE on keys at integer positions.
- **Phase 1:** RMS-normalized MSE to receiver native cache (keys compared pre-RoPE). 5,000 steps, AdamW lr 3e-4 cosine, wd 1e-3, clip 1.0.
- **Phase 2:** NLL of target y given (predicted cache, q, y_<t)), gradients through the frozen receiver into the mapper only. 24,000 steps, constant lr 3e-6.
- First pass: cut both phases to 1k / 4k steps to verify the pipeline, then run full.
- Receiver activation checkpointing on; sender runs under `no_grad`.

Report wall time per phase and the phase-1 loss curve.

## 6. Method C — Ours

Same architecture, data, and phase-2 as B. Two changes, each ablated:

**C1. Closed-form init.** Before training, initialize per block from HeteroFold's Recolor maps:
- W_mem_b: fit the Recolor map from the concatenated S_b features to the receiver's R_b features (one big affine map for the group, same whiten/Procrustes/recolor recipe), then factor it so that W_mem_b · (identity path through the block) · W_out_b ≈ that affine map. Simplest version: set W_mem_b to the Recolor map into the D-dim hidden via the top-D principal directions, and W_out_b to the matching reconstruction. Initialize attention and FFN residual branches near zero (zero-init output projections) so the block starts as an identity on top of the affine map. Document the exact factorization you use.
- Everything else as in B.

**C2. Attention-aware phase-1 loss.** Replace phase-1 MSE with: MSE + HeteroFold's per-layer key loss (attention KL + key-induced output error) + value loss, computed with the receiver's native queries on the question span. Same step count as B's phase 1.

**C = C1 + C2.** Then phase-2 NLL as in B.

Runs: B (baseline), C1 only, C2 only, C. Hypothesis to test: C reaches B's final retention with materially fewer phase-2 steps, and its phase-1 attention KL is lower than B's at equal steps. Log attention KL per layer at every eval checkpoint so this is visible.

## 7. Execution order and gates

1. §2 model inspection → report → **wait for my go-ahead**
2. §3.1 cache round-trip sanity test → must pass
3. §3.2 TA + unit tests → report mismatch numbers
4. §3.6 eval harness, run receiver-native and sender-alone baselines → report the denominator table
5. Method A end to end → report retention table + Fig-1 reconstruction + per-layer KL
6. Method B short pass (1k/4k steps) → check loss curves are sane → full B
7. C1, C2, C with the same step budgets as B
8. Final table: method × benchmark, score and retention %, plus wall-clock train time and per-layer KL summary. Save as `results/summary.md` and `results/summary.csv`.

Between gates, send a short status message with the numbers. Don't start a multi-hour training run without reporting the preceding gate first.

## 8. Things that will go wrong — handle, don't paper over

- **Stored-cache format mismatch** (pre/post k_norm, pre/post RoPE) will silently produce garbage with a plausible-looking loss. Sanity test §3.1 and a "receiver→receiver identity mapper" check (mapper = identity on the receiver's own cache must reproduce native logits) catch this.
- **Sliding-window layers** on either side: see §3.3/§3.4.
- **Very small sender:** 1.5B → 31B is a larger capacity gap than either paper tested (their smallest ratio ≈ 1:3.5 sender:receiver by params; this is ≈ 1:20). Expect retention below the papers' numbers, especially on short reasoning tasks (GSM8K, MMLU) where the receiver's own knowledge matters more than the context. Report it honestly; don't tune on the eval sets.
- **HeteroFold underdetermined covariance** — §4 item 2.
- **Receiver memory** during NLL backprop — reduce seq len before reducing batch to 0.

Deliverables: the code repo, `results/summary.md`, per-method loss curves, and a one-page `NOTES.md` listing every place you deviated from the papers and why.
