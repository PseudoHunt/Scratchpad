# Brief for Claude Code: Stage 2 recovery, range-split KV compression, Qwen2.5-7B on B300

Repo: github.com/PseudoHunt/kfactor (scripts: kfactor.py, fold_eval.py, rsplit.py, rsplit_prep.py, attn_profile.py, lbablate/). Reuse the existing range-split attention patch and diagnostics. Do not rewrite them. Paths are hard-coded to /home/kfactor, so change the ROOT variable first.

Only start this if the Qwen1.5-0.5B stage 2 run has passed its go/no-go. Use the same training script with a model flag.

## Question this run answers

Training-free, R32+LR at the same K-compression ratio is worse than plain truncation+window (on Qwen1.5-7B @8192: +0.78 vs +0.46 ΔPPL). Does 30M tokens of distillation close that gap on a modern GQA model? If it does not, range-split only buys absorbability, not quality.

## 0. Pre-flight (15 min)

1. Check that PyTorch supports B300 (CUDA 12.8 or newer, Blackwell). Run a bf16 matmul and an SDPA forward+backward. Do not depend on flash-attn wheels. Use SDPA.
2. Read the model's config.json and report the real values. Expected: 28 layers, 28 query heads, 4 KV heads, head_dim 128, rope_theta 1e6, bias on q/k/v projections. Use base Qwen/Qwen2.5-7B, not Instruct.

## 1. What differs from the earlier MHA models (handle all four)

- **GQA.** Total K width is 4 x 128 = 512 per token per layer, so the earlier r=600 is impossible. Use one joint latent over the 4 KV heads. Pick r so the K-bytes-per-token ratio matches the definition in rsplit/RESULTS.md (the 4.3x rows). Expect r around 120. Report the exact ratio. Seven query heads share each KV head, so fold the latent into each query head's projection and count that cost.
- **k_proj bias.** The key is xW + b. Fold b into the factorization (for example, append a constant 1 to the latent). Verify that full-rank, no-drop output matches the HF model to within noise before any compression.
- **rope_theta 1e6.** Pair wavelengths differ from Llama-2/Qwen1.5, so "drop pairs 0-31" is not validated here. Choose the cutoff from the training-free sweep below, using the delta-bucket error, not the pair index.
- **K-only compression on GQA.** The uncompressed KV cache is already about 7x smaller than MHA. Report K ratio and total KV ratio separately.

## 2. Stage 0: training-free sweep (30-40 min)

Configs: full, AW64, GW64, R16/R24/R32/R40/R48 (window + sinks exact), and RS (R-cutoff + LR). Evaluate PPL at 2048 and 8192. Also run the delta-bucket error, the attention-mass profile (sink and window coverage), and werr for layers 2-27. Put the results in a table next to the Qwen1.5-7B numbers.

Stop and report before training if R32 training-free ΔPPL @8192 is above about +1.0 (Qwen1.5-7B was +0.31). That is a proposed threshold, so adjust it if needed.

## 3. Stage 2 configs

All configs keep an exact 64-token window plus 4 sinks, and leave layers 0-1 uncompressed. All are matched on K-bytes-per-token.

| Name | Far-token keys (delta >= 64) |
|---|---|
| AW64 | low-rank truncation + window |
| GW64 | NoPE low-frequency (32 pairs) + window |
| RS | drop high-frequency pairs below the Stage 0 cutoff + low-rank, absorbed |
| RS+heads | RS with a per-head cutoff, chosen from calibration werr, budget-matched to RS |

I inferred these definitions from my notes. Check them against rsplit.py and fix any mismatch before running.

## 4. Training recipe

If the 0.5B script already fixes any of these, copy it and ignore my defaults.

- **Teacher:** same weights, full cache, frozen, bf16.
- **Loss:** KL(teacher || student) on logits. Optionally cache the teacher's top-128 logits once and reuse them for all four configs. That removes the teacher forward pass and its memory.
- **Trainable:** the compression factors plus LoRA (r=64) on q, v, o. Teacher and base weights are frozen.
- **Data:** 30M tokens per config. Same corpus as the 0.5B run. Pack to 4096, with about 20% of sequences at 8192. Keep eval documents disjoint from training.
- **Optimizer:** AdamW, bf16, lr 1e-4 for LoRA and 2e-5 for the factors, cosine schedule, 5% warmup, no weight decay, grad clip 1.0. Global batch about 128k tokens (about 230 steps). Use gradient checkpointing.
- **Run order:** one config at a time. Running all four at once fits in memory only with LoRA and checkpointing, and it does not reduce total time.
- **Throughput:** my estimate is about 15k tokens/s, so about 35 min per config and about 2.5-3 h in total. Measure it over the first 20 steps. If it is below 8k tokens/s, stop and report.
- **Checkpoints:** save at 25%, 50% and 100%. Run a quick PPL @2048 check at each one.

## 5. Evaluation after training

For full, each stage 2 config, and its training-free version:

- PPL on WikiText-2 and C4 validation at 2048, 8192 and 16384.
- Needle-in-a-haystack at 4k to 32k, 5 depths, 10 trials each.
- A RULER subset at 8k and 16k: niah_single_1/2/3, niah_multikey_1, vt, cwe.
- Report the K ratio and total KV ratio next to every number.

This measures quality at matched compression. No decode kernel is involved, so there are no speed claims.

## 6. Decision rule (proposed, adjust to match the 0.5B thresholds)

Proceed to a paper if all of these hold:

1. RS is within +0.15 ΔPPL @2048 and +0.4 @8192 of full.
2. RS is at or below AW64 and GW64 @8192.
3. NIAH accuracy is at least 95% of full up to 16k.
4. RS+heads is at least as good as RS. If it is not, drop the per-head variant.

If RS only ties AW64, the contribution reduces to analysis. Say so plainly in the report. Use one seed first. If any gap between configs is under 0.1 PPL, rerun with two more seeds.

## Deliverable

A short report with the Stage 0 table, the four stage 2 curves, the eval table, the exact ratios, and any mismatch you found against my config definitions.
