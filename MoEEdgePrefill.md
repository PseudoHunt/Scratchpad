# MoE Edge Prefill — Phase 1 Kill Tests (Claude Code brief)

As of 2026-10-23

## Context and goal

Phase 1 is three measurement-only kill tests for a new MoE deployment method. No system building yet. Each test ends in a go/no-go number. Total GPU budget: at most 4 hours on one 48 GB GPU.

The method under test (for your context, not for implementation now):

1. **Prefill expert admission under a byte budget.** On edge devices, prefill streams nearly the whole expert pool because thousands of prompt tokens activate almost every expert. We want to load experts in demand order up to a byte budget, and handle the tail with substitutes.
2. **Peer-combination surrogates.** A non-admitted expert D is approximated as a fitted linear mix of the experts the token already uses: D(x) ≈ Σ_j α_Dj·E_j(x). Fitted by least squares. A weighted variant fits the error only along the directions the next layer's router reads (routing-preserving).
3. **Routing-projected sketches for cross-layer prefetch.** A cross-layer predictor applies layer N's router to the state at layer N-1. Its error comes from the MoE output of layer N-1 and attention at layer N, which it ignores. We want to know how much of that error the MoE term explains.

Phase 1 answers three questions:

- KT1: Is the per-layer expert demand histogram in prefill long-tailed enough that a byte budget saves real bytes?
- KT2: Can an expert's output be reconstructed from its co-routed peers, and does the routing-weighted fit cut next-layer routing flips?
- KT3: Is the MoE term the dominant error of a cross-layer predictor?

If all three pass, Phase 2 builds the method. If KT1 fails, the whole direction stops. KT2 and KT3 can fail independently and the method shrinks accordingly.

## Models, data, hardware

Run everything on OLMoE-1B-7B first, end to end. Only then repeat on Qwen3-30B-A3B. Do not start Qwen3 until all three OLMoE reports exist on disk.

| Model | Experts / top-k | MoE layers | Hidden dim | Load as | Role |
| --- | --- | --- | --- | --- | --- |
| allenai/OLMoE-1B-7B-0924 | 64 / 8 | 16 | 2048 | bf16 (~14 GB) | Primary, fast iteration |
| Qwen/Qwen3-30B-A3B | 128 / 8 | 48 | 2048 | int8 or bf16 with CPU offload of experts | Confirmation only |

If Qwen3-30B-A3B does not fit, substitute openai/gpt-oss-20b (32 experts, top-4) and say so in the report.

**Calibration set (KT2, KT3):** 200 sequences of 512 tokens each, about 100k tokens, equal thirds from WikiText-103 (prose), a code corpus (e.g. the-stack-smol, Python), and UltraChat (chat). Fix the seed. Save the tokenized set to disk once.

**Prefill prompt set (KT1):** 60 prompts, 2k to 8k tokens, equal thirds from the same three domains (long PG-19 or Wikipedia passages, whole source files, multi-turn chat transcripts). Also run each prompt twice: bare, and with a shared 300-token system prompt prepended.

**Hardware rules:**

- One GPU, up to 48 GB. Instances can be reclaimed without warning. Write every per-layer result to disk as soon as it exists, and make every script resumable from the last saved layer.
- Never store per-token expert outputs across layers. Accumulate statistics in a streaming pass (Gram matrices, counts). Memory per layer must stay under a few hundred MB.
- Use HF Transformers with eager attention. Hook the MoE block forward, do not rewrite the model.
- Target under 4 GPU-hours total. Report actual GPU time used per test.

## Instrumentation

Write one hook per MoE block that exposes, for every token in the batch, the quantities below. All three tests read from the same hook.

Per MoE layer l, per token x:

- `h_in`: the hidden state entering the block (post-attention residual, before the MoE pre-norm).
- `router_logits`: the full E-dim router logits of this layer, computed on norm(h_in).
- `topk_ids`, `topk_gates`: the k selected experts and their normalized gate weights.
- `expert_out[j]` for each selected j: that expert's raw output on this token, before the gate weight is applied. Needed only during the layer's own pass; never kept across layers.
- `moe_out`: Σ_j gate_j·expert_out[j] (plus shared expert output if the model has one, kept separate).
- `h_out`: `h_in + moe_out`.

Also read once per model: every layer's router weight matrix `W_g[l]` (E × d) and its pre-norm scale.

**Streaming Gram accumulation for KT2.** For each layer, keep per expert D:

- `n_D`: token count.
- `mean_D`: running mean of `expert_out[D]` (d-dim).
- `G_D`: an E × E matrix of accumulated dot products between co-routed peer outputs, `G_D[j,k] += z_j·z_k`, where z_j = expert_out[j] for peers j, k of a token routed to D (zeros for experts not co-routed on that token).
- `b_D`: an E-dim vector, `b_D[j] += z_j·expert_out[D]`.
- `GM_D`, `bM_D`: the same two, but with every output first multiplied by `M = W_g[l+1]` (the next layer's router matrix, E × d), so dot products are taken in router-score space. For the last MoE layer use the LM head's top-1000 rows by norm as M, or skip the weighted fit for that layer.

Center every output by subtracting the expert's `mean` from a first pass, so no constant term is needed. Two passes over the calibration set: pass 1 means and counts, pass 2 Grams. This is 16 layers × 64 experts × (64×64) floats for OLMoE, trivial.

Save hook outputs as `.npz` per layer under `runs/<model>/<test>/layer_<l>.npz`. Log GPU seconds per layer.

## KT1 — Prefill demand histogram tail

Question: in a single long prompt, how many experts per layer serve almost no tokens? Every expert weighs the same, so the fraction of experts is the fraction of bytes.

Procedure, per prompt, per MoE layer:

1. Run prefill once (no generation). From `topk_ids`, count `demand[e]` = number of token-slots routed to expert e. Total slots = tokens × k.
2. Sort experts by demand, descending. Compute the cumulative share of slots covered by the top m experts, for every m.
3. Record: `frac_experts_lt_1pct` (experts whose demand is under 1% of slots), `frac_experts_lt_5pct`, and `m_90`, `m_95`, `m_99` (the smallest m covering 90/95/99% of slots), each as a fraction of E.
4. Repeat with the 300-token system prompt prepended and report the same numbers.

Aggregate: median and interquartile range across prompts, per layer and per domain. One plot: cumulative slot coverage vs fraction of experts loaded, one curve per layer, median over prompts. One table: per domain, median `frac_experts_lt_1pct` and median `m_95`.

Also record, for the tail experts (demand under 1%), how far their routed tokens' router score sits above the (k+1)-th candidate. Report the median margin as a fraction of the top-1 score. This is a preview of how cheap substitution is.

**Go/no-go.** Go if the median layer has `frac_experts_lt_1pct ≥ 0.30` or `m_95 ≤ 0.70·E`. No-go if both fail on the median layer for all three domains. Report the verdict per domain; a go in two of three domains is a go with a note.

## KT2 — Peer-combination surrogate and routing-preserving fit

Question: when expert D is missing, how well can the token's other selected experts reconstruct D's output, and does fitting in router-score space cut routing flips in the next layer?

**Fit (offline, closed form).** For each layer and each expert D, solve from the Grams of the Instrumentation section:

- Plain: `α_D = argmin Σ_x ‖D(x) − Σ_j α_Dj z_j(x)‖²` → `α_D = (G_D + λI)⁻¹ b_D`, λ = 1e-3·trace(G_D)/E.
- Weighted: same with `GM_D`, `bM_D` (errors measured after multiplying by M = next-layer router).

Split calibration tokens 80/20 by sequence. Fit on 80, evaluate on 20.

**Substitution rules to compare** (all applied to the same held-out tokens, replacing D's contribution only):

| Rule | What replaces gate_D·D(x) |
| --- | --- |
| Drop | nothing; renormalize remaining gates to sum to 1 |
| Best single peer (SERE-like) | gate_D·E_j*(x), j* = the co-routed peer with highest plain R² for D |
| Plain LS mix | gate_D·Σ_j α_Dj·z_j(x) |
| Weighted LS mix | same with the weighted α |

**Metrics, per layer, median across experts (weight experts by token count):**

1. `R2_plain`, `R2_weighted`: held-out fraction of D's output variance explained, measured in raw space and in M-space respectively. Also report plain-α R² measured in M-space, so the two fits are compared on the same yardstick.
2. `flip_rate`: fraction of substituted tokens whose next-layer top-k set changes (Jaccard < 1) under each rule. Compute by actually running the next layer's router on the perturbed `h_out`. Do not propagate further than one layer in this test.
3. `flip_rate_at_equal_residual`: bin tokens by plain-space residual norm (5 bins) and report flip rate per bin for plain vs weighted. This is the comparison that matters.
4. `margin_vs_flip`: for substituted tokens, flip rate as a function of the next-layer predicted margin (k-th minus (k+1)-th score), 10 bins. This calibrates the certificate used in Phase 2.

One plot: flip rate per residual bin, plain vs weighted, one panel per 4 layers. One table: per layer, median R2_plain, median R2_weighted, flip rate for the four rules.

**Go/no-go.**

- Surrogate go: median `R2_plain ≥ 0.70` on at least half the layers. Fall back to best-single-peer if 0.40–0.70. Below 0.40 on most layers: substitution is not viable, keep only drop with importance gating.
- Routing-preserving go: weighted LS flip rate is at most half of plain LS flip rate in at least 3 of 5 residual bins, on at least half the layers. If it fails, the direction-dependence claim is weak; keep plain LS and drop the routing-preserving framing.

## KT3 — Cross-layer predictor error decomposition

Question: a cross-layer predictor guesses layer N's experts from the state at layer N-1. Its error comes from two things it skips: the MoE output of layer N-1, and attention at layer N. Which one dominates?

For each pair of consecutive MoE layers (N-1, N) and each calibration token, compute three next-layer top-k predictions, all using layer N's real router `W_g[N]` and pre-norm:

1. **Plain cross-layer gate**: `W_g[N]·norm(h_in[N-1])`. This is what Fate-style predictors do.
2. **Plus true MoE term**: `W_g[N]·norm(h_out[N-1])`, i.e. the state after layer N-1's MoE but before layer N's attention. This is the best any MoE-side sketch could reach.
3. **Truth**: `router_logits[N]`, the real routing.

Metrics per layer pair, averaged over tokens:

- `hit@k`: fraction of truth's top-k experts present in the prediction's top-k, for (1) and (2).
- `hit@2k`: same with the prediction's top-2k (the prefetch-with-slack case).
- `gap_closed = (hit_2 − hit_1) / (hit_3 − hit_1)` where hit_3 = 1 by definition. This is the fraction of the predictor's error the MoE term explains.
- Per-expert routing influence: for each expert e at layer N-1, `‖W_g[N]·(E_e(x)·gate_e)‖ / ‖W_g[N]·h_in‖` averaged over its tokens. Sort experts by it. Report the ratio between the top-10% and bottom-10% experts. A ratio above 3 means "routing shapers" and "routing neutral" experts really exist.
- Margin check: among tokens where prediction (2) is wrong, what fraction had a predicted margin under 5% of the top-1 score? High means errors concentrate at near-ties, which a certificate can catch.

One plot: `hit@k` for (1) and (2) per layer pair. One table: per layer pair, hit@k for (1), (2), gap_closed, influence ratio.

**Go/no-go.** Go if median `gap_closed ≥ 0.60` across layer pairs. If `gap_closed` is between 0.30 and 0.60, the sketch helps but attention is the larger error; keep only the certificate part of the Phase 2 design. Below 0.30: drop the prefetch fusion entirely.

## Deliverables and run order

Run order, strictly: hook + smoke test on 2 sequences → KT1 OLMoE → KT2 OLMoE → KT3 OLMoE → write `runs/olmoe/REPORT.md` → only then Qwen3-30B-A3B, same order.

Deliverables per model, under `runs/<model>/`:

- [ ] `data/`: tokenized calibration set and prompt set, with the seed and source splits recorded.
- [ ] `kt1/`: per-layer `.npz` of demand histograms, `coverage.png`, `kt1_table.md`, verdict line.
- [ ] `kt2/`: per-layer Grams and fitted α (plain and weighted), `flip_vs_residual.png`, `kt2_table.md`, verdict lines (surrogate, routing-preserving).
- [ ] `kt3/`: per-layer-pair `.npz`, `hitk.png`, `kt3_table.md`, verdict line.
- [ ] `REPORT.md`: the three verdicts first, each as one sentence with its number. Then the tables. Then GPU time per test. Then anything that surprised you, in at most five bullets.

Rules for the report: numbers with units, no adjectives. If a number is from a reduced run (fewer prompts, fewer layers), say so next to it. If a test could not be completed, say which step failed and what was saved.

**Do not** build the admission system, the sketches, the CPU tier, or any end-to-end latency measurement. Those are Phase 2 and depend on these verdicts.

**Phase 2 preview (not for now):** if KT2 passes, fit rank-8 and rank-16 low-rank patches on the weighted residual and re-measure flip rate. If KT3 passes, fit per-expert sketches `S_e: x → W_g[N]·E_e(x)` at rank 8–16 and measure the corrected predictor's hit@k against (1) and (2). Then end-to-end prefill admission under byte budgets with LongBench and GSM8K quality, and TTFT on a throttled-bandwidth or CPU-offload setup.
