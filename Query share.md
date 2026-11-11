# Read-in-place cross-model cache reading: kill test

Brief for Claude Code. Read fully before writing code. Where this says "ask", stop and ask me before continuing.

## 0. What we are testing

Every published cross-model KV transfer method (SwaT arXiv 2610.15095, HeteroFold arXiv 2609.32259, Heo et al. arXiv 2608.03893, CacheBridge arXiv 2609.00891) translates the sender's KV cache into the receiver's format: one mapped key and value per document token, per receiver layer.

We do the opposite. The sender's cache stays exactly as the sender wrote it. The receiver gets a small trained **reader** per layer that:

1. maps the receiver's query into the sender's key space, so the receiver can score the sender's keys directly;
2. maps the attention-weighted sum of the sender's values back into the receiver's value space.

No translated cache is ever built. The receiver never sees the document text.

This brief is a **kill test**: one public model pair, about one GPU-day of training, one go/no-go decision. Do not build beyond what the gates ask for.

## 1. Setup

- **Sender (S):** `Qwen/Qwen3-4B` (non-thinking mode)
- **Receiver (R):** `meta-llama/Llama-3.1-8B-Instruct`
- Both frozen, bf16. Only reader parameters train.
- Hardware: one B300. Both models fit together with room to spare.
- This is HeteroFold's exact pair, so its Table 1 numbers are our reference. SwaT reports Qwen 3 4B → Llama 3 8B (3.0, not 3.1), so treat its numbers as indicative only.
- If code from `kv_transfer_brief.md` exists in this workspace (cache capture hooks, eval harness, HeteroFold reimplementation), reuse it. If not, build only what this brief needs.

Reference retention (transfer score ÷ receiver-native score), computed from the papers:

| Benchmark | HeteroFold | SwaT |
|---|---|---|
| QASPER (F1) | 65% | 104% |
| HotpotQA (F1) | 67% | 105% |
| QuALITY (acc) | 84% | 100% |
| GSM8K (acc) | 64% | 90% |
| ARC-C (acc) | 95% | 106% |

## 2. Gate 0: inspect and report (before any code)

From both `config.json` files and the modeling code, report: layer count, hidden size, query heads, KV heads, head dim, RoPE theta, any q/k normalization, and **what exactly is stored in `past_key_values`** (I expect post-RoPE keys and raw values; confirm). Also report bytes per token of KV cache for each model and the ratio.

**Ask** if either model stores keys in a form other than "k_proj output with RoPE applied".

## 3. The reader

Notation: receiver layer ℓ, receiver query head h (H_R of them), receiver KV head g(h) (G_R of them), receiver head dim d_R. Sender KV heads G_S, sender head dim d_S.

### 3.1 Sources
Each receiver layer ℓ reads three sender layers: the proportional-depth match π(ℓ) = round(ℓ · (L_S − 1) / (L_R − 1)) and offsets −4 and +4, clipped to valid layers (HeteroFold's neighborhood). From each, all G_S KV heads. So M = 3 · G_S sources.

Concatenate the sources per document token:
- wide key K_cat[i] ∈ R^(M·d_S): the stored sender keys of the M sources at sender token i
- wide value V_cat[i] ∈ R^(M·d_S): same for values

These are views into the sender's cache. Do not copy or modify the cache.

### 3.2 Query side
Let q_h be the receiver's query vector for head h **before** the receiver's RoPE (d_R dims).

- q_wide = A_{ℓ,h} · q_h, with A_{ℓ,h} ∈ R^((M·d_S) × d_R), trainable.
- Apply the **sender's** RoPE (sender theta, sender head dim) to each of the M blocks of q_wide, at the query token's position in the sender's frame (see 3.5).
- Foreign score for document token i: s_for[i] = τ_{ℓ,h} · (q_wide · K_cat[i]) / sqrt(d_S) + β_{ℓ,h}. τ and β are trainable scalars (init τ = 1, β = 0).

The stored sender keys already carry the sender's RoPE at their own positions, so this is an ordinary RoPE dot product in the sender's frame. Nothing is de-rotated.

### 3.3 Joint softmax
The receiver also has its own tokens (the question, the template, generated tokens) with native keys and values. Native scores s_own are computed exactly as the receiver normally does. Attention weights are one softmax over the concatenation [s_for ; s_own].

### 3.4 Value side
- Native part: Σ_j p_j · v^R_{g(h), j} over own tokens, as usual.
- Foreign part: u = Σ_i p_i · V_cat[i] (one vector of M·d_S dims), then B_{ℓ,g(h)} · u + c_{ℓ,g(h)} · (Σ_i p_i). B ∈ R^(d_R × (M·d_S)) and c ∈ R^(d_R) are trainable, one per receiver KV head (as native values are).
- Head output = native part + foreign part, then the receiver's own o_proj, unchanged.

### 3.5 Positions and visibility
- Sender prefills [its template prefix + document d] with its own tokenizer. Let P_S be the sender position after the last document token.
- The receiver's own sequence is its chat template with the document slot left empty: [prefix tokens, question q, assistant header, then generated tokens].
- Only receiver tokens **after** the empty document slot may read the foreign cache. Prefix tokens before the slot get no foreign scores.
- A receiver token that is the n-th token after the slot uses sender-frame position P_S + n for the query rotation in 3.2.
- No token alignment between tokenizers is needed anywhere.

### 3.6 Size
Report the trainable parameter count. If it exceeds about 400M, factor A and B as rank-64 products and report that you did.

Init: A and B Gaussian std 0.02, c = 0.

### 3.7 Implementation notes
- Patch the receiver's attention forward. The foreign part needs explicit scores (no flash-attn); the native part can stay as is if you combine logits before softmax, otherwise do the whole layer manually.
- The sum over sources is one matmul because of the concatenation: [H, T, M·d_S] × [S, M·d_S]ᵀ.
- Generation must work with the HF cache for the receiver's own tokens while the foreign cache is held separately and passed to every layer.

## 4. Gate 1: plumbing identity check (must pass before training)

Use the receiver as its own sender: R prefills [prefix + d], then a second R instance path reads that cache in place with sources = the same layer only, A = identity (per KV-head grouping), B = identity, c = 0, τ = 1, β = 0. Next-token logits on the question tokens must match the native run on [prefix + d + q] to bf16 tolerance.

This proves the joint softmax, the position bookkeeping and the source-frame RoPE are right. Report max abs logit difference. Do not continue until it passes.

## 5. Data

Each example is (document d, question q, target y).

- Long context: train splits of QASPER, QuALITY, HotpotQA, MuSiQue. d = context, q = question (+ options), y = gold answer.
- Continuation: PG-19 and WikiText-103 chunks. d = first 1–2K tokens, q empty, y = next 128 tokens.
- Short: ARC-Challenge and GSM8K train. d = the full question (+ options), q empty, y = answer (for GSM8K, a greedy reasoning trace from the receiver, kept only when its final number is correct).
- First pass about 5M tokens total. Document length capped at 4K sender tokens.
- Evaluation sets must not overlap training. LongBench HotpotQA and QASPER subsets come from different splits than the train splits; verify by text hash and drop any overlap.

## 6. Training

### Phase 1: local attention-output distillation (no backprop through the model)
Teacher: the receiver run natively on [prefix + d + q + y], with hooks capturing, for every layer, the attention sublayer input and output at the tokens after the document, plus those tokens' native K and V.

Student, per layer independently: feed the teacher's attention input at those tokens into the reader-augmented attention (own-token K/V from the teacher, foreign cache from the sender) and compare the attention output:

L_ℓ = ||O_ℓ − O*_ℓ||² / ||O*_ℓ||²

Layers train independently, so batch them or run them in parallel. AdamW, lr 1e-3 with cosine decay, wd 1e-4, 3,000 steps. Log L_ℓ per layer, next to the same error when the foreign part is switched off (the "no document" error). The ratio tells us how much of the document's effect each layer recovers.

### Phase 2: end-to-end
Receiver runs free on [prefix + q + y] with readers in every layer, reading the sender cache. Loss = KL(native teacher next-token distribution ‖ student) on y positions + 0.5 · NLL of y. Gradients pass through the frozen receiver into all readers. lr 1e-4 for 500 warm-up steps then 3e-5, 8,000 steps, clip 1.0, activation checkpointing if needed.

## 7. Evaluation

Benchmarks, greedy decoding, same prompts for every row:
- QASPER and HotpotQA (LongBench 200-item subsets, token F1)
- QuALITY dev, 500 items (accuracy)
- GSM8K test, 500 items (final-number exact match)
- ARC-Challenge test (answer-token likelihood)

Rows:
1. Receiver native (prefills d itself): the denominator
2. Receiver with no document (question only): the floor
3. Sender alone
4. Read-in-place after Phase 1 only
5. Read-in-place after Phase 2
6. **Shuffled donor**: same as 5 but the sender cache comes from a different document. Must fall to about row 2. If it does not, the reader is not using the cache.
7. HeteroFold on this pair: run our reimplementation if it exists; otherwise list the published retention from §1.

Report score and retention for each.

Cost measurements at 4K, 16K and 32K document tokens:
- hand-off time from "sender finished prefill" to "receiver ready to decode" (expected: about zero)
- receiver time to first token
- receiver per-token decode latency, vs native decode at the same context length
- KV bytes held for the document, vs what the receiver's native cache would hold

## 8. Decision rule

- **PASS:** row 5 retention ≥ HeteroFold on all three long-context benchmarks (65% / 67% / 84%), and the shuffled-donor control collapses.
- **STRONG PASS:** mean long-context retention ≥ 90%.
- **Below the bar:** run one rescue arm before calling it. Replace the linear query map with q_wide = A·q_h + MLP(x), where x is the receiver's normed attention input and the MLP is D_R → 256 → M·d_S with GELU, zero-init last layer. Retrain both phases. If still below HeteroFold, stop and write up what the per-layer Phase 1 errors show.

Report after every gate. Do not start Phase 2 before reporting Phase 1 results.

## 9. Ablations (only after a PASS, each at the Phase 2 budget)

- Sources: 1 sender layer vs 3
- Source pruning: keep the 2 sources per receiver head with the largest ‖A block‖, fine-tune 1,000 steps; report retention and decode latency
- Phase 1 only vs Phase 2 only vs both
- Linear vs MLP query map (if not already run as the rescue arm)

## 10. Things likely to go wrong

- **Position frame errors** look exactly like a weak reader. Gate 1 exists to rule them out; rerun it after any refactor of the attention patch.
- **Score scale mismatch** between foreign and native logits can starve one side of the softmax. Log the mean attention mass on document tokens per layer, for teacher and student. They should be close after Phase 1.
- **Short-task collapse:** on ARC and GSM8K the receiver's only view of the question is through the reader. Expect these to lag the long-context tasks; report them but do not gate on them.
- **Phase 1 looks good, Phase 2 start is bad:** errors compound across layers once teacher forcing is removed. That is what Phase 2 is for; do not judge the method on the first few hundred Phase 2 steps.

## 11. Deliverables

- `results/read_in_place_summary.md` and `.csv`: the §7 table, the cost table, the decision
- per-layer Phase 1 error plot and document-attention-mass plot
- `NOTES.md`: every deviation from this brief and why
