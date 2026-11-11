# Method D — Compact-cache output (Gauss 4 1.5B → Gemma 4 31B)

Brief for Claude Code. This extends `kv_transfer_brief.md`; everything there (shared infrastructure §3, Methods A/B/C, gates) still applies and is a prerequisite. Read both before writing code. Where this says "ask", stop and ask me.

## 0. Goal

Make the mapper emit a **compact** receiver cache of U entries directly, instead of one entry per receiver token followed by a separate compaction step. The sender (Gauss 4 1.5B) prefills the long document; the receiver (Gemma 4 31B) never sees the document text and decodes from U cache entries plus the query.

Headline claims to test, at a matched cache budget U:

1. D matches or beats **SwaT + Attention Matching** on task score, with sender-side cost in milliseconds instead of seconds.
2. D beats **text summarization** (sender writes a summary, receiver prefills it) at matched token budget.
3. D is competitive with **500xCompressor** run same-model on Gemma, while never running the 31B model on the document.

No KV quantization in this phase.

## 1. Prerequisites

- Gates 1–7 of `kv_transfer_brief.md` passed: cache round-trip sanity test, TA unit tests, receiver-native baselines, Methods A/B/C trained and evaluated.
- Use the **Method C checkpoint** as the starting point for D. If C did not beat B, use the B checkpoint and say so.
- SwaT's AM compaction must be runnable on the full-length C output (needed as a baseline). The AM code is referenced from SwaT's repo; if it's not there, implement from arXiv 2602.16284 and **ask** before spending more than a day on it.

## 2. Architecture change (relative to Method C)

Keep everything from C except the following.

**Input rows.** Replace h₀ = W_in · E_R(receiver tokens) (R rows) with h₀ = Z, a trainable matrix of shape U × D (D = 1024, same as the mapper width). Z is shared across all documents and learned. Initialize Z from the mean of W_in · E_R over training tokens plus small Gaussian noise, so slots start in the same region of space the C mapper already handles.

**Self-attention over slots.** Bidirectional (drop the causal mask). Positions for self-attention RoPE: integer 0..U−1.

**Cross-attention into sender memory m_b.** Every slot attends to every sender token: drop the TA causal mask. Keep the sender-side fractional positions from TA for the memory (they still carry order information). Slot-side query positions for cross-attention RoPE: spread evenly over the document's receiver-token range, i.e. slot u gets position u · R / U. Ablate against "no RoPE on slot queries" (see §6).

**Output.** W_out_b, per-layer-head RMS rescale, and receiver RoPE on predicted keys: unchanged. Receiver positions for the U cache entries: evenly spaced, position of slot u = round(u · R / U). The receiver's query q then starts at position R (as if the full document had been there). Ablate against compact positions 0..U−1 with q starting at U.

**Sliding-window receiver layers.** With U entries spread over positions 0..R, a sliding-window layer may see none or few of them. Handle by assigning the compact positions (0..U−1) for sliding-window layers only, or by placing the slots in the last window. Report which, and **ask** if Gemma 4's window layout makes neither clean.

Parameter count: C's ~300M minus W_in, plus Z (U × 1024, negligible).

## 3. Budgets

Train separate D models for U ∈ {32, 64, 128, 256}. Also train one **variable-U** model: at each training step sample U uniformly from {32, 64, 128, 256} and use the first U rows of a 256-row Z (nested slots). Evaluate it at each U. If it is within ~1 point of the fixed-U models, use it for everything downstream; it's one mapper instead of four.

Documents longer than the chunk size (set chunk = 2048 sender tokens for a first pass): the sender prefills the **whole** document once (so each chunk's KV carries cross-chunk context), then the mapper runs per chunk on that chunk's slice of sender KV, producing U slots per chunk. Concatenate slot sets in document order; receiver positions per chunk follow the same even-spacing rule within the chunk's token range. This is SwaT Appendix F's scheme.

## 4. Training

Two phases, both with gradients through the frozen receiver into the mapper only.

**Phase D1 — warm start (short).** Loss = NLL of the target y given (U-entry cache, q, y_<t). Learning rate 1e-4 constant, 2,000 steps, AdamW wd 1e-3, clip 1.0, bf16. Z, the cross-attention projections, and W_out train; freeze everything else from C for this phase. Purpose: let the slots learn to read the memory before the rest of the network moves.

**Phase D2 — full.** Same NLL loss, all mapper parameters trainable, lr 3e-6 constant (SwaT's phase-2 rate), 16,000 steps. Same data mix as C's phase 2 (`kv_transfer_brief.md` §3.5). Add a document-reconstruction auxiliary: with weight 0.1, train a small frozen-receiver decoding of "repeat the document" from the compact cache on 10% of batches. This discourages slots from collapsing onto only the question-relevant parts of training documents. Ablate it (see §6).

There is no MSE/attention-matching phase for D: with U ≠ R there is no token-aligned receiver target to match.

**Slot collapse check.** Every 1,000 steps compute, on 64 held-out documents, the pairwise cosine similarity among the U output key vectors at a middle receiver layer. If the mean off-diagonal similarity exceeds 0.9, the slots have collapsed; stop and **ask**. A cheap fix to try first: add an orthogonality penalty on Z with weight 0.01.

## 5. Baselines (all at matched budget)

For each benchmark and each U:

1. **Receiver native, full context** — the ceiling (denominator for retention).
2. **Receiver native, truncated** — receiver sees only the first U receiver tokens of the document. The "do-nothing" baseline.
3. **SwaT + AM** — Method B (or C) full-length output, then AM compaction to U entries. Record AM wall-clock on the sender GPU separately.
4. **T2T summary** — sender writes a summary with a prompt targeting ~U receiver tokens; receiver prefills summary + q. Use two summary prompts and pick the one whose realized length is closest to U. This is SwaT's T2T setup.
5. **500xCompressor, same-model on Gemma 4 31B** — follow arXiv 2408.03094: LoRA-tuned copy of the receiver as encoder, U compressed tokens, frozen receiver decodes from their KV. Train on the same data mix as D with the same step budget. This costs a full 31B forward+backward per step, so expect it to be the slowest run; use rank-16 LoRA and start it early. Report its training time next to D's.
6. **Method D** — fixed-U and variable-U.

## 6. Ablations (one pair of settings each, U = 64, on three benchmarks)

- Slot query RoPE positions: evenly spread vs none.
- Receiver positions for slots: spread over 0..R vs compact 0..U−1.
- Reconstruction auxiliary: on vs off.
- Init from C checkpoint vs init from scratch (random).
- Z init from mean embedding vs random.

## 7. Evaluation

Benchmarks: the six from `kv_transfer_brief.md` §3.6 plus LongBench NarrativeQA (F1) and SQuAD (F1), since those are where SwaT+AM gains most over text. 500 samples each, greedy.

Report per benchmark × U × method:

- score and retention % vs receiver native full context
- **cache entries** given to the receiver (should equal U, or U × chunks)
- **sender GPU time per request**: sender prefill + mapper (+ AM for baseline 3, + summary generation for baseline 4). Separate columns for prefill, mapper, AM/summary.
- **receiver TTFT**: time from receiving the cache to first output token, including prefill of q.
- receiver cache bytes

Diagnostics on 200 held-out docs: next-token KL between receiver-native-full and D at the first 16 answer positions; per-layer attention entropy over the U slots (flags dead slots).

Also run the HeteroFold Figure-1 reconstruction test from the compact cache: how much of a GSM8K question can the receiver restate from U = 32 slots. Report LCS-F1.

## 8. Execution order and gates

1. Implement §2 on top of the C codebase; run the identity check: with U = R and Z initialized to W_in · E_R(tokens) for one document, the D forward pass must reproduce C's output for that document to bf16 tolerance. **Report before training.**
2. Phase D1 at U = 64 only; check the slot-collapse metric and that NLL is falling. Report.
3. Phase D2 at U = 64; evaluate against baselines 1–4. **Report and wait** — if D is more than 5 points below SwaT+AM here, we rethink before spending GPU time on other budgets.
4. Start baseline 5 (500xCompressor) in parallel with step 5; it is the long pole.
5. Remaining U values and the variable-U model.
6. Ablations.
7. Final tables to `results/method_d_summary.md` and `.csv`, with a `NOTES.md` listing every deviation from this brief and why.

## 9. Known risks

- **Position assignment is the most likely silent failure.** If the receiver's RoPE sees slot positions that don't match where q starts, attention from q to the slots will be scaled wrong and the score will look like a weak mapper. The two position ablations in §6 exist to catch this; run them before concluding D is weak.
- **Slots overfitting to QA training distribution** (only storing what questions typically ask). The reconstruction auxiliary and the Figure-1 test are the checks.
- **AM baseline fidelity**: AM is per-cache optimized and may beat D on score at small U. That is an acceptable result if D is within a few points at ≥10× lower sender time; report both honestly.
- **Memory**: U × chunks entries can exceed the full-length cache for very short documents. Skip compaction when R ≤ U; note it in the eval.
