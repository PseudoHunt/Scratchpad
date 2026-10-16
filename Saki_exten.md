# Handoff: attention-weighted SAKI index, then sparse attention in llama.cpp for KV caches that don't fit in GPU memory

Self-contained brief for a new Claude session (chat and Claude Code) with no history. Read fully before touching code.

---

## 1. Summary

Two stages. **Stage A** tests one change to SAKI's low-rank key index: weight its objective by attention mass so the basis concentrates on the keys that actually get attended to. It is a one-afternoon offline experiment on existing tensors. **Stage B** uses whichever index wins (attention-weighted SAKI, or plain SAKI if A fails) inside llama.cpp, for the case where the KV cache does not fit in GPU memory and attention runs on the CPU from DDR. That is the one regime where index quality converts directly into tokens per second. Stage B is the paper; Stage A decides whether the index is a contribution or a cited baseline.

## 2. Background

- **Loki** (arXiv 2406.02542): rank-r PCA basis on pre-RoPE keys, same basis for queries and keys, scores queries against projected keys, keeps top-k. About 1.45x attention-only speedup, no end-to-end gain on a GPU-resident cache.
- **CrossCov-U** (github.com/8BitSpacemanSpiff/loki-crossconv): same mechanics; basis from the SVD of the aligned cross-moment C = (1/N) Σₙ kₙqₙᵀ, pooled per KV head for GQA. Beat Loki on Mistral-7B WikiText-2, especially at aggressive compression. **In the owner's I0 test, SAKI beat CrossCov-U on recall at matched rank.** CrossCov-U is therefore a baseline, not a contribution.
- **SAKI** (arXiv 2608.03228, Aug 2026): minimizes expected attention-score distortion from rank-r key compression; closed-form asymmetric factorization from the SVD of a covariance-weighted query-key operator; random-matrix noise correction; 512 calibration tokens; beats key-PCA at every rank on four models (LLaMA-3.1-8B recall 0.748 to 0.799 at r=32). Reports recall only, no timing. **Read the paper first and confirm the exact operator and whether keys are pre- or post-RoPE.** The reconstruction below is from the abstract: summing the score distortion over all query-key pairs gives ‖Q(I−PPᵀ)Kᵀ‖²_F = tr((I−PPᵀ) R_k (I−PPᵀ) R_q) with R_k = E[kkᵀ], R_q = E[qqᵀ], optimized by the SVD of R_q^{1/2} R_k^{1/2} with asymmetric maps on the two sides. If the paper's operator differs, use the paper's.
- **Serving facts that motivate Stage B:** sparsity only becomes throughput when the full cache leaves HBM; Fathom (2609.17652) beats Loki-class index scans by 1.67x only when offloaded; break-even for GPU-resident sparse decode is about 10x sparsity (2605.24168). llama.cpp keeps KV for non-offloaded layers in CPU RAM and runs attention there with ggml kernels, DDR-bandwidth bound: for an 8B GQA model at 128K context the per-step KV read is about 16 GB, roughly 0.25 s per token at 60 GB/s. A selector touching 3-5% of the cache cuts that 20-30x with no PCIe transfer.
- **Prior art for Stage B:** RetrievalAttention (CPU ANN index), MagicPIG (2410.16179, LSH sampling on CPU), ShadowKV (2410.21465), InfiniGen, Fathom: all server offload with GPU attention. KTransformers, HeadInfer: CPU-GPU hybrids for weights and heads. TierKV (2609.21172), KVSwap (2511.11907): on-device tiering and swapping, not score-aware selection. The context-intensive failure study (2604.08426) shows low-rank keys degrade on context-intensive tasks; include those tasks. Search llama.cpp issues, PRs and forks for sparse attention or top-k KV before claiming there is none.

## 3. Stage A: attention-weighted SAKI (offline, about 1 GPU-hour)

### 3.1 The change
Replace R_k in SAKI's objective with the attention-mass-weighted key second moment

    R̃_k = Σⱼ w̄ⱼ kⱼ kⱼᵀ / Σⱼ w̄ⱼ,   w̄ⱼ = mean over calibration queries of the softmax attention weight on key j

and solve the same closed form with R̃_k in place of R_k. Rationale: top-k recall is decided by high-score pairs; SAKI weights all pairs equally. Variants to include: (a) w̄ⱼ from full softmax weights; (b) w̄ⱼ = fraction of queries for which key j is in their top-k (k = 5% of context); (c) tempered, w̄ⱼ^α with α in {0.5, 1}. Also a query-side analogue (weight R_q by each query's attention entropy or max weight) as a secondary variant.

### 3.2 Data
Reuse the owner's saved tensors (pre-RoPE keys and queries; the CrossCov-U repo's saver format). Add two long-context sets if not already captured: 10 RULER-style prompts at 16K and 32K, and 5 context-intensive prompts of the 2604.08426 type. Fit on held-in chunks, evaluate on held-out chunks, same split as the CrossCov-U README diagnostics. Models: Mistral-7B-v0.1 (existing tensors) and LLaMA-3.1-8B-Instruct (SAKI's main model).

### 3.3 Rows and metrics
Rows: PCA (Loki), CrossCov-U pooled, SAKI as reconstructed, SAKI with the paper's operator if it differs, attention-weighted SAKI variants (a), (b), (c). Ranks 8/16/32. Selection k = 5% and 12.5% of context, per KV head, runtime-fixed k.
Metrics per row: top-k recall, attention-mass recall, attention-output relative L2 against dense, and bytes-to-95%-recall (the smallest k at which recall reaches 0.95, times bytes per token). Paired per (layer, head, query) with bootstrap intervals; report per-layer too, since SAKI's gains concentrate in deeper layers.

### 3.4 Gate
Pass: the best attention-weighted variant beats plain SAKI on bytes-to-95%-recall by at least 10% at r=16 on both models, with the interval clear of zero, and does not lose on the context-intensive set. Then the index is a contribution and goes into Stage B under its own name. Fail: use plain SAKI in Stage B, cite it, and report the negative in the appendix.

Before writing anything, do one targeted prior-art search: "attention-weighted low-rank key", "importance-weighted PCA KV cache", "Fisher-weighted SVD KV cache" (Palu, LoRC, ReCalKV use activation or Fisher weighting for KV *compression*; check that none does it for a retrieval index).

## 4. Stage B: sparse attention in llama.cpp when the KV cache doesn't fit

### 4.1 Step B0: is there a ceiling worth chasing (one afternoon, no new code)
Run stock llama.cpp on the work machine with a model whose KV cache exceeds VRAM, or with `--no-kv-offload`, at 32K, 64K and 128K context, Q8_0 or F16 KV. Profile a decode step (ggml's built-in timing or `perf`). Record the share of step time in CPU attention. If it is under 50% at 128K, stop: the ceiling is too low. If it is 70%+ at 128K, continue. Record DDR bandwidth achieved versus the machine's peak.

### 4.2 Step B1: PyTorch emulation of the regime (about 10 GPU-hours, mostly CPU)
Before any C++, emulate the system in PyTorch: KV on pinned host memory, index (rank-r projected keys, fp16) in host or GPU memory, attention computed on the CPU over the selected tokens only. Models: Llama-3.1-8B-Instruct, Qwen3-8B. Contexts 32K/64K/128K. Methods: dense CPU attention; PCA index; SAKI index; the Stage A winner. Selection per GQA group, k in {2%, 5%, 10%}; reuse the previous step's selection for a fraction of layers (IndexCache/PIVOT style) as an ablation.
Metrics: RULER 32K and 64K, two context-intensive tasks, LongBench subset; decode step time; bytes of KV read per step; index scan time. Pass: the winning index reaches dense accuracy (within 1 point) at a k where step time is at least 5x below dense CPU attention, and beats the PCA index on accuracy at matched k.

### 4.3 Step B2: ggml implementation (2-4 weeks engineering; a junior can own it after B1)
llama.cpp specifics to handle:
- Keys are stored post-RoPE and often quantized (Q8_0/Q4_0). Compute the rank-r index from the **pre-RoPE** key at write time, store it as a separate fp16 tensor per layer and KV head. For a rank-32 index this is 64 bytes per token per head.
- Query side: project the pre-RoPE query with the index's query-side map each step (small matmul, negligible).
- Selection once per GQA group so the gather is contiguous; gather at page granularity (16 or 32 tokens) to keep CPU reads sequential.
- A new ggml op or a modified flash-attention CPU path that takes a token index list; the sink tokens and the recent window are always included.
- Keep positions original. Handle the KV cache's ring-buffer and sequence-slot semantics; test with `llama-cli` and `llama-server`.
- Expose `--sparse-kv-rank`, `--sparse-kv-topk`, and an off switch.
Measurement: tokens/s and accuracy at 32K/64K/128K on at least two machines (the work box and one consumer-class laptop with unified memory if available), dense vs sparse, F16 and Q8_0 KV. Report where time goes after the change.

### 4.4 Paper
One line: "Score-aware sparse attention for KV caches that don't fit in GPU memory: real tokens/s on consumer hardware at 128K context." Headline numbers: tokens/s dense vs sparse at 128K; accuracy within 1 point; bytes read per step. The index section reports Stage A (positive or negative) honestly. Baselines: dense, PCA index, SAKI index, and a plain recency window (StreamingLLM) as the cheap floor. Target venues: MLSys, or an efficient-inference workshop if the headline is below 3x. Cite Loki, SAKI, CrossCov-U (as a baseline), ShadowKV, RetrievalAttention, MagicPIG, Fathom, TierKV, KVSwap, 2604.08426, 2605.24168.

## 5. Harness rules (from earlier projects)
- Tokenizer guard: read `tokenizer.json` directly and assert "Hello world" survives an encode/decode round trip.
- Use SDPA or explicit fp32 softmax, never bf16 eager attention, for any PyTorch reference.
- Verify the emulation against dense HF decoding on 50 steps before any run.
- Every result with paired bootstrap intervals; every run records git commit, environment, hardware, seeds, and GPU/CPU hours.
- Pre-register each gate in `runs/<stage>/PLAN.md` before launching.

## 6. First message to Claude Code
"Read this document fully. Clone the CrossCov-U repository and read `pca_analysis/crosscov.py`, `pca_analysis/recall_eval.py` and the README's diagnostics section; reuse its tensor format and recall harness. Read arXiv 2608.03228 and write down SAKI's exact operator and RoPE handling in `notes/saki.md`. Implement SAKI and the three attention-weighted variants from Section 3.1 as new basis builders in the same format. Run Stage A on the existing Mistral-7B tensors first, then LLaMA-3.1-8B. Report the gate in Section 3.4 with numbers before starting Stage B. Do not start any llama.cpp work until B0 is measured and reported."
