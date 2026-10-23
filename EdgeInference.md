# MoE Deployment Research Direction — Idea Brief, Literature Survey, and Plan

Prepared 2026-10-23. Purpose: a self-contained handoff for review and continuation in a new workspace. It covers the research direction, the fused method for review, the literature survey behind it (with links), and the experiment plan.

Scope rules that shaped this: deployment-side only (no training-side ideas), not quantization-based, feasible on one 48 GB GPU plus an 8 GB consumer GPU with CPU RAM, and aimed at a positive-result paper (ICLR/ICML/MLSys/ICASSP tier).

---

## 1. One-page summary of the direction

**The gap.** Every deployment lane for Mixture-of-Experts (MoE) got crowded in 2026 — expert offloading/prefetch, decode-time expert skipping, speculative decoding for MoE, expert-parallel serving, and routing-aware KV compression. One thing stayed open: **the prefill byte floor on edge devices.**

Decode is sparse. Prefill is not. Thousands of prompt tokens per layer activate nearly every expert, so a prefill pass streams almost the entire expert pool across the interconnect (PCIe or flash). A 30B-A3B model at 4-bit is ~15 GB; from phone flash at 1–2 GB/s that is a ~10 s time-to-first-token (TTFT) floor on every turn, no matter how sparse decode is. The 2026 systems (FreeToken, Sirius, FlashMoE, MoE-CORE) hide compute under the transfer or cache state across turns. None reduce the bytes. Reducing per-token k does not help either — prompt length, not per-token sparsity, drives the floor.

**The method: prefill expert admission under a byte budget.**

1. Per MoE layer, run the router on all prompt tokens first (one small matmul). This gives a demand count per expert.
2. Load experts in demand order until a byte budget B = target TTFT × bandwidth. Stop.
3. Tokens whose expert was not admitted get a substitute (Section 3).
4. Decide *which* tokens may be cheapened using an attention-importance score (the same family as KV-eviction scores). Tokens that eviction would discard anyway get cheap routing; tokens that stay in the cache keep exact routing.
5. Per-layer budgets, since layers differ in sensitivity.

**Tiers** (cheapest to most exact):

- Admitted experts → accelerator, exact.
- Tail expert, high-importance tokens → CPU computes just those tokens, exact, zero transfer (Fiddler's insight: for a handful of tokens, computing on CPU beats moving the expert). Applies when experts live in host RAM; on a phone with experts in flash this tier collapses to surrogate-only.
- Tail expert, low-importance tokens → surrogate or drop.

**Admission cost** (knapsack): `cost_D = Σ_{tokens routed to D} importance(x) × r̂_D`, where `r̂_D` is the offline-measured residual of D's surrogate. Admit by `bytes / cost` until the budget is hit.

**Timing bonus.** Qwen3-Next-class models (512 experts, ~10 active, hybrid linear attention) make the prefill transfer : active-compute ratio worse, and no deployment papers on that class were found as of this survey.

---

## 2. A team finding and where it fits

**The finding.** With top-4 routing, if the 4th- and 5th-ranked router scores are nearly tied and the 5th expert is already loaded (it is top-1 for other tokens), skip loading the 4th and reuse the 5th.

**Prior art — this exact pattern is published, under the name "cache-aware routing" or "expert substitution":**

- Mixture of Cache-Conditional Experts (Qualcomm, Dec 2024): Cumsum picks candidates from top gating scores and prefers cached ones; Cache-Prior boosts gating scores of cached experts.
- Importance-Driven Expert Scheduling (Aug 2025): substitutes low-score active experts with GPU-resident inactive experts of comparable score; prefetches top-score experts.
- MoE-CORE (Oct 2026): low-score cache misses may use resident substitutes chosen by routing-score proximity; high-score misses load exactly.
- BuddyMoE (2025): offline-calibrated replacement pairs of interchangeable experts.
- SliceMoE (Dec 2025) surveys the category and notes cache-aware routing beats LRU but still misses too often for practical on-device use.

So the finding cannot be a contribution on its own. It can be a component.

**Where it fits.** It is the re-routing rule in step 3 of the admission method, with one twist that makes it new in that spot: in decode the swap is greedy (cache state is whatever it is); in prefill the whole demand histogram is known before loading, so "already resident" becomes "already admitted," and the swap can be *planned* as the admission knapsack rather than applied per token.

**Two fixes before using it:**

1. Score closeness ≠ output closeness. ACE's observation: router gates show relative preference among candidates, not each expert's actual contribution. Use expert functional similarity (BuddyMoE/SERE style) as the gate and score gap as a tiebreaker — or better, the fitted residual of Section 3.
2. Gate the swap by token importance. A low-importance prompt token can take a bigger gap; a token that will stay in KV should keep exact routing.

Position in the paper: one ablation row (planned vs greedy substitution), not the headline.

---

## 3. The substitution rule: peer-combination surrogates (an extension of SERE)

SERE (ICLR 2026) re-routes a token from a missing expert to that expert's single most similar resident expert, using a fixed expert-pair similarity. Three things are left on the table.

**Setup (one token, top-4).** Token `x` routes to A, B, C, D with gates `gA, gB, gC, gD`. D is not admitted.

`y = gA·A(x) + gB·B(x) + gC·C(x) + gD·D(x)`

SERE replaces `D(x)` with `D'(x)`, D's nearest admitted neighbor. Drop-and-renormalize replaces it with nothing and rescales the other gates.

**Extension 1 — combine the peers already computed.** `A(x), B(x), C(x)` are already in hand. Approximate D as a weighted mix of them:

`D(x) ≈ α_DA·A(x) + α_DB·B(x) + α_DC·C(x) + c_D`

Fit `α` and `c_D` by ordinary least squares, offline, on a few thousand calibration tokens that route to D. Closed form. Store one E×E coefficient table per layer (for 128 experts × 48 layers ≈ 800K numbers). SERE is the special case where α is one-hot; drop-and-renormalize is the case where α is gate-proportional. Zero extra expert loads; zero extra expert compute if the mix is restricted to the token's own admitted peers.

**Extension 2 — fit to what the next layer reads (ACE's direction-dependence).** Plain least squares treats every error direction equally, but only some directions matter downstream: those that (a) flip routing in the next layer and (b) change this token's K and V, which every later token reads.

Tiny case: D's output is 2-D, `D(x) = (d1, d2)`. The next router reads only direction 1: `score = 1·h1 + 0·h2`. An error of 1.0 in `d2` changes nothing; an error of 0.1 in `d1` can flip an expert choice. Plain LS would trade a small `d1` error for a big `d2` gain — backwards.

Fix: weighted least squares. Minimize `‖ M · (D(x) − surrogate(x)) ‖²` where `M` stacks the next layer's router rows and its K/V projections (after norm). Still closed form. This gives *routing-preserving substitution*, which directly targets the propagation risk MoEZip identified (KV perturbation changes expert selection).

**Extension 3 — a tiny low-rank patch.** For the leftover residual, fit a rank-r term (r ≈ 8–16) with the same weighted objective:

`D(x) ≈ Σ_j α_Dj·Ej(x) + U_D·(V_Dᵀ·x) + c_D`

Fit peers first, then the low-rank term on what is left. ~1–2% of an expert's bytes, always resident.

**Extension 4 — replace "score gap" with "fitted residual" in the knapsack.** The fit yields an expected residual `r̂_D` per expert. Use `cost = r̂_D × importance`, so admission minimizes actual output error rather than a proxy.

**Extension 5 — tier by token importance.** Low importance → drop and renormalize; medium → peer surrogate; high → exact (admit, or compute on CPU for the few tail tokens).

**Stretch — cross-layer demand shaping.** When two substitutes are nearly equal under the weighted objective, pick the one whose output pushes the token toward next-layer experts already in high demand. Section 4 makes this computable.

---

## 4. The fused method for review: routing-projected expert sketches

This fuses two observations: (a) ACE's point that an expert's effect on routing is direction-dependent, and (b) cross-layer predictors (Fate-style) that guess layer N's experts from the state at layer N-1 to prefetch early. Both are about the same quantity, which is the key to fusing them.

### What the predictor is missing

A cross-layer predictor applies layer N's router to the token's state at layer N-1. It works because the residual stream changes slowly. Its error comes from exactly what hasn't happened yet: the output of the experts about to run at layer N-1, plus attention at layer N. The predictor ignores both.

ACE's direction point says an expert's effect on routing is not its output size, but its output's projection onto what the router reads. So the predictor's missing term is computable, cheaply, before the expert is loaded — if you keep a sketch of each expert along the directions the next router reads.

### Tiny case: layer N has two experts

Router rows `w1, w2`. Token state entering layer N is `h` (ignore norm for now).

- Scores: `s1 = w1·h`, `s2 = w2·h`. Say `s1 > s2`, so expert 1 is picked. Margin: `m = s1 − s2`.
- But `h` isn't final. Expert `e` at layer N-1 is about to add `Δ = g_e·e(x)`.
- New scores: `s1' = s1 + w1·Δ`, `s2' = s2 + w2·Δ`.
- New margin: `m' = m + (w1 − w2)·Δ`.
- Routing flips only if `(w1 − w2)·Δ < −m`.

So one number decides it: `p = (w1 − w2)·Δ` — the projection of the expert's output onto the difference of the two router rows. A huge `Δ` in any other direction changes nothing. That is ACE's observation made exact for routing.

With E experts and top-k, you need `Δs = W_g^N · Δ`, a vector of E numbers. Every boundary pair's margin shift is a difference of two entries in it.

### The method

1. **Offline, per expert e at layer l:** fit a small map `S_e: x → Δs` that predicts how e's output shifts layer l+1's router scores. Target is `W_g^{l+1}·norm(h + g_e·e(x)) − W_g^{l+1}·norm(h)`, measured on calibration tokens. Rank 8–16, closed-form least squares. This stores "the expert as the next router sees it" — a d→E map, a few percent of the expert's bytes, always resident.
2. **Online, right after router l fires:** the token's selected experts are known. Predicted next-layer scores:
   `ŝ_N = W_g^N·norm(h'_{N-1}) + Σ_{e selected} S_e(x)`
   The first term is the plain cross-layer gate. The second is the correction the predictor was missing. Prefetch from `ŝ_N`.
3. **A certificate, not a learned confidence:** for each token, compare the predicted margin at the k/(k+1) boundary against the sketch's known residual error (measured offline). Margin ≫ residual → commit: run the prefetched experts without re-fetching, as Speculating Experts does. Margin ≈ residual → prefetch k+1 or verify. Per expert, `‖S_e‖` says whether it is a "routing shaper" or "routing neutral"; tokens passing through neutral experts are safe to commit always.

### Why this is a method, not just a better predictor

- `S_e` is the same artifact as the M-weighted surrogate in Section 3 (M = next router rows). One calibration, three uses: prediction correction, commit certificate, substitution cost.
- In prefill, step 2 gives layer N's demand histogram while layer N-1 is still running. Admission for layer N can be planned and its loads started early. Demand shaping becomes a real objective: the substitution objective at layer N-1 can include "keep `ŝ_N` concentrated on experts already in high demand."

Learned predictors (ProMoE, PreScope, PROBE's distilled lookahead predictor) can absorb some of this implicitly. The difference: this is explicit, compositional across models, needs no predictor training, and gives a margin you can reason about.

### Kill test (half a day)

At each layer, compare three next-layer top-k predictions: (i) plain cross-layer gate, (ii) cross-layer gate + the true MoE output of layer N-1 but no attention, (iii) truth. If (ii) closes most of the gap between (i) and (iii), the MoE term is the dominant error and the sketch can capture it. If attention at layer N is the dominant error, the sketch has little to add and only the certificate part survives.

### Questions a reviewer should push on

1. RMSNorm sits between the residual add and the router. The 2-expert case ignores it. For small Δ it is near-linear, and the sketch is fitted on the true post-norm target, but does norm rescaling break the "difference of router rows" argument at large Δ?
2. Shared experts (DeepSeek/Moonlight-style) always run. Should their contribution be folded into the base term rather than sketched? Likely yes; verify.
3. Attention at layer N is unobservable from the token alone. Is there any cheap proxy (e.g., the previous layer's attention output direction) worth including, or is KT3 the right place to measure and move on?
4. How does `S_e` interact with the k-th/(k+1)-th boundary when many candidates are near-tied (fine-grained routers with 128–512 experts)? The certificate may rarely fire there. Measure margin distributions early.
5. Is the "routing shaper vs routing neutral" split stable across domains (code vs prose vs chat)? If not, the always-commit rule needs a domain guard.
6. Novelty check: does any 2026 predictor explicitly condition on the current layer's selected expert IDs? PreScope's "expert-gate" predictor and PROBE's distilled lookahead predictor are the closest; read both fully before claiming the sketch is new.

---

## 5. Literature survey (searches run 2026-10-23)

Method: targeted web searches across the MoE inference lanes below, reading abstracts and key sections. Not exhaustive; each lane lists what was found and its status for this direction. Dates are arXiv/venue dates as found.

### 5.1 Surveys

| Paper | Venue / date | Link |
| --- | --- | --- |
| A Survey on Inference Optimization Techniques for MoE Models | ACM Computing Surveys, Mar 2026 (arXiv 2412.14219) | https://dl.acm.org/doi/10.1145/3794845 |
| A Survey of MoE LLM Inference Optimization | HKUST-GZ, 2026 | https://dsa.hkust-gz.edu.cn/blog/2026/05/28/a-survey-of-mixture-of-experts-llm-inference-optimization/ |
| A Comprehensive Survey of MoE: Algorithms, Theory, Applications | arXiv 2503.07137 (v4 Jan 2026) | https://arxiv.org/pdf/2503.07137 |

### 5.2 Expert offloading and prefetch (edge / single device) — **crowded**

| Paper | Date | What it does | Relevance |
| --- | --- | --- | --- |
| MoE-Infinity | Jan 2024 | Sparsity-aware expert cache for personal machines; notes KV cache competes with expert buffer at long context | Baseline system |
| ProMoE | Oct 2024 | Learned predictor for proactive expert caching; chunked prefetching | Learned-predictor baseline for KT3 |
| HOBBIT | Nov 2024 | Mixed-precision expert offloading (loads low-precision versions of cache-miss experts) | Quantization-based; out of scope but a baseline |
| Fate: Cross-Layer Gate | Feb 2025 | Uses previous gate's input for next gate's expert selection; ~97% prefetch accuracy claim | The plain cross-layer predictor in Section 4 |
| ExpertFlow / Routing Path Predictor | Oct 2024 | Predicts full routing path before computation; token rebatching | Predictor baseline |
| CommitMoE | AAAI 2026 | Commits to predicted experts without fallback; output-weight robustness | Related to the certificate idea |
| Importance-Driven Expert Scheduling (SMoE) | Aug 2025 | Substitutes low-score active experts with resident ones; prefetches top-score | Prior art for the team finding |
| CoMoE | Aug 2025 | Joint expert aggregation + offloading at mobile edge | Context |
| PreScope | Sep 2025 | Prediction-driven cross-layer scheduling; "expert-gate" predictor; prefetch-aware scheduling | Closest learned-predictor competitor to Section 4 |
| ZipMoE | Jan 2026 | Lossless compression + cache-affinity scheduling on-device | Context |
| FlashMoE | Jan 2026 | ML-based cache replacement for SSD-resident experts | Context (flash tier) |
| MoE-SpAc | Feb 2026 | Uses speculative decoding as a lookahead sensor for expert demand; online integer optimization | Context |
| Speculating Experts Accelerates Inference | Mar 2026 | Prefetch from internal representations; executing speculated experts keeps accuracy; 5–14% TPOT reduction | The "commit" mechanism in Section 4 |
| FluxMoE | Apr 2026 | Budget-aware residency planner trading expert residency against KV cache under memory pressure | Closes the "expert vs KV memory budget" idea |
| MoE-CORE | Oct 2026 | Coordinated offloading and residency; phase-specific prefill/decode paths; score-proximity substitution; MTP depth 2 | Prior art for team finding; nearest system to compare prefill against |
| Not all models suit expert offloading (local routing consistency) | May 2025 | Metric for which MoE architectures are offload-friendly | Use to pick target models |
| Mixture of Cache-Conditional Experts (Qualcomm) | Dec 2024 | Cache-Prior / Cumsum cache-aware routing; on-device demo (Qwen-MoE on phone) | Prior art for team finding |
| SliceMoE | Dec 2025 | Bit-sliced expert caching under miss-rate constraints; surveys cache-aware routing (Cache-Prior, BuddyMoE) | Prior art summary |
| SpecMD (cache simulator) | Feb 2026 | Expert-cache policy simulator; score-based prefetch beats top-k | Tooling reference |

Links: MoE-Infinity https://arxiv.org/pdf/2401.14361 · ProMoE https://www.alphaxiv.org/overview/2410.22134v3 · HOBBIT https://arxiv.org/abs/2411.01433 · Fate https://arxiv.org/html/2502.12224v2 · ExpertFlow https://arxiv.org/pdf/2410.17954v2 · CommitMoE https://ojs.aaai.org/index.php/AAAI/article/view/39454 · Importance-Driven https://arxiv.org/html/2508.18983v2 · CoMoE https://arxiv.org/html/2508.09208v1 · PreScope https://arxiv.org/pdf/2509.23638v1 · ZipMoE https://arxiv.org/pdf/2601.21198 · FlashMoE https://arxiv.org/html/2601.17063v1 · MoE-SpAc https://arxiv.org/abs/2603.09983 · Speculating Experts https://arxiv.org/abs/2603.19289 · FluxMoE https://arxiv.org/pdf/2604.02715 · MoE-CORE https://arxiv.org/pdf/2610.01950 · Routing consistency https://arxiv.org/abs/2505.16056 · Cache-Conditional https://arxiv.org/html/2412.00099v1 · SliceMoE https://arxiv.org/pdf/2512.12990 · SpecMD https://www.arxiv.org/pdf/2602.03921

### 5.3 Decode-time expert skipping / re-routing — **crowded**

| Paper | Date | What it does | Relevance |
| --- | --- | --- | --- |
| Lynx | Nov 2024 | Batch-aware dynamic expert selection; finds prefill far more sensitive to expert changes than decode | Key risk for prefill substitution |
| OEA: Opportunistic Expert Activation | Nov 2025 | Batch-aware routing minimizing unique active experts per decode batch; roofline shows latency ∝ unique experts | Cost model reference |
| MoDES | Nov 2025 / CVPR 2026 | Training-free skipping for MoE multimodal LLMs; global layer importance × local gating | Context |
| SERE | ICLR 2026 (Feb 2026) | Similarity-based re-routing from secondary experts to most similar primary expert in batch decode | The baseline Section 3 extends |
| SMIDT | AAAI 2026 | Inference framework for dynamic top-k routing under pipeline parallelism | Context |
| ASET ("Faster MoE LLM Inference for Extremely Large Models") | ACL Findings 2026 | Static layer-wise top-k schedules + entropy-penalized adaptive skipping; 10–78% throughput gains | Context |
| ACE | Sep 2026 | Calibration-free dynamic expert skipping; observations: router gates ≠ contribution; contribution is direction-dependent | Source of the direction-dependence idea in Sections 3–4 |

Links: Lynx https://arxiv.org/html/2411.08982v1 · OEA https://arxiv.org/html/2511.02237v1 · MoDES https://arxiv.org/abs/2511.15690v1 · SERE https://arxiv.org/pdf/2602.07616 · SMIDT https://ojs.aaai.org/index.php/AAAI/article/view/39403 · ASET https://preview.aclanthology.org/ingest-acl/2026.findings-acl.2140/ · ACE https://arxiv.org/pdf/2609.05228

### 5.4 Speculative decoding × MoE — **very crowded (seven papers since Feb 2026)**

| Paper | Date | What it does |
| --- | --- | --- |
| SpecMoEOff | Aug 2025 | SD to enlarge per-expert workload under offloading; hot-expert caching |
| SP-MoE | Oct 2025 | Drafting-stage prefetch from draft attention + target gates; cutoff layer |
| MoE-SpeQ | Nov 2025 | Quantized on-device draft + proactive expert prefetch |
| MoE-Spec | Feb 2026 | Verification-time method; large draft trees activate many experts |
| EVICT | May 2026 | Lossless adaptive verification; verify fewer nodes to shrink expert union |
| EcoSpec | Jul 2026 | Cost-aware draft selection using predicted marginal expert-activation cost |
| DraftExpert | Jul 2026 | Self-speculative decoding for expert-offloaded MoE; fixed-footprint shared+top-1+draft-expert; evaluated on CPU-GPU and Flash-NPU |
| S2-MoE | Aug 2026 | Self-speculative decoding for MoE on edge |

Common finding across the lane: MoE verification cost scales with the union of distinct experts touched by the draft tokens, not with token count. Avoid this lane.

Links: SpecMoEOff https://arxiv.org/html/2508.21706v1 · SP-MoE https://arxiv.org/pdf/2510.10302 · MoE-Spec https://www.emergentmind.com/topics/moe-spec · EVICT https://arxiv.org/pdf/2605.00342 · EcoSpec https://www.alphaxiv.org/abs/2607.12696 · DraftExpert https://arxiv.org/pdf/2607.24434 · S2-MoE https://arxiv.org/pdf/2608.15018

### 5.5 Multi-GPU serving (EP load balance, PD-disaggregation) — **needs a cluster; not for this track**

| Paper | Date | What it does |
| --- | --- | --- |
| metro | Dec 2025 | Balance activated experts, not tokens, in memory-bound decode; up to 4.11× over EPLB at fixed SLO |
| Janus | Dec 2025 | Disaggregates attention and experts; activated-expert-balanced scheduling |
| PROBE | Jan 2026 | Gate-initialized lookahead predictor + hardware-aware balance planning + phase-locked co-scheduling |
| MoEless | Mar 2026 | Serverless MoE serving with expert-load predictors |
| NanoCP | May 2026 | Request-level dynamic context parallelism for DP-EP decoding |
| ELDR | Jul 2026 | Expert-locality-aware decode routing from prefill expert signatures; prefill/decode expert overlap Spearman 0.70–0.92 |
| Doubleword blog: expert co-activations | 2026 | Reordering inputs so similar prompts batch together cuts expert loads 12–21% |

ELDR's finding (prefill expert activations predict decode activations) is useful evidence that per-prompt demand histograms are structured by domain.

Links: metro https://arxiv.org/html/2512.09277v1 · Janus https://arxiv.org/pdf/2512.13525 · PROBE https://arxiv.org/abs/2602.00509v1 · MoEless https://arxiv.org/pdf/2603.06350v1 · NanoCP https://www.alphaxiv.org/abs/2605.21100 · ELDR https://www.emergentmind.com/papers/2607.00466 · Doubleword https://blog.doubleword.ai/moe-expert-coactivations

### 5.6 KV cache × MoE — **partly taken**

| Paper | Date | What it does | Relevance |
| --- | --- | --- | --- |
| MoEZip | NeurIPS 2026 | Routing-aware KV compression; Fisher-information metric for routing sensitivity to KV eviction; +24/+20 on LongBench/RULER at 10% retention | Shows KV perturbation flips routing; the mirror of our prefill risk |
| PiKV | Aug 2025 | Expert-sharded distributed KV; cache-miss penalty in routing | Context |
| MoE-XBench | Aug 2026 | Benchmark for stacked compression (pruning, quant, KV) in MoE | Evaluation reference |
| MoE-nD | Apr 2026 | Per-layer routing over KV compression configs (not an MoE-model paper) | Tangential |

Links: MoEZip https://neurips.cc/virtual/2026/poster/152692 · PiKV https://arxiv.org/html/2508.06526v1 · MoE-XBench https://arxiv.org/pdf/2608.21693 · MoE-nD https://arxiv.org/abs/2604.17695

### 5.7 Prefill / TTFT on edge — **the open spot**

| Paper | Date | What it does | Gap left |
| --- | --- | --- | --- |
| MoE-Gen | Mar 2025 | Module-based batching on a single GPU; full KV offload to cut expert fetch traffic | Throughput, not TTFT |
| Sirius | 2026 | Collaborative prefill over edge devices; interleaved partitioning; speculative prefetch | Distributes bytes, does not reduce them |
| FreeToken | Aug 2026 | Edge-native MoE serving; full-layer double buffering makes prefill transfer-bound; semantic state caching for agentic re-prefill; shows reducing per-token k does not fix prefill transfer | Hides compute under transfer; bytes unchanged |
| MoE-CORE | Oct 2026 | Phase-specific prefill/decode offloading paths within a shared memory budget | Same |
| MobileMoE | May 2026 | On-device MoE scaling law + smartphone runtime (expert-ordered grouped GEMMs) | Architecture co-design; context |
| SlimCaching | Jul 2025 | Edge-network expert caching | Context |

Links: MoE-Gen https://arxiv.org/pdf/2503.09716 · Sirius https://crad.ict.ac.cn/en/article/doi/10.7544/issn1000-1239.202660131 · FreeToken https://arxiv.org/pdf/2608.16157 · MobileMoE https://api.emergentmind.com/topics/mobilemoe · SlimCaching https://arxiv.org/html/2507.06567v3

### 5.8 Items to verify from memory (not found in this session's searches)

- "Capacity-Aware Inference: Mitigating the Straggler Effect in MoE" (2025) — capacity-limited token drop/re-route for EP stragglers. Nearest prior for planned re-routing under a per-expert constraint. Confirm and cite.
- BuddyMoE (2025) — referenced by SliceMoE and MELINOE; read the original.
- Fiddler (CPU-GPU hybrid MoE) — basis for the CPU-exact tier. Read the original.

---

## 6. Experimental plan

Hardware assumed: one 48 GB GPU (A6000-class) for measurement; an 8 GB consumer GPU with ≥32 GB host RAM as a real edge testbed for TTFT. Rented instances may be reclaimed, so every script checkpoints per layer.

Models: OLMoE-1B-7B (64 experts, top-8, 16 layers) as primary; Qwen3-30B-A3B (128 experts, top-8, 48 layers) at int8 for confirmation; GPT-OSS-20B (32 experts, top-4) as fallback; DeepSeek-V2-Lite / Moonlight-16B-A3B if shared-expert behavior matters.

### Phase 1 — kill tests (measurement only, ≤4 GPU-hours)

A separate Claude Code brief ("MoE Edge Prefill — Phase 1 Kill Tests") specifies these in detail. Summary:

| Test | Question | Go threshold |
| --- | --- | --- |
| KT1 | Is the per-layer prefill demand histogram long-tailed? | Median layer: ≥30% of experts serve <1% of token-slots, or 95% of slots covered by ≤70% of experts |
| KT2 | Can peers reconstruct a missing expert, and does M-weighted LS cut next-layer routing flips? | Median R² ≥ 0.70 on half the layers; weighted flip rate ≤ half of plain in 3/5 residual bins |
| KT3 | Is the MoE term the dominant cross-layer predictor error? | Median gap_closed ≥ 0.60 |

Decision tree: KT1 fail → stop the direction. KT2 partial (R² 0.4–0.7) → fall back to best-single-peer. KT3 partial (0.3–0.6) → keep the certificate, drop the sketch correction.

### Phase 2 — method build

- Fit rank-8/16 low-rank patches on the weighted residual; re-measure flip rate.
- Fit per-expert sketches `S_e` (rank 8–16); measure corrected predictor hit@k vs plain cross-layer gate and vs ProMoE-style learned predictor.
- Calibrate the certificate: commit precision vs margin threshold.
- Implement admission knapsack with per-layer budgets and importance tiers (SnapKV-style importance from the last window of queries).

### Phase 3 — end-to-end

- Prefill admission at byte budgets of 30/50/70% of the expert pool; quality on LongBench (subset), RULER, GSM8K with long prompts; compare against full streaming (FreeToken-style), uniform k-reduction, and greedy cache-aware substitution (MoE-CORE-style).
- TTFT on the 8 GB GPU + CPU-offload testbed, and under throttled bandwidth to simulate flash.
- CPU-exact tier ablation (laptop setting only).
- Ablations: planned vs greedy substitution (team finding), plain vs weighted LS, with/without low-rank patch, with/without demand shaping.

---

## 7. Risks and open questions

1. **Prefill sensitivity.** Lynx reports prefill is far more sensitive than decode to expert changes because errors propagate through KV. Importance gating and the routing-preserving fit are the mitigations; KT2's flip-rate metric is the early warning.
2. **Flat demand.** Fine-grained routers trained with aux-loss-free balancing may give flat per-prompt demand. KT1 decides.
3. **Attention-dominated predictor error.** If KT3 shows attention at layer N dominates, the sketch's correction is small.
4. **Incremental-systems perception.** Reviewers may read "apply re-routing to prefill" as glue. The defense is the cost model (TTFT = bytes/bandwidth), the routing-preserving objective, and the one-artifact-three-uses structure.
5. **Novelty collisions to check before writing:** PreScope and PROBE predictors (conditioning on selected experts), MoE-CORE (score-proximity substitution), Capacity-Aware Inference (planned re-routing).
6. **Model availability.** Qwen3-Next-80B does not fit a 48 GB GPU at bf16; use Qwen3.5-35B-A3B at int8 or stay with Qwen3-30B-A3B for the hybrid-architecture angle.
