Experiment handoff: interaction-aware circuit ordering for dynamic MoE pruning

Date: 2026-11-06. Status: proposed experiment; no results claimed.

Instruction to the executing Codex agent

Implement and run this bounded experiment on one NVIDIA RTX A6000 48 GB. Read the repository’s AGENTS.md first, preserve existing work, create an isolated experiment directory or branch, and proceed autonomously within the scope below. This document authorizes the described local experiment, not cloud rentals or publication. Do not expand into quantization, recovery training, online compensation, or a cache-serving system. Record failures and negative results. If no suitable GPU is available, prepare runnable code and report the blocker; do not pretend to have run it.

The user wants a better OFFLINE sensitivity/order algorithm. At inference, each expert has ONE stored permutation of its intermediate channels. A memory budget selects a prefix width; the suffix is omitted. All three associated projection slices move together. There must be no inference-time optimization, recalibration, compensation branch, or model variant per width.

1. Research question and scope

Does retaining circuit-interaction information under HEAPr’s output-space metric produce better channel orderings than sorting individual HEAPr scores? Does optimizing the same ordering across several prefix widths improve further?

Separate these two hypotheses:

• H1: interactions improve ranking at matched per-expert widths.
• H2: an ordering optimized across widths improves the aggregate quality-versus-width curve over an interaction-aware greedy ordering.

This is a sensitivity pilot, not a novelty proof or a deployment speedup paper. Full Fisher matrices already represent interactions. The proposed change is selection/ordering using circuit cross terms. The pilot uses within-expert interactions only; it does not claim to optimize cross-expert or cross-layer interactions.

2. Model and hardware

Use allenai/OLMoE-1B-7B-0924 BASE, not Instruct. Resolve main to an exact Hugging Face revision and record model/tokenizer revisions before running. Official config currently gives:

|Property                   |Value                  |
|---------------------------|-----------------------|
|Total parameters           |Approximately 7 billion|
|Layers                     |16                     |
|Experts per layer          |64                     |
|Experts selected per token |8                      |
|Hidden dimension d         |2048                   |
|Expert intermediate width m|1024                   |
|Expert activation          |SiLU gated MLP         |
|Router top-k normalization |`norm_topk_prob=false` |

Verify these against downloaded config and actual module shapes; fail loudly on a mismatch. The BF16 weights are approximately 14 GB decimal (13 GiB), not a promise about backward peak memory. This is why we use OLMoE rather than a 16B+ MoE for the first test.

Use one A6000, BF16 model weights, FP32 statistics, batch size 1, sequence length 512, eval(), no KV cache, no optimizer, and all parameter gradients disabled. Enable input/activation gradients explicitly for calibration. Confirm the gradient graph exists even though weights are frozen. Use SDPA or an available attention backend with a correctness check; do not require an unsupported fused MoE kernel.

Target peak GPU allocation below 42 GiB, leaving headroom. Do not keep a second model on GPU. Stream teacher outputs and reuse the same immutable base checkpoint for all arms. Keep statistics on CPU/disk and process one expert’s matrix at a time. Inspect host RAM and disk first; target at least 32 GB available host RAM and 40 GB free disk, or use memmaps and smaller bounded caches.

Hard cap: 12 hours of GPU-occupied experiment time, including failed jobs. This is a pilot guardrail, not a runtime forecast. Run a measured throughput/memory smoke test before scheduling the full phase. Reserve at least 25% of remaining time for final held-out evaluation and reporting. Stop cleanly with resumable artifacts if the cap is reached. Do not silently reduce data or change precision between arms to fit the cap.

3. HEAPr reference and mathematical contract

Read the HEAPr paper and inspect its official implementation. Pin its commit if code is reused; respect its license. Port scoring to OLMoE if necessary. Do not label the result an exact official reproduction if architecture support, gradient reduction, sampling, or normalization differs. Call the core comparator HEAPr-compatible per-expert ordering and list deviations.

Within expert e, use column-vector notation:

h_j(x) = SiLU(w_gate[j] x) * (w_up[j] x)
c_j(x) = h_j(x) d_j
E_e(x) = sum_j c_j(x)

Here d_j is column j of down_proj.weight. In PyTorch prune/permutate ROW j of gate/up and COLUMN j of down. Do not infer parameter names blindly; inspect the installed model source.

Let g_e be the gradient of the chosen calibration loss with respect to the expert’s UNWEIGHTED output. This gradient already includes routing weights through the mixture. Do not multiply it or c_j by the router weight a second time.

For n routed calibration samples in an expert, collect activation matrix A in R^(n x m) and expert-output gradient matrix B in R^(n x d). Define:

F_e = B^T B / n
C_e = A^T A / n
D_e = down weight, shape (d, m)
K_e = C_e elementwise-multiply (D_e^T F_e D_e)

Equivalently calculate V = B D_e, then:

K_e = (A^T A / n) elementwise-multiply (V^T V / n)

This avoids explicitly forming the d x d matrix F_e. Accumulate in FP32. Symmetrize K; retain signed off-diagonal entries. Never take absolute values of cross terms. K is positive semidefinite in exact arithmetic (Schur product), although rounding can introduce small numerical deviations.

The HEAPr-compatible individual score is s_j = K[j,j]/2. The group surrogate for a removal indicator z is:

Q_e(z) = 0.5 * z^T K_e z

Conditional removal cost:

delta_Q(j | P) = 0.5*K[j,j] + sum_{i in P} K[i,j]

These are exact algebraic relationships for the fixed HEAPr-style surrogate, NOT exact changes in network loss. F_e is an empirical gradient second moment; do not claim it equals the true Hessian generally. It is averaged separately from activations, as in HEAPr. The paired statistic mean[(g(x)^T c_i(x))*(g(x)^T c_j(x))] is a DIFFERENT estimator; do not substitute it in only one arm.

HEAPr Eq. 7 concerns mixed derivatives of expert output. Their vanishing does not force mixed derivatives of loss to vanish: J_i^T H_y J_j can remain. Treat this as motivation for measuring cross terms, not evidence that their method fails empirically.

All arms use the same expert-level normalization and the same routed samples. Our primary experiment keeps the same width in every expert, so it does not test HEAPr’s global allocation rule. Do not conflate allocation gains with ordering gains.

4. Calibration, development, and locked evaluation data

Use pinned allenai/c4, English configuration, with document IDs/hashes and tokenizer IDs recorded. Stream a bounded number of documents rather than downloading the full corpus. If unavailable, document the issue and use a fixed accessible English corpus for ALL arms; do not silently mix sources.

• Calibration A: 128 disjoint documents, one deterministic 512-token window per document, seed 1729 (65,536 input tokens).
• Calibration B: same size, disjoint documents, seed 2718, used only for replication if the full phase is reached.
• Development: 32 disjoint documents/windows, 512 tokens each.
• Locked test: 128 disjoint documents/windows, 512 tokens each, never used to select swaps or settings.
• Secondary evaluation: WikiText-2 raw test, using a fixed documented tokenization/chunking protocol. Run after the primary evaluation if time allows.

Choose documents with enough tokens; do not create overlapping windows across sets. Check document hashes for leakage. Save manifests before comparisons. The primary test is a sampled C4 evaluation, not a claim about full C4 perplexity.

Use causal next-token cross-entropy SUM over valid predicted tokens for calibration gradients; exclude padding and router auxiliary losses. This gives invariant per-position scaling across batches. Expert-output gradients from sequence loss include downstream token dependencies; identify this as the HEAPr-style gradient statistic, not an exact per-example full Fisher. Record source-code differences from HEAPr’s loss reduction. No dropout. Clear activation gradients each batch.

Maintain a deterministic uniform reservoir of at most 512 routed rows PER EXPERT containing paired (h, g) from the same forward/backward. Exclude invalid/padded positions. Use the same reservoir for all arms. Preserve router weights in g through autograd; do not reweight reservoirs by expert usage. Log total routed count, stored count, gradient norms, and routing frequency per expert. Never fill inactive-expert gradients with invented values. If fewer than 64 samples are present, mark that expert undercovered and use the diagonal ordering in ALL proposed arms for that expert. Report how often this fallback occurs.

For full collection, reservoir storage for 1024 experts at 512 rows of (1024+2048) BF16 values is about 3 GiB. FP32 K matrices for all experts are about 4 GiB. Stream by layer/expert and avoid Python lists of huge temporary tensors. These estimates exclude model, framework, and host overhead.

5. Required arms

Use identical per-expert retained widths k in every compared arm:

optimized widths: [256, 512, 768] (25%, 50%, 75% retained)
unseen widths:   [384, 640, 896]
full control:    1024

All are multiples of 128. Do not allocate different widths to different experts in the primary result.

|ID      |Ordering                                                                 |
|--------|-------------------------------------------------------------------------|
|FULL    |Unmodified BF16 model                                                    |
|RANDOM  |Deterministic random channel permutation, one seed; sanity baseline      |
|H-DIAG  |Descending K diagonal; HEAPr-compatible baseline                         |
|H-GREEDY|Sequential minimum conditional removal using full K; reverse removal list|
|H-NESTED|Optimize fixed prefix ordering over the three optimized widths using K   |

For H-GREEDY, start z=0 and v=Kz=0. Repeatedly remove the remaining j minimizing 0.5*K[j,j]+v[j], then v += K[:,j]. Deterministic ties use original channel index. Continue to produce a complete permutation. Negative marginal scores are permitted. Greedy failure to initiate cancelling pairs is a reason for H-NESTED, not an implementation error.

For H-NESTED, minimize per expert:

J_e(pi) = mean_{k in [256,512,768]} 0.5*z_k(pi)^T K_e z_k(pi)

z_k is 1 on the suffix after retained position k. Initialize from BOTH H-DIAG and H-GREEDY; optimize each with the same fixed search budget, then choose the smaller calibration objective. Record each start/end objective and chosen initialization. This choice must not use test or development quality.

Implement deterministic local exchanges between positions in different width intervals. For any swap with mask change delta:

delta_Q = delta^T (Kz) + 0.5*delta^T K delta

Cache Kz for the three widths. A proposed swap affects only cutoffs between its two positions. Accept only strictly improving aggregate moves with a numerical tolerance scaled to initial J. Update cached vectors exactly. Objective cannot increase after an accepted move.

Concrete bounded search per initialization: four sweeps. Per sweep, screen 1024 single swaps (half sampled near cutoffs within 128 positions, half uniformly across different intervals), rank by exact delta_J, and accept up to 32 improvements sequentially while recomputing their current gains. Also screen 128 pair exchanges per sweep: exchange two retained-side channels with two discarded-side channels across one selected cutoff; evaluate all affected widths exactly. Half of pair proposals should use strongly negative normalized K correlations among candidate channels, half uniform. Do not move two channels as an indivisible block at inference. Record seeds, proposed/accepted counts, wall time, and objective trace. Finish early if a complete sweep accepts nothing. Batch calculations and stream one K at a time.

If pair search is too costly, complete the single-swap arm and label pair search unrun; do not call it completed. Channel ordering inside each optimized-width interval is underdetermined by J; preserve inherited relative order except accepted exchanges, and explicitly evaluate unseen widths.

No downstream-KL-based candidate search in the primary experiment: that would give proposed arms extra supervision and obscure whether the K interactions explain gains. KL is an evaluation metric here.

6. Meaningful correctness checks before quality runs

1. Full-width permutation invariance: identical routing, expert outputs, and model logits within measured BF16 numerical tolerance. Measure repeat-forward noise first. Compare relative L2 and max error; investigate discrepancies above roughly 1e-3 relative L2 rather than weakening tolerances silently.
2. In FP64 synthetic examples, verify circuit sums, gate/up/down permutation orientation, explicit versus factorized K, diagonal HEAPr scores, direct versus cached swap deltas, and cancelling-pair behavior.
3. On real expert samples, compare K[j,j]/2 to explicit mean[c_j^T F c_j]/2 for several channels, relative tolerance about 1e-4 in FP32. Compare random group Q to explicit quadratic output residuals.
4. Verify expert-output gradient against router-weighted mixture-output gradient on a tiny real batch, preserving token dispatch order. Include the unnormalized top-k router convention. Gradient hooks must not reorder routed rows.
5. Check hooks are called once per intended computation. Gradient checkpoint recomputation must not duplicate reservoir entries. Frozen parameters must have no .grad allocation; input/activation gradients must be finite and nonzero.
6. A retained-prefix implementation must match an equivalent binary channel mask at the same width. Masking is acceptable for quality evaluation, but cannot substantiate cache-memory or speedup claims.
7. Every arm’s permutations are bijections. The retained set at k is a subset of that at larger k. Switching widths back to full recovers full output at a fixed input.
8. All weights, routing parameters, normalization parameters, and quantization state remain unchanged apart from permutation/slicing. No accidental recovery training.

7. Execution phases and decision gates

Phase A: smoke test and local diagnostic

Start with 8 calibration windows and inspect gradients, memory, and throughput. Then collect Calibration A statistics for layers [3,7,11,15] (zero-based), all 64 experts in each selected layer. Computing K only for these layers limits initial expense. Use a hook-friendly implementation; do not spend the pilot writing high-performance kernels.

Run H-DIAG, H-GREEDY, H-NESTED and FULL on development data with ONLY ONE selected layer pruned at a time, at k=512 and 768. Use actual network suffix execution and teacher-forced tokens. The other layers remain full. Report layer-output relative MSE, surrogate Q, actual final-distribution KL, and NLL.

Screening rule, fixed in advance: advance to full-model Phase B if either proposed arm improves mean development KL over H-DIAG by at least 3% at either tested width, and improves at least 3 of the 4 layer cases at that width. This is a compute-allocation gate, not statistical proof. If the comparator KL is numerically negligible, mark the ratio unstable and do not claim success from it. If the gate fails, still evaluate the selected-layer comparisons on locked test data and report a local null; do not claim that full-model pruning has been disproved.

Phase B: full-model quality curves

If the gate passes and projected work fits the cap, collect remaining layers’ statistics, construct all orderings, and evaluate simultaneous pruning in all MoE layers. Reuse collected statistics without changing reservoir sampling rules. Prefer A arms at optimized widths first, then unseen widths, then RANDOM and secondary checks if time is tight. Label all missing cells.

At each width, every arm must have the same retained parameter count. Record held-out NLL, perplexity, and full-vocabulary teacher-to-pruned KL. Store per-sequence metrics. Use normal downstream routing in the pruned network; do not freeze the whole network’s teacher routing. Our local surrogate freezes layer inputs/routing, but quality evaluation must reveal actual upstream and routing changes.

Replicate H-DIAG and the proposed arms with Calibration B if time allows. Reuse the SAME locked evaluation set for paired comparisons and report the two calibration runs separately; do not treat them as independent test examples. An unreplicated finding is provisional.

Phase C: minimal dynamic-width validation

For H-NESTED, run fixed teacher-forced inputs through a deterministic width schedule over successive chunks: [1024,768,512,256,640,384,896,1024]. Compare each chunk to an independently executed static-width reference using identical context and no KV cache. This checks switching correctness, not cached autoregressive consistency or system throughput. Export one permutation per expert and a width selector. Do not claim real cache eviction savings from mask-only execution.

8. Evaluation, statistical rules, and interpretation

Primary full-phase endpoint: mean excess NLL over FULL across k=[256,512,768], equal width weights, for each arm. Also report paired NLL differences at every width. Use excess NLL rather than relative perplexity as the primary scalar.

For confidence intervals, bootstrap independent document/windows 2000 times, preserving pairing between methods and widths. Do not bootstrap individual tokens as independent observations. Report a 95% interval for J_test(method)-J_test(H-DIAG), where J_test is the empirical mean across the three widths. Record degradation at each width so an average cannot hide severe regression.

Decision rules for this pilot:

• H1 supported provisionally: H-GREEDY or H-NESTED improves aggregate excess NLL by at least 5% relative to H-DIAG, paired 95% interval for the absolute difference is below zero, and no optimized width worsens by more than 0.02 nats/token. These thresholds are practical pilot criteria, not established field standards.
• H2 supported provisionally: H-NESTED beats H-GREEDY on aggregate excess NLL with an interval below zero and satisfies the same regression guard. Show magnitude even if below 5%.
• Stronger pilot evidence requires the gain direction to replicate with Calibration B and the unseen-width curve to avoid major regressions. Report any failure explicitly.
• If H-DIAG excess NLL is nearly zero, use absolute differences and call relative improvements unstable.
• Lower calibration Q alone is insufficient. Lower Q with worse held-out KL/NLL indicates a surrogate/generalization problem, not success.
• A successful pilot on OLMoE alone does not establish generality or novelty. A negative pilot under the time cap should be reported without an unbounded rescue search.

Teacher KL: compute softmax/log-softmax in FP32 across the complete vocabulary, chunking positions as needed. Do not compare only top-k logits unless separately labeled approximate. Cache teacher logits one small batch at a time or bounded disk shards; do not keep all teacher logits on GPU. Report teacher-forced distribution fidelity, not claims about generation quality.

Secondary diagnostics: calibration-vs-held-out ordering of group damage; signed off-diagonal energy; interaction strength versus observed benefit; routing coverage; expert frequency; search runtime; statistics memory; actual GPU peak. Optional within-layer heterogeneous widths can be tested only after primary comparisons and must be clearly labeled, since joint expert interactions were not optimized.

9. Implementation and deliverables

Provide a runnable experiment package with commands equivalent to:

python -m experiment.prepare --config configs/olmoe_a6000.yaml
python -m experiment.collect --phase pilot --seed 1729
python -m experiment.rank --arms h_diag h_greedy h_nested
python -m experiment.evaluate --phase pilot
python -m experiment.run_full_if_gate_passes
python -m experiment.report

These names are a suggested interface, not a claim that files already exist. Implement the actual commands and document them. Provide resume support and skip completed artifacts only after verifying model/config/data hashes.

Save:

• Exact environment versions, GPU identity, checkpoint/HEAPr revisions, calibration/test manifests, and preregistered config.
• Correctness-test outputs, memory/throughput smoke result, and a time ledger including failures.
• Per-expert statistics or sufficient resumable inputs, routing sample counts, orderings, and optimizer traces.
• Per-document NLL/KL, width-by-arm tables, paired intervals, and quality-versus-retained-parameter plots.
• REPORT.md stating what ran, whether each hypothesis passed, regressions, limitations, and next experiment justified by results.
• A compact checkpoint-independent artifact containing permutations, source-model hash, supported widths, and prefix application code. Avoid duplicating a full model per width.

Never present synthetic estimates as measurements. Do not tune on test data. Do not change the method after seeing locked test results and reuse that test as a fresh confirmation. If a bug invalidates results, retain the old artifacts with an invalidated label, fix it, and rerun the affected matched arms.

10. Sources and prior-work boundaries

1. OLMoE model card: https://huggingface.co/allenai/OLMoE-1B-7B-0924
2. Architecture config: https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json
3. HEAPr paper (inspect Eqs. 7–16): https://arxiv.org/html/2509.22299v2
4. HEAPr official repository: https://github.com/LLIKKE/HEAPr
5. Combinatorial Brain Surgeon: https://proceedings.mlr.press/v162/yu22f.html
6. FlexMoE, close prior work for nested channel orderings: https://arxiv.org/abs/2606.27866
7. CRISP, close prior work for multiple widths and memory-constrained scheduling: https://arxiv.org/abs/2610.37442

Borrow HEAPr’s circuit representation and output-space metric with attribution. Circuit grouping, interactions in pruning, and dynamic nested widths are not independently new. The candidate contribution being tested is their particular combination: a task-weighted interaction objective used to optimize one reusable circuit order across cache budgets without online repair. Establish performance first; perform a deeper novelty audit before writing a paper claim.
