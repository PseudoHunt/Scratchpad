# Brief: 2x2 hash-feature test for SimBin (pre/post-RoPE x full/low-band)

You already have a working SimBin codebase. Do not change its default behavior. Add a flag and run the study below. No training is involved.

## Question

SimBin hashes keys with SimHash and uses Hamming distance to a query hash to decide which tokens to keep. Does the choice of what goes into the hash change how well it predicts true attention?

Two independent choices:

- **Position:** hash pre-RoPE vectors (content only) or post-RoPE vectors (content + position rotation).
- **Band:** hash all 128 dims or only the slow-rotating (low-frequency) RoPE pairs.

Two uses to test separately:

- **Current mode:** predict attention from the queries that were hashed (selection now).
- **Future mode:** predict attention from queries that arrive after the hash is computed (eviction, which decides what future queries can see).

Hypothesis: post-RoPE full is the exact signal for current mode. Future queries sit at different offsets, so the fast pairs get rescrambled, and position-robust features (low band, pre-RoPE) should win in future mode. The test checks whether this is true.

## Step 0: report what SimBin does today (5 lines, before building)

1. Are hashed keys pre-RoPE or post-RoPE? Which query vector is hashed, and at which position(s)?
2. Number of hash bits b, how the random projection is drawn, and the bucket structure.
3. The scoring function that turns Hamming distance into keep/evict decisions.
4. Model(s) the harness runs, and whether it uses GQA.
5. Whether the harness has a task-boundary data format.

## Step 1: feature definitions (add `--hash_feat`)

Values: `post_full`, `post_low`, `pre_full`, `pre_low`.

- **Pre-RoPE capture:** hook the q_proj/k_proj outputs before rotary is applied. Qwen has a bias on q/k/v, so the hook must be after the bias. Assert that applying rotary to your captured pre-RoPE tensors reproduces the post-RoPE tensors the cache holds.
- **Post-RoPE:** use the vectors as the cache stores them.
- **Low band:** RoPE pairs are (dim j, dim j + d/2) in the HF rotate_half layout, with j = 0 the fastest. Low band = pairs j >= J, which means dims {j, j + d/2 for j in J..d/2-1}. Do NOT just slice dims [J:]. If the codebase uses interleaved pairs, adapt the index set and say so. Default J = 32 for d = 128. Sensitivity runs: J = 16 and J = 48.
- **Query side must match the key side:** `pre_*` uses pre-RoPE q. `post_*` uses post-RoPE q.
- **Shared random matrix:** draw one Gaussian R of shape (d, b) per seed. The low-band arms use the rows of the same R that correspond to the low-band dims. This pairs the arms and removes projection noise from the comparison.
- **GQA:** hash per KV head. The query side of a group is its query heads, each hashed separately.

## Step 2: ground truth

For each document, take context length N in {4096, 8192} and T = 256 more tokens. Run one forward pass over N + T tokens, teacher-forced on the true text.

- **Candidates:** keys in M = [4, N - 64). The first 4 sinks and the last 64 positions are always kept, so exclude them.
- **Future truth A_fut[i]:** mean attention probability onto key i over queries at positions [N, N + T) and over the query heads in the KV group. The softmax must be normalized over all keys the query can see (including the earlier future positions). Do not materialize L x L for every query. Compute only the T probe rows per layer.
- **Current truth A_cur[i]:** the same, over the last alpha = 8 context queries (the positions that get hashed).
- **Headroom:** also report what fraction of total attention mass falls inside M. On Llama-2, sinks hold 40-47% of mass, so M may hold only 25-40%.

If the SimBin harness has task-boundary data (context, then a task prompt, then a response), use it. Set N at the boundary and use the next 256 tokens as the future queries.

## Step 3: scores and metrics

For each key i in M, score_i = -(mean Hamming distance between the code of key i and the codes of the alpha probe queries). Average across the group's query heads.

- **Break Hamming ties randomly.** Do not break ties by index, because that favors old or recent tokens and fakes a signal.
- **Selection:** take the top-B keys in M by score, with B in {32, 64, 128, 256, 512}.
- **Mass recall ratio:** (sum of A over the top-B by score) / (sum of A over the oracle top-B by A). Do this for both A_cur and A_fut.
- **Overlap@B:** |top-B by score intersect top-B by A| / B.
- **Controls (same four feature sets, no hashing):** exact cosine on the same features. This separates "the features are bad" from "hashing loses information."
- **Floor:** random selection.
- **Diagnostics per arm:** per-bit balance (mean of each bit, ideal 0.5), number of nonempty buckets, and bucket-size Gini. If an arm has unbalanced bits, a recall difference may be a bucket-utilization effect, not a feature effect.
- **Optional extra arm:** mean-centered keys (subtract the per-layer, per-head mean over the context) before hashing. Keys often share one dominant direction that makes SimHash bits nearly constant.

Sweep b in {8, 16, 32, 64, 128}. Use 5 projection seeds and report mean and std. Layers: skip 0-1 and report layers 2 to end by group (early, mid, late), plus the per-head spread.

## Step 4: data and models

- 100 documents per length from the SimBin harness eval set, plus 100 long documents from a generic corpus (PG19 or FineWeb-Edu long).
- **Model 1:** whatever the SimBin harness already runs.
- **Model 2:** Qwen2.5-7B (GQA, rope_theta 1e6, bias on q/k/v).

## Step 5: downstream run (only if Stage 1 passes the gate)

Gate (proposed, adjust as needed): in future mode, the best non-baseline arm beats `post_full` by at least 2 points of mass recall ratio, averaged over layers 2 to end at SimBin's default b. The paired difference over seeds is positive in at least 4 of 5 seeds, and the sign holds on both models.

If the gate passes, run SimBin end to end with each `--hash_feat` at its existing budgets, using the harness's own task metric. Compare against the numbers already in the codebase.

## How to read the outcome (put this in the report summary)

| Outcome | Meaning |
|---|---|
| Future mode: low band or pre-RoPE beats post_full, current mode: no | Position-robustness helps eviction, as hypothesized |
| post_full wins in both modes | Position-specific signal matters on this model. Drop the idea |
| Exact-cosine low band beats full but the hashed version does not | Hashing at this b loses the gain. Try larger b |
| All arms within about 1 point | Hash features are not the bottleneck. Stop, skip Step 5 |
| Gains appear only in bit-balance diagnostics | Bucket utilization effect, not a feature effect |

## Deliverable

`results.md` with tables per ground-truth mode (rows = arms and controls, columns = b and B), layer-group breakdown, headroom numbers, diagnostics, and the Step 0 answers. Also a raw `results.csv`. List any deviation from this brief, including any J or layout change.
