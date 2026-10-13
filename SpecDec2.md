# Handoff: drafter long-output decay (S2 + calibration vs Test-Time Speculation)

Written 2026-10-13 for a new coding agent (Claude Code / Codex) on a new machine and account.
Read this whole file first. Then read the repo's `HANDOFF.md`, `results/summary.md`, `RUNLOG.md`, `NOTES.md`.

---

## 1. What this project is, in one paragraph

Speculative decoding with the DFlash block-diffusion drafter (`z-lab/Qwen3-4B-DFlash-b16`) on target `Qwen/Qwen3-4B`
loses acceptance as reasoning outputs grow (thinking mode on, up to 16K tokens). Test-Time Speculation (TTS,
arXiv 2605.09329) fixes this with one Adam step on the drafter after every verification round, at 69.6 ms/round.
We found the decay has **two separate causes** and fixed both **without online updates**:
(1) a **thinking-mode mismatch** (the released drafter was trained with thinking off), fixed by a **one-time
offline calibration** of the released weights on 59 disjoint thinking-on traces (≤0.75 GPU-h); and
(2) a **long-context failure inside the drafter**, fixed by the **S2 context layout**: the drafter sees only the
prompt + the last 1,024 tokens + 16 far 32-token chunks chosen by the *target's* attention.
**S2+cal beats TTS on 49 held-out traces: τ 4.408 vs 4.208, +0.200 [95% CI +0.179, +0.222], at 0 ms/round.**
Ablations are done: calibration alone (cal-full), S2 alone, SharpDraft (a training-free competitor from
4 Oct 2026), a public thinking-mode-trained drafter, and a cheap hybrid. Remaining work: generality (second target
and drafter), real-loop wall-clock, and three baseline rows. Then the paper.

## 2. Where everything is

- **Repo:** `github.com/PseudoHunt/rls-drafter-killtest` (private, branch `main`). Everything except model weights,
  traces and calibrated checkpoints is in git.
- **Key files in the repo:**
  - `HANDOFF.md` — last instance state, code map, one-command rerun.
  - `results/summary.md` — all results, newest session first (session 4 at the top).
  - `RUNLOG.md` — every command, every pre-registration, GPU hours. **Append to it; never edit past entries.**
  - `NOTES.md` — DFlash internals (§1–4), context policies (§9–12), batched update + calibration (§13),
    SharpDraft and related work (§14–15).
  - `KILL_TEST_BRIEF.md` — the original brief with amendments (replay design, rules).
  - `results/s3_heldout.md`, `results/s4_{anchor,sharpdraft,thinking,calfull}.md` — per-arm reports with CIs.
  - `results/raw_test*/`, `results/raw_a30*/` — per-round JSONL logs for every arm (in git).
  - `results/cal_disjoint.txt` — proof the calibration problems don't overlap dev/test.
  - `rlsdk/` — the code: `replay.py` (round simulation, leakage guard), `window.py` (context policies),
    `adapters.py` (arms: Baseline, FarSel=S2, SharpDraft, TTS), `batched.py` (exact batched TTS update),
    `calibrate.py` (offline calibration, `--layout s2|full`), `pool.py` (resumable workers), `analyze.py`
    (loop truncation, bins, bootstrap).
  - `scripts/` — `resume.sh`, `rerun_s2cal.sh`, `check_cal_disjoint.py`, `s3_report.py`, `s4_report.py`,
    `heldout_report.py` (contains `Boot`, the bootstrap used everywhere), `run_s4_item*.sh` launchers.
  - `tests/` — 143 CPU unit tests on a tiny random model (leakage guard, RLS, batched gradient equality,
    SharpDraft identities, resumability). All must pass before any GPU run.
- **Not in git (must be restored):**
  - Calibrated checkpoints: `ckpt/cal/lr1e-05/` (S2+cal, **frozen**, sha256
    `cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7`), `ckpt/cal/lr3e-06/`,
    `ckpt/calfull/lr1e-05/` (sha256 `1f3ba7a4bddc1f17bade5608eebfb7e6f512b393cb61ff01d39283615a05b797`).
  - Traces: `traces/traces.jsonl` (62 dev/test), `traces/traces_cal.jsonl` (60 calibration),
    `traces/traces_nothink.jsonl` (6 thinking-off controls).
  - These were to be uploaded as GitHub release `artifacts-s4` with `results/ARTIFACTS.md` listing sha256s.
    **Check `gh release view artifacts-s4` first.** If the release does not exist, the files are gone with the
    old instance and must be regenerated (see §4, "if artifacts are missing").
  - Model weights (re-downloadable): `Qwen/Qwen3-4B`, `z-lab/Qwen3-4B-DFlash-b16`,
    `jgeuter/qwen3-4b-dpard-thinking-b16` @ `78baf299fc6dc793aab6c35ce44676a980e266d1`.
- **Papers:** SharpDraft arXiv 2610.05106 (re-download the PDF; NOTES §15 has the equations).

## 3. Setup on the new machine

```bash
git clone git@github.com:PseudoHunt/rls-drafter-killtest.git && cd rls-drafter-killtest
# Two venvs (uv): env-dflash (torch, transformers, dflash 0.1.0 via `pip install dflash`), env-vllm (vLLM).
# scripts/resume.sh documents the exact versions used (torch 2.13, transformers 5.15, vLLM 0.31) — match them
# as closely as possible; vLLM version changes can change greedy traces.
source scripts/resume.sh            # should print RESUME OK; set HF_HOME to a local cache
pytest -q tests/                    # 143 must pass before any GPU work
gh release download artifacts-s4 -D /tmp/art
mkdir -p ckpt/cal ckpt/calfull traces
tar xzf /tmp/art/s2cal_lr1e-05.tar.gz -C ckpt/cal
tar xzf /tmp/art/s2cal_lr3e-06.tar.gz -C ckpt/cal
tar xzf /tmp/art/calfull_lr1e-05.tar.gz -C ckpt/calfull
tar xzf /tmp/art/traces.tar.gz -C traces
sha256sum ckpt/cal/lr1e-05/model.safetensors   # must equal cc25cc15…8ca5d7
bash scripts/rerun_s2cal.sh         # ≈0.4 A30-h; replays S2+cal on the test traces, prints τ next to 4.408 → MATCH
```

The `rerun_s2cal.sh` match is the acceptance test for the new environment. If it does not match within 0.5%,
stop and diagnose (dtype path, attention kernel, dflash version) before running anything new.

**If artifacts are missing:** regenerate traces with vLLM (greedy, thinking on, 16K cap; seed-0 split as in
`KILL_TEST_BRIEF.md` §3), then recalibrate with
`python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 1e-5 --steps 1415 --batch 16 --out ckpt/cal`.
This reproduces the recipe but not the hash. **Then every arm in the table must be rerun** on the new traces
before any number is reported; mixing old and new traces is not allowed. Say so in RUNLOG.md.

## 4. Definitions (so the agent uses the same words)

- **τ** = mean tokens committed per verification round (accepted draft tokens + 1). The primary metric.
- **Replay:** greedy decoding commits exactly the target's greedy trace, so we replay rounds offline from the saved
  trace plus one teacher-forced target pass. Verified exact against the real loop (mean τ bias −0.41%, 98%+ of
  rounds identical). All methods share this bias.
- **Test set:** 50 test traces (24 AIME-24, 26 LCB v6 hard), seed-0 split; 1 dropped → **49**. Dev = 12.
- **Loop rule:** a trace is truncated at loop onset (1,000-token-window rule, ≥80% repeated). The "decision
  variant" uses truncated traces and excludes the dropped one. Never change this.
- **Bins:** output position 0–2K, 2–4K, 4–8K, ≥8K.
- **Bootstrap:** paired, stratified by dataset, 10,000 resamples, seed 0 (`Boot` in `scripts/heldout_report.py`).
  "A beats B" means the lower CI bound of (A − B) > 0.
- **Baseline:** the released drafter, full context.
- **Window:** drafter sees prompt + last 1,024 tokens (contiguous positions).
- **S2:** window + 16 far 32-token chunks scored by the *target's* attention (layers 17/25/33, last committed
  query, summed over heads). The drafter's own attention is NOT used for selection (it loses to the plain window).
- **cal / S2+cal:** released weights fine-tuned once, offline, in the S2 layout: teacher-forced, random anchors,
  DFlash-weighted KL over 15 slots (w_k = exp(−(k−1)/7)), Adam lr 1e-5, 1,415 steps × batch 16, on 59
  calibration traces (AIME-2025 + unused LCB, disjoint from dev/test). Frozen; 0 ms/round at inference.
- **cal-full:** same recipe in the full-context layout, run without a window.
- **TTS stride 1:** one Adam step (lr 1e-5, dev-chosen) on all drafter parameters after every round, KL to the
  target's distributions on the accepted prefix. Stale drafter cache (refresh checked: −0.07%, no better).
- **full:** S2+cal + exact batched pooled TTS at stride 10 (one batched backward per 10 rounds). 11.4 ms/round.
- **SharpDraft:** arXiv 2610.05106; scales drafter queries by β̂ = min{e^0.5, 1 + log(r)/5},
  r = max{(N−32)/(2048−32), 1}, N = visible keys + 16 block keys. Implemented as a q_norm hook (NOTES §15).
- **R** = (τ_method − τ_base) / (τ_TTS − τ_base), share of TTS's gain recovered.

## 5. Results so far (49 test traces; − TTS and − S2+cal are paired 95% CIs)

| arm | τ | 0–2K | 2–4K | 4–8K | ≥8K | − TTS | − S2+cal | update ms/round |
|---|---|---|---|---|---|---|---|---|
| baseline (released, full ctx) | 3.100 | | | | | | | 0 |
| window (1024 + prompt) | 3.896 | | | | | | | 0 |
| S2 | 4.069 | 3.516 | 4.020 | 4.185 | 4.416 | −0.139 [−0.173, −0.107] | −0.339 | 0 |
| TTS stride 1, lr 1e-5 | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | — | −0.200 | 69.63 |
| **S2+cal** | **4.408** | 3.779 | 4.370 | 4.578 | 4.756 | **+0.200 [+0.179, +0.222]** | — | **0** |
| SharpDraft (full ctx) | 3.644 | 3.517 | 3.980 | 3.630 | 3.557 | −0.564 [−0.624, −0.502] | −0.764 | 0 |
| SharpDraft + S2+cal | 4.410 | 3.781 | 4.369 | 4.579 | 4.761 | +0.202 | +0.002 [+0.001, +0.004] | 0 |
| thinking drafter (jgeuter, 3L) | 3.588 | 3.248 | 3.769 | 3.898 | 3.438 | −0.620 | −0.820 [−0.893, −0.742] | 0 |
| thinking drafter + S2 | 3.790 | 3.247 | 3.759 | 3.917 | 4.109 | −0.418 | −0.618 | 0 |
| cal-full | 4.166 | 3.763 | 4.322 | 4.342 | 4.187 | −0.042 [−0.077, −0.001] | −0.242 [−0.272, −0.206] | 0 |
| full (S2+cal + batched TTS s10) | 4.548 | 3.802 | 4.483 | 4.741 | 5.007 | +0.340 [+0.319, +0.360] | +0.140 [+0.122, +0.157] | 11.36 |

Per dataset: S2+cal AIME 4.607 / LCB 4.209; TTS 4.353 / 4.063; cal-full 4.365 / 3.968; full 4.722 / 4.374.

**Readings already settled (do not re-litigate):**
- S2+cal beats TTS at zero online cost. Both pre-registered rules ("beats", "cheaper") met.
- Two causes, two fixes: calibration alone is the largest single factor (+1.066 over baseline, nearly matches
  TTS) but still decays at ≥8K; S2 adds +0.242 on top, almost all late (≥8K +0.568). S2 alone +0.969,
  calibration alone +1.066, both +1.308 (sub-additive).
- Mechanism (4 dev traces): ~72% of the window's gain is from removing far middle keys, ~28% from position
  distance. Target-picked far chunks beat random chunks (+0.110) and a bigger window (+0.106).
- SharpDraft works on the full-context drafter (+0.544) but loses to S2 alone by 0.425; on top of S2+cal it adds
  +0.002 because S2 keeps N below N0 = 2048 (β̂ = 1 on every AIME round).
- The public thinking-mode drafter is 0.820 below S2+cal; S2 helps it too (+0.202).
- full: +0.140 over S2+cal for 11.4 ms/round = 16.3% of TTS's cost. **Fails the pre-registered ≤10% rule.**
  Reported as the cost–τ point, not as "cheaper." The 10% line was not relaxed and must not be.
- Negative results (closed): RLS on hidden states (R<0), KL output adapter (R=+0.12), VPA soft bias (all β>0
  hurt), drafter-attention top-k (below plain window).

## 6. Rules of the game (non-negotiable)

1. **Pre-register** every arm, its reading rule, and its budget in `RUNLOG.md` before any GPU run.
   Never change thresholds, metrics, splits, loop rule, bins or bootstrap after seeing results.
2. **Reuse saved arms**; never rerun S2, TTS, S2+cal, full, etc. on the same traces. New arms only.
3. **Tune only on dev.** lr for calibration was picked on 4 dev reference traces (aime24_II_14, aime24_I_10,
   lcb_abc393_d, lcb_arc191_d); none is in the test set. Keep it that way.
4. **Every new arm reports:** τ overall, per bin, per dataset, paired CI vs TTS stride 1 and vs S2+cal, update
   ms/round, draft ms/round, and the exact command in RUNLOG.md.
5. **Stage gates:** stop and report after each numbered item; post the table in RUNLOG.md; commit and push.
6. **Hygiene:** one output file per (trace, method, config), skip if exists, write-then-rename; tmux/nohup; run
   the 143 CPU tests before any GPU run; `results/ENV.md` records pip freeze, GPU, driver, commit.
7. **Honesty:** a failed pre-registered rule is reported as failed. Report partial runs as NOT RUN, not as
   "no effect."
8. **Secrets:** no provider API tokens anywhere in the repo, logs, or shell history. Pause/stop machines by hand.

## 7. Remaining experiments, in priority order

Estimates are A30-hours; scale by your GPU (A100 40GB ≈ 2× A30, H100/H200 ≈ 3–4×).

**A. Generality: second target (decides workshop vs main track).** Qwen/Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16`
(check it is a 5-layer drafter with the same feature-layer pattern [1,9,17,25,33]; if only a different shape
exists, use it and footnote). Same prompts, split, loop rule, bins, bootstrap, lr 1e-5, steps; no lr search.
Steps: config check (CPU) → 62 traces (vLLM, 16K, ~1.0) → 60 calibration traces (8K, ~0.4) + disjointness check →
baseline + S2 replay (~0.6) → calibrate in S2 layout, freeze hash (~1.0) → S2+cal replay (~0.4) → TTS stride 1 on
all 49 (~3.0, one worker: Adam state ≈ 15 GB) → cal-full (~1.5) → report. ~8 A30-h. Reading fixed: "transfers"
if S2+cal − TTS lower CI > 0 on 8B. If GPU time is short, run TTS on a pre-registered 16-trace subset
(8 AIME + 8 LCB, seed 0) first and the full 49 later.

**B. Real-loop wall-clock (reviewers will ask).** Actual DFlash speculative `generate` (greedy) on 3–4 prompts
(2 AIME, 1–2 LCB, 4096–8192 new tokens): baseline, S2+cal, TTS stride 1. Report tokens/s and τ; τ must match the
replay within 1%. S2's selection costs 0.6–1.1 ms/round unfused; show it does not eat the gain. Also run the
pending real (non-replay) loop check on 3 dev traces (`results/loop_check_proposal.md`). ~1 A30-h.

**C. Three baseline rows (one table row each).**
  1. TTS stride 1 on the S2+cal weights, S2 layout (~2.5). Strongest adaptive baseline. Our claim stays
     "matches or beats TTS at zero online cost," whatever this gives.
  2. SharpDraft + S2+cal at N0 = 1024 (~0.4). Sensitivity check: the "no-op under S2" result depends on
     N0 = 2048; the paper's own sweep shows 1K ≈ 2K, and S2's AIME visible count (≤1,965) would be above 1K.
  3. S2+cal with D-PACE slot weights (arXiv 2605.18810, Eq. 8, α = 0.5, detached; keep the soft KL) (~1.2).
     Note Eq. 8 and App. B.1 differ; use Eq. 8.

**D. Second drafter family (stretch).** EAGLE-3 on Qwen3-8B or Llama-3.1-8B. New code path (autoregressive
drafter, not block diffusion); only if A–C are done.

**E. Thinking drafter + TTS (~1.3)** — low priority; completes the jgeuter row.

Cut order if budget is short: E, D, C3, C2, C1, then B's loop check.

## 8. Related work the paper must cite (one line each on how we differ)

- **SharpDraft** (2610.05106, 4 Oct 2026): same problem and diagnosis (attention-mass dilution), training-free
  query scaling. We change which keys the drafter sees instead; SharpDraft is below S2 and negligible on top.
  Their "Full Adaptation" baseline cites TTS and gets no speedup; supports our cost argument.
- **Windowed-MTP** (2607.21535): sink + sliding window on MTP heads at 1M-token prompts, for draft-KV cost;
  acceptance barely moves; rejects dynamic selection. Ours is for acceptance on long outputs; S2's target-guided
  selection adds +0.17.
- **AngelSpec** (2607.25852) Table 6: no-think drafter on high-think prompts 4.79 → 3.29; their fix is a separate
  drafter. We close the mismatch with a 0.33–0.75 GPU-h calibration of released weights.
- **D-PACE** (2605.18810): acceptance-weighted slot CE, +10% τ in full training. Candidate weighting (C3).
- **H-Spec** (2609.24197): trained drafter with a 2,048-token window over target KV. Window is a known design
  choice when training from scratch.
- **Attention Drift** (2605.09992): drafter attention moves from prompt to recent tokens. Explains why the
  window loses little.
- **Test-Time Speculation** (2605.09329): the baseline.
- Not overlapping: Strong Drafts Need Compact Memories (2608.30252), KVShot (2604.26412), Draft-OPD (2605.29343),
  SpecStream (2609.33184).

The "pure window" paper is already published twice over; the paper leads with **the two-cause decomposition,
target-guided selection, and zero-cost calibration beating online adaptation**.

## 9. Paper skeleton (start drafting now; A changes a table, not the story)

1. Problem: acceptance decay on long reasoning outputs; TTS fixes it at 69.6 ms/round.
2. Diagnosis: two causes. Position-resolved gap table (early = mode mismatch, late = context). 72/28 split.
3. Method: S2 layout (target-guided far chunks) + one-time calibration. Both cheap.
4. Results: the table in §5. S2+cal beats TTS at 0 ms. cal-full and S2-alone decomposition.
5. Baselines reviewers expect: SharpDraft, thinking-trained drafter, bigger window, random chunks, drafter-side
   selection, hybrid cost–τ curve.
6. Generality (from A) and wall-clock (from B).
7. Limitations: replay-based, one drafter family until D, N0 sensitivity, greedy only.

## 10. How to work with the agent

- Give it this file, then say: "Set up per §3, run the 143 tests and the rerun check, report, stop."
- Then one item at a time from §7, each pre-registered. Ask for the table after each item.
- Budget caps per session; the agent must report projected hours and stop if the projection exceeds the cap.
- It should never relax a rule in §6 or re-run a saved arm.
