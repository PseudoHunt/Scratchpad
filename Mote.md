# MOTE — handoff for continued development

**Purpose of this file.** Everything needed to continue developing MOTE, the agent-native kernel language for phone LLM decode. It covers the motivation, what was decided, what was built and verified (as of 11 Oct 2026), the open questions, and the next steps. The three source files (`mote.lark`, `examples.mote`, `check.py`) are included in full at the end, so this file is self-contained. Copy them out and run `python3 check.py` (needs `lark` and `numpy`).

**Owner:** Sanjay (LLM efficiency / hardware co-design, Samsung; not on the Exynos side, so no NPU toolchain access — targets are Arm CPU first, mobile GPU via OpenCL/Vulkan second).

---

## 1. The idea in one paragraph

MOTE is a tiny kernel language made for LLM agents, not humans, aimed at batch-1 LLM decode on phones. The agent writes *what* to compute (tiles and math, indexed by bare names) plus a *schedule* made of nine fixed knobs. The compiler handles the chip: tiling defaults from a chip profile, bounds and tail handling, NEON/SME2 or GPU subgroup mapping. Every mistake comes back as a one-line error with a fix. Every run comes back as a five-line report that includes the memory-bandwidth floor, so the agent knows when to stop tuning. The headline claim to prove: **agents produce faster mobile kernels, using far fewer tokens, than when they write C++/OpenCL directly.**

MOTE is a working name.

## 2. Why this direction (how we got here)

1. Starting point was "improve a CuTe-like DSL with Halide ideas." That conversation found Hexcute (NVIDIA + U Toronto) already does layout synthesis for CuTe; the open gaps were bounds inference / auto-predication, a first-class searchable schedule, and paged + quantized decode.
2. Sanjay ruled out testing/verification tools (equivalence checkers, benchmarks, cost models) as low-value for his context. Don't propose those again.
3. The reframing that stuck: if LLMs write most kernel code, the winning tools are the ones LLMs write *well and cheaply* (DHH's point about Rails and token counts). So: a compact, strict language with defaults, not a verifier.
4. Scope was narrowed to **batch-1 decode for 1–3B models**, Arm CPU first, Adreno/Mali second. Low-bit weights and KV are just input *types* here, not the research contribution (Sanjay does not want this to be a quantization project).

## 3. Prior work to position against (all checked Oct 2026)

| Work | What it does | Why MOTE is different |
| --- | --- | --- |
| µCUTLASS (NVIDIA, arXiv 2603.29010, Mar 2026) | Compact DSL learned in context + "speed-of-light" guidance for agents; weaker models beat stronger baseline agents at lower token cost; SOL budgeting saves 19–43% of tokens | GPU/CUTLASS only; it is a config layer over CUTLASS, not a language with its own compiler. MOTE borrows both principles (in-prompt grammar, floor feedback) and brings them to phones. |
| Ave (arXiv 2604.18616) | Tile-based Pythonic DSL for agents with data-flow invariants, AMD MI300X | Datacenter GPU; MOTE's knobs are phone-specific (`cores`, `dot i8`). |
| AutoMegaKernel (arXiv 2606.09682, Jun 2026) | Agent harness that builds whole-model persistent CUDA megakernels; static schedule validator; W8A16 beats cuBLAS bf16 on inference-class GPUs | CUDA only. Supports the "fused decode step" direction for CPU later. |
| MobileKernelBench / MoKA (ICLR 2026, arXiv 2603.11935) | First mobile kernel-gen benchmark (MNN CPU backend). Plain LLMs fail to compile >54% of the time (hallucinated APIs, data scarcity). MoKA agent: 93.7% compile, 27.4% of kernels beat native libs | **This is the baseline to beat.** It shows the problem (agents can't write mobile kernels) and the gap (a small in-prompt language removes the data-scarcity problem). Nobody has done an agent-facing language for phones. |
| AKG kernel agent (arXiv 2512.23424), GEAK (AMD), KernelCraft | Multi-agent kernel generation across Triton/TileLang/CUDA, GPUs and NPUs | Agents over existing languages; none designs the language for the agent. |
| Halide | Algorithm/schedule split, bounds inference | Borrowed directly (rules 1–2 below). |

## 4. Metrics and the 12-week plan (as proposed)

Measure three things:
- Tokens per working kernel (vs the agent writing C++/OpenCL, and vs MoKA)
- Share of kernels that beat native libraries (MoKA's 27.4% is the bar)
- End-to-end tokens/sec vs llama.cpp on a real phone

Plan: weeks 1–4 language + Arm CPU compiler for ~6 decode ops; weeks 5–8 agent loop with on-phone timing, head-to-head vs MoKA; weeks 9–12 fused decode step, Adreno backend, live phone demo.

## 5. Design rules (decided)

1. **No index math, pointers, masks or barriers.** Tensors are indexed by bare names. The compiler works out bounds and loop tails.
2. **What is split from how.** The kernel says what to compute. A separate `schedule` block says how, with nine fixed knobs. Anything else is a parse error.
3. **Layout lives in the type.** Quantization (`q4g32`) and paging (`paged(table, 16)`) are part of a tensor's type. Change the type; the body stays the same.
4. **Every reduction is a merge.** A merge = `init` (empty state) + `lift` (one element → state) + associative `join` + output. So the compiler can split any reduction across workers.
5. **Defaults come from the chip profile.** A kernel with no schedule still runs. The agent overrides only what it wants.
6. **Feedback is short and fixed.** One-line errors with a fix. Five-line run reports with the bandwidth floor.

**Naming rule:** dims are UPPERCASE sizes; the lowercase name is the index over that dim (`t` over `T`, `kv` over `KV`). The checker knows every index's range without declarations and catches swapped indices.

**Types:** `f32 f16 bf16 i8 i32`; `qNgM` = N-bit codes with one f16 scale per M values along the last dim (symmetric; N ∈ {2,4,8}, M ∈ {32,64,128}); `paged(table, P)` on the first dim: row `t` lives in page `table[t / P]`, slot `t % P`. `let` values are f32; outputs round to their declared type. A size like `D` can be used as a number (`rsqrt(D)`).

## 6. The nine schedule knobs

| Knob | Does | CPU | GPU | Default |
| --- | --- | --- | --- | --- |
| `parallel x` | spread output index x over workers | threads | workgroups | outermost output index |
| `split r N` | N partial results for reduction r, joined at the end | one partial per thread | one per workgroup + small merge pass | off unless output work can't fill workers |
| `tile x S` | S values of x per step | register/L1 blocking | values per workgroup step | one step fits half of L1 |
| `order a b ...` | loop order, outer→inner | loop nest | loop nest in a workgroup | outputs outside, reductions inside |
| `vector x [W]` | x across W lanes | NEON lanes | workgroup threads + subgroup reduce | innermost index, W from profile |
| `unroll x N` | unroll | unroll | unroll | off |
| `stage v per L` | compute let-value v once per L (`worker`, `launch`, or an index), keep it close | registers/L1 | registers/local mem/pre-pass | per worker |
| `dot f32/f16/i8` | inner-product math | f32 FMA / FP16 FMA / SDOT, i8mm | f32 / f16 / int8 dot | f32 |
| `cores prime/big/all` | which CPU clusters | thread pinning | not allowed | big |

Only `dot f16/i8` changes numerics; the report shows error vs an f32 reference. In v0 a knob applies to every loop over its index.

## 7. The two example kernels

**Kernel 1 — RMSNorm fused into a 4-bit matvec** (`norm_matvec`): three lines of math, CPU and GPU schedules. The interesting schedule choice is `stage xn per worker` (CPU: 4 threads each re-normalize x, cheap) vs `stage xn per launch` (GPU: at N=8192, 4,096 workgroups each redoing the norm doubles the arithmetic, so a pre-pass is the better first guess).

**Kernel 2 — paged GQA decode attention** (`decode_attn`): two lines of math cover paging, grouped heads and online softmax. GQA lives in the shape (`q: [KV, G, D]`, `K[t, kv, d]` doesn't depend on `g`; `order kv t g d` loads each K/V tile once for all G heads). `split t 4` is flash-decoding as one line. Switching K/V to `q8g32` is a type change only; the body is unchanged (verified). At T=4096, KV=8, D=128, G=3 the bytes per call drop from 16.79 MB (f16) to 8.93 MB.

Full source is in `examples.mote` below.

## 8. Chip profile and report format

```
# phone-a.profile (example values; `mote calibrate` would measure them)
cpu  cores    prime:1 big:3 little:4
cpu  simd     neon128 fp16 dotprod i8mm
cpu  cache    l1:64K l2:512K l3:8M
cpu  bw       60GB/s      # measured streaming read, prime + big cores
gpu  subgroup 64
gpu  local    32K
gpu  bw       55GB/s
```

```
norm_matvec on phone-a/cpu   K=3072 N=8192   cores big (4 threads)
check   ok, max |err| 2.1e-3 vs f32 reference (dot i8 changes numerics)
time    0.410 ms, median of 50 runs after warm-up
bytes   14.18 MB, floor 0.236 ms at 60 GB/s
score   58% of floor
```

Bytes and floor are computed from sizes and profile (14,184,448 bytes at K=3072, N=8192 is real). **Time and error are placeholders: there is no compiler yet.** Proposed stop rule: stop at 90% of floor, or after three tries without a gain.

## 9. What has been built and verified (11 Oct 2026)

Files: `mote.lark` (grammar, 46 non-comment lines, Lark/Earley), `examples.mote` (3 merges, 2 kernels, 4 schedules), `check.py` (parser, static checks, one-line error messages, float64 reference interpreter, negative tests).

Verified:
- Both kernels parse with **zero ambiguous parses** (Earley with `ambiguity="explicit"`).
- Static checks pass: index binding, dim/index naming, `let` scoping, paged layout needs an `i32` table, schedule legality (`parallel` needs an output index, `split` needs a reduced index, `stage` needs a `let`, `cores` is CPU-only, no duplicate knobs).
- Reference interpreter vs plain NumPy (float64):

| Run | max abs error |
| --- | --- |
| norm_matvec K=64 N=8, no split | 3.8e-15 |
| norm_matvec, split k 3 | 2.9e-15 |
| decode_attn T=100 paged (shuffled page table), no split | 2.2e-16 |
| decode_attn, split t 7 | 1.4e-16 |
| decode_attn, split t 4 + tile t 64 (2 of 4 splits empty) | 2.2e-16 |
| same, but join run on the empty splits | **NaN** |
| decode_attn with K,V as q8g32 (D=32), split t 4 tile t 64 | 2.8e-16 |

- Five common agent mistakes each return one actionable line:

| Agent wrote | Checker says |
| --- | --- |
| `y[n] = W[n, k] * x[k]` | index k is not bound here. Wrap it in a reduction, e.g. sum(k: ...) |
| `sum(k: W[k, n] * x[k])` | W[k, n]: dim 0 of W is N, so the index there must be n (got k) |
| `y[k] = x[k+1]` | indices are bare names, no index math (got '+' inside [...]). |
| `split n 4` | split needs a reduced index; n is an output index. Did you mean 'parallel n' or 'tile n 4'? |
| `swizzle k 4` | 'swizzle' is not a knob. Knobs: parallel, split, tile, order, vector, unroll, stage, dot, cores. |

**One real finding from testing:** with `split t 4`, `tile t 64`, T=100, two splits get no keys. Joining two empty online-softmax states computes `0 × exp(−inf + inf)` = NaN. Language rule adopted: `init` must be the identity, and the compiler/interpreter *skips* a join with an empty state instead of running it. This must carry into the compiler (empty splits are common on GPU with `split t 16`).

Not built: any code generation, any on-device timing, the agent loop, the chip-profile calibrator, GBNF conversion for constrained decoding (the reserved-word lookahead in `NAME` needs a rewrite for GBNF).

## 10. Open questions (not decided)

- **RoPE needs pairs of values.** Add a `view` (e.g. `x as [2, D/2]`) so rotate-half needs no index math?
- **Fast path for custom merges.** The compiler can special-case `softmax_sum` (tile max first, then exp). Should `merge` let users declare a tile-level lift so their own merges get the same?
- **Per-statement schedules.** `norm_matvec` may want `vector k` for the matvec but not the norm. v0 applies a knob to every loop over its index.
- **Zero points.** Asymmetric types (scale + zero point) for GPTQ/AWQ-style weights?
- **Where sizes bind.** Sizes come from a model config at build time. Would a `sizes` line in the source help the agent?
- **Whole-layer kernels.** Several outputs and phases, so a full decode layer becomes one CPU megakernel. (Mobile GPUs don't guarantee cross-workgroup sync, so GPU gets multi-op fused kernels, not a true megakernel.)
- **SME2.** Map `tile` and `dot` onto SME2 outer products where available.

## 11. Suggested next steps, in order

1. **Arm CPU backend for `norm_matvec`** (C + NEON intrinsics, or C with pragmas as a first cut). Get real `time` and `check` lines in the report. Start with `dot f32`, then `dot i8` via SDOT with per-32-group quantization of `xn`.
2. **`mote calibrate`**: measure streaming bandwidth on prime+big cores, L1/L2 sizes, SIMD features; write the profile file.
3. **Default inference** from the profile so a schedule-less kernel runs.
4. **Agent loop**: grammar + profile + last report in the prompt; parse → check → compile → run → report; stop rule.
5. **`decode_attn` on CPU** with split/merge of partial softmax states; confirm the empty-split rule in generated code.
6. **Head-to-head vs MoKA** on the MobileKernelBench op set where it overlaps decode; measure tokens per working kernel and share beating native.
7. Then GPU (OpenCL on Adreno), fused decode step, phone demo.

---

## Appendix A — `mote.lark`

```
// Mote v0 core grammar (Lark, Earley). Kept small on purpose: it goes in the agent's prompt.
// The grammar itself enforces two rules: indices are bare names (no index math),
// and a schedule can only use the nine knobs listed below.

start: item*
?item: kernel | merge | schedule

// ---- kernel: what to compute ----
kernel: "kernel" NAME "(" params ")" "->" params "{" stmt+ "}"
params: param ("," param)*
param: NAME ":" type
type: ELEM shape layout?
shape: "[" NAME ("," NAME)* "]"
layout: "paged" "(" NAME "," INT ")"

stmt: "let" target "=" expr      -> let
    | target "=" expr            -> store
target: NAME ("[" NAME ("," NAME)* "]")?

?expr: arith
?arith: term | arith "+" term -> add | arith "-" term -> sub
?term: unary | term "*" unary -> mul | term "/" unary -> div
?unary: atom | "-" unary -> neg
?atom: NUMBER                                  -> num
     | NAME                                    -> var
     | NAME "." NAME                           -> field
     | NAME "[" NAME ("," NAME)* "]"           -> load
     | NAME "(" expr ("," expr)* ")"           -> call
     | NAME "(" NAME ":" expr ("," expr)* ")"  -> reduce
     | "(" expr ")"

// ---- merge: a reduction as init, lift, join, out. sum, max and softmax_sum are merges ----
merge: "merge" NAME "(" NAME ("," NAME)* ")" "->" expr "{" "init" sets "lift" sets "join" "(" NAME "," NAME ")" sets "}"
sets: set ("," set)*
set: NAME "=" expr

// ---- schedule: how to compute it. A closed list; anything else is a parse error ----
schedule: "schedule" NAME "on" TARGET "{" knob* "}"
?knob: "parallel" NAME          -> parallel
     | "split" NAME INT         -> split
     | "tile" NAME INT          -> tile
     | "order" NAME NAME+       -> order
     | "vector" NAME INT?       -> vector
     | "unroll" NAME INT        -> unroll
     | "stage" NAME "per" NAME  -> stage
     | "dot" DOTMODE            -> dot
     | "cores" CORESET          -> cores

TARGET: "cpu" | "gpu"
DOTMODE: "f32" | "f16" | "i8"
CORESET: "prime" | "big" | "all"
ELEM: /(f32|f16|bf16|i8|i32|q[248]g(32|64|128))\b/
NAME: /(?!(kernel|let|merge|init|lift|join|schedule|on|paged|parallel|split|tile|order|vector|unroll|stage|per|dot|cores)\b)[A-Za-z_][A-Za-z0-9_]*/
INT: /[0-9]+/
NUMBER: /[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?/
COMMENT: /#[^\n]*/
%import common.WS
%ignore WS
%ignore COMMENT
```

## Appendix B — `examples.mote`

```
# Mote v0 examples: the prelude merges, two decode kernels, and their schedules.

# ---------- prelude: every reduction is a merge ----------
merge sum(x) -> a {
  init a = 0
  lift a = x
  join(p, q) a = p.a + q.a
}

merge max(x) -> m {
  init m = -inf
  lift m = x
  join(p, q) m = max(p.m, q.m)
}

# Online softmax as a merge. State: running max m, normalizer l, weighted sum a.
merge softmax_sum(s, v) -> a / l {
  init m = -inf, l = 0, a = 0
  lift m = s,    l = 1, a = v
  join(x, y) m = max(x.m, y.m),
             l = x.l * exp(x.m - m) + y.l * exp(y.m - m),
             a = x.a * exp(x.m - m) + y.a * exp(y.m - m)
}

# ---------- kernel 1: RMSNorm fused into a 4-bit matvec ----------
# y = W . (rmsnorm(x) * g), for one decode token
kernel norm_matvec(x: f16[K], g: f16[K], W: q4g32[N, K]) -> y: f16[N] {
  let r = rsqrt(sum(k: x[k] * x[k]) / K + 1e-5)
  let xn[k] = x[k] * g[k] * r
  y[n] = sum(k: W[n, k] * xn[k])
}

schedule norm_matvec on cpu {
  cores    big             # prime + big cores; little cores would straggle
  parallel n
  tile     n 4             # 4 rows per step: each xn load feeds 4 rows
  vector   k               # lane count comes from the chip profile
  stage    xn per worker   # normalize once per thread, keep it in L1
  dot      i8              # int8 dot product; xn is quantized per 32-group
}

schedule norm_matvec on gpu {
  parallel n
  tile     n 2             # 2 rows per workgroup
  vector   k 64            # 64 lanes split k, then a subgroup reduce
  stage    xn per launch   # one small pre-pass; per workgroup it would double the math
  dot      f16
}

# ---------- kernel 2: paged GQA decode attention ----------
# o = softmax(q . K^T / sqrt(D)) . V for one new token. G query heads share each KV head.
kernel decode_attn(q: f16[KV, G, D],
                   K: f16[T, KV, D] paged(table, 16),
                   V: f16[T, KV, D] paged(table, 16),
                   table: i32[NB]) -> o: f16[KV, G, D] {
  let s[kv, g, t] = sum(d: q[kv, g, d] * K[t, kv, d]) * rsqrt(D)
  o[kv, g, d] = softmax_sum(t: s[kv, g, t], V[t, kv, d])
}

schedule decode_attn on cpu {
  cores    big
  parallel kv
  split    t 4             # flash-decoding: 4 partial states per head, merged at the end
  tile     t 64            # 64 keys per step
  order    kv t g d        # t outside g: each K/V tile is loaded once for all G heads
  vector   d
  stage    s per t         # scores for one 64-key tile stay in registers
}

schedule decode_attn on gpu {
  parallel kv
  split    t 16            # a GPU needs far more workgroups than a CPU has cores
  tile     t 32
  order    kv t g d
  vector   d
  stage    s per t
}
```

## Appendix C — `check.py`

```python
#!/usr/bin/env python3
"""Mote v0: parse, static checks, and a tiny reference interpreter.

    python3 check.py            # needs lark and numpy

1. Parses examples.mote with mote.lark.
2. Runs the static checks (index naming, binding, schedule legality).
3. Shows the one-line errors the agent gets for common mistakes.
4. Runs both kernels in a slow reference interpreter and compares with plain
   NumPy, including split reductions with empty splits.
"""
import math
import os
import re

import numpy as np
from lark import Lark, Token
from lark.exceptions import UnexpectedInput

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "mote.lark")) as f:
    GRAMMAR = f.read()
PARSER = Lark(GRAMMAR, parser="earley", propagate_positions=True)

FUNCS = {
    "exp": (1, math.exp), "log": (1, math.log), "sqrt": (1, math.sqrt),
    "rsqrt": (1, lambda v: 1.0 / math.sqrt(v)), "abs": (1, abs), "tanh": (1, math.tanh),
    "sigmoid": (1, lambda v: 1.0 / (1.0 + math.exp(-v))),
    "silu": (1, lambda v: v / (1.0 + math.exp(-v))),
    "max": (2, max), "min": (2, min),
}
CONSTS = {"inf": math.inf}
KNOBS = ["parallel", "split", "tile", "order", "vector", "unroll", "stage", "dot", "cores"]


class MoteError(Exception):
    pass


def err(line, msg):
    raise MoteError(f"line {line}: {msg}")


# ------------------------------------------------------------------ parsing
def terminal_text(name):
    """Show a terminal as the agent would type it ('dot', '}'), not Lark's internal name."""
    try:
        pat = PARSER.get_terminal(name).pattern
        return pat.value if pat.type == "str" else name.lower()
    except KeyError:
        return name.lower()


def parse(src):
    try:
        return PARSER.parse(src)
    except UnexpectedInput as e:
        lines = src.splitlines()
        text = lines[e.line - 1][e.column - 1:] if 0 < e.line <= len(lines) else ""
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)
        got = m.group(0) if m else "end of input"
        allowed = getattr(e, "allowed", None) or getattr(e, "expected", None) or set()
        allowed = {terminal_text(a) for a in allowed}
        if allowed and allowed <= set(KNOBS) | {"}"}:
            err(e.line, f"'{got}' is not a knob. Knobs: {', '.join(KNOBS)}.")
        if allowed and allowed <= {",", "]"}:
            err(e.line, f"indices are bare names, no index math (got '{got}' inside [...]).")
        err(e.line, f"unexpected '{got}'; expected one of: {', '.join(sorted(allowed))}")


def sets(tree):
    return [(str(s.children[0]), s.children[1]) for s in tree.children]


class Param:
    def __init__(self, t):
        self.name, ty = str(t.children[0]), t.children[1]
        self.elem = str(ty.children[0])
        self.dims = [str(d) for d in ty.children[1].children]
        self.layout = None
        if len(ty.children) > 2:
            lay = ty.children[2]
            self.layout = (str(lay.children[0]), int(lay.children[1]))
        self.line = t.meta.line


class Kernel:
    def __init__(self, t):
        c = t.children
        self.name, self.line = str(c[0]), t.meta.line
        self.inputs = [Param(p) for p in c[1].children]
        self.outputs = [Param(p) for p in c[2].children]
        self.stmts = []
        for s in c[3:]:
            tgt, expr = s.children
            idxs = [str(x) for x in tgt.children[1:]]
            self.stmts.append((str(s.data), str(tgt.children[0]), idxs, expr, s.meta.line))
        self.dims, self.lets, self.out_idx, self.red_idx = set(), {}, set(), set()


class Merge:
    def __init__(self, t):
        c = t.children
        self.name, self.line = str(c[0]), t.meta.line
        i, self.params = 1, []
        while isinstance(c[i], Token):
            self.params.append(str(c[i]))
            i += 1
        self.out, self.init, self.lift = c[i], c[i + 1], c[i + 2]
        self.x, self.y, self.join = str(c[i + 3]), str(c[i + 4]), c[i + 5]


class Schedule:
    def __init__(self, t):
        c = t.children
        self.kernel, self.target, self.line = str(c[0]), str(c[1]), t.meta.line
        self.knobs = [(str(k.data), [str(x) for x in k.children], k.meta.line) for k in c[2:]]


class Program:
    def __init__(self, src, base=None):
        self.merges = dict(base.merges) if base else {}
        self.kernels, self.schedules = {}, []
        for item in parse(src).children:
            if item.data == "merge":
                m = Merge(item)
                self.merges[m.name] = m
            elif item.data == "kernel":
                k = Kernel(item)
                self.kernels[k.name] = k
            else:
                self.schedules.append(Schedule(item))
        for m in self.merges.values():
            check_merge(m)
        for k in self.kernels.values():
            check_kernel(self, k)
        for s in self.schedules:
            check_schedule(self, s)


# ------------------------------------------------------------ static checks
def check_merge(m):
    fields = [f for f, _ in sets(m.init)]
    for part in (m.lift, m.join):
        if [f for f, _ in sets(part)] != fields:
            err(m.line, f"merge {m.name}: init, lift and join must set the same fields ({', '.join(fields)})")


def check_indices(line, name, dims, idxs):
    if len(idxs) != len(dims):
        err(line, f"{name} has {len(dims)} dims [{', '.join(dims)}] but got {len(idxs)} indices")
    for j, (i, d) in enumerate(zip(idxs, dims)):
        if i != d.lower():
            err(line, f"{name}[{', '.join(idxs)}]: dim {j} of {name} is {d}, "
                      f"so the index there must be {d.lower()} (got {i})")


def unbound(line, i):
    err(line, f"index {i} is not bound here. Wrap it in a reduction, e.g. sum({i}: ...)")


def check_kernel(prog, k):
    params = {p.name: p for p in k.inputs + k.outputs}
    for p in params.values():
        for d in p.dims:
            if not d.isupper():
                err(p.line, f"dim '{d}' of {p.name} must be UPPERCASE; its index is then '{d.lower()}'")
            k.dims.add(d)
        if p.layout and (p.layout[0] not in params or params[p.layout[0]].elem != "i32"):
            err(p.line, f"paged({p.layout[0]}, ...) needs an i32 parameter named {p.layout[0]}")
    tensors = {n: p.dims for n, p in params.items()}
    outputs, stored = {o.name for o in k.outputs}, set()
    for kind, name, idxs, expr, line in k.stmts:
        if len(set(idxs)) != len(idxs):
            err(line, f"{name}: an index appears twice")
        if kind == "store":
            if name not in outputs:
                err(line, f"{name} is not an output; write 'let {name} ...' for a temporary")
            check_indices(line, name, params[name].dims, idxs)
            stored.add(name)
        else:
            if name in tensors:
                err(line, f"{name} is already defined")
            for i in idxs:
                if not i.islower() or i.upper() not in k.dims:
                    err(line, f"index '{i}' has no matching dim '{i.upper()}' in the parameters")
            tensors[name] = [i.upper() for i in idxs]
            k.lets[name] = idxs
        k.out_idx |= set(idxs)
        check_expr(prog, k, tensors, expr, set(idxs), line)
    for o in k.outputs:
        if o.name not in stored:
            err(o.line, f"output {o.name} is never written")


def check_expr(prog, k, tensors, e, bound, line):
    t, ch = str(e.data), e.children
    if t == "num":
        return
    if t == "var":
        n = str(ch[0])
        if n in bound or n in CONSTS or n in k.dims:
            return
        if n in tensors:
            if tensors[n]:
                err(line, f"{n} is a tensor; index it, e.g. {n}[{', '.join(d.lower() for d in tensors[n])}]")
            return
        if n.islower() and n.upper() in k.dims:
            unbound(line, n)
        err(line, f"unknown name '{n}'")
    if t == "field":
        err(line, f"'{ch[0]}.{ch[1]}': fields only exist inside a merge")
    if t == "load":
        n, idxs = str(ch[0]), [str(x) for x in ch[1:]]
        if n not in tensors:
            err(line, f"unknown tensor '{n}'")
        check_indices(line, n, tensors[n], idxs)
        for i in idxs:
            if i not in bound:
                unbound(line, i)
        return
    if t == "call":
        f = str(ch[0])
        if f not in FUNCS:
            err(line, f"unknown function '{f}'. Functions: {', '.join(FUNCS)}")
        if len(ch) - 1 != FUNCS[f][0]:
            err(line, f"{f} takes {FUNCS[f][0]} argument(s)")
        for a in ch[1:]:
            check_expr(prog, k, tensors, a, bound, line)
        return
    if t == "reduce":
        f, i = str(ch[0]), str(ch[1])
        if f not in prog.merges:
            err(line, f"'{f}' is not a merge. Merges: {', '.join(prog.merges)}")
        if len(ch) - 2 != len(prog.merges[f].params):
            err(line, f"{f} takes {len(prog.merges[f].params)} operand(s)")
        if not i.islower() or i.upper() not in k.dims:
            err(line, f"cannot reduce over '{i}': there is no dim {i.upper()}")
        if i in bound:
            err(line, f"index {i} is already bound, so it cannot be reduced here")
        k.red_idx.add(i)
        for a in ch[2:]:
            check_expr(prog, k, tensors, a, bound | {i}, line)
        return
    for a in ch:  # add, sub, mul, div, neg
        check_expr(prog, k, tensors, a, bound, line)


def check_schedule(prog, s):
    if s.kernel not in prog.kernels:
        err(s.line, f"no kernel named {s.kernel}")
    k = prog.kernels[s.kernel]
    idx, seen = k.out_idx | k.red_idx, set()
    for knob, args, line in s.knobs:
        key = knob if knob in ("dot", "cores", "order") else f"{knob} {args[0]}"
        if key in seen:
            err(line, f"'{key}' is given twice")
        seen.add(key)

        def need(i):
            if i not in idx:
                err(line, f"{k.name} has no index '{i}'. Indices: {', '.join(sorted(idx))}")

        if knob == "parallel":
            need(args[0])
            if args[0] not in k.out_idx:
                err(line, f"parallel needs an output index; {args[0]} is reduced. "
                          f"Use 'split {args[0]} N' to spread a reduction over workers.")
        elif knob == "split":
            need(args[0])
            if args[0] not in k.red_idx:
                err(line, f"split needs a reduced index; {args[0]} is an output index. "
                          f"Did you mean 'parallel {args[0]}' or 'tile {args[0]} {args[1]}'?")
            if int(args[1]) < 2:
                err(line, "split count must be at least 2")
        elif knob in ("tile", "vector", "unroll"):
            need(args[0])
            if len(args) > 1 and int(args[1]) < 1:
                err(line, f"{knob} size must be at least 1")
        elif knob == "order":
            for i in args:
                need(i)
            if len(set(args)) != len(args):
                err(line, "order lists an index twice")
        elif knob == "stage":
            if args[0] not in k.lets:
                err(line, f"stage needs a 'let' value; {args[0]} is not one. Lets: {', '.join(k.lets) or 'none'}")
            if args[1] not in ("worker", "launch") and args[1] not in idx:
                err(line, f"'stage {args[0]} per {args[1]}': use worker, launch, or an index")
        elif knob == "cores" and s.target != "cpu":
            err(line, "cores only applies on cpu")


# --------------------------------------------------- reference interpreter
EMPTY = object()  # state of a split that got no elements


class Ctx:
    def __init__(self, prog, sizes, splits, skip_identity):
        self.prog, self.sizes, self.splits = prog, sizes, splits
        self.skip_identity, self.tensors = skip_identity, {}


def logical(p, arrays, sizes):
    """The logical value of an input, exactly as its type defines it."""
    a = arrays[p.name]
    if p.elem.startswith("q"):  # qNgM: integer codes times one f16 scale per group of M
        codes, scales = a
        group = int(p.elem.split("g")[1])
        a = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), group, axis=-1)
    else:
        a = np.asarray(a, dtype=np.float64)
    if p.layout:  # paged: logical row t lives in page table[t // P], slot t % P
        tbl, page = p.layout
        table = arrays[tbl]
        a = np.stack([a[table[t // page], t % page] for t in range(sizes[p.dims[0]])])
    return a


def ev(e, scope, cx):
    t, ch = str(e.data), e.children
    if t == "num":
        return float(ch[0])
    if t == "var":
        n = str(ch[0])
        if n in scope:
            return scope[n]
        if n in cx.sizes:
            return float(cx.sizes[n])
        if n in CONSTS:
            return CONSTS[n]
        return cx.tensors[n]
    if t == "field":
        return scope[str(ch[0])][str(ch[1])]
    if t == "load":
        return cx.tensors[str(ch[0])][tuple(scope[str(i)] for i in ch[1:])]
    if t == "call":
        return FUNCS[str(ch[0])][1](*[ev(a, scope, cx) for a in ch[1:]])
    if t == "reduce":
        return reduce_(e, scope, cx)
    a = [ev(x, scope, cx) for x in ch]
    if t == "neg":
        return -a[0]
    return {"add": a[0] + a[1], "sub": a[0] - a[1], "mul": a[0] * a[1],
            "div": a[0] / a[1] if a[1] != 0 else math.copysign(math.inf, a[0]) if a[0] else math.nan}[t]


def run_sets(tree, scope, cx):
    st = {}
    for f, x in sets(tree):
        st[f] = ev(x, {**scope, **st}, cx)
    return st


def join(m, a, b, cx):
    if cx.skip_identity:  # init is the identity, so an empty side is skipped
        if a is EMPTY:
            return b
        if b is EMPTY:
            return a
    a = run_sets(m.init, {}, cx) if a is EMPTY else a
    b = run_sets(m.init, {}, cx) if b is EMPTY else b
    return run_sets(m.join, {m.x: a, m.y: b}, cx)


def reduce_(e, scope, cx):
    m = cx.prog.merges[str(e.children[0])]
    i, args = str(e.children[1]), e.children[2:]
    n = cx.sizes[i.upper()]
    nsplit, tile = cx.splits.get(i, (1, 1))
    chunk = -(-n // nsplit)
    chunk = -(-chunk // tile) * tile  # each split covers whole tiles
    parts = []
    for c in range(nsplit):
        st = EMPTY
        for v in range(c * chunk, min(n, (c + 1) * chunk)):
            ops = [ev(a, {**scope, i: v}, cx) for a in args]
            st = join(m, st, run_sets(m.lift, dict(zip(m.params, ops)), cx), cx)
        parts.append(st)
    while len(parts) > 1:  # merge split partials as a tree, like parallel workers would
        parts = [join(m, parts[j], parts[j + 1], cx) if j + 1 < len(parts) else parts[j]
                 for j in range(0, len(parts), 2)]
    st = parts[0] if parts[0] is not EMPTY else run_sets(m.init, {}, cx)
    return ev(m.out, st, cx)


def run(prog, kname, sizes, arrays, splits=None, skip_identity=True):
    k = prog.kernels[kname]
    cx = Ctx(prog, sizes, splits or {}, skip_identity)
    for p in k.inputs:
        cx.tensors[p.name] = logical(p, arrays, sizes)
    for _, name, idxs, expr, _ in k.stmts:
        out = np.empty([sizes[i.upper()] for i in idxs])
        for pos in np.ndindex(*out.shape):
            out[pos] = ev(expr, dict(zip(idxs, pos)), cx)
        cx.tensors[name] = out if idxs else float(out)
    return cx.tensors[k.outputs[0].name]


# --------------------------------------------------------------- the checks
MV = "kernel mv(x: f16[K], W: q4g32[N, K]) -> y: f16[N] {\n  y[n] = sum(k: W[n, k] * x[k])\n}\n"
NEGATIVE = [
    ("index left unbound", "kernel mv(x: f16[K], W: q4g32[N, K]) -> y: f16[N] {\n  y[n] = W[n, k] * x[k]\n}\n"),
    ("transposed index", "kernel mv(x: f16[K], W: q4g32[N, K]) -> y: f16[N] {\n  y[n] = sum(k: W[k, n] * x[k])\n}\n"),
    ("index math", "kernel shift(x: f16[K]) -> y: f16[K] {\n  y[k] = x[k+1]\n}\n"),
    ("split on an output index", MV + "schedule mv on cpu {\n  split n 4\n}\n"),
    ("knob outside the list", MV + "schedule mv on cpu {\n  swizzle k 4\n}\n"),
]


def quantize(w, group):
    rows, cols = w.shape
    wg = w.reshape(rows, cols // group, group)
    scales = (np.abs(wg).max(axis=2) / 7).astype(np.float16)
    codes = np.clip(np.round(wg / scales.astype(np.float64)[..., None]), -8, 7)
    return codes.astype(np.int8).reshape(rows, cols), scales


def main():
    src = open(os.path.join(HERE, "examples.mote")).read()
    prog = Program(src)
    gl = [ln for ln in GRAMMAR.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    print(f"grammar   {len(gl)} lines (not counting comments)")
    print(f"parse     examples.mote: {len(prog.merges)} merges, {len(prog.kernels)} kernels, "
          f"{len(prog.schedules)} schedules")
    print("check     all static checks pass")
    for k in prog.kernels.values():
        print(f"          {k.name}: output indices {sorted(k.out_idx)}, reduced {sorted(k.red_idx)}")

    print("\nrejected (the one-line error the agent gets back):")
    for label, snippet in NEGATIVE:
        try:
            Program(snippet, base=prog)
            print(f"  {label:26s} NOT REJECTED")
        except MoteError as e:
            print(f"  {label:26s} {e}")

    rng = np.random.default_rng(0)
    print("\nreference interpreter vs plain NumPy (float64):")

    K, N = 64, 8
    x = rng.standard_normal(K).astype(np.float16)
    g = (1 + 0.1 * rng.standard_normal(K)).astype(np.float16)
    codes, scales = quantize(rng.standard_normal((N, K)), 32)
    wd = codes * np.repeat(scales.astype(np.float64), 32, axis=1)
    xf, gf = x.astype(np.float64), g.astype(np.float64)
    y_ref = wd @ (xf * gf / np.sqrt(np.mean(xf * xf) + 1e-5))
    arrays, sizes = {"x": x, "g": g, "W": (codes, scales)}, {"K": K, "N": N}
    for label, splits in [("no split", None), ("split k 3", {"k": (3, 1)})]:
        y = run(prog, "norm_matvec", sizes, arrays, splits)
        print(f"  norm_matvec  K={K} N={N}  {label:44s} max|err| {np.abs(y - y_ref).max():.1e}")

    KV, G, D, T, P = 2, 3, 8, 100, 16
    NB = -(-T // P)
    phys = NB + 3
    q = rng.standard_normal((KV, G, D)).astype(np.float16)
    kp = rng.standard_normal((phys, P, KV, D)).astype(np.float16)
    vp = rng.standard_normal((phys, P, KV, D)).astype(np.float16)
    table = rng.permutation(phys)[:NB].astype(np.int32)
    kl = np.concatenate([kp[p] for p in table])[:T].astype(np.float64)
    vl = np.concatenate([vp[p] for p in table])[:T].astype(np.float64)
    o_ref = np.empty((KV, G, D))
    for a in range(KV):
        for b in range(G):
            sc = kl[:, a, :] @ q[a, b].astype(np.float64) / np.sqrt(D)
            w = np.exp(sc - sc.max())
            o_ref[a, b] = (w / w.sum()) @ vl[:, a, :]
    arrays = {"q": q, "K": kp, "V": vp, "table": table}
    sizes = {"KV": KV, "G": G, "D": D, "T": T, "NB": NB}
    cases = [
        ("no split", None, True),
        ("split t 7", {"t": (7, 1)}, True),
        ("split t 4, tile t 64 (2 of 4 splits empty)", {"t": (4, 64)}, True),
        ("same, but join run on the empty splits", {"t": (4, 64)}, False),
    ]
    for label, splits, skip in cases:
        o = run(prog, "decode_attn", sizes, arrays, splits, skip)
        res = "NaN" if np.isnan(o).any() else f"{np.abs(o - o_ref).max():.1e}"
        print(f"  decode_attn  T={T} paged  {label:44s} max|err| {res}")


if __name__ == "__main__":
    main()
```
