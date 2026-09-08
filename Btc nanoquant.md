Yes. I think BTC-style feature-space rotation is a much better next experiment for NanoQuant than continuing with Gauge-NQ. And importantly, it attacks a different failure mode.

NanoQuant represents a weight matrix as

W \approx D_1\,U_{\pm1}V_{\pm1}^{T}\,D_2,

with binary low-rank factors U,V, and uses Hessian-aware LB-ADMM followed by STE block reconstruction. 

BTC instead changes the coordinate system before binarization:

y=xW^T

becomes

y=(xR\Lambda)\,
\mathcal Q\!\left(\Lambda^{-1}R^TW^T\right).

R is learned orthogonal and \Lambda is diagonal. They optimize these transformations specifically to reduce the output error after binary quantization. 

For NanoQuant, define

W_R=W R\Lambda^{-1}.

Then run ordinary NanoQuant on W_R:

W_R
\approx
D_1 U_{\pm1}V_{\pm1}^TD_2.

Therefore the effective approximation to the original weight is

\boxed{
W\approx
D_1 U_{\pm1}V_{\pm1}^TD_2\Lambda R^T
}

rather than

W\approx
D_1 U_{\pm1}V_{\pm1}^TD_2.

That gives NanoQuant a much friendlier coordinate system in which to find its binary factors.

Why this is different from our Gauge experiment

Our Gauge-NQ idea exploited

UV^T=(UR)(VR)^T

for an orthogonal R.

So we were merely selecting another basis inside the existing latent rank-r space:

U\rightarrow UR,\qquad
V\rightarrow VR.

The continuous matrix being approximated remained exactly the same.

That’s why it was possible for Gauge to improve the pre-Step-3 initialization but have NanoQuant’s common Step 3 simply erase the benefit: it had found another representation of essentially the same factorization problem.

BTC rotation instead does:

\boxed{W\rightarrow WR}

and

\boxed{X\rightarrow XR}.

It changes the actual feature coordinate system in which the discrete binary factorization problem is solved.

For example, NanoQuant might currently face a difficult right factor resembling

V_{\rm FP}=
\begin{bmatrix}
4.9&0.02\\
0.1&0.01\\
0.04&3.7\\
0.02&0.1
\end{bmatrix},

which is highly coherent/spiky and horrible to map to \pm1.

A rotation can redistribute that geometry into something much more balanced before NanoQuant ever sees it:

V'_{\rm FP}\sim
\begin{bmatrix}
1.2&-0.9\\
0.8&1.1\\
-1.0&0.9\\
1.1&1.0
\end{bmatrix}.

Then

\operatorname{sign}(V')

can preserve much more information.

BTC explicitly reports that learned orthogonal transformation plus scaling gives its strongest transformation result, and says the learned transform redistributes activation/weight outliers to reduce binary quantization error. 

There is one particularly nice observation for NanoQuant

I actually wouldn’t initially copy BTC’s \Lambda.

Why?

NanoQuant already has

D_2=\operatorname{diag}(s_2)

as a learned input-channel scale.

Our transformed representation would contain

D_2\Lambda.

Since both are diagonal,

D_2\Lambda=D_2',

so much of BTC’s diagonal scaling freedom is already present in NanoQuant.

That means the genuinely new degree of freedom is largely

\boxed{R}.

So a clean first algorithm is simply:

\boxed{\text{Rotated NanoQuant}}

with

W_R=WR

followed by the unchanged NanoQuant factorization.

That makes the scientific question much cleaner:

Does an activation-aware orthogonal feature rotation make NanoQuant’s low-rank binary factorization intrinsically easier?

⸻

And I wouldn’t use a random/Hadamard rotation first as the main method

Use Hadamard as an ablation.

BTC learns R with Cayley SGD while preserving

R^TR=I

and minimizes actual binary output reconstruction error. 

Our objective should ideally be even more NanoQuant-specific:

\boxed{
\min_R
\left\|
XW^T-
(XR)\widehat W_{\rm NQ}(R)^T
\right\|_F^2
}

where

\widehat W_{\rm NQ}(R)
=
D_1U_{\pm1}V_{\pm1}^TD_2

is the NanoQuant approximation of

WR.

That’s different from BTC, because BTC optimizes its transform around an in-place ARB binary quantizer; we’d optimize it around a low-rank double-binary factorization. BTC explicitly says its current binary operation follows ARB. 

That could be meaningful.

How I would implement it without differentiating through ADMM

Don’t attempt

\nabla_R \text{ADMM}(WR)

initially. That’s unnecessary pain.

Do alternating optimization:

R = I
1. Rotate target
      W' = W R
2. Run NanoQuant LB-ADMM on W'
      → U, V, s1, s2
3. Hold U,V roughly fixed
   Optimize R + scales on calibration activations
   with Cayley SGD:
      min ||Y_FP - Y_RotNQ||²
4. Re-run LB-ADMM using new W R
5. Run the SAME NanoQuant Step 3
6. Freeze R, signs and scales

Potentially just two rotation↔ADMM cycles would tell us whether the mechanism exists.

If it works, only then make it more sophisticated.

⸻

The experiment I’d run on our existing harness

Keep the exact same ~0.986-bpw setting and all the controls we used for Gauge-NQ.

Arm	Method
0a	Original NanoQuant
R0	NanoQuant + random orthogonal R
R1	NanoQuant + fixed Hadamard R
R2	NanoQuant + BTC-style learned R
R3	Learned R + one ADMM↔rotation alternation
R4	Learned R + extra STE, compute-matched control

And record three numbers separately:

E_{\rm ADMM}

immediately after factorization,

E_{\rm Step3}

after the common NanoQuant refinement, and finally held-out PPL/task accuracy.

This distinction is crucial because of what happened with Gauge. A result like

E_{\rm ADMM}: 0.140\rightarrow0.110

but

E_{\rm Step3}:0.115\rightarrow0.114

means rotation didn’t really help NanoQuant.

What we want is something like:

0.115\rightarrow0.105

after the common Step 3, followed by a real held-out/PPL improvement.

⸻

There is already nearby prior art—but not exactly this experiment

LittleBit-2 uses internal latent rotation + Joint-ITQ to align low-rank latent geometry with the binary hypercube. 

That’s basically the family our Gauge experiment was probing.

What I did not find in the current literature search is a published method that takes NanoQuant’s Hessian-aware PTQ low-rank binary factorization and precedes it with a BTC-style learned feature-space rotation optimized for the factorized binary reconstruction.

So there is a useful conceptual distinction:

\text{LittleBit-2 / Gauge}
=
\boxed{\text{rotate inside latent factor space}}

versus

\text{BTC-NanoQuant}
=
\boxed{\text{rotate feature space before solving the binary factorization}}

I think the second one has a considerably better chance of surviving NanoQuant Step 3.

And inference overhead may not be as bad as it first appears

BTC specifically designs its transformations so that many of the diagonal/orthogonal transforms are merged into surrounding weights, reporting no additional parameter/storage cost for those cases; it notes particular non-fusible locations such as Q/K and FFN down-projection where Hadamard transforms are used instead. 

So eventually I’d aim for

\boxed{\text{Fuseable Rotation-NanoQuant}}

rather than carrying an arbitrary dense R multiplication at runtime.

But I wouldn’t solve that yet.

First test the hypothesis with R2: take BTC’s learned orthogonal rotation machinery, remove the codebook entirely, rotate the targets/activations, and feed those targets through the unchanged NanoQuant LB-ADMM + Step 3 pipeline.

If that one experiment beats 0a after Step 3, I think we’ve found a much better vein to mine than Gauge-NQ.
