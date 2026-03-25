# MonoidEmbed specification

**Version:** 1.3.1
**Status:** Normative
**Date:** 2026-01-13

**Abstract:** This specification defines MonoidEmbed, a CPU-only streaming embedding model and inference procedure that converts streams of 8-bit codes into Matryoshka-compatible 512-dimensional INT8 embeddings with an explicit per-embedding scale. It defines the model configuration, parameter tensors, fixed-point arithmetic, deterministic reference algorithms, a single-core and multi-core execution model, stability constraints, initialization strategies, and a deterministic method for deriving custom model variants from either (layer depth, state width) inputs or a parameter-budget range.

**Keywords:** streaming embeddings, fixed-point inference, affine monoid, deterministic execution, CPU parallelism, Matryoshka, stability constraints

---

## 1. Scope

### 1.1 Overview
This document specifies the MonoidEmbed inference architecture, its fixed-point numerical semantics, and the observable outputs for a single stream instance processing an unbounded sequence of microblocks. The specification is written to be implementable on modern general-purpose CPUs and to produce bitwise-identical outputs across conforming implementations.

### 1.2 Purpose
MonoidEmbed is designed to replace byte-granularity recurrent embedding paths that are limited by per-step state traffic. It compiles per-symbol affine transforms over microblocks so the stream state is read and written a constant number of times per microblock, enabling maximum performance on a single CPU core and deterministic scaling across multiple CPU cores.

### 1.3 Design rationale
The byte-rate Mamba-2 path that MonoidEmbed replaces is slow primarily because it is a serial recurrence with heavy per-step state traffic:

1. **Serial dependency at byte granularity:** Per-token recurrence implies unavoidable serial work. At byte-rate, per-step overhead dominates.
2. **Memory and cache behavior dominates:** State and per-layer parameters are touched every step. The loop becomes memory-bound with frequent loads/stores.
3. **Quantization overhead inside the inner loop:** Per-step scaling, dequant, and conversion costs are amplified at byte-rate.
4. **Poor amortization:** Any fixed per-token bookkeeping becomes expensive when the token is a byte.

The bottleneck is mostly state traffic plus serial dependency. The recurrence is therefore composable over blocks to remove token-level serial work while keeping hot sets in L1/L2.

### 1.4 Architecture summary
MonoidEmbed replaces token-by-token recurrence with a composable affine monoid compiled over microblocks, plus an explicit deterministic nonlinearity and a low-cost cross-tile exchange.

For each input code c (byte or mapped TEMPEST code), the per-symbol state update is:
- `s ← a[c] ⊙ s + b[c]`

where a[c] and b[c] are code-conditioned vectors, and ⊙ is element-wise multiplication.

Because affine transforms compose associatively in exact arithmetic, the per-symbol transforms across a microblock can be compiled into a single transform (A, B) and applied once per microblock (see 7.2).

A pure affine recurrence plus linear mixing can behave like an effectively linear dynamical system end-to-end; depth does not add expressivity without nonlinearities in deep linear networks [3]. Therefore, this specification requires applying a deterministic activation at microblock boundaries (and optionally once after exchange) to prevent "linearity collapse," while preserving the monoid property inside each microblock (see 7.3).

### 1.5 Non-goals
- This document does not define training datasets, teacher models, or quality targets; however, informative guidance is provided in Annex F.
- This document does not define the TEMPEST compression format, audio feature extraction, or any modality parser that produces codes; MonoidEmbed consumes already-produced 8-bit codes (see 8.7).
- This document does not define a specific on-disk container or serialization format for model artifacts; it defines required tensors and metadata (see Clause 9).
- This document does not require or assume GPU, NPU, DSP, or other accelerators.

## 2. Normative references

[1] S. Bradner, "Key words for use in RFCs to Indicate Requirement Levels," RFC 2119, Mar. 1997.
[2] B. Leiba, "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words," RFC 8174, May 2017.

## 3. Informative references

[3] A. M. Saxe, J. L. McClelland, S. Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," arXiv:1312.6120, 2013.
[4] A. Gu, K. Goel, C. Ré, "Efficiently Modeling Long Sequences with Structured State Spaces," arXiv:2111.00396, 2021.
[5] T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, "Spectral Normalization for Generative Adversarial Networks," arXiv:1802.05957, 2018.
[6] Q. V. Le, N. Jaitly, G. E. Hinton, "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units," arXiv:1504.00941, 2015.
[7] J. Koutník, K. Greff, F. Gomez, J. Schmidhuber, "A Clockwork RNN," ICML 2014; arXiv:1402.3511, 2014.

## 4. Definitions

For the purposes of this document, the following terms and definitions apply.

**4.1 bank:** A named interpretation of input codes that supplies bank-specific constants (for example activation and shift constants) and any required per-microblock metadata.

**4.2 code:** An unsigned 8-bit symbol in the set {0..255} that indexes per-code parameters a and b.

**4.3 microblock:** A fixed-length sequence of K codes processed as a unit. One embedding is emitted per processed microblock (see 7.7 and Clause 8).

**4.4 stream instance:** A stateful MonoidEmbed inference context that processes an unbounded sequence of microblocks while maintaining persistent state (see 8.6).

**4.5 state:** The persistent recurrent vector s of length D = d_state stored as signed 16-bit integers (int16).

**4.6 tile:** One of N_TILES disjoint interleaved partitions of the state dimensions (see 6.1).

**4.7 canonical dimension order:** The dimension ordering d = 8*i + t (tile-interleaved) used for output embeddings and for defining prefixes (see 6.1 and 6.6).

**4.8 hemisphere:** One of two fixed tile-groups used by the exchange mechanism: hemisphere 0 owns tiles {0,1,2,3}, hemisphere 1 owns tiles {4,5,6,7} (see 5.2 and 7.5).

**4.9 model artifact:** The set of tensors and metadata that fully determines MonoidEmbed inference for a specific trained model variant (see Clause 9).

## 5. Conformance

### 5.1 Conformance language
The key words **shall**, **shall not**, **should**, **should not**, and **may** are to be interpreted as described in RFC 2119 and clarified by RFC 8174.

### 5.2 Determinism scope
A conforming implementation shall be *bitwise deterministic* for all observable outputs and persisted state. Specifically, given:
- a model artifact (Clause 9),
- an initial persisted state (8.6),
- a sequence of microblocks, where each microblock provides: (a) K codes, (b) a bank identifier, and (c) required per-microblock metadata for that bank (8.7), and
- the build-time flag `post_exchange_activation_enabled` from the model artifact (9.1),
the implementation shall produce bitwise-identical:
- emitted embeddings (e_int8, scale_q15) for each microblock, and
- the final persisted state after processing the sequence.

Determinism shall hold regardless of:
- CPU ISA choice (for example scalar vs SIMD),
- thread count and scheduling (see Clause 8), and
- memory layout choices internal to the implementation,
provided the implementation follows the deterministic integer semantics and the reference-equivalence requirements in this document (see 6.3 and 7.2.4).

### 5.3 Conventions
Unless otherwise specified:
- All indexing is 0-based.
- All integers are two's-complement with exact widths: int8, int16, int32, int64 are signed 8, 16, 32, 64-bit types; uint8, uint32, uint64 are unsigned 8, 32, 64-bit types.
- Division and shifting semantics are defined by this document and shall not rely on language-defined behavior for negative right-shifts or signed overflow (see 6.3).
- The model vocabulary size is fixed at 256 codes.

## 6. Model configuration and parameters

### 6.1 Configuration fields
A MonoidEmbed model variant shall be defined by the configuration tuple:
- `n_layers` (L): number of stacked recurrent blocks, integer L >= 1.
- `d_state` (D): total state dimension, integer D.
- `microblock_size` (K): microblock length in codes, integer K.
- `exchange_dim` (E): exchange vector dimension, integer E.

All configuration fields shall satisfy the validity constraints in 6.3.

### 6.2 Fixed structural constants
The following constants are fixed by this specification:
- `VOCAB_SIZE` = 256.
- `EMBED_DIM` = 512.
- `N_TILES` = 8.
- `EXCHANGE_GROUP_SIZE` (G) = 4.
- `N_HEMISPHERES` = 2 with tile groups:
  - hemisphere 0 tiles: {0,1,2,3}
  - hemisphere 1 tiles: {4,5,6,7}

### 6.3 Derived dimensions and validity constraints
Given D = d_state and K = microblock_size, the following shall hold for a valid configuration:
- D shall be divisible by N_TILES.
- `tile_dim = D / N_TILES`.
- `tile_dim` shall be a power of two.
- `tile_dim` shall be divisible by G.
- The exchange dimension shall be:
  - `E = 2 * (tile_dim / G)`.
  - Equivalently (since N_TILES = 8 and G = 4): `E = D / 16`.
- K shall be a power of two and K >= 64.

Numeric safety: Any accumulation specified as int32 shall be widened to a type wide enough to avoid overflow under worst-case valid inputs (for example int64), while preserving the same floor-division and clamping semantics.

### 6.4 Trainable parameter sets
For each layer index l in {0..L-1}, the model artifact shall provide the following trainable tensors:
- Code-conditioned affine tables:
  - `a[l][c][d]` as int8 for c in {0..255}, d in {0..D-1}.
  - `b[l][c][d]` as int8 for c in {0..255}, d in {0..D-1}.
- Exchange matrix and shifts:
  - `M[l][r][j]` as int8 for r,j in {0..E-1}.
  - `m_shift[l][r]` as non-negative integer for r in {0..E-1}.

In addition, for each inter-layer LayerNorm instance (there are L-1 of them), the model artifact shall provide:
- `ln_gamma[l][d]` as int8 and `ln_beta[l][d]` as int8 for d in {0..D-1}, where l indexes the normalization applied after block l (so l in {0..L-2}).
- `ln_gamma_shift[l]` and `ln_beta_shift[l]` as non-negative integers (scalar shifts).

If D != EMBED_DIM, the model artifact shall provide a projection head:
- `P_out[k][d_out]` as int8 for k in {0..D-1}, d_out in {0..511}.
- `p_shift` as a non-negative integer.

### 6.5 Parameter count
Let V = 256, O = 512, D = d_state, E = exchange_dim, L = n_layers. The trainable parameter count shall be:
- `per_layer = 2*V*D + E*E`.
- `layernorm_params = (L-1) * 2*D`.
- `projection_head = 0` if D == O, else `D*O`.
- `total_params = L*per_layer + layernorm_params + projection_head`.

Trainable parameter counts in this clause exclude non-trainable metadata (for example shift arrays and fixed LUTs).

### 6.6 Reference presets
The following presets are defined for interoperability and benchmarking:

| Preset | n_layers (L) | d_state (D) | microblock_size (K) | exchange_dim (E) | total_params |
|---|---:|---:|---:|---:|---:|
| small | 1 | 512 | 256 | 32 | 263,168 |
| medium | 1 | 2048 | 128 | 128 | 2,113,536 |
| base | 15 | 2048 | 128 | 128 | 17,080,320 |
| large | 27 | 2048 | 128 | 128 | 29,908,992 |

Notes:
- small → medium is approximately 8× parameters, and medium → base is approximately 8× parameters.
- The next exact 8× step from base would exceed 30M; therefore large is the maximum supported preset ≤ 30M.

### 6.7 Deterministic custom-variant derivation
This specification defines two deterministic derivation procedures for producing a complete configuration tuple (L,D,K,E) when only partial sizing inputs are provided.

#### 6.7.1 Derivation from (n_layers, d_state)
Given integers L and D, a conforming implementation shall derive K and E as follows:
1. Validate that L >= 1 and that D satisfies the validity constraints in 6.3 (including that E = D/16 is an integer). If validation fails, the derivation shall fail.
2. Set `E = D / 16`.
3. Set `K` using the deterministic default:
   - if D <= 1024 then K = 256, else K = 128.
4. Validate K against 6.3. If validation fails, the derivation shall fail.
5. Return (L,D,K,E).

#### 6.7.2 Derivation from a parameter-budget range
Given integers P_min and P_max with 0 <= P_min <= P_max, a conforming implementation shall select a configuration as follows:
1. Define the candidate state widths set: D in {512 * 2^k | k is a non-negative integer}.
2. For each candidate D in increasing order:
   a. Set `E = D / 16`.
   b. Compute `per_layer = 2*256*D + E*E`.
   c. Compute `projection_head = 0` if D == 512 else `D*512`.
   d. Compute `A = per_layer + 2*D`.
   e. Compute `L_max = floor( (P_max - projection_head + 2*D) / A )` using floor division over integers with A > 0.
   f. If L_max == 0, stop the search (larger D will not fit).
3. For each candidate D with L_max >= 1, compute `total_params` using 6.5 with L = L_max. If `P_min <= total_params <= P_max`, add the candidate configuration (L_max, D) to the feasible set.
4. If the feasible set is empty, the derivation shall fail.
5. Select the feasible candidate with the largest total_params. Break ties deterministically in the following order:
   a. larger d_state D wins
   b. if still tied, larger n_layers L wins
6. For the selected (L,D), set E = D/16 and set K using 6.7.1 step 3.
7. Return (L,D,K,E).

## 7. State layout and numerics

### 7.1 State layout and tiling
The state s is an int16 vector of length D split into N_TILES = 8 tiles of equal size:
- `tile_dim = D / 8`.

Canonical mapping between tile coordinates and canonical dimension index:
- For tile t in {0..7} and within-tile index i in {0..tile_dim-1}:
  - `d = 8*i + t`.
- Tile t owns precisely those canonical indices d such that `d mod 8 == t`.

Implementations may store tile data in tile-major contiguous order for performance, but shall preserve the canonical mapping above whenever converting between (t,i) and canonical index d.

### 7.2 Fixed-point conventions
Unless otherwise specified:
- State elements s[d] are int16 in Q0 (integer) format.
- Stored multipliers a[l][c][d] are int8 interpreted as unsigned Q7 with the export restriction: `a_int8` in [0,127].
- Runtime multipliers are represented as int16 Q15: `a_q15 = a_int8 * 2^8`.
- Stored additive terms b[l][c][d] are int8. A bank-specific shift `b_shift[bank]` is applied at runtime to obtain int16 Q0 values (see 8.2.2).

### 7.3 Deterministic integer helpers
The following helper functions are normative.

**7.3.1 clamp functions**
- `clamp_int8(x)` shall clamp integer x to the range [-128, 127].
- `clamp_int16(x)` shall clamp integer x to the range [-32768, 32767].

**7.3.2 mul_pow2**
- `mul_pow2(x, k)` for integer x and integer k >= 0 shall compute the mathematical product x * 2^k using int64 intermediate arithmetic (prior to any later clamping).

**7.3.3 floor_div_pow2**
- `floor_div_pow2(x, k)` for integer x and integer k >= 0 shall return floor(x / 2^k), where floor is toward negative infinity.
- Implementations shall not use language-defined right-shift of negative values unless it is proven equivalent to the required floor semantics on the target platform.
- `floor_div(x, y)` for integer x and integer y > 0 shall return floor(x / y), where floor is toward negative infinity.
- A reference definition is:
  - if x >= 0: `floor_div(x,y) = x / y` using integer truncation,
  - if x < 0: `floor_div(x,y) = -(((-x) + y - 1) / y)` using integer truncation in the positive division.

**7.3.4 Q15 multiply helpers**
Define the constant `R15 = 2^14` (16384).
- `mul_q15(x_q0, a_q15)` shall compute:
  - `clamp_int16( floor_div_pow2( (int32)x_q0*(int32)a_q15 + R15, 15 ) )`.
- `mul_q15_q15(x_q15, y_q15)` shall compute:
  - `clamp_int16( floor_div_pow2( (int32)x_q15*(int32)y_q15 + R15, 15 ) )`.

**7.3.5 round_nearest_div_pow2_15**
For integer x:
- If x >= 0: `round_nearest_div_pow2_15(x) = (x + 2^14) >> 15`.
- If x < 0: `round_nearest_div_pow2_15(x) = -(((-x + 2^14) >> 15))`.

**7.3.6 div_trunc**
For integers x and y with y > 0, `div_trunc(x, y)` shall return truncation toward zero:
- `div_trunc(x,y) = sign(x) * floor(|x| / y)`, where `sign(x)` is -1 if x < 0, 0 if x == 0, and +1 if x > 0.

**7.3.7 isqrt**
`isqrt(n)` shall return floor(sqrt(n)) for integer n >= 0. A reference algorithm is provided in Annex A.

### 7.4 Saturation points
State elements shall be saturated to int16 at all of the following points:
- after monoid apply (8.2.3),
- at each sum and difference operation within the mixer (8.4),
- after each activation (8.3),
- after exchange injection (8.5.3),
- after residual addition (8.1.3),
- after LayerNorm output (7.5).

### 7.5 LayerNorm (deterministic fixed-point)
LayerNorm is applied between blocks for l < L-1 (see 8.1). It shall normalize across all D dimensions (not per-tile).

For a vector x[d] in int16, LayerNorm shall be computed as follows:
1. Compute `sum = Σ x[d]` in int64.
2. Compute `sum_sq = Σ (int64)x[d]*(int64)x[d]` in int64.
3. Compute `mean = floor_div(sum, D)` where floor_div is floor toward negative infinity.
4. Compute `ex2 = sum_sq / D` using integer truncation (sum_sq is non-negative).
5. Compute `var = ex2 - (int64)mean*(int64)mean`. If var < 0 then set var = 0.
6. Let `eps = 1` (in the same squared-units as var). Let `std = isqrt(var + eps)`.
7. Compute `inv_std_q15 = clamp_int16( round_nearest( (2^15) / max(std,1) ) )` where round_nearest uses half-up rounding:
   - `inv_std_q15 = min(32767, floor( (2^15 + floor(std/2)) / std ))` for std >= 1.
8. For each dimension d:
   a. `delta = (int32)x[d] - (int32)mean`.
   b. `norm_q0 = mul_q15(delta, inv_std_q15)`.
   c. `gamma_q15 = clamp_int16( (int32)ln_gamma[l][d] << ln_gamma_shift[l] )`.
   d. `beta_q0 = clamp_int16( (int32)ln_beta[l][d] << ln_beta_shift[l] )`.
   e. `y = clamp_int16( (int32)mul_q15(norm_q0, gamma_q15) + (int32)beta_q0 )`.
9. The LayerNorm output is y[d].

### 7.6 Output projection, normalization, and quantization
After processing each microblock, exactly one embedding shall be emitted (see 8.7). The embedding shall be represented as:
- `e_int8[512]` (int8 vector), and
- `scale_q15` (int16 scalar).

Let s_final denote the post-layer state after the last block for the microblock, indexed in canonical order (see 7.1). Emission shall proceed as follows.

**7.6.1 Optional projection to 512 dimensions**
Construct an intermediate int16 vector e[0..511] as:
- If D == 512: `e[d] = s_final[d]`.
- Else, for each output dimension d_out in {0..511}:
  - `acc = Σ_{k=0..D-1} (int64)P_out[k][d_out] * (int64)s_final[k]` in int64.
  - `e[d_out] = clamp_int16( floor_div_pow2(acc, p_shift) )`.

**7.6.2 L2 norm and scale**
Compute:
- `sumsq = Σ_{d=0..511} (int64)e[d]*(int64)e[d]` in int64.
- `norm = isqrt(sumsq)` as int32 (norm >= 0).

Compute the per-embedding quantization scale:
- `numer = 127 * 2^15` (int32).
- `denom = max(norm, 1)` (int32).
- `scale_raw = numer / denom` using integer truncation toward zero (denom > 0).
- `scale_q15 = (int16)max(1, min(32767, scale_raw))`.

**7.6.3 Quantize to INT8**
For each d in {0..511}:
- `x = (int32)e[d] * (int32)scale_q15`.
- `e_int8[d] = clamp_int8( round_nearest_div_pow2_15(x) )`.

**7.6.4 Dequantization interpretation**
The emitted pair (e_int8, scale_q15) shall be interpreted such that the dequantized approximation is:
- `e_hat[d] = div_trunc( (int32)e_int8[d] * 2^15, scale_q15 )` for scale_q15 >= 1.

**7.6.5 Matryoshka prefixes**
Prefixes are defined as:
- 128D prefix: `e_int8[0..127]`
- 256D prefix: `e_int8[0..255]`
- 512D embedding: `e_int8[0..511]`

The same `scale_q15` applies to all prefixes. Prefix validity follows from the canonical interleaving in 7.1.

## 8. Inference computation

### 8.1 Stacked recurrent blocks
Let K = microblock_size and let c[0..K-1] be the microblock codes. For each microblock, the model applies L stacked blocks sequentially. For layer index l in {0..L-1}:

#### 8.1.1 Block input
Let `s_in` be the state entering layer l for this microblock.

#### 8.1.2 Block transform
Compute `s_block = Block_l(s_in, c, bank_id, microblock_meta, phase, exchange_scheduled)` where Block_l performs, in order:
1. affine monoid compilation over the microblock (8.2.4),
2. monoid apply to s_in (8.2.3),
3. local mixer (8.4),
4. activation (8.3),
5. if exchange_scheduled is true: cross-tile exchange (8.5), and if enabled in the model artifact, a post-exchange activation (8.3.4).

#### 8.1.3 Residual
Compute `s_res[d] = clamp_int16( (int32)s_in[d] + (int32)s_block[d] )` for all d.

#### 8.1.4 Normalization
- If l < L-1: set `s_out = LayerNorm_l(s_res)` using the deterministic algorithm in 7.5.
- If l == L-1: set `s_out = s_res`.

The next layer uses s_out as its input state. After l = L-1, the resulting state is s_final for the microblock and becomes the persisted stream state (after clock update in 8.6).

### 8.2 Affine monoid over a microblock
For a microblock of length K, define the per-code transform for layer l at microblock position i:
- `T_{l,i}(s) = a[l][c[i]] ⊙ s + b[l][c[i]]`,
where ⊙ is element-wise multiplication.

The compiled microblock transform is an element-wise affine transform:
- `s <- A_l ⊙ s + B_l`,
where (A_l, B_l) is the composition `T_{l,K-1} ∘ ... ∘ T_{l,0}`.

#### 8.2.1 Composition rule
Composition of two element-wise affine transforms (A2,B2) ∘ (A1,B1) shall be:
- `A = A2 ⊙ A1`
- `B = B2 + A2 ⊙ B1`

#### 8.2.2 Runtime representation of a and b
During inference:
- A shall be represented as int16 in Q15.
- B shall be represented as int16 in Q0.

For each bank_id, the model artifact shall provide a non-negative integer `b_shift[bank_id]`. For a stored b value `b_int8` (int8), its runtime Q0 representation shall be:
- `b_q0 = clamp_int16( mul_pow2( (int32)b_int8, b_shift[bank_id] ) )`.

For a stored a value `a_int8` (int8) with `a_int8` in [0,127], its runtime Q15 representation shall be:
- `a_q15 = (int16)mul_pow2((int16)a_int8, 8)`.

#### 8.2.3 Monoid apply
Given compiled A[d] (Q15) and B[d] (Q0), monoid apply shall compute for each dimension d:
- `s_out[d] = clamp_int16( (int32)mul_q15(s_in[d], A[d]) + (int32)B[d] )`.

#### 8.2.4 Reference compilation order
Under fixed-point rounding, compilation order affects results. Therefore, the compiled (A,B) for each layer shall be computed using a reference-equivalent procedure.

The reference compilation algorithm for a subset of dimensions is:
1. Initialize `A[d] = 32767` (Q15 unity) and `B[d] = 0` for all d in the subset.
2. For i from 0 to K-1 in increasing order, update each d in the subset:
   - `A[d] = mul_q15_q15(a_q15(l,c[i],d), A[d])`
   - `B[d] = clamp_int16( (int32)b_q0(l,c[i],d) + (int32)mul_q15(B[d], a_q15(l,c[i],d)) )`

An implementation may use SIMD, unrolling, or different memory layouts, but shall produce bitwise-identical A and B to the reference algorithm for all valid inputs.

### 8.3 Deterministic activation
#### 8.3.1 Requirement
A deterministic nonlinearity shall be applied at microblock boundaries to prevent linear collapse in deep affine-only stacks [3].

#### 8.3.2 Bank constants and constraints
For each bank_id, the model artifact shall provide:
- `T[bank_id]` as int16 (Q0), an output bound parameter.
- `act_shift[bank_id]` as a non-negative integer.

Define `S[bank_id] = 2^(act_shift[bank_id] + 7)`. The model artifact shall satisfy the stability constraint:
- `T[bank_id] <= S[bank_id]`.

**Default bank constants (informative):**
Unless overridden by the model artifact, conforming implementations should use:
- BANK_RAW256: T = 24576
- BANK_AUDIO_CFF256: T = 20480

#### 8.3.3 LUT definition and per-element activation
The activation shall use a fixed lookup table `tanh_LUT[0..255]` stored in Q15, where index k corresponds to x8 = k-128 and tanh_LUT[k] approximates tanh(x8/128). The normative table values are specified in Annex A.

For each input x (int16) the activation output y (int16) shall be:
1. `x8 = clamp_int8( floor_div_pow2(x, act_shift[bank_id]) )`.
2. `t = tanh_LUT[x8 + 128]` (Q15).
3. `y = clamp_int16( floor_div_pow2( (int32)T[bank_id] * (int32)t, 15 ) )`.

The activation shall not use data-dependent branches on x.

#### 8.3.4 Application points
Activation shall be applied:
- after the monoid apply and local mixer, and
- if `post_exchange_activation_enabled` is true in the model artifact, also after exchange injection (8.5.3).

### 8.4 Local mixer
Within each tile, a fixed butterfly diffusion network shall be applied in-place.

Let x[0..tile_dim-1] be the tile vector. For stage from 0 to log2(tile_dim)-1:
- For base from 0 to tile_dim-1 in steps of 2^(stage+1):
  - For j from 0 to 2^stage - 1:
    - `u = x[base + j]`
    - `v = x[base + j + 2^stage]`
    - `x[base + j] = clamp_int16(u + v)`
    - `x[base + j + 2^stage] = clamp_int16(u - v)`

No per-stage scaling is permitted.

### 8.5 Cross-tile exchange
Exchange provides low-cost cross-tile mixing subject to deterministic integer semantics.
Let `s_tile(t)[i]` denote the within-tile element of the current state for tile t in {0..7} and index i in {0..tile_dim-1}, where `s_tile(t)[i] = s[d]` with `d = 8*i + t` (see 7.1).


#### 8.5.1 Summary vector u construction
Let G = 4. Let E = exchange_dim. Let `phase` be defined in 8.6. For each hemisphere h in {0,1}, construct `u_h[0..E/2-1]` as follows:
- For group index j in {0..E/2-1}:
  - `acc = 0` in int32.
  - For each tile t in hemisphere h:
    - For i in {0..G-1}:
      - `acc += s_tile(t)[ (G*j + phase + i) mod tile_dim ]`.
  - `u_h[j] = clamp_int16(acc)`.

Concatenate `u = [u_0, u_1]` as an int16 vector of length E.

#### 8.5.2 Global mixing v = M_l u
For layer l, compute v = M[l] * u with per-row shifts. For each output row r in {0..E-1}:
- `acc = Σ_{j=0..E-1} (int64)M[l][r][j] * (int64)u[j]` accumulated in int64.
- `v[r] = clamp_int16( floor_div_pow2(acc, m_shift[l][r]) )`.

#### 8.5.3 Injection back into state
The model artifact shall provide `inj_shift` as a non-negative integer.

**Default value (informative):** Unless overridden by the model artifact, conforming implementations should use inj_shift = 3.

For hemisphere h, let `v_h` be its half of v (length E/2). For each tile t in hemisphere h and each group index j in {0..E/2-1}:
- `inj = floor_div_pow2(v_h[j], inj_shift)`.
- Let:
  - `idx0 = (G*j + phase + 0) mod tile_dim`
  - `idx1 = (G*j + phase + 1) mod tile_dim`
  - `idx2 = (G*j + phase + 2) mod tile_dim`
  - `idx3 = (G*j + phase + 3) mod tile_dim`
- Update with fixed signs:
  - `s_tile(t)[idx0] = clamp_int16( (int32)s_tile(t)[idx0] + (int32)inj )`
  - `s_tile(t)[idx1] = clamp_int16( (int32)s_tile(t)[idx1] + (int32)inj )`
  - `s_tile(t)[idx2] = clamp_int16( (int32)s_tile(t)[idx2] - (int32)inj )`
  - `s_tile(t)[idx3] = clamp_int16( (int32)s_tile(t)[idx3] - (int32)inj )`

### 8.6 Barrel shift and modality-aware clock
The exchange grouping is varied over time using a phase in {0,1,2,3}. Multi-rate recurrent architectures motivate modality-dependent update rates [7].

#### 8.6.1 Persisted clock state
Each stream instance shall maintain the following persisted clock state:
- `tick_accum_q16` as uint32 in Q16.16,
- `tick_count` as uint64,
- `phase` as uint8 with phase in {0,1,2,3}.

#### 8.6.2 Tick update
For each processed microblock, compute deterministically:
1. Compute `Δtick_q16 = Δtick(bank_id, microblock_meta)` in Q16.16 (see 8.6.3).
2. Compute `tick_accum_q16' = (uint64)tick_accum_q16 + (uint64)Δtick_q16` in uint64.
3. `k = floor(tick_accum_q16' / 2^16)` (integer completed ticks in this microblock).
4. `tick_accum_q16_next = (uint32)(tick_accum_q16' - k*2^16)` and `0 <= tick_accum_q16_next < 2^16`.
5. `tick_count_next = tick_count + k`.
6. `phase_next = (phase + (k mod 4)) mod 4`.

The values with suffix _next shall become the persisted clock state for the next microblock.

#### 8.6.3 Default Δtick schedules
The following default tick schedules are defined:
- BANK_RAW256: `Δtick_q16 = 1 << 16`.
- BANK_AUDIO_CFF256: `Δtick_q16 = floor( frames_in_microblock * 2^16 / frames_per_tick )`, where `frames_per_tick` is a positive integer provided in the model artifact.

**Default frames_per_tick (informative):** Unless overridden by the model artifact, conforming implementations should use frames_per_tick = 2.

#### 8.6.4 Exchange cadence
For each bank_id, the model artifact shall provide a positive integer `E_ticks[bank_id]`. Exchange is scheduled for a microblock if and only if:
- k > 0, and
- `floor(tick_count_next / E_ticks[bank_id]) > floor(tick_count / E_ticks[bank_id])`.

**Default E_ticks values (informative):** Unless overridden by the model artifact:
- BANK_RAW256: E_ticks = 1
- BANK_AUDIO_CFF256: E_ticks = 2

`phase` used within the microblock shall be the value at microblock start and shall remain constant for all layers of that microblock.

### 8.7 Per-microblock step function and emission
For each microblock, a conforming implementation shall:
1. Determine `exchange_scheduled` using 8.6.4 and the current clock state.
2. Process layers l = 0..L-1 as defined in 8.1, using the phase value at microblock start.
3. Emit exactly one embedding using the post-layer state (7.6).
4. Update the persisted clock state using 8.6.2.

Embedding emission in step 3 shall not depend on the clock update in step 4.

## 9. Stability and drift

### 9.1 Intrinsic stability constraints for a[c,d]
Stability is enforced by construction:
- For each dimension d, a[c,d] shall be bounded such that the effective magnitude is ≤ 1.
- Training parameterization shall enforce bounds in float space and export to INT8 Q7 for inference (see 7.2).

This reflects stability control through constrained state transitions in structured state space modeling [4].

### 9.2 Spectral constraint on M
The exchange matrix M shall be spectrally constrained in training (e.g., via spectral normalization) to bound its spectral norm and thus its Lipschitz contribution [5].

Training-time requirement:
- Spectral normalization shall be applied to the float master weights before INT8 export.
- Exported INT8 M shall include per-row power-of-two shifts (m_shift) to preserve the bound as closely as practical (see 8.5.2).

### 9.3 Activation as a stability component
The tanh-squash activation is bounded and has slope ≤ 1 (under the constraint in 8.3.2); therefore it is an intrinsic component of stability and quantization-noise control.

## 10. Initialization strategy

Recurrent initialization shall prevent early vanishing or exploding gradients. Identity-like initialization is a standard approach for stabilizing long-horizon gradient flow in recurrences [6].

### 10.1 Initialization for a[c,d]
- Choose per-dimension base decays a_base[d] close to 1 with log-spaced half-lives.
- Initialize all codes:
  - a[c,d] = a_base[d] (no per-code variation at initialization).

### 10.2 Initialization for b[c,d]
- Initialize b to 0 or very small random INT8 with tight scale.

### 10.3 Initialization for M
- Initialize M = 0 (no exchange) for an initial warmup window (e.g., 10k steps), then allow learning under spectral constraints.

### 10.4 Mixer and activation
- Butterfly mixer is fixed and deterministic.
- tanh-squash LUT is fixed.
- Only the bank-level constants (T, act_shift, b_shift) and the global injection shift (inj_shift) are tunable hyperparameters.

## 11. Runtime execution model and performance requirements

### 11.1 CPU-only and bounded-memory execution
A conforming implementation:
- shall execute on CPU only and shall not require a GPU or accelerator,
- shall support unbounded streaming by maintaining bounded memory that does not grow with the number of processed microblocks, and
- shall not rely on large batching for throughput; latency shall be controlled by microblock_size K and shall not require aggregating multiple microblocks before producing an embedding.

### 11.2 Single-core fast path
A conforming implementation shall support a single-thread execution mode (`thread_count = 1`) in which the full step function in 8.7 executes without waiting for other internal threads. In steady state, a stream instance in this mode shall not have any additional runnable internal threads.

### 11.3 Multi-core deterministic scaling
A conforming implementation shall support a multi-thread execution mode with `thread_count = T` where T >= 2. In this mode:
- Computation shall be partitioned across threads so that adding threads can reduce wall-clock time.
- Observable outputs shall remain bitwise identical for any supported T (see 5.2).

At minimum, the implementation shall support T in {1..N_TILES} by assigning tiles to threads deterministically as described in 11.4.

### 11.4 Deterministic work partitioning
Let T be the configured thread_count. Define `T_eff = min(T, N_TILES)`. Tile ownership shall be assigned deterministically by:
- `owner_thread(t) = t mod T_eff` for tile t in {0..7}.

For operations in 8.1 that are tile-local (monoid compile and apply, mixer, activation, residual addition, and exchange injection), only the owner thread of a tile shall write that tile's state elements.

For operations requiring global reductions (LayerNorm statistics in 7.5 and exchange vectors in 8.5), threads may compute partial results over the tiles they own, and the partials shall be combined in a deterministic order:
- combine tile partials in increasing tile index order (0 to 7), and
- for each tile partial, combine within-tile contributions in increasing within-tile index order.

### 11.5 Synchronization requirements
In multi-thread mode, synchronization shall ensure correctness without affecting determinism.

A conforming implementation shall provide barriers at the following points:
- Exchange:
  - a barrier after all threads have completed their contributions to u (8.5.1) and before any thread uses the full u, and
  - a barrier after v has been fully computed (8.5.2) and before injection begins (8.5.3).
- LayerNorm:
  - a barrier after all partial sums and sum_sq values have been produced, and
  - a barrier after the global mean and inv_std_q15 have been computed and published for use in normalization.

Synchronization primitives shall not introduce additional numerical operations. Implementations should avoid OS mutexes in the per-layer hot path; spin barriers or user-space barriers are acceptable provided they do not alter results.

### 11.6 Stream initialization and persistence
For a new stream instance with no prior persisted state:
- s[d] shall be initialized to 0 for all d.
- tick_accum_q16 shall be initialized to 0.
- tick_count shall be initialized to 0.
- phase shall be initialized to 0.

If a stream instance is resumed, the persisted values s, tick_accum_q16, tick_count, and phase shall be restored bitwise exactly.

### 11.7 Input banks and required microblock metadata
This specification defines the following bank identifiers (bank_id is an unsigned 8-bit integer):
- `BANK_RAW256` (bank_id = 0): codes are raw byte values 0..255. microblock_meta is empty.
- `BANK_AUDIO_CFF256` (bank_id = 1): codes are 0..255 values produced by an external audio-to-code mapper. For this bank, microblock_meta shall include `frames_in_microblock` as a positive integer. The `frames_in_microblock` value shall be a positive integer and shall be provided for every microblock processed under this bank.
- `BANK_JPEG_DCT` (bank_id = 2): reserved for future use; not required for conformance to this version.

### 11.8 Reserved bank: BANK_JPEG_DCT
BANK_JPEG_DCT is reserved for a future milestone and is not trained in v1.

Three-field mapping to 256 codes:
- TYPE in 0..63 → codes 0..63
- POS in 0..63 → codes 64..127
- VAL in 0..127 → codes 128..255

This preserves the runtime shape without requiring v1 teacher precompute for images.

## 12. TEMPEST code mapping for audio

### 12.1 Requirement
Inference must be fast. The runtime shall not run a neural mapper per symbol.

### 12.2 Learned mapping exported as LUT
Training:
- Learn a discrete assignment from structured audio symbols to codes 0..255.
- Use STE or Gumbel-softmax during training to learn assignments end-to-end.

Export:
- Compile the final mapping into a deterministic LUT keyed by the symbol fields.
- Runtime mapping becomes O(1) table lookup.

## 13. Model artifact requirements

A model artifact shall include all tensors and metadata necessary to execute Clauses 6 to 11.

### 13.1 Required metadata
The artifact shall include:
- configuration tuple (L,D,K,E) (6.1),
- `post_exchange_activation_enabled` as a boolean,
- per-bank constants for each supported bank_id:
  - `b_shift[bank_id]` (non-negative integer),
  - `act_shift[bank_id]` (non-negative integer),
  - `T[bank_id]` (int16),
  - `E_ticks[bank_id]` (positive integer),
  - if bank_id == BANK_AUDIO_CFF256: `frames_per_tick` (positive integer),
- `inj_shift` (non-negative integer),
- if D != 512: `p_shift` (non-negative integer),
- LayerNorm shifts `ln_gamma_shift[l]` and `ln_beta_shift[l]` for l in {0..L-2},
- Exchange shifts `m_shift[l][r]` for l in {0..L-1}, r in {0..E-1}.

### 13.2 Required tensors
The artifact shall include the tensors listed in 6.4 with the exact shapes implied by the configuration.

### 13.3 Tensor indexing
All tensors indexed by state dimension d (for example a[l][c][d]) shall use canonical dimension order (see 7.1). Implementations may transpose tensors internally for performance, but the observable behavior shall match canonical indexing.

---

## Annex A
(normative)

**Reference algorithms and fixed tables**

### A.1 Reference integer square root (isqrt)
A conforming implementation shall implement isqrt(n) as floor(sqrt(n)) for n >= 0. The following reference algorithm is normative and may be used directly.

Pseudocode (unsigned 64-bit arithmetic is recommended):

```text
function isqrt(n: uint64) -> uint32:
    # Returns floor(sqrt(n)).
    x = n
    res = 0
    bit = 1 << 62  # The second-to-top bit set (even bit index).
    while bit > x:
        bit >>= 2
    while bit != 0:
        if x >= res + bit:
            x = x - (res + bit)
            res = (res >> 1) + bit
        else:
            res = res >> 1
        bit >>= 2
    return (uint32)res
```

### A.2 tanh_LUT table (Q15)
tanh_LUT has 256 entries. Index k corresponds to x8 = k-128. Each entry is an int16 in Q15. The following values are normative.

```text
-24955, -24847, -24738, -24627, -24515, -24401, -24287, -24171, -24053, -23935, -23815, -23693, -23570, -23446, -23320, -23193
-23065, -22935, -22804, -22671, -22537, -22401, -22264, -22126, -21986, -21844, -21701, -21557, -21411, -21263, -21114, -20964
-20812, -20658, -20503, -20347, -20189, -20029, -19868, -19706, -19541, -19376, -19208, -19040, -18869, -18697, -18524, -18349
-18173, -17995, -17815, -17634, -17451, -17267, -17081, -16894, -16706, -16515, -16324, -16130, -15936, -15740, -15542, -15343
-15142, -14940, -14737, -14532, -14325, -14118, -13908, -13698, -13486, -13273, -13058, -12842, -12625, -12406, -12186, -11965
-11742, -11519, -11294, -11067, -10840, -10611, -10382, -10151,  -9919,  -9686,  -9452,  -9216,  -8980,  -8743,  -8505,  -8265
 -8025,  -7784,  -7542,  -7299,  -7056,  -6811,  -6566,  -6320,  -6073,  -5825,  -5577,  -5328,  -5079,  -4828,  -4578,  -4326
 -4075,  -3822,  -3570,  -3317,  -3063,  -2809,  -2555,  -2300,  -2045,  -1790,  -1535,  -1279,  -1024,   -768,   -512,   -256
     0,    256,    512,    768,   1024,   1279,   1535,   1790,   2045,   2300,   2555,   2809,   3063,   3317,   3570,   3822
  4075,   4326,   4578,   4828,   5079,   5328,   5577,   5825,   6073,   6320,   6566,   6811,   7056,   7299,   7542,   7784
  8025,   8265,   8505,   8743,   8980,   9216,   9452,   9686,   9919,  10151,  10382,  10611,  10840,  11067,  11294,  11519
 11742,  11965,  12186,  12406,  12625,  12842,  13058,  13273,  13486,  13698,  13908,  14118,  14325,  14532,  14737,  14940
 15142,  15343,  15542,  15740,  15936,  16130,  16324,  16515,  16706,  16894,  17081,  17267,  17451,  17634,  17815,  17995
 18173,  18349,  18524,  18697,  18869,  19040,  19208,  19376,  19541,  19706,  19868,  20029,  20189,  20347,  20503,  20658
 20812,  20964,  21114,  21263,  21411,  21557,  21701,  21844,  21986,  22126,  22264,  22401,  22537,  22671,  22804,  22935
 23065,  23193,  23320,  23446,  23570,  23693,  23815,  23935,  24053,  24171,  24287,  24401,  24515,  24627,  24738,  24847
```

---

## Annex B
(informative)

**Revision history**

### B.1 Changes in version 1.3.1

#### B.1.1 Restored design rationale section
Added Clause 1.3 (Design rationale) and Clause 1.4 (Architecture summary) describing the root-cause analysis of the baseline and the monoid architecture motivation.

#### B.1.2 Added informative references
Added Clause 3 with informative academic references for design decisions (Saxe 2013, Gu 2021, Miyato 2018, Le 2015, Koutník 2014).

#### B.1.3 Restored stability constraints
Added Clause 9 (Stability and drift) specifying intrinsic stability constraints on a[c,d], spectral constraints on M, and activation as a stability component.

#### B.1.4 Restored initialization strategy
Added Clause 10 (Initialization strategy) specifying initialization for a, b, M, and fixed components.

#### B.1.5 Restored TEMPEST code mapping
Added Clause 12 (TEMPEST code mapping for audio) specifying LUT export requirements.

#### B.1.6 Restored reserved bank details
Added Clause 11.8 (Reserved bank: BANK_JPEG_DCT) with three-field mapping specification.

#### B.1.7 Added default constant values
Restored informative default values throughout:
- 8.3.2: Default T values (T=24576 for RAW256, T=20480 for AUDIO_CFF256)
- 8.5.3: Default inj_shift = 3
- 8.6.3: Default frames_per_tick = 2
- 8.6.4: Default E_ticks values (1 for RAW256, 2 for AUDIO_CFF256)

#### B.1.8 Added training guidance annex
Added Annex F (informative) with teacher source (Gemma 3n E4B family), Matryoshka validation (EmbeddingGemma), training recipe, and throughput guidance. Teacher input processing specifies audio at 6.25 tokens per second per v1.2.2.

#### B.1.9 Added three-thread reference pattern
Added Annex G (informative) documenting the Core 0/1/2 (Orchestrator/Worker A/Worker B) role assignment for three-thread deployments, preserving implementation clarity from v1.2.2.

#### B.1.10 Strengthened frames_in_microblock requirement
Clause 11.7 now explicitly states that frames_in_microblock shall be provided for every microblock processed under BANK_AUDIO_CFF256.

#### B.1.11 Simplified preset naming (cosmetic)
Reference preset names in Clause 6.6 use plain identifiers (small, medium, base, large) rather than method-call syntax (MonoidEmbedConfig.small()). This is a documentation simplification; implementations may use either naming convention.

### B.2 Changes in version 1.3.0

#### B.2.1 Adopted IEEE-style clause structure
Rewrote the document into IEEE-style clauses, definitions, and conformance requirements for clarity and interoperability.

#### B.2.2 Removed fixed three-core requirement
Removed the requirement to use exactly three CPU cores per stream instance. The execution model now specifies a single-core fast path and deterministic scaling across multiple cores (Clause 11).

#### B.2.3 Defined deterministic custom-variant derivation
Added deterministic algorithms to derive complete configurations from (n_layers, d_state) or from a min/max parameter budget (6.7).

#### B.2.4 Made LayerNorm fully specified
Replaced the prior 'model must document' LayerNorm approximation requirement with a single deterministic fixed-point LayerNorm algorithm (7.5).

---

## Annex C
(informative)

**Design rationale**

### C.1 Removing fixed core counts

#### C.1.1 Issue
A fixed core count constrains deployment and prevents using available CPU resources on systems with more or fewer cores.

#### C.1.2 Risk
Hard-coding a specific thread topology can reduce throughput on both single-core constrained environments and multi-core servers, and can complicate integration in systems with their own schedulers.

#### C.1.3 Resolution
The model remains CPU-first and bounded-memory, but the runtime is specified for thread_count = 1 (fast path) and deterministic multi-thread execution with tile-parallel partitioning and explicit barriers where required for correctness (Clause 11).

#### C.1.4 Remaining uncertainty
None.

### C.2 Expressivity and nonlinearity

#### C.2.1 Issue
A pure affine recurrence plus linear mixing can behave like an effectively linear dynamical system end-to-end.

#### C.2.2 Risk
Depth does not add expressivity without nonlinearities in deep linear networks [3].

#### C.2.3 Resolution
Apply a deterministic activation at microblock boundaries to prevent "linearity collapse," while preserving the monoid property inside each microblock.

#### C.2.4 Remaining uncertainty
None.

---

## Annex D
(informative)

**Issue disposition**

| Issue ID | Description | Resolution | Clause |
|---|---|---|---|
| D-1 | Fixed three-core execution constraint limited scalability | Removed; replaced with single-core + multi-core model | 11 |
| D-2 | LayerNorm approximation left to per-model documentation | Fully specified deterministic LayerNorm | 7.5 |
| D-3 | Custom variant sizing not defined | Added deterministic derivation procedures | 6.7 |
| D-4 | Missing stability constraints from v1.2.2 | Restored in Clause 9 | 9 |
| D-5 | Missing initialization guidance from v1.2.2 | Restored in Clause 10 | 10 |
| D-6 | Missing default constant values | Restored as informative defaults | 8.3.2, 8.5.3, 8.6.3, 8.6.4 |
| D-7 | Three-thread role pattern lost implementation clarity | Added informative Annex G with Core 0/1/2 mapping | Annex G |
| D-8 | frames_in_microblock requirement split across sections | Strengthened explicit requirement in bank definition | 11.7 |

---

## Annex E
(informative)

**Considered alternatives**

### E.1 Variable N_TILES for additional parallelism

**Considered:** Making N_TILES a configuration parameter to increase parallelism beyond eight tiles.

**Rejected:** Changing N_TILES affects state layout, prefix interleaving properties, and exchange dimensionality, increasing compatibility risk for existing trained variants.

**Resolution:** Keep N_TILES fixed at 8 for this version and scale via tile-parallel execution (Clause 11).

---

## Annex F
(informative)

**Training guidance**

This annex provides informative guidance for training MonoidEmbed models. These recommendations are not normative requirements.

### F.1 Teacher source

#### F.1.1 Teacher model
Teacher model: Gemma 3n E4B family.

#### F.1.2 Matryoshka targets
Matryoshka-style truncation is validated for EmbeddingGemma: the model produces a larger embedding (documented as 768 dimensions) and supports truncation to 512, 256, and 128 dimensions via Matryoshka Representation Learning, with re-normalization after truncation.

#### F.1.3 Teacher input processing
Model documentation indicates:
- audio input is encoded at 6.25 tokens per second from a single channel
- images are encoded to 256 tokens per image (not used in v1 training)

#### F.1.4 Teacher projection head
Train a projection head P_teacher that maps teacher hidden states to 512-d targets.

**Paired dataset:**
- Audio-text pairs: audiobook segments with transcripts; podcasts with captions
- Text-text positives: adjacent paragraphs, summary pairs, paraphrases

**Contrastive loss:**
InfoNCE with τ = 0.05, in-batch negatives, and optional hard-negative mining from a queue of recent embeddings.

**Completion criterion:**
Stop when Recall@10 (audio↔text) stabilizes on a held-out set.

### F.2 Training recipe

#### F.2.1 Dataset composition
Total 30M segments:
- Text 80% (24M)
- Audio 20% (6M)

Images are excluded from v1 to reduce teacher precompute time; BANK_JPEG_DCT remains reserved.

#### F.2.2 Microblock and unroll parameters
- Microblock size shall equal microblock_size in the chosen config.
- Default unroll length: 16 microblocks.
- Batch size: B = 256.

#### F.2.3 Optimizer and schedule
- AdamW
- Warmup: 2% steps
- Peak LR: 2e-3
- Cosine decay to 2e-4

#### F.2.4 Losses
Losses are computed on the canonical (interleaved) output embedding e_out:
- L512, L256, L128 cosine alignment losses
- Default weights:
  - w512 = 1.0
  - w256 = 0.7
  - w128 = 0.5

Regularizers:
- stability regularizer on a
- saturation rate penalty
- optional Lipschitz proxy penalty (in addition to spectral constraint on M)

### F.3 Throughput targets

The monoid structure amortizes per-symbol work; the design goal is to exceed the 4 KB/s baseline by at least 10× on comparable hardware.

Non-normative indicative targets:
- small (~0.263M): extreme throughput (e.g., on the order of tens of MB/s on a modern CPU)
- large (~30M): high-capacity mode with substantially lower throughput but still far above the baseline

### F.4 Training milestones

#### F.4.1 Teacher precompute scaling assumption
Use empirical benchmarks and system-level profiling to bound teacher precompute throughput, since performance depends on kernels, precision, batching, and software stack.

#### F.4.2 Explicit training milestones
**Checkpoint A (engineering gate): 200k steps**
- validate training stack
- confirm stable streaming dynamics under quantization
- confirm Matryoshka prefix behavior
- produce a usable early-integration model

**Checkpoint B (quality gate): 1.5–2.5M steps**
- reach full embedding-geometry alignment
- confirm long-horizon stability and drift resilience under quantization
- produce ship-quality targets

---

## Annex G
(informative)

**Reference three-thread deployment pattern**

This annex describes a reference thread assignment for three-thread deployments (thread_count = 3), preserving implementation clarity from earlier specification versions.

### G.1 Thread roles
For a three-thread deployment, the following role assignment is recommended:
- **Thread 0 (Orchestrator):**
  - builds microblocks (assigns bank_id and microblock_meta)
  - maintains modality-aware clock state, computes exchange_scheduled for each microblock
  - publishes the current phase for worker threads
  - emits embeddings and Matryoshka prefixes
  - coordinates LayerNorm global statistics
  - may run read-only health/drift monitors (shall not modify model state except for specified saturations)
- **Thread 1 (Worker A):**
  - owns tiles {0,1,2,3} (hemisphere 0)
  - compiles and applies monoid transforms per layer for owned tiles
  - runs mixer, activation, local summary, exchange compute, injection for owned tiles
- **Thread 2 (Worker B):**
  - owns tiles {4,5,6,7} (hemisphere 1)
  - same responsibilities as Worker A for its tiles

### G.2 Exchange barrier mapping
For the three-thread pattern, the exchange barrier in 11.5 maps as follows:
- Thread 1 computes E/2 output dimensions of v (rows 0..E/2-1)
- Thread 2 computes the other E/2 output dimensions of v (rows E/2..E-1)
- Thread 0 coordinates barrier synchronization

### G.3 Relationship to general model
This pattern is a specific instance of the general thread_count model in Clause 11. Conforming implementations using three threads should follow this pattern for consistency with prior implementations, but may use alternative partitioning strategies provided they satisfy the determinism requirements in 5.2.

---

**References**

[1] S. Bradner, "Key words for use in RFCs to Indicate Requirement Levels," RFC 2119, 1997.
[2] B. Leiba, "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words," RFC 8174, 2017.
[3] A. M. Saxe, J. L. McClelland, S. Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," arXiv:1312.6120, 2013.
[4] A. Gu, K. Goel, C. Ré, "Efficiently Modeling Long Sequences with Structured State Spaces," arXiv:2111.00396, 2021.
[5] T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, "Spectral Normalization for Generative Adversarial Networks," arXiv:1802.05957, 2018.
[6] Q. V. Le, N. Jaitly, G. E. Hinton, "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units," arXiv:1504.00941, 2015.
[7] J. Koutník, K. Greff, F. Gomez, J. Schmidhuber, "A Clockwork RNN," ICML 2014; arXiv:1402.3511, 2014.