# MonoidEmbed

A CPU-only streaming embedding model that converts streams of 8-bit byte codes into 512-dimensional INT8 embeddings with explicit per-embedding scaling.

MonoidEmbed replaces token-by-token recurrence with a composable affine monoid compiled over microblocks. Per-symbol affine transforms are composed associatively within each microblock and applied as a single fused update, minimizing state traffic and enabling deterministic, bitwise-identical outputs across conforming implementations.

## Architecture

For each input code `c`, the per-symbol state update is:

```
s ← a[c] ⊙ s + b[c]
```

where `a[c]` and `b[c]` are code-conditioned vectors. Because affine transforms compose associatively, the per-symbol transforms across a microblock are compiled into a single transform `(A, B)` applied once per microblock. A deterministic tanh activation at microblock boundaries prevents linearity collapse, and a low-cost cross-tile exchange provides inter-tile mixing.

**Key properties:**

- **CPU-only** — no GPU/NPU/accelerator dependencies
- **Streaming** — processes unbounded sequences via persistent recurrent state
- **Deterministic** — fixed-point arithmetic with bitwise-identical outputs across threads and platforms
- **Matryoshka-compatible** — supports progressive dimensionality (512, 256, 128)
- **Quantized** — INT8 embeddings with explicit Q15 scale factors

## Model Presets

| Preset | Layers | State Dim | Microblock | Approx. Params | 1T CPU (B=1, ms) |
|--------|--------|-----------|------------|----------------|------------------|
| `small` | 1 | 512 | 256 | ~0.5M | 0.17 |
| `small_l2` | 2 | 512 | 256 | ~1M | 0.33 |
| `small_l5` | 5 | 512 | 256 | ~2.5M | 0.92 |
| `medium` | 1 | 2048 | 64 | ~4M | 0.96 |
| `medium_deep` | 8 | 512 | 128 | ~4M | 1.59 |
| `base` | 15 | 2048 | 128 | ~30M | 11.58 |
| `large` | 27 | 2048 | 128 | ~55M | 21.21 |
| `xlarge` | 216 | 2048 | 128 | ~440M | 172.73 |

`1T CPU` is single-document float-kernel latency (`B=1`) from [`.snapshots/embed_bench_20260114_031044.md`](.snapshots/embed_bench_20260114_031044.md).

## Installation

Requires Python 3.10+ and PyTorch 2.9.1+.

```bash
# Create virtual environment
uv venv --system-site-packages .venv

# Install the package
uv pip install --python .venv/bin/python -e .
```

The C++ CPU kernel is compiled on demand via `torch.utils.cpp_extension` with architecture-specific optimizations (AVX-512, AVX2, ARM NEON).

### Kernel Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONOID_CPU_MARCH` | `native` | Target CPU architecture |
| `MONOID_CPU_OPENMP` | `1` | Enable OpenMP parallelism |
| `MONOID_CPU_FAST_MATH` | `1` | Enable fast math optimizations |
| `MONOID_CPU_FAST_TANH` | follows `FAST_MATH` | Fast tanh approximation |
| `MONOID_CPU_THREADS` | auto | Thread count override |
| `MONOID_PRESET` | — | Config preset selection |

## Usage

### Training

```bash
source .env.sh
uv run --python .venv/bin/python -- scripts/run_embed_training.py \
    --use_wandb --run_name <name>
```

Training uses knowledge distillation from teacher models (EmbeddingGemma-300M for text, Gemma-3n-E4B for audio) into a unified 512-dimensional embedding space. Supported losses include geometric distillation, hardness-weighted contrastive, RKD, VICReg, and more.

### Evaluation

```bash
# MTEB benchmark
uv run --python .venv/bin/python -- scripts/run_mteb_eval.py

# Retrieval evaluation
uv run --python .venv/bin/python -- scripts/run_retrieval_eval.py

# Performance benchmarking
uv run --python .venv/bin/python -- scripts/run_embed_bench.py
uv run --python .venv/bin/python -- scripts/run_inference_benchmark.py
```

### Verification

```bash
# Verify CPU kernel correctness against float reference
uv run --python .venv/bin/python -- scripts/verify_kernel_correctness.py

# Verify INT8 quantization sanity
uv run --python .venv/bin/python -- scripts/verify_stacked_int8_sanity.py

# Regression testing
uv run --python .venv/bin/python -- scripts/run_embed_regression.py
```

## Project Structure

```
src/monoid/
├── embed/
│   ├── model.py                  # MonoidEmbed model and config
│   └── monoid_cpu/
│       ├── __init__.py           # CPU kernel Python wrapper
│       ├── extension.py          # C++ extension loader
│       └── monoid_cpu.cpp        # SIMD-optimized C++ kernel
└── training/
    └── embed/
        ├── data.py               # Multimodal dataset classes
        ├── loss.py               # Distillation and contrastive losses
        ├── teacher.py            # Teacher model handler
        └── teacher_cache.py      # Embedding cache

scripts/                          # Training, eval, and benchmark scripts
docs/
├── spec.md                       # MonoidEmbed specification (v1.3.1)
└── experiments.md                # Benchmark results and analysis
```

## Status

This project is a work in progress. Training was not completed due to GPU credit constraints — no fully trained checkpoint is available. The architecture, spec, CPU kernel, and training infrastructure are functional, but published benchmark numbers in [docs/experiments.md](docs/experiments.md) reflect partially trained models.

**Last training checkpoint:** step **9500**, epoch **0** (2026-01-17).

```bash
source .env.sh
uv run --python .venv/bin/python -- scripts/run_embed_training.py \
    --use_wandb --use_real_teacher \
    --run_name train-mbe-6m-phase1-bs256-clip20-freeze20k-cap001 \
    --preset mbe_6m \
    --resume checkpoints/train-mbe-6m-phase1-bs256-clip20-freeze20k-cap001/9500_weights.pt \
    --resume_step 0 \
    --output_dir checkpoints/train-mbe-6m-phase1-bs256-clip20-freeze20k-cap001 \
    --streaming --max_steps 10000 \
    --batch_size 256 --grad_accum_steps 1 \
    --num_workers 1 --persistent_workers --prefetch_factor 16 \
    --dataset_mix \
        allenai/c4:en:train:0.55 \
        wikimedia/wikipedia:20231101.en:train:0.15 \
        Skylion007/openwebtext::train:0.15 \
        sentence-transformers/all-nli:pair:train:0.10 \
        sentence-transformers/wikianswers-duplicates:pair:train:0.05 \
    --train_retrieval --retrieval_train_pairs 20000 \
    --retrieval_eval --retrieval_eval_every 500 --retrieval_queries 500 \
    --retrieval_ramp_steps 1000 --retrieval_mult_start 1 --retrieval_mult_target 1 \
    --freeze_exchange_steps 20000 \
    --modality multimodal --text_audio_ratio 1.0 \
    --text_prompt_name document --text_prompt_mix document --text_prompt_mix_mode alternate \
    --optimizer adamw --lr 0.002 --lr_schedule cosine \
    --peak_lr 0.0005 --min_lr 0.0002 --warmup_frac 0.02 \
    --rkd_distance_weight 0.5 \
    --contrast_scale_cap 0.01 --contrast_start_cos 1.0 \
    --max_grad_norm 20 \
    --log_jsonl logs/train-mbe-6m-phase1-bs256-clip20-freeze20k-cap001.jsonl \
    --log_every 20 --disable_tqdm
```

| Group | Metric | Value |
|-------|--------|-------|
| Loss | `loss/total` | 0.255 |
| | `loss/distill` | 0.110 |
| | `loss/contrast` | 1.506 |
| | `loss/var` / `loss/var_text` | 0.960 |
| | `loss/L512` / `L256` / `L128` | 0.134 / 0.103 / 0.089 |
| Optim | `train/lr` | 2.02e-4 |
| | `train/grad_norm` | 0.344 |
| Geometry | `geom/cos_T_S_128` (EMA) | 0.911 (0.696) |
| | `geom/cos_T_S_256` (EMA) | 0.897 (0.675) |
| | `geom/cos_T_S_512` (EMA) | 0.866 (0.636) |
| | `geom/neighborhood_overlap@5` / `@10` | 0.691 / 0.727 |
| | `geom/rank_correlation` | 0.887 |
| Retrieval (eval) | sim stats (min / mean / max) | −0.257 / 0.114 / 0.659 |
| | positive sims (mean ± std, n) | 0.301 ± 0.140 (n=300) |
| | `retrieval/recall@1` / `@5` / `@10` | 0.29 / 0.39 / 0.44 |

Full scalar dump from that step (W&B-style log; the original export was truncated after teacher off-diagonal percentiles):

```json
{
  "step": 9500,
  "epoch": 0,
  "loss/total": 0.254952996969223,
  "loss/distill": 0.10967721045017242,
  "loss/spread": 0.04214286431670189,
  "loss/spkd": 0.0038256391417235136,
  "loss/var": 0.959797203540802,
  "loss/var_text": 0.959797203540802,
  "loss/var_audio": 0.0,
  "loss/rkd_distance": 0.0004408185195643455,
  "loss/rkd_angle": 0.0006056143320165575,
  "loss/contrast": 1.506019949913025,
  "loss/pairwise": 0.01167137548327446,
  "loss/consistency": 0.0,
  "loss/neighborhood": 0.0008841946255415678,
  "loss/cross_modal": 0.0,
  "loss/mm_distill": 0.0,
  "loss.total": 0.254952996969223,
  "loss.distill": 0.10967721045017242,
  "loss.contrast": 1.506019949913025,
  "loss.neighborhood": 0.0008841946255415678,
  "loss.rkd_d": 0.0004408185195643455,
  "loss.rkd_a": 0.0006056143320165575,
  "loss.saturation_penalty": 0.0,
  "train/lr": 0.00020192273248783248,
  "lr.value": 0.00020192273248783248,
  "sched.align_mult": 1.0,
  "sched.retrieval_mult": 1.0,
  "loss.align_block": 0.10967721045017242,
  "loss.retrieval_block": 0.0028043421916663647,
  "train/grad_norm": 0.34423139691352844,
  "grad.norm.global": 0.34423139691352844,
  "grad.norm.global_post_clip": 0.3442314031222116,
  "grad.clipped": 0.0,
  "grad.nan_or_inf": 0.0,
  "train/alpha_hardness": 5.0,
  "train/contrast_scale": 0.01,
  "loss/L512": 0.1342424601316452,
  "loss.L512": 0.1342424601316452,
  "loss/L256": 0.1028900146484375,
  "loss.L256": 0.1028900146484375,
  "loss/L128": 0.08851106464862823,
  "loss.L128": 0.08851106464862823,
  "loss/align_weighted": 0.2505210041999817,
  "loss.align_weighted": 0.2505210041999817,
  "loss/total_recomputed": 0.254952996969223,
  "loss/total_recompute_err": 0.0,
  "loss/tau": 0.05,
  "grad.norm.a": 0.003263222001230031,
  "grad.norm.b": 0.33253607371717936,
  "grad.norm.M": 0.0,
  "grad.is_none.M_param": 1.0,
  "grad.norm.M_param": 0.0,
  "grad_norm_M": 0.0,
  "M.lr_mult": 1.0,
  "param.name.M_param": "blocks.0.exchange.parametrizations.weight.original",
  "exchange.frozen": 1.0,
  "embed/student_norm_mean": 1.0,
  "embed/student_norm_std": 2.60770320892334e-08,
  "embed/teacher_norm_mean": 1.0,
  "embed/teacher_norm_std": 1.3301947099364497e-08,
  "embed/student_norm_pre_mean": 1.0,
  "embed/student_norm_pre_std": 2.60770320892334e-08,
  "embed/student_norm_post_mean": 1.0,
  "embed/student_norm_post_std": 1.4307248719092058e-08,
  "embed/teacher_norm_pre_mean": 1.0,
  "embed/teacher_norm_pre_std": 1.3301947099364497e-08,
  "embed/teacher_norm_post_mean": 1.0,
  "embed/teacher_norm_post_std": 6.4523919540704355e-09,
  "embed.l2_norm_mean": 1.0,
  "embed.l2_norm_p95": 1.0,
  "embed.scale_q15_mean": 32767.0,
  "embed.scale_q15_p95": 32767.0,
  "a.max_abs": 0.9989794492721558,
  "a.mean_abs": 0.9866040945053101,
  "M.spectral_norm_after": 0.0,
  "M.sn_isfinite": 1.0,
  "M.sn_max": 1.0,
  "activation.pre_clip_max": 3840.31494140625,
  "activation.post_clip_max": 1.0,
  "activation.sat_frac_gt_0p99": 0.4836239218711853,
  "sched.m_lr_mult": 1.0,
  "M_freeze_brake_active": 0.0,
  "geom/cos_T_S_128": 0.911488950252533,
  "geom_ema/cos_T_S_128": 0.6961604515331971,
  "geom/cos_T_S_256": 0.8971099853515625,
  "geom_ema/cos_T_S_256": 0.6749343545704763,
  "geom/cos_T_S_512": 0.8657575249671936,
  "geom_ema/cos_T_S_512": 0.6360452391510023,
  "geom/batch_text_count": 512,
  "geom/batch_audio_count": 0,
  "geom/cos_T_S_512_text": 0.8657575249671936,
  "geom/text_audio_matched_count": 0,
  "geom/neighborhood_overlap@5": 0.6906250715255737,
  "geom/neighborhood_overlap@10": 0.7265625,
  "geom/rank_correlation": 0.8874073624610901,
  "geom/teacher_cos_offdiag_mean": 0.20372143387794495,
  "geom/teacher_cos_offdiag_std": 0.11394510418176651,
  "geom/student_cos_offdiag_mean": 0.16356907784938812,
  "geom/student_cos_offdiag_std": 0.1177237257361412,
  "geom/cos_mean_gap": -0.040152356028556824,
  "geom/cos_std_gap": 0.003778621554374695,
  "embed/teacher_pairwise_cos_mean": 0.20372143387794495,
  "embed/teacher_pairwise_cos_std": 0.11394510418176651,
  "embed/student_pairwise_cos_mean": 0.16356907784938812,
  "embed/student_pairwise_cos_std": 0.1177237257361412,
  "geom/teacher_cos_offdiag_p10": 0.06189243495464325,
  "geom/teacher_cos_offdiag_p50": 0.19737178087234497,
  "geom/teacher_cos_offdiag_p90": 0.3522588610649109
}
```

Retrieval eval log (same step):

```
Retrieval eval sim stats: min=-0.2568 mean=0.1140 max=0.6586
Retrieval eval norms: query mean=1.0000 std=0.0000 | doc mean=1.0000 std=0.0000
Retrieval eval positive sims: mean=0.3013 std=0.1400 (n=300)
```

```json
{"retrieval/recall@1": 0.29, "retrieval/recall@5": 0.38666666666666666, "retrieval/recall@10": 0.44333333333333336, "step": 9500}
```

If you're interested in sponsoring GPU compute to finish training, reach out at gabewillen@gmail.com — I'd happily continue.

## Specification

The full normative specification is in [docs/spec.md](docs/spec.md). It defines the model configuration, parameter tensors, fixed-point arithmetic, deterministic reference algorithms, single-core and multi-core execution models, stability constraints, and initialization strategies.

## License

See [LICENSE](LICENSE) for details.
