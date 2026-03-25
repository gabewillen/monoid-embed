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

| Preset | Layers | State Dim | Microblock | Approx. Params |
|--------|--------|-----------|------------|----------------|
| `small` | 1 | 512 | 256 | ~0.5M |
| `small_l2` | 2 | 512 | 256 | ~1M |
| `small_l5` | 5 | 512 | 256 | ~2.5M |
| `medium` | 1 | 2048 | 64 | ~4M |
| `medium_deep` | 8 | 512 | 128 | ~4M |
| `base` | 15 | 2048 | 128 | ~30M |
| `large` | 27 | 2048 | 128 | ~55M |
| `xlarge` | 216 | 2048 | 128 | ~440M |

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

If you're interested in sponsoring GPU compute to finish training, reach out at gabewillen@gmail.com — I'd happily continue.

## Specification

The full normative specification is in [docs/spec.md](docs/spec.md). It defines the model configuration, parameter tensors, fixed-point arithmetic, deterministic reference algorithms, single-core and multi-core execution models, stability constraints, and initialization strategies.

## License

See [LICENSE](LICENSE) for details.
