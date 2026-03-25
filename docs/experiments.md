# Experiments

## 2026-01-14: Phase 7 baseline rerun (pinned + fast-math sweep)

### Benchmark setup
- Host: AMD EPYC 9124 (32 threads).
- Input: `docs/embed.md` truncated to 4096 bytes.
- Batch: 1 (single-doc kernel throughput).
- Kernel: `MonoidCpuKernel.forward_full_precision`.
- Pinning: `taskset -c 0` (threads=1) and `taskset -c 0-31` (threads=32), plus `--pin` in the script.
- Threads: `OMP_NUM_THREADS` set to thread count, `MKL_NUM_THREADS=1`.
- Fast math: `MonoidCpuConfig(fast_math=0/1)` (fast tanh follows `fast_math`).

### Results (threads=1)
| Preset | fast_math=0 ms | fast_math=0 KB/s | fast_math=1 ms | fast_math=1 KB/s |
| --- | --- | --- | --- | --- |
| base | 29.82 | 134.14 | 12.11 | 330.34 |
| large | 55.61 | 71.93 | 22.10 | 181.02 |
| medium | 3.54 | 1128.44 | 1.07 | 3752.45 |
| medium_deep | 3.96 | 1009.93 | 1.46 | 2743.17 |
| small | 0.34 | 11870.67 | 0.18 | 21788.59 |
| small_2l | 0.68 | 5862.74 | 0.35 | 11554.56 |
| small_l2 | 0.65 | 6162.05 | 0.38 | 10605.07 |
| small_l3 | 1.02 | 3937.08 | 0.51 | 7806.99 |
| small_l4 | 1.31 | 3055.22 | 0.67 | 5940.24 |
| small_l5 | 1.69 | 2364.10 | 0.85 | 4730.42 |
| xlarge | 443.28 | 9.02 | 181.52 | 22.04 |

### Results (threads=32)
| Preset | fast_math=0 ms | fast_math=0 KB/s | fast_math=1 ms | fast_math=1 KB/s |
| --- | --- | --- | --- | --- |
| base | 34.76 | 115.09 | 15.93 | 251.13 |
| large | 63.71 | 62.78 | 29.23 | 136.87 |
| medium | 4.18 | 957.37 | 1.65 | 2428.38 |
| medium_deep | 6.15 | 650.01 | 3.54 | 1129.90 |
| small | 0.46 | 8670.40 | 0.32 | 12394.52 |
| small_2l | 0.92 | 4363.16 | 0.60 | 6644.44 |
| small_l2 | 1.22 | 3270.80 | 0.61 | 6606.24 |
| small_l3 | 1.41 | 2836.67 | 0.92 | 4335.20 |
| small_l4 | 1.82 | 2192.07 | 1.26 | 3180.03 |
| small_l5 | 3.27 | 1223.29 | 1.54 | 2598.86 |
| xlarge | 547.69 | 7.30 | 254.96 | 15.69 |

Notes:
- threads=32 table updated with `repeat=5` for stability. fast_math=0 still trails fast_math=1 on base/large by ~2.2–2.4x.

## 2026-01-14: Control-flow + correctness parity checks

### Scope
- Verify `activation_T<=0` skips tanh (no NaNs) and matches kernel behavior.
- Verify exchange scheduling and pooling (`exchange_every=2`, `pool=last`) parity.
- Re-run float correctness for fast_math=0/1 and quant edge tests.
- Confirm thread-count determinism (`threads=1` vs `8`).

### Results
- `activation_T=0` parity (manual check): max diff ~7e-08, cosine=1.0 (no NaNs).
- `exchange_every=2` parity (manual check): max diff ~6e-08, cosine=1.0.
- `pool=last` parity (manual check): max diff ~7e-08, cosine=1.0.
- Float correctness (`scripts/verify_kernel_correctness.py`) passes for `small`, `small_l2` with fast_math=0 and fast_math=1.
- Quant + int8 correctness + edge tests pass for `small` preset (hi_gain, hi_shift).
- Thread determinism: `threads=1` vs `threads=8` max diff 0 for float + quant.

Commands:
```bash
source .env.sh; MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small small_l2 --modes float --batch-sizes 2 --seq-lens 512 \
  --normalize-output 1 --pool-strategies mean --use-exchange 1 --use-second-activation 0

source .env.sh; MONOID_CPU_FAST_MATH=1 MONOID_CPU_FAST_TANH=1 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small small_l2 --modes float --batch-sizes 2 --seq-lens 512 \
  --normalize-output 1 --pool-strategies mean --use-exchange 1 --use-second-activation 0

source .env.sh; MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small --modes quant quant_int8 --batch-sizes 2 --seq-lens 512 \
  --normalize-output 1 --pool-strategies mean --use-exchange 1 --use-second-activation 0 --edge-tests
```

## 2026-01-14: Phase 11 random grid correctness

### Float grid
- Presets: `small` (K=256), `medium` (K=64), `base` (K=128).
- Batches: 1, 2, 8, 64 (plus 256 for `small` only).
- Seq lens: 64, 256.
- Toggles: pool mean/last, normalize on/off, exchange on/off, second activation on/off.
- Result: all PASS (0 failures).

### Quant + int8 grid (small preset)
- Batches: 1, 8, 64.
- Seq lens: 64, 256.
- Toggles: pool mean/last, normalize on/off, exchange on/off, second activation on/off.
- Result: all PASS (0 failures).

Logs:
- `tmp/phase11_random_float.txt`
- `tmp/phase11_random_float_b256.txt`
- `tmp/phase11_random_quant.txt`

Commands:
```bash
source .env.sh; MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small medium base --modes float --batch-sizes 1 2 8 64 \
  --seq-lens 64 256 --pool-strategies mean last --normalize-output 0 1 --use-exchange 0 1 --use-second-activation 0 1

source .env.sh; MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small --modes float --batch-sizes 256 --seq-lens 64 256 \
  --pool-strategies mean last --normalize-output 0 1 --use-exchange 0 1 --use-second-activation 0 1

source .env.sh; MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- \
  scripts/verify_kernel_correctness.py --presets small --modes quant quant_int8 --batch-sizes 1 8 64 --seq-lens 64 256 \
  --pool-strategies mean last --normalize-output 0 1 --use-exchange 0 1 --use-second-activation 0 1
```

## 2026-01-13: Phase 3 CPU kernel refactor (per-sample blocks + pooled output)

### Changes
- Restructured float, stacked, and quant kernels to process per-sample blocks in one pass (avoid empty-tail work).
- Accumulate pooled outputs directly in `output` and normalize in-place (remove per-batch sum buffer).
- Reuse thread-local scratch buffers for exchange/temp data (remove per-iteration allocations).

### Benchmark setup
- Host: AMD EPYC 9124 (32 threads).
- Input: `docs/embed.md` truncated to 4096 bytes.
- Batch: 64.
- Checkpoints: random presets (`tmp/random_ckpts/*.pt`).
- Kernel: `MonoidCpuKernel.forward_full_precision`.
- Timed with `warmup=3`, `repeat=5`.

### Before vs After (B=64, L=4096)

Single core (threads=1):
| Preset | Before | After |
| --- | --- | --- |
| small | 11.02 ms (23.24 MB/s) | 11.29 ms (22.13 MB/s) |
| small_l2 | 23.04 ms (11.11 MB/s) | 23.09 ms (10.83 MB/s) |
| medium | 76.94 ms (3.25 MB/s) | 76.74 ms (3.26 MB/s) |
| base | 895.61 ms (0.28 MB/s) | 902.05 ms (0.28 MB/s) |

Multi-core (threads=32):
| Preset | Before | After (refactor) |
| --- | --- | --- |
| small | 1.04 ms (245.29 MB/s) | 1.79 ms (139.78 MB/s) |
| small_l2 | 1.97 ms (129.98 MB/s) | 3.45 ms (72.41 MB/s) |
| medium | 1.31 ms (3.05 MB/s) | 5.63 ms (44.43 MB/s) |
| base | 1.97 ms (129.98 MB/s) | 65.10 ms (3.84 MB/s) |

Notes:
- Single-core performance is roughly unchanged.
- Multi-core throughput regressed in this run; likely needs follow-up on parallelization strategy and/or cache behavior after the refactor.
- SIMD butterfly shows modest gains in float (more visible at medium/base), and clearer gains in quantized throughput (especially multi-core).
- Attempted 64-byte aligned scratch buffers; multi-thread benchmarks crashed (SIGSEGV), so the change was reverted and not included in results.
- Exchange-row parallelism now uses shared scratch for `batch==1` to avoid TLS cross-thread access; multi-thread `run_inference_benchmark.py` completes for small_l2 (threads=2/8/32).

### Quantized experiments (small preset, B=64, L=4096)

Quantized uses `MonoidCpuKernel.forward` with `emit_int8` toggled.

Single core (threads=1):
| Mode | Before | After (refactor) | After (SIMD butterfly) | After (exchange precompute) | After (SIMD norm/LN) | After (quant SIMD matvec) | After (SIMD recurrence) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int16->float | 17.04 ms (15.03 MB/s) | 18.38 ms (13.60 MB/s) | 16.15 ms (15.48 MB/s) | 15.17 ms (16.48 MB/s) | 15.08 ms (16.58 MB/s) | 15.26 ms (16.39 MB/s) | 10.65 ms (23.47 MB/s) |
| int8 | 17.07 ms (14.99 MB/s) | 18.17 ms (13.76 MB/s) | 16.09 ms (15.53 MB/s) | 15.14 ms (16.51 MB/s) | 15.14 ms (16.52 MB/s) | 15.17 ms (16.48 MB/s) | 10.68 ms (23.41 MB/s) |

Multi-core (threads=32):
| Mode | Before | After (refactor) | After (SIMD butterfly) | After (exchange precompute) | After (SIMD norm/LN) | After (quant SIMD matvec) | After (SIMD recurrence) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int16->float | 17.02 ms (15.04 MB/s) | 2.63 ms (94.89 MB/s) | 2.32 ms (107.54 MB/s) | 1.60 ms (156.12 MB/s) | 2.29 ms (108.99 MB/s) | 2.24 ms (111.43 MB/s) | 1.71 ms (146.25 MB/s) |
| int8 | 17.07 ms (15.00 MB/s) | 2.40 ms (104.30 MB/s) | 2.40 ms (104.22 MB/s) | 1.73 ms (144.75 MB/s) | 2.33 ms (107.52 MB/s) | 2.30 ms (108.89 MB/s) | 1.79 ms (139.69 MB/s) |

### Float experiments (B=1, L=4096)

Single core (threads=1):
| Preset | Before | After (SIMD butterfly) | After (exchange precompute) | After (SIMD norm/LN) | After (quant SIMD matvec) | After (SIMD recurrence) |
| --- | --- | --- | --- | --- | --- | --- |
| small | 0.18 ms (21.90 MB/s) | 0.18 ms (22.39 MB/s) | 0.16 ms (24.67 MB/s) | 0.16 ms (24.55 MB/s) | 0.16 ms (24.33 MB/s) | 0.18 ms (22.09 MB/s) |
| small_l2 | 0.36 ms (11.24 MB/s) | 0.35 ms (11.29 MB/s) | 0.32 ms (12.69 MB/s) | 0.32 ms (12.54 MB/s) | 0.33 ms (12.26 MB/s) | 0.34 ms (11.75 MB/s) |
| medium | 1.29 ms (3.11 MB/s) | 1.16 ms (3.46 MB/s) | 0.92 ms (4.33 MB/s) | 0.92 ms (4.37 MB/s) | 0.94 ms (4.24 MB/s) | 1.08 ms (3.71 MB/s) |
| base | 14.30 ms (0.28 MB/s) | 13.49 ms (0.30 MB/s) | 11.69 ms (0.34 MB/s) | 11.50 ms (0.35 MB/s) | 11.57 ms (0.35 MB/s) | 11.67 ms (0.34 MB/s) |

Multi-core (threads=32):
| Preset | Before | After (SIMD butterfly) | After (exchange precompute) | After (SIMD norm/LN) | After (quant SIMD matvec) | After (SIMD recurrence) |
| --- | --- | --- | --- | --- | --- | --- |
| small | 0.18 ms (21.79 MB/s) | 0.19 ms (21.58 MB/s) | 0.17 ms (22.99 MB/s) | 0.17 ms (23.56 MB/s) | 0.19 ms (20.66 MB/s) | 0.19 ms (21.55 MB/s) |
| small_l2 | 0.36 ms (10.99 MB/s) | 0.37 ms (10.93 MB/s) | 0.34 ms (11.91 MB/s) | 0.32 ms (12.32 MB/s) | 0.33 ms (12.05 MB/s) | 0.35 ms (11.47 MB/s) |
| medium | 1.31 ms (3.05 MB/s) | 1.17 ms (3.42 MB/s) | 0.94 ms (4.26 MB/s) | 0.96 ms (4.18 MB/s) | 0.95 ms (4.19 MB/s) | 0.95 ms (4.23 MB/s) |
| base | 14.72 ms (0.27 MB/s) | 13.41 ms (0.30 MB/s) | 13.83 ms (0.29 MB/s) | 11.83 ms (0.34 MB/s) | 11.43 ms (0.35 MB/s) | 12.09 ms (0.33 MB/s) |

After exchange-row scratch fix (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18001.30 KB/s |
| small_2l | 0.43 ms | 9399.00 KB/s |
| small_l2 | 0.42 ms | 9425.40 KB/s |
| small_l3 | 0.63 ms | 6362.24 KB/s |
| small_l4 | 0.86 ms | 4632.03 KB/s |
| small_l5 | 1.07 ms | 3737.41 KB/s |
| medium | 1.21 ms | 3298.71 KB/s |
| medium_deep | 1.84 ms | 2175.47 KB/s |
| base | 17.00 ms | 235.35 KB/s |
| large | 29.36 ms | 136.24 KB/s |
| xlarge | 240.30 ms | 16.65 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10796.15 KB/s |
| small_2l | 0.72 ms | 5533.38 KB/s |
| small_l2 | 0.85 ms | 4686.37 KB/s |
| small_l3 | 1.09 ms | 3656.76 KB/s |
| small_l4 | 1.45 ms | 2758.05 KB/s |
| small_l5 | 2.26 ms | 1768.82 KB/s |
| medium | 1.95 ms | 2049.50 KB/s |
| medium_deep | 4.62 ms | 865.25 KB/s |
| base | 21.99 ms | 181.93 KB/s |
| large | 42.89 ms | 93.26 KB/s |
| xlarge | 324.00 ms | 12.35 KB/s |

After aligned scratch reintroduced (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18375.92 KB/s |
| small_2l | 0.43 ms | 9404.27 KB/s |
| small_l2 | 0.42 ms | 9478.65 KB/s |
| small_l3 | 0.65 ms | 6186.29 KB/s |
| small_l4 | 0.84 ms | 4739.33 KB/s |
| small_l5 | 1.08 ms | 3701.13 KB/s |
| medium | 1.25 ms | 3203.59 KB/s |
| medium_deep | 1.87 ms | 2144.60 KB/s |
| base | 16.82 ms | 237.88 KB/s |
| large | 33.39 ms | 119.81 KB/s |
| xlarge | 242.68 ms | 16.48 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14450.66 KB/s |
| small_2l | 0.56 ms | 7185.10 KB/s |
| small_l2 | 0.55 ms | 7259.72 KB/s |
| small_l3 | 0.82 ms | 4861.55 KB/s |
| small_l4 | 1.11 ms | 3602.58 KB/s |
| small_l5 | 1.38 ms | 2901.63 KB/s |
| medium | 1.38 ms | 2902.63 KB/s |
| medium_deep | 2.90 ms | 1379.25 KB/s |
| base | 18.01 ms | 222.13 KB/s |
| large | 31.94 ms | 125.23 KB/s |
| xlarge | 262.56 ms | 15.23 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.30 ms | 13168.93 KB/s |
| small_2l | 0.62 ms | 6500.28 KB/s |
| small_l2 | 0.60 ms | 6721.64 KB/s |
| small_l3 | 0.91 ms | 4373.62 KB/s |
| small_l4 | 1.25 ms | 3211.57 KB/s |
| small_l5 | 1.53 ms | 2615.72 KB/s |
| medium | 1.55 ms | 2584.69 KB/s |
| medium_deep | 3.21 ms | 1247.84 KB/s |
| base | 19.94 ms | 200.61 KB/s |
| large | 33.84 ms | 118.21 KB/s |
| xlarge | 273.97 ms | 14.60 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10859.04 KB/s |
| small_2l | 0.71 ms | 5614.86 KB/s |
| small_l2 | 0.74 ms | 5424.25 KB/s |
| small_l3 | 1.09 ms | 3656.76 KB/s |
| small_l4 | 1.48 ms | 2698.60 KB/s |
| small_l5 | 1.95 ms | 2054.52 KB/s |
| medium | 1.85 ms | 2166.20 KB/s |
| medium_deep | 4.25 ms | 940.27 KB/s |
| base | 22.37 ms | 178.84 KB/s |
| large | 40.42 ms | 98.97 KB/s |
| xlarge | 391.28 ms | 10.22 KB/s |

After invariant hoisting (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18579.42 KB/s |
| small_2l | 0.42 ms | 9559.67 KB/s |
| small_l2 | 0.42 ms | 9489.38 KB/s |
| small_l3 | 0.64 ms | 6260.16 KB/s |
| small_l4 | 0.83 ms | 4816.89 KB/s |
| small_l5 | 1.07 ms | 3744.91 KB/s |
| medium | 1.24 ms | 3227.63 KB/s |
| medium_deep | 1.81 ms | 2213.06 KB/s |
| base | 16.45 ms | 243.11 KB/s |
| large | 29.74 ms | 134.51 KB/s |
| xlarge | 241.21 ms | 16.58 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14074.85 KB/s |
| small_2l | 0.58 ms | 6929.87 KB/s |
| small_l2 | 0.55 ms | 7288.10 KB/s |
| small_l3 | 0.83 ms | 4833.54 KB/s |
| small_l4 | 1.14 ms | 3519.45 KB/s |
| small_l5 | 1.40 ms | 2861.54 KB/s |
| medium | 1.44 ms | 2782.75 KB/s |
| medium_deep | 2.91 ms | 1376.31 KB/s |
| base | 18.23 ms | 219.46 KB/s |
| large | 32.36 ms | 123.60 KB/s |

## 2026-01-13: Phase 5 correctness smoke tests

### Setup
- Command: `uv run --python .venv/bin/python -- scripts/verify_kernel_correctness.py --presets small --modes float quant quant_int8 --batch-sizes 2 --seq-lens 256 --determinism`
- Kernel: `MonoidCpuKernel` (full precision + quant)
- Reference: PyTorch `MonoidEmbed`

### Results (small, B=2, L=256, mean pool, normalize=1, exchange=1, second_act=0)
| Mode | Max abs err | Mean abs err | Cosine | Determinism max diff | Int8/scale match |
| --- | --- | --- | --- | --- | --- |
| float | 5.96e-08 | 1.18e-08 | 1.0 | 0.0 | n/a |
| quant | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| quant_int8 | 0.0 | 0.0 | 0.0 | 0.0 | yes |

## 2026-01-13: Phase 5 correctness grid (strict)

### Setup
- Env: `MONOID_CPU_FAST_MATH=0`, `MONOID_CPU_FAST_TANH=0`
- Float grid:
  - `uv run --python .venv/bin/python -- scripts/verify_kernel_correctness.py --presets small medium base --modes float --batch-sizes 1 2 8 64 256 --seq-lens 256 --pool-strategies mean last --normalize-output 1 0 --use-exchange 1 0 --use-second-activation 0 1 --determinism --determinism-runs 2 --determinism-tol 0 --continue-on-fail`
- Quant grid:
  - `uv run --python .venv/bin/python -- scripts/verify_kernel_correctness.py --presets small medium --modes quant quant_int8 --batch-sizes 1 2 8 64 256 --seq-lens 256 --pool-strategies mean last --normalize-output 1 0 --use-exchange 1 0 --use-second-activation 0 1 --determinism --determinism-runs 2 --determinism-tol 0 --continue-on-fail`

### Results
- Float: 240 cases, 0 failures.
- Quant: 160 cases, 0 failures.
- Note: quant tests skip `medium` because the quant kernel does not apply projection.

### Edge/saturation checks (quant, small)
- Command: `MONOID_CPU_FAST_MATH=0 MONOID_CPU_FAST_TANH=0 uv run --python .venv/bin/python -- scripts/verify_kernel_correctness.py --presets small --modes quant quant_int8 --batch-sizes 1 --seq-lens 256 --pool-strategies mean --normalize-output 1 --use-exchange 1 --use-second-activation 0 --determinism --determinism-runs 2 --determinism-tol 0 --edge-tests --continue-on-fail`
- Cases: `hi_gain` (activation_shift=0, activation_T_q15=32767, b_shift=15, inj_shift=0, second_act=1) and `hi_shift` (activation_shift=12, activation_T_q15=16384, b_shift=0, inj_shift=8, second_act=0)
- Result: 4/4 edge cases passed (quant + quant_int8).
| xlarge | 273.58 ms | 14.62 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.30 ms | 13315.25 KB/s |
| small_2l | 0.60 ms | 6697.49 KB/s |
| small_l2 | 0.62 ms | 6500.28 KB/s |
| small_l3 | 0.92 ms | 4341.93 KB/s |
| small_l4 | 1.20 ms | 3342.74 KB/s |
| small_l5 | 1.50 ms | 2674.09 KB/s |
| medium | 1.47 ms | 2723.13 KB/s |
| medium_deep | 3.21 ms | 1247.19 KB/s |
| base | 19.91 ms | 200.95 KB/s |
| large | 33.30 ms | 120.12 KB/s |
| xlarge | 276.20 ms | 14.48 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.36 ms | 11169.92 KB/s |
| small_2l | 0.72 ms | 5560.89 KB/s |
| small_l2 | 0.71 ms | 5637.51 KB/s |
| small_l3 | 1.17 ms | 3405.16 KB/s |
| small_l4 | 1.47 ms | 2716.08 KB/s |
| small_l5 | 1.95 ms | 2052.76 KB/s |
| medium | 1.81 ms | 2209.27 KB/s |
| medium_deep | 4.42 ms | 904.38 KB/s |
| base | 21.76 ms | 183.82 KB/s |
| large | 39.54 ms | 101.16 KB/s |
| xlarge | 331.89 ms | 12.05 KB/s |

After grain tuning (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18275.83 KB/s |
| small_2l | 0.42 ms | 9559.67 KB/s |
| small_l2 | 0.42 ms | 9521.69 KB/s |
| small_l3 | 0.63 ms | 6364.65 KB/s |
| small_l4 | 0.88 ms | 4556.55 KB/s |
| small_l5 | 1.08 ms | 3704.40 KB/s |
| medium | 1.18 ms | 3397.57 KB/s |
| medium_deep | 1.83 ms | 2180.84 KB/s |
| base | 16.78 ms | 238.32 KB/s |
| large | 29.73 ms | 134.54 KB/s |
| xlarge | 236.94 ms | 16.88 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14146.05 KB/s |
| small_2l | 0.55 ms | 7319.90 KB/s |
| small_l2 | 0.55 ms | 7288.10 KB/s |
| small_l3 | 0.82 ms | 4860.14 KB/s |
| small_l4 | 1.13 ms | 3552.99 KB/s |
| small_l5 | 1.40 ms | 2853.75 KB/s |
| medium | 1.44 ms | 2779.53 KB/s |
| medium_deep | 2.77 ms | 1442.08 KB/s |
| base | 18.37 ms | 217.79 KB/s |
| large | 32.68 ms | 122.39 KB/s |
| xlarge | 262.35 ms | 15.25 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.30 ms | 13262.62 KB/s |
| small_2l | 0.59 ms | 6803.41 KB/s |
| small_l2 | 0.59 ms | 6765.01 KB/s |
| small_l3 | 0.89 ms | 4479.90 KB/s |
| small_l4 | 1.22 ms | 3284.50 KB/s |
| small_l5 | 1.48 ms | 2700.78 KB/s |
| medium | 1.46 ms | 2736.01 KB/s |
| medium_deep | 3.63 ms | 1103.11 KB/s |
| base | 19.54 ms | 204.67 KB/s |
| large | 33.76 ms | 118.48 KB/s |
| xlarge | 276.01 ms | 14.49 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.36 ms | 11081.38 KB/s |
| small_2l | 0.74 ms | 5427.76 KB/s |
| small_l2 | 0.73 ms | 5506.14 KB/s |
| small_l3 | 1.07 ms | 3736.57 KB/s |
| small_l4 | 1.66 ms | 2415.73 KB/s |
| small_l5 | 1.80 ms | 2227.46 KB/s |
| medium | 1.78 ms | 2243.54 KB/s |
| medium_deep | 4.15 ms | 962.82 KB/s |
| base | 21.82 ms | 183.28 KB/s |
| large | 39.25 ms | 101.90 KB/s |
| xlarge | 440.94 ms | 9.07 KB/s |

After false sharing padding (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.23 ms | 17296.10 KB/s |
| small_2l | 0.43 ms | 9388.48 KB/s |
| small_l2 | 0.45 ms | 8867.45 KB/s |
| small_l3 | 0.68 ms | 5919.98 KB/s |
| small_l4 | 0.88 ms | 4534.38 KB/s |
| small_l5 | 1.11 ms | 3617.34 KB/s |
| medium | 1.36 ms | 2934.11 KB/s |
| medium_deep | 1.95 ms | 2046.75 KB/s |
| base | 16.44 ms | 243.37 KB/s |
| large | 30.61 ms | 130.67 KB/s |
| xlarge | 243.52 ms | 16.43 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14098.50 KB/s |
| small_2l | 0.59 ms | 6822.78 KB/s |
| small_l2 | 0.57 ms | 6984.69 KB/s |
| small_l3 | 0.88 ms | 4554.08 KB/s |
| small_l4 | 1.09 ms | 3678.41 KB/s |
| small_l5 | 1.43 ms | 2789.69 KB/s |
| medium | 1.39 ms | 2874.78 KB/s |
| medium_deep | 2.77 ms | 1445.69 KB/s |
| base | 19.91 ms | 200.95 KB/s |
| large | 32.97 ms | 121.34 KB/s |
| xlarge | 263.71 ms | 15.17 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.31 ms | 12797.27 KB/s |
| small_2l | 0.61 ms | 6561.29 KB/s |
| small_l2 | 0.59 ms | 6795.15 KB/s |
| small_l3 | 0.88 ms | 4523.38 KB/s |
| small_l4 | 1.19 ms | 3363.52 KB/s |
| small_l5 | 1.49 ms | 2676.65 KB/s |
| medium | 1.45 ms | 2765.32 KB/s |
| medium_deep | 3.13 ms | 1277.29 KB/s |
| base | 18.44 ms | 216.96 KB/s |
| large | 33.16 ms | 120.64 KB/s |
| xlarge | 277.22 ms | 14.43 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10951.19 KB/s |
| small_2l | 0.70 ms | 5698.78 KB/s |
| small_l2 | 0.75 ms | 5363.56 KB/s |
| small_l3 | 1.09 ms | 3661.55 KB/s |
| small_l4 | 1.62 ms | 2470.87 KB/s |
| small_l5 | 1.88 ms | 2132.06 KB/s |
| medium | 1.78 ms | 2243.24 KB/s |
| medium_deep | 4.60 ms | 870.37 KB/s |
| base | 22.41 ms | 178.49 KB/s |
| large | 41.85 ms | 95.59 KB/s |
| xlarge | 430.69 ms | 9.29 KB/s |

After guarded kernels (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18375.92 KB/s |
| small_2l | 0.43 ms | 9208.13 KB/s |
| small_l2 | 0.47 ms | 8559.80 KB/s |
| small_l3 | 0.64 ms | 6269.51 KB/s |
| small_l4 | 0.87 ms | 4596.50 KB/s |
| small_l5 | 1.16 ms | 3447.85 KB/s |
| medium | 1.25 ms | 3207.27 KB/s |
| medium_deep | 1.97 ms | 2034.84 KB/s |
| base | 17.26 ms | 231.76 KB/s |
| large | 31.10 ms | 128.61 KB/s |
| xlarge | 239.39 ms | 16.71 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.27 ms | 14847.09 KB/s |
| small_2l | 0.53 ms | 7543.71 KB/s |
| small_l2 | 0.52 ms | 7706.58 KB/s |
| small_l3 | 0.79 ms | 5065.58 KB/s |
| small_l4 | 1.05 ms | 3795.75 KB/s |
| small_l5 | 1.35 ms | 2963.13 KB/s |
| medium | 1.42 ms | 2808.84 KB/s |
| medium_deep | 2.65 ms | 1511.46 KB/s |
| base | 17.92 ms | 223.18 KB/s |
| large | 33.26 ms | 120.25 KB/s |
| xlarge | 267.64 ms | 14.95 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.30 ms | 13252.15 KB/s |
| small_2l | 0.58 ms | 6859.04 KB/s |
| small_l2 | 0.58 ms | 6850.64 KB/s |
| small_l3 | 0.88 ms | 4554.08 KB/s |
| small_l4 | 1.17 ms | 3430.92 KB/s |
| small_l5 | 1.45 ms | 2749.91 KB/s |
| medium | 1.56 ms | 2561.41 KB/s |
| medium_deep | 3.07 ms | 1301.37 KB/s |
| base | 18.81 ms | 212.63 KB/s |
| large | 33.84 ms | 118.21 KB/s |
| xlarge | 266.70 ms | 15.00 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10852.02 KB/s |
| small_2l | 0.71 ms | 5631.83 KB/s |
| small_l2 | 0.72 ms | 5592.41 KB/s |
| small_l3 | 1.07 ms | 3727.44 KB/s |
| small_l4 | 1.42 ms | 2818.28 KB/s |
| small_l5 | 1.80 ms | 2227.75 KB/s |
| medium | 1.78 ms | 2251.37 KB/s |
| medium_deep | 4.54 ms | 880.79 KB/s |
| base | 21.98 ms | 181.96 KB/s |
| large | 40.13 ms | 99.66 KB/s |
| xlarge | 328.66 ms | 12.17 KB/s |

After runtime dispatch (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18176.83 KB/s |
| small_2l | 0.42 ms | 9414.82 KB/s |
| small_l2 | 0.42 ms | 9462.61 KB/s |
| small_l3 | 0.67 ms | 6002.58 KB/s |
| small_l4 | 0.86 ms | 4635.87 KB/s |
| small_l5 | 1.08 ms | 3696.24 KB/s |
| medium | 1.24 ms | 3235.10 KB/s |
| medium_deep | 1.86 ms | 2148.45 KB/s |
| base | 16.64 ms | 240.39 KB/s |
| large | 29.76 ms | 134.40 KB/s |
| xlarge | 238.65 ms | 16.76 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14500.62 KB/s |
| small_2l | 0.55 ms | 7250.31 KB/s |
| small_l2 | 0.54 ms | 7433.41 KB/s |
| small_l3 | 0.83 ms | 4832.15 KB/s |
| small_l4 | 1.07 ms | 3724.96 KB/s |
| small_l5 | 1.39 ms | 2868.39 KB/s |
| medium | 1.47 ms | 2729.33 KB/s |
| medium_deep | 2.81 ms | 1424.57 KB/s |
| base | 18.37 ms | 217.79 KB/s |
| large | 33.79 ms | 118.37 KB/s |
| xlarge | 264.19 ms | 15.14 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.29 ms | 13763.10 KB/s |
| small_2l | 0.59 ms | 6743.25 KB/s |
| small_l2 | 0.58 ms | 6935.60 KB/s |
| small_l3 | 0.94 ms | 4234.53 KB/s |
| small_l4 | 1.30 ms | 3079.52 KB/s |
| small_l5 | 1.51 ms | 2642.08 KB/s |
| medium | 1.46 ms | 2748.11 KB/s |
| medium_deep | 3.09 ms | 1294.74 KB/s |
| base | 18.75 ms | 213.34 KB/s |
| large | 33.04 ms | 121.07 KB/s |
| xlarge | 276.91 ms | 14.45 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.36 ms | 11059.47 KB/s |
| small_2l | 0.82 ms | 4904.18 KB/s |
| small_l2 | 0.69 ms | 5755.48 KB/s |
| small_l3 | 1.06 ms | 3783.77 KB/s |
| small_l4 | 1.45 ms | 2760.32 KB/s |
| small_l5 | 1.93 ms | 2077.42 KB/s |
| medium | 1.81 ms | 2204.63 KB/s |
| medium_deep | 4.10 ms | 975.02 KB/s |
| base | 22.36 ms | 178.87 KB/s |
| large | 40.83 ms | 97.96 KB/s |
| xlarge | 381.44 ms | 10.49 KB/s |

After kernel registry dispatch (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.21 ms | 18787.48 KB/s |
| small_2l | 0.42 ms | 9570.57 KB/s |
| small_l2 | 0.41 ms | 9675.44 KB/s |
| small_l3 | 0.63 ms | 6396.19 KB/s |
| small_l4 | 0.92 ms | 4332.96 KB/s |
| small_l5 | 1.06 ms | 3774.40 KB/s |
| medium | 1.19 ms | 3354.10 KB/s |
| medium_deep | 1.85 ms | 2160.62 KB/s |
| base | 16.45 ms | 243.20 KB/s |
| large | 29.67 ms | 134.81 KB/s |
| xlarge | 234.32 ms | 17.07 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14376.36 KB/s |
| small_2l | 0.54 ms | 7390.84 KB/s |
| small_l2 | 0.54 ms | 7387.59 KB/s |
| small_l3 | 0.79 ms | 5059.47 KB/s |
| small_l4 | 1.06 ms | 3780.36 KB/s |
| small_l5 | 1.33 ms | 3011.53 KB/s |
| medium | 1.37 ms | 2909.68 KB/s |
| medium_deep | 2.74 ms | 1458.89 KB/s |
| base | 18.12 ms | 220.75 KB/s |
| large | 32.15 ms | 124.44 KB/s |
| xlarge | 256.92 ms | 15.57 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.29 ms | 13640.01 KB/s |
| small_2l | 0.60 ms | 6639.18 KB/s |
| small_l2 | 0.59 ms | 6792.40 KB/s |
| small_l3 | 0.91 ms | 4401.16 KB/s |
| small_l4 | 1.18 ms | 3390.71 KB/s |
| small_l5 | 1.58 ms | 2525.55 KB/s |
| medium | 1.53 ms | 2621.44 KB/s |
| medium_deep | 3.19 ms | 1253.43 KB/s |
| base | 19.23 ms | 208.00 KB/s |
| large | 34.45 ms | 116.10 KB/s |
| xlarge | 279.61 ms | 14.31 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10873.11 KB/s |
| small_2l | 0.70 ms | 5706.54 KB/s |
| small_l2 | 0.75 ms | 5314.29 KB/s |
| small_l3 | 1.13 ms | 3552.24 KB/s |
| small_l4 | 1.77 ms | 2261.99 KB/s |
| small_l5 | 1.80 ms | 2222.44 KB/s |
| medium | 2.04 ms | 1957.44 KB/s |
| medium_deep | 4.40 ms | 909.78 KB/s |
| base | 21.84 ms | 183.13 KB/s |
| large | 39.75 ms | 100.62 KB/s |
| xlarge | 317.07 ms | 12.62 KB/s |

After restrict + alignment hints (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18157.16 KB/s |
| small_2l | 0.44 ms | 9078.58 KB/s |
| small_l2 | 0.45 ms | 8830.11 KB/s |
| small_l3 | 0.71 ms | 5673.73 KB/s |
| small_l4 | 0.87 ms | 4601.54 KB/s |
| small_l5 | 1.11 ms | 3611.89 KB/s |
| medium | 1.23 ms | 3243.23 KB/s |
| medium_deep | 2.05 ms | 1953.11 KB/s |
| base | 16.86 ms | 237.26 KB/s |
| large | 30.17 ms | 132.58 KB/s |
| xlarge | 249.26 ms | 16.05 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14364.05 KB/s |
| small_2l | 0.56 ms | 7185.10 KB/s |
| small_l2 | 0.61 ms | 6597.41 KB/s |
| small_l3 | 0.81 ms | 4954.88 KB/s |
| small_l4 | 1.14 ms | 3497.44 KB/s |
| small_l5 | 1.42 ms | 2823.97 KB/s |
| medium | 1.50 ms | 2671.96 KB/s |
| medium_deep | 2.88 ms | 1391.26 KB/s |
| base | 18.58 ms | 215.32 KB/s |
| large | 33.70 ms | 118.69 KB/s |
| xlarge | 260.61 ms | 15.35 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.30 ms | 13464.86 KB/s |
| small_2l | 0.60 ms | 6628.69 KB/s |
| small_l2 | 0.57 ms | 7049.25 KB/s |
| small_l3 | 0.90 ms | 4438.42 KB/s |
| small_l4 | 1.14 ms | 3518.71 KB/s |
| small_l5 | 1.57 ms | 2547.41 KB/s |
| medium | 1.51 ms | 2647.92 KB/s |
| medium_deep | 3.04 ms | 1314.73 KB/s |
| base | 18.76 ms | 213.18 KB/s |
| large | 34.26 ms | 116.75 KB/s |
| xlarge | 274.10 ms | 14.59 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.35 ms | 11358.98 KB/s |
| small_2l | 0.77 ms | 5195.79 KB/s |
| small_l2 | 0.68 ms | 5901.24 KB/s |
| small_l3 | 1.05 ms | 3819.95 KB/s |
| small_l4 | 1.38 ms | 2892.12 KB/s |
| small_l5 | 1.76 ms | 2270.87 KB/s |
| medium | 1.76 ms | 2274.26 KB/s |
| medium_deep | 3.95 ms | 1011.71 KB/s |
| base | 22.66 ms | 176.51 KB/s |
| large | 41.76 ms | 95.80 KB/s |
| xlarge | 323.50 ms | 12.36 KB/s |

After SIMD recurrence (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.23 ms | 17494.49 KB/s |
| small_2l | 0.42 ms | 9500.12 KB/s |
| small_l2 | 0.43 ms | 9213.19 KB/s |
| small_l3 | 0.67 ms | 6006.88 KB/s |
| small_l4 | 0.85 ms | 4704.77 KB/s |
| small_l5 | 1.11 ms | 3601.03 KB/s |
| medium | 1.29 ms | 3094.86 KB/s |
| medium_deep | 1.92 ms | 2085.68 KB/s |
| base | 16.52 ms | 242.15 KB/s |
| large | 30.45 ms | 131.36 KB/s |
| xlarge | 243.77 ms | 16.41 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.27 ms | 14716.86 KB/s |
| small_2l | 0.53 ms | 7560.71 KB/s |
| small_l2 | 0.54 ms | 7440.01 KB/s |
| small_l3 | 0.79 ms | 5050.34 KB/s |
| small_l4 | 1.08 ms | 3711.77 KB/s |
| small_l5 | 1.34 ms | 2991.66 KB/s |
| medium | 1.39 ms | 2869.86 KB/s |
| medium_deep | 2.73 ms | 1467.18 KB/s |
| base | 18.02 ms | 222.04 KB/s |
| large | 32.45 ms | 123.25 KB/s |
| xlarge | 265.19 ms | 15.08 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.31 ms | 13066.37 KB/s |
| small_2l | 0.59 ms | 6748.68 KB/s |
| small_l2 | 0.58 ms | 6842.26 KB/s |
| small_l3 | 0.88 ms | 4524.60 KB/s |
| small_l4 | 1.17 ms | 3429.52 KB/s |
| small_l5 | 1.59 ms | 2513.06 KB/s |
| medium | 1.56 ms | 2568.47 KB/s |
| medium_deep | 3.12 ms | 1282.27 KB/s |
| base | 18.49 ms | 216.34 KB/s |
| large | 33.39 ms | 119.78 KB/s |
| xlarge | 309.17 ms | 12.94 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.35 ms | 11374.38 KB/s |
| small_2l | 0.72 ms | 5544.35 KB/s |
| small_l2 | 0.69 ms | 5801.25 KB/s |
| small_l3 | 1.04 ms | 3842.70 KB/s |
| small_l4 | 1.38 ms | 2897.12 KB/s |
| small_l5 | 1.74 ms | 2298.88 KB/s |
| medium | 1.71 ms | 2341.88 KB/s |
| medium_deep | 4.08 ms | 980.03 KB/s |
| base | 22.96 ms | 174.24 KB/s |
| large | 38.79 ms | 103.11 KB/s |
| xlarge | 339.93 ms | 11.77 KB/s |

After SIMD butterfly (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.21 ms | 18703.70 KB/s |
| small_2l | 0.42 ms | 9527.10 KB/s |
| small_l2 | 0.43 ms | 9409.54 KB/s |
| small_l3 | 0.65 ms | 6136.51 KB/s |
| small_l4 | 0.84 ms | 4754.10 KB/s |
| small_l5 | 1.07 ms | 3721.65 KB/s |
| medium | 1.23 ms | 3253.92 KB/s |
| medium_deep | 1.83 ms | 2187.67 KB/s |
| base | 17.13 ms | 233.51 KB/s |
| large | 30.00 ms | 133.35 KB/s |
| xlarge | 241.69 ms | 16.55 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.28 ms | 14364.05 KB/s |
| small_2l | 0.54 ms | 7466.50 KB/s |
| small_l2 | 0.55 ms | 7225.33 KB/s |
| small_l3 | 0.82 ms | 4885.62 KB/s |
| small_l4 | 1.12 ms | 3556.76 KB/s |
| small_l5 | 1.47 ms | 2714.32 KB/s |
| medium | 1.44 ms | 2771.72 KB/s |
| medium_deep | 2.84 ms | 1408.19 KB/s |
| base | 18.51 ms | 216.09 KB/s |
| large | 32.56 ms | 122.84 KB/s |
| xlarge | 263.23 ms | 15.20 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.32 ms | 12671.61 KB/s |
| small_2l | 0.64 ms | 6234.57 KB/s |
| small_l2 | 0.62 ms | 6423.13 KB/s |
| small_l3 | 0.92 ms | 4329.60 KB/s |
| small_l4 | 1.26 ms | 3175.70 KB/s |
| small_l5 | 1.62 ms | 2461.81 KB/s |
| medium | 1.56 ms | 2559.84 KB/s |
| medium_deep | 3.27 ms | 1223.99 KB/s |
| base | 19.35 ms | 206.73 KB/s |
| large | 35.31 ms | 113.28 KB/s |
| xlarge | 279.66 ms | 14.30 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10824.01 KB/s |
| small_2l | 0.75 ms | 5365.28 KB/s |
| small_l2 | 0.71 ms | 5669.89 KB/s |
| small_l3 | 1.08 ms | 3706.85 KB/s |
| small_l4 | 1.51 ms | 2651.69 KB/s |
| small_l5 | 1.91 ms | 2091.40 KB/s |
| medium | 1.85 ms | 2164.80 KB/s |
| medium_deep | 4.69 ms | 853.67 KB/s |
| base | 22.95 ms | 174.32 KB/s |
| large | 40.69 ms | 98.30 KB/s |
| xlarge | 328.82 ms | 12.16 KB/s |

After fast tanh (B=1, L=4096, full precision, `run_inference_benchmark_all_presets.py`, `MONOID_CPU_FAST_TANH=1`):

Single core (threads=1):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.22 ms | 18236.10 KB/s |
| small_2l | 0.42 ms | 9457.28 KB/s |
| small_l2 | 0.47 ms | 8542.37 KB/s |
| small_l3 | 0.63 ms | 6321.48 KB/s |
| small_l4 | 0.85 ms | 4692.93 KB/s |
| small_l5 | 1.06 ms | 3771.01 KB/s |
| medium | 1.21 ms | 3311.07 KB/s |
| medium_deep | 1.84 ms | 2172.09 KB/s |
| base | 16.75 ms | 238.84 KB/s |
| large | 29.83 ms | 134.08 KB/s |
| xlarge | 237.84 ms | 16.82 KB/s |

Multi-core (threads=8):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.27 ms | 14601.58 KB/s |
| small_2l | 0.53 ms | 7506.58 KB/s |
| small_l2 | 0.57 ms | 6964.39 KB/s |
| small_l3 | 0.82 ms | 4857.33 KB/s |
| small_l4 | 1.14 ms | 3518.71 KB/s |
| small_l5 | 1.35 ms | 2963.13 KB/s |
| medium | 1.43 ms | 2793.41 KB/s |
| medium_deep | 2.96 ms | 1353.33 KB/s |
| base | 18.10 ms | 221.03 KB/s |
| large | 32.58 ms | 122.76 KB/s |
| xlarge | 264.11 ms | 15.15 KB/s |

Multi-core (threads=16):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.29 ms | 13617.87 KB/s |
| small_2l | 0.58 ms | 6842.26 KB/s |
| small_l2 | 0.66 ms | 6024.14 KB/s |
| small_l3 | 1.01 ms | 3976.59 KB/s |
| small_l4 | 1.18 ms | 3386.60 KB/s |
| small_l5 | 1.47 ms | 2713.00 KB/s |
| medium | 1.52 ms | 2628.42 KB/s |
| medium_deep | 3.14 ms | 1273.99 KB/s |
| base | 19.19 ms | 208.46 KB/s |
| large | 37.58 ms | 106.43 KB/s |
| xlarge | 280.39 ms | 14.27 KB/s |

Multi-core (threads=32):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10887.23 KB/s |
| small_2l | 0.71 ms | 5633.72 KB/s |
| small_l2 | 0.73 ms | 5513.38 KB/s |
| small_l3 | 1.11 ms | 3604.90 KB/s |
| small_l4 | 1.55 ms | 2579.52 KB/s |
| small_l5 | 1.87 ms | 2133.69 KB/s |
| medium | 1.92 ms | 2081.02 KB/s |
| medium_deep | 4.14 ms | 965.48 KB/s |
| base | 22.76 ms | 175.72 KB/s |
| large | 40.92 ms | 97.75 KB/s |
| xlarge | 332.32 ms | 12.04 KB/s |

### Core-count ablation (B=64, L=4096, full precision)

| Threads | small (MB/s) | small_l2 (MB/s) | medium (MB/s) | base (MB/s) |
| --- | --- | --- | --- | --- |
| 1 | 19.04 | 9.41 | 3.58 | 0.23 |
| 8 | 88.38 | 62.67 | 28.01 | 1.58 |
| 16 | 278.73 | 143.73 | 52.24 | 3.33 |
| 24 | 199.84 | 105.79 | 33.32 | 2.41 |
| 32 | 309.94 | 156.40 | 48.99 | 3.31 |

Per-thread raw timings:
| Threads | small (ms) | small_l2 (ms) | medium (ms) | base (ms) |
| --- | --- | --- | --- | --- |
| 1 | 13.13 | 26.56 | 69.74 | 1098.99 |
| 8 | 2.83 | 3.99 | 8.93 | 158.06 |
| 16 | 0.90 | 1.74 | 4.79 | 75.11 |
| 24 | 1.25 | 2.36 | 7.50 | 103.55 |
| 32 | 0.81 | 1.60 | 5.10 | 75.57 |
Note: core-count ablation uses `torch.set_num_interop_threads(1)` to allow varying intra-op thread counts.

### Pinned (taskset, NUMA0)

Full precision, `MONOID_CPU_FAST_TANH=0`, `MKL_NUM_THREADS=1`:

B=1, L=4096 (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.26 ms | 15278.57 KB/s |
| small_2l | 0.50 ms | 7951.61 KB/s |
| small_l2 | 0.51 ms | 7900.21 KB/s |
| small_l3 | 0.76 ms | 5287.15 KB/s |
| small_l4 | 1.03 ms | 3889.56 KB/s |
| small_l5 | 1.26 ms | 3166.43 KB/s |
| medium | 1.25 ms | 3212.45 KB/s |
| medium_deep | 2.75 ms | 1454.85 KB/s |
| base | 13.62 ms | 293.68 KB/s |
| large | 24.96 ms | 160.28 KB/s |
| xlarge | 202.96 ms | 19.71 KB/s |

B=1, L=4096 (pinned `taskset -c 0-31`, `OMP_NUM_THREADS=32`):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.37 ms | 10701.41 KB/s |
| small_2l | 0.64 ms | 6212.30 KB/s |
| small_l2 | 1.05 ms | 3802.39 KB/s |
| small_l3 | 0.95 ms | 4220.97 KB/s |
| small_l4 | 1.24 ms | 3216.43 KB/s |
| small_l5 | 1.56 ms | 2569.87 KB/s |
| medium | 1.58 ms | 2535.14 KB/s |
| medium_deep | 3.68 ms | 1086.76 KB/s |
| base | 16.05 ms | 249.16 KB/s |
| large | 30.38 ms | 131.67 KB/s |
| xlarge | 256.14 ms | 15.62 KB/s |

B=64, L=4096 (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.74 ms | 336.90 MB/s |
| small_l2 | 1.49 ms | 168.36 MB/s |
| medium | 3.86 ms | 64.78 MB/s |
| base | 48.77 ms | 5.13 MB/s |

B=64, L=4096 (pinned `taskset -c 0-31`, `OMP_NUM_THREADS=32`):
| Preset | Elapsed | Throughput |
| --- | --- | --- |
| small | 0.75 ms | 332.90 MB/s |
| small_l2 | 1.40 ms | 178.50 MB/s |
| medium | 4.56 ms | 54.87 MB/s |
| base | 56.55 ms | 4.42 MB/s |

Batch size sweep (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, L=4096, presets `small`, `small_l2`, `medium`, `base`):

Latency (avg ms):
| B | small | small_l2 | medium | base |
| --- | --- | --- | --- | --- |
| 1 | 0.26 | 0.50 | 1.25 | 13.87 |
| 2 | 0.34 | 0.75 | 1.80 | 23.16 |
| 4 | 0.68 | 1.34 | 3.62 | 46.79 |
| 8 | 1.33 | 2.61 | 6.96 | 99.21 |
| 16 | 2.66 | 5.22 | 14.22 | 190.40 |
| 32 | 0.90 | 1.68 | 3.37 | 24.27 |
| 64 | 1.72 | 3.38 | 3.86 | 48.94 |
| 128 | 2.37 | 3.34 | 7.57 | 100.17 |
| 256 | 6.08 | 5.96 | 15.58 | 203.66 |

Throughput (MB/s):
| B | small | small_l2 | medium | base |
| --- | --- | --- | --- | --- |
| 1 | 15.18 | 7.76 | 3.11 | 0.28 |
| 2 | 22.83 | 10.46 | 4.33 | 0.34 |
| 4 | 22.87 | 11.66 | 4.32 | 0.33 |
| 8 | 23.57 | 11.99 | 4.49 | 0.31 |
| 16 | 23.47 | 11.96 | 4.39 | 0.33 |
| 32 | 138.15 | 74.43 | 37.12 | 5.15 |
| 64 | 145.40 | 73.97 | 64.77 | 5.11 |
| 128 | 211.15 | 149.85 | 66.07 | 4.99 |
| 256 | 164.34 | 167.91 | 64.19 | 4.91 |

Length + chunking sweep (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, preset `small_l2`, doc_bytes=52228, throughput is effective doc bytes):

Chunk aggregate: mean

Stride = L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 817 | 20.00 | 2.49 |
| 128 | 409 | 12.84 | 3.88 |
| 256 | 205 | 7.85 | 6.34 |
| 512 | 103 | 7.36 | 6.77 |
| 1024 | 52 | 6.66 | 7.48 |
| 2048 | 26 | 6.46 | 7.71 |
| 4096 | 13 | 6.32 | 7.88 |
| 8192 | 7 | 6.78 | 7.34 |
| 16384 | 4 | 6.12 | 8.14 |

Stride = 3/4 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1088 | 26.26 | 1.90 |
| 128 | 544 | 16.52 | 3.02 |
| 256 | 272 | 10.64 | 4.68 |
| 512 | 136 | 9.68 | 5.15 |
| 1024 | 68 | 9.00 | 5.53 |
| 2048 | 34 | 8.82 | 5.65 |
| 4096 | 17 | 8.47 | 5.88 |
| 8192 | 9 | 8.17 | 6.10 |
| 16384 | 4 | 7.95 | 6.26 |

Stride = 1/2 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1632 | 40.57 | 1.23 |
| 128 | 816 | 24.51 | 2.03 |
| 256 | 408 | 16.04 | 3.11 |
| 512 | 204 | 14.81 | 3.36 |
| 1024 | 102 | 13.59 | 3.67 |
| 2048 | 51 | 13.03 | 3.82 |
| 4096 | 25 | 12.78 | 3.90 |
| 8192 | 12 | 11.72 | 4.25 |
| 16384 | 6 | 11.07 | 4.50 |

Chunk aggregate: max

Stride = L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 817 | 20.21 | 2.46 |
| 128 | 409 | 11.96 | 4.17 |
| 256 | 205 | 8.35 | 5.96 |
| 512 | 103 | 7.23 | 6.89 |
| 1024 | 52 | 7.05 | 7.07 |
| 2048 | 26 | 6.65 | 7.49 |
| 4096 | 13 | 6.25 | 7.96 |
| 8192 | 7 | 6.39 | 7.79 |
| 16384 | 4 | 6.57 | 7.58 |

Stride = 3/4 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1088 | 26.29 | 1.89 |
| 128 | 544 | 15.84 | 3.14 |
| 256 | 272 | 10.96 | 4.54 |
| 512 | 136 | 9.72 | 5.12 |
| 1024 | 68 | 9.07 | 5.49 |
| 2048 | 34 | 8.71 | 5.72 |
| 4096 | 17 | 8.56 | 5.82 |
| 8192 | 9 | 8.36 | 5.96 |
| 16384 | 4 | 7.81 | 6.38 |

Stride = 1/2 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1632 | 39.06 | 1.28 |
| 128 | 816 | 23.62 | 2.11 |
| 256 | 408 | 16.11 | 3.09 |
| 512 | 204 | 14.54 | 3.43 |
| 1024 | 102 | 13.93 | 3.57 |
| 2048 | 51 | 13.02 | 3.82 |
| 4096 | 25 | 12.91 | 3.86 |
| 8192 | 12 | 11.47 | 4.34 |
| 16384 | 6 | 11.69 | 4.26 |

Chunk aggregate: last

Stride = L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 817 | 18.98 | 2.62 |
| 128 | 409 | 11.30 | 4.41 |
| 256 | 205 | 7.75 | 6.43 |
| 512 | 103 | 6.98 | 7.13 |
| 1024 | 52 | 6.79 | 7.34 |
| 2048 | 26 | 6.72 | 7.42 |
| 4096 | 13 | 6.30 | 7.91 |
| 8192 | 7 | 6.54 | 7.61 |
| 16384 | 4 | 6.40 | 7.78 |

Stride = 3/4 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1088 | 25.25 | 1.97 |
| 128 | 544 | 16.02 | 3.11 |
| 256 | 272 | 10.60 | 4.70 |
| 512 | 136 | 9.64 | 5.16 |
| 1024 | 68 | 8.99 | 5.54 |
| 2048 | 34 | 8.66 | 5.75 |
| 4096 | 17 | 8.50 | 5.86 |
| 8192 | 9 | 8.76 | 5.68 |
| 16384 | 4 | 7.73 | 6.45 |

Stride = 1/2 L:
| L | chunks | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- |
| 64 | 1632 | 37.26 | 1.34 |
| 128 | 816 | 22.81 | 2.18 |
| 256 | 408 | 15.72 | 3.17 |
| 512 | 204 | 14.20 | 3.51 |
| 1024 | 102 | 13.15 | 3.79 |
| 2048 | 51 | 12.86 | 3.87 |
| 4096 | 25 | 12.41 | 4.01 |
| 8192 | 12 | 11.69 | 4.26 |
| 16384 | 6 | 12.32 | 4.04 |

Input distribution sweep (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=1, L=4096, preset `small_l2`, inputs in `tmp/bench_inputs`):

| Input | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| ASCII English | 0.50 | 7.74 |
| Random bytes | 0.59 | 6.60 |
| SciFact-style abstract | 0.51 | 7.67 |
| Token-dense | 0.54 | 7.23 |
| UTF-8 heavy | 0.49 | 7.94 |
| Whitespace-dense | 0.45 | 8.76 |

Input distribution sweep (quant, pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=1, L=4096, preset `small`):

| Input | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| ASCII English | 0.23 | 16.72 |
| Random bytes | 0.25 | 15.69 |
| SciFact-style abstract | 0.23 | 16.67 |
| Token-dense | 0.23 | 16.64 |
| UTF-8 heavy | 0.25 | 15.87 |
| Whitespace-dense | 0.24 | 16.52 |

Input distribution sweep (quant_int8, pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=1, L=4096, preset `small`):

| Input | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| ASCII English | 0.24 | 16.27 |
| Random bytes | 0.24 | 16.24 |
| SciFact-style abstract | 0.23 | 17.20 |
| Token-dense | 0.23 | 16.72 |
| UTF-8 heavy | 0.24 | 16.32 |
| Whitespace-dense | 0.24 | 16.54 |

Threading + pinning matrix (B=64, L=4096, preset `small_l2`):

Unpinned (OS scheduling):
| Threads | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| 16 | 2.61 | 95.79 |
| 32 | 2.52 | 99.27 |

Pinned cores `0-15` (`taskset -c 0-15`):
| Threads | Avg (ms) | Throughput (MB/s) | Efficiency |
| --- | --- | --- | --- |
| 1 | 21.60 | 11.57 | 1.000 |
| 2 | 10.49 | 23.83 | 1.030 |
| 4 | 7.16 | 34.94 | 0.755 |
| 8 | 6.13 | 40.76 | 0.440 |
| 16 | 2.51 | 99.71 | 0.539 |

Pinned cores `0-31` (`taskset -c 0-31`):
| Threads | Avg (ms) | Throughput (MB/s) | Efficiency |
| --- | --- | --- | --- |
| 1 | 20.87 | 11.98 | 1.000 |
| 2 | 10.95 | 22.82 | 0.953 |
| 4 | 5.43 | 46.01 | 0.960 |
| 8 | 4.87 | 51.28 | 0.535 |
| 16 | 2.58 | 96.79 | 0.505 |
| 24 | 3.36 | 74.50 | 0.259 |
| 32 | 2.53 | 98.65 | 0.257 |

Kernel mode sweep (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, preset `small`):

B=1, L=4096:
| Mode | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| float | 0.26 | 15.12 |
| quant | 0.24 | 16.35 |
| quant_int8 | 0.24 | 16.23 |

B=64, L=4096:
| Mode | Avg (ms) | Throughput (MB/s) |
| --- | --- | --- |
| float | 1.75 | 142.65 |
| quant | 1.60 | 156.65 |
| quant_int8 | 1.64 | 152.45 |

Model shape sweep (pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=64, L=4096, exchange_dim fixed to d_state/16; d_state=768 skipped because tile_dim must be power-of-two):

| layers | d_state | microblock | avg (ms) | throughput (MB/s) |
| --- | --- | --- | --- | --- |
| 1 | 256 | 64 | 0.86 | 289.84 |
| 2 | 256 | 64 | 0.88 | 284.58 |
| 3 | 256 | 64 | 1.32 | 189.31 |
| 4 | 256 | 64 | 1.69 | 148.18 |
| 5 | 256 | 64 | 2.08 | 120.47 |
| 1 | 256 | 128 | 0.76 | 330.68 |
| 2 | 256 | 128 | 0.77 | 323.26 |
| 3 | 256 | 128 | 1.16 | 215.57 |
| 4 | 256 | 128 | 1.51 | 165.97 |
| 5 | 256 | 128 | 1.87 | 133.78 |
| 1 | 256 | 256 | 0.72 | 345.55 |
| 2 | 256 | 256 | 0.75 | 334.15 |
| 3 | 256 | 256 | 1.06 | 236.58 |
| 4 | 256 | 256 | 1.40 | 178.63 |
| 5 | 256 | 256 | 1.84 | 135.66 |
| 1 | 256 | 512 | 0.61 | 407.30 |
| 2 | 256 | 512 | 0.73 | 344.56 |
| 3 | 256 | 512 | 1.00 | 248.98 |
| 4 | 256 | 512 | 1.41 | 177.46 |
| 5 | 256 | 512 | 1.73 | 144.16 |
| 1 | 512 | 64 | 0.87 | 286.76 |
| 2 | 512 | 64 | 1.84 | 135.61 |
| 3 | 512 | 64 | 2.59 | 96.45 |
| 4 | 512 | 64 | 3.75 | 66.65 |
| 5 | 512 | 64 | 4.17 | 59.99 |
| 1 | 512 | 128 | 0.76 | 327.51 |
| 2 | 512 | 128 | 1.45 | 172.05 |
| 3 | 512 | 128 | 2.25 | 111.02 |
| 4 | 512 | 128 | 2.93 | 85.35 |
| 5 | 512 | 128 | 3.67 | 68.15 |
| 1 | 512 | 256 | 0.72 | 346.36 |
| 2 | 512 | 256 | 1.39 | 179.34 |
| 3 | 512 | 256 | 2.10 | 119.08 |
| 4 | 512 | 256 | 2.78 | 89.99 |
| 5 | 512 | 256 | 3.88 | 64.38 |
| 1 | 512 | 512 | 0.72 | 348.32 |
| 2 | 512 | 512 | 1.44 | 174.06 |
| 3 | 512 | 512 | 1.99 | 125.53 |
| 4 | 512 | 512 | 2.66 | 94.16 |
| 5 | 512 | 512 | 3.46 | 72.36 |
| 1 | 1024 | 64 | 2.13 | 117.38 |
| 2 | 1024 | 64 | 3.71 | 67.37 |
| 3 | 1024 | 64 | 5.94 | 42.12 |
| 4 | 1024 | 64 | 7.06 | 35.41 |
| 5 | 1024 | 64 | 8.59 | 29.11 |
| 1 | 1024 | 128 | 1.74 | 143.91 |
| 2 | 1024 | 128 | 3.22 | 77.68 |
| 3 | 1024 | 128 | 4.84 | 51.63 |
| 4 | 1024 | 128 | 6.19 | 40.42 |
| 5 | 1024 | 128 | 7.74 | 32.31 |
| 1 | 1024 | 256 | 1.64 | 152.70 |
| 2 | 1024 | 256 | 3.14 | 79.63 |
| 3 | 1024 | 256 | 4.73 | 52.84 |
| 4 | 1024 | 256 | 6.19 | 40.40 |
| 5 | 1024 | 256 | 7.13 | 35.08 |
| 1 | 1024 | 512 | 1.57 | 158.98 |
| 2 | 1024 | 512 | 2.89 | 86.49 |
| 3 | 1024 | 512 | 4.31 | 58.03 |
| 4 | 1024 | 512 | 5.64 | 44.34 |
| 5 | 1024 | 512 | 6.83 | 36.58 |

Perf profile (cycles, pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=64, L=4096, preset `small_l2`, fast_math=0, `MONOID_CPU_DEBUG=1`, `perf record -F 199 -g`, `repeat=2000`):

| Symbol | Overhead | Notes |
| --- | --- | --- |
| (anonymous namespace)::recurrence_f32_avx512 | 37.23% | `monoid_cpu_ext_dbg.so` |
| expm1f32 | 13.10% | `libm.so.6` (tanh path) |
| tanhf32 | 11.16% | `libm.so.6` (tanh path) |
| monoid_forward_float_stacked_impl<...>::operator() | 2.31% | `monoid_cpu_ext_dbg.so` |
| (anonymous namespace)::butterfly_mix_float_interleaved_stages | 1.27% | `monoid_cpu_ext_dbg.so` |
| (anonymous namespace)::dot_f32_avx512 | 0.67% | `monoid_cpu_ext_dbg.so` |
| blas_thread_server | 0.97% | `libscipy_openblas64_-56d6093b.so` |

Perf profile (cycles, pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, B=64, L=4096, preset `small_l2`, fast_math=1, `MONOID_CPU_DEBUG=1`, `perf record -F 199 -g`, `repeat=20000`):

| Symbol | Overhead | Notes |
| --- | --- | --- |
| (anonymous namespace)::recurrence_f32_avx512 | 75.34% | `monoid_cpu_ext_fm_ft_dbg.so` |
| monoid_forward_float_stacked_impl<...>::operator() | 5.47% | `monoid_cpu_ext_fm_ft_dbg.so` |
| (anonymous namespace)::butterfly_mix_float_interleaved_stages | 3.01% | `monoid_cpu_ext_fm_ft_dbg.so` |
| (anonymous namespace)::dot_f32_avx512 | 1.24% | `monoid_cpu_ext_fm_ft_dbg.so` |
| blas_thread_server | 0.16% | `libscipy_openblas64_-56d6093b.so` |

Memory behavior (RSS deltas, pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, L=4096, cache evict 256MB between cold/warm):

| B | Preset | RSS delta (KB) | Warm RSS spread (KB) | Cold (ms) | Warm avg (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | base | 0 | 0 | 14.32 | 13.98 |
| 1 | medium | 0 | 0 | 1.68 | 1.31 |
| 1 | small | 0 | 0 | 0.41 | 0.26 |
| 1 | small_l2 | 0 | 0 | 0.67 | 0.51 |
| 64 | base | 0 | 0 | 48.37 | 47.83 |
| 64 | medium | 0 | 0 | 4.72 | 3.88 |
| 64 | small | 0 | 0 | 2.89 | 1.17 |
| 64 | small_l2 | 0 | 0 | 2.77 | 2.45 |

### Toggles

- `fast_math`: enable fast tanh by passing `MonoidCpuConfig(fast_math=True)` or setting `MONOID_CPU_FAST_TANH=1` before loading the extension.
- `threads`: set `MonoidCpuConfig(threads=N)` or `MONOID_CPU_THREADS=N` (applied via `torch.set_num_threads` at kernel init).

### Quantized CPU kernel (int16->float, Track A behavior-preserving)

Quantized kernel supports single-layer presets only (`small`, `medium`).

B=1, L=4096 (`MonoidCpuKernel.forward`):
| Threads | small (ms) | small (KB/s) | medium (ms) | medium (KB/s) |
| --- | --- | --- | --- | --- |
| 1 | 0.16 | 24636.15 | 0.94 | 4236.24 |
| 8 | 0.22 | 17806.43 | 1.13 | 3529.07 |
| 16 | 0.24 | 16898.89 | 1.21 | 3312.64 |
| 32 | 0.32 | 12411.02 | 1.68 | 2382.18 |

B=64, L=4096:
| Threads | small (MB/s) | medium (MB/s) |
| --- | --- | --- |
| 1 | 24.96 | 4.16 |
| 8 | 117.13 | 32.29 |
| 16 | 365.05 | 63.91 |
| 24 | 261.52 | 45.98 |
| 32 | 365.18 | 64.95 |

Per-thread raw timings:
| Threads | small (ms) | medium (ms) |
| --- | --- | --- |
| 1 | 10.02 | 60.09 |
| 8 | 2.13 | 7.74 |
| 16 | 0.68 | 3.91 |
| 24 | 0.96 | 5.44 |
| 32 | 0.68 | 3.85 |

### Quantized CPU kernel (int8 output, Track A behavior-preserving)

B=1, L=4096 (`MonoidCpuKernel.forward` with `emit_int8=True`):
| Threads | small (ms) | small (KB/s) | medium (ms) | medium (KB/s) |
| --- | --- | --- | --- | --- |
| 1 | 0.17 | 24153.78 | 0.95 | 4192.63 |
| 8 | 0.22 | 18067.22 | 1.14 | 3496.27 |
| 16 | 0.24 | 16627.57 | 1.23 | 3252.66 |
| 32 | 0.31 | 12750.58 | 1.66 | 2408.79 |

B=64, L=4096:
| Threads | small (MB/s) | medium (MB/s) |
| --- | --- | --- |
| 1 | 24.74 | 4.11 |
| 8 | 110.76 | 31.65 |
| 16 | 342.63 | 60.52 |
| 24 | 256.46 | 44.48 |
| 32 | 349.08 | 61.92 |

Per-thread raw timings:
| Threads | small (ms) | medium (ms) |
| --- | --- | --- |
| 1 | 10.11 | 60.89 |
| 8 | 2.26 | 7.90 |
| 16 | 0.73 | 4.13 |
| 24 | 0.97 | 5.62 |
| 32 | 0.72 | 4.04 |

### Quant activation bottleneck check

Test: fixed L=4096, B=64, threads=16, exchange off, compare microblock sizes (MB=64 vs MB=512) to change block count without changing token count. Pinned `taskset -c 0-15`, `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=1`.

| Mode | microblock | Avg (ms) | p50 (ms) | Embeddings/s |
| --- | --- | --- | --- | --- |
| quant | 64 | 4.94 | 1.43 | 12959.26 |
| quant | 512 | 1.12 | 1.03 | 57173.72 |
| float | 64 | 11.38 | 9.30 | 5622.21 |
| float | 512 | 0.74 | 0.67 | 86905.65 |

Derived per-block overhead (p50) for additional 56 blocks (4096/64 - 4096/512):
- quant: +0.41 ms total → ~7.24 us per extra block.
- float: +8.62 ms total → ~153.99 us per extra block.

Conclusion: this experiment does not support the claim that the quantized activation loop is the dominant bottleneck; the per-block overhead is higher in the float path here. This test captures combined per-block costs (butterfly + activation), not the activation loop in isolation.

### Kitchen Sink Matrix

Pinned runs: `taskset -c 0-15`, `OMP_NUM_THREADS` = threads, `MKL_NUM_THREADS=1`, `MONOID_CPU_THREADS` = threads.

| Input | Mode | Batch | Threads | small emb/s | small_l2 emb/s |
| --- | --- | --- | --- | --- | --- |
| ascii_english_4096 | float | 1 | 1 | 5729.50 | 2997.81 |
| ascii_english_4096 | float | 1 | 16 | 3704.02 | 2034.23 |
| ascii_english_4096 | float | 64 | 1 | 6151.60 | 3092.46 |
| ascii_english_4096 | float | 64 | 16 | 38600.35 | 18985.23 |
| ascii_english_4096 | quant | 1 | 1 | 6089.78 | - |
| ascii_english_4096 | quant | 1 | 16 | 4107.13 | - |
| ascii_english_4096 | quant | 64 | 1 | 6486.78 | - |
| ascii_english_4096 | quant | 64 | 16 | 39531.05 | - |
| ascii_english_4096 | quant_int8 | 1 | 1 | 6099.53 | - |
| ascii_english_4096 | quant_int8 | 1 | 16 | 4252.55 | - |
| ascii_english_4096 | quant_int8 | 64 | 1 | 6419.50 | - |
| ascii_english_4096 | quant_int8 | 64 | 16 | 35260.06 | - |
| utf8_heavy_4096 | float | 1 | 1 | 5657.56 | 3063.40 |
| utf8_heavy_4096 | float | 1 | 16 | 3532.05 | 2011.10 |
| utf8_heavy_4096 | float | 64 | 1 | 6160.94 | 3167.25 |
| utf8_heavy_4096 | float | 64 | 16 | 52358.68 | 27501.49 |
| utf8_heavy_4096 | quant | 1 | 1 | 6125.78 | - |
| utf8_heavy_4096 | quant | 1 | 16 | 4142.09 | - |
| utf8_heavy_4096 | quant | 64 | 1 | 6489.61 | - |
| utf8_heavy_4096 | quant | 64 | 16 | 44942.36 | - |
| utf8_heavy_4096 | quant_int8 | 1 | 1 | 6140.25 | - |
| utf8_heavy_4096 | quant_int8 | 1 | 16 | 4054.46 | - |
| utf8_heavy_4096 | quant_int8 | 64 | 1 | 6466.68 | - |
| utf8_heavy_4096 | quant_int8 | 64 | 16 | 49617.87 | - |

Best emb/s in this grid:
- ascii_english_4096: float small 38600.35 (B=64, T=16), float small_l2 18985.23 (B=64, T=16), quant small 39531.05 (B=64, T=16), quant_int8 small 35260.06 (B=64, T=16).
- utf8_heavy_4096: float small 52358.68 (B=64, T=16), float small_l2 27501.49 (B=64, T=16), quant small 44942.36 (B=64, T=16), quant_int8 small 49617.87 (B=64, T=16).
- Quant modes skip `small_l2` because quantized kernels require `n_layers == 1`.

### Correctness
- `scripts/verify_kernel_correctness.py` passes for presets: `small`, `base`.
- Quant + edge tests: `scripts/verify_kernel_correctness.py --presets small --batch-sizes 2 --seq-lens 128 --modes quant --pool-strategies mean --normalize-output 1 --use-exchange 1 --use-second-activation 0 --edge-tests` (all cases pass).
- Phase 14 refactor sanity: `scripts/verify_kernel_correctness.py --presets small --batch-sizes 2 --seq-lens 128 --modes quant --pool-strategies mean --normalize-output 1 --use-exchange 1 --use-second-activation 0` (max_abs_err 0.0).
- Phase 15 stacked quant sanity: `monoid_forward_quantized_stacked` with `n_layers=1` matches `monoid_forward_quantized` (max_abs_err 0.0 on preset `small`, B=2, L=128; see `tmp/stacked_quant_test.pt`).
- Phase 16 per-layer exchange sanity: stacked quant with 2 layers and per-layer int8 exchange weights produces different output when weights are swapped (`max_abs_diff_swap` 0.16468 with random int8 weights).
- Phase 16B LayerNorm sanity: stacked quant with `n_layers=2`, `a_q15=1`, `b_int8=0`, `L=1` shows non-zero delta when swapping per-layer LN weights/bias (`max_abs_diff_ln_swap` 0.399).
- Phase 16C projection sanity: stacked quant output changes when swapping projection matrices (`max_abs_diff_proj_swap` 0.542 on random weights, out_dim=32, d_state=128).
- Phase 17 wrapper sanity: `MonoidCpuKernel.forward` auto-routes to stacked quant for `small_l2` and returns `(B, d_state)` without errors (B=2, L=64).
- Phase 17 stacked int8 sanity: `MonoidCpuKernel.forward` with `emit_int8=True` on `small_l2` returns `(float, int8, scale)` with shapes `(2, 512)`, `(2, 512)`, `(2,)`.
- Phase 18 stacked int8 harness: `scripts/verify_stacked_int8_sanity.py --preset small_l2 --batch-size 2 --seq-len 64` reports float_match max_abs_err 0.0, dequant max_abs_err 0.0, float_vs_quant max_abs_err 0.0477.
- Phase 18 regression (n_layers=1): `scripts/verify_stacked_int8_sanity.py --preset small --batch-size 2 --seq-len 64` reports stacked_vs_single max_abs_err 0.0 and float_vs_quant max_abs_err 0.1490.
- Phase 18 shape checks: added `MonoidCpuKernel` validation for quant tensor shapes (stacked vs single, exchange, LN, proj).
- Phase 18 vectorization pass: added SIMD saturating residual add + int16→int32 widening, and skipped temp allocation when `do_pool=false` (rechecked with `verify_stacked_int8_sanity.py --preset small_l2 --batch-size 2 --seq-len 64`).
- Phase 18 parallelization tweak: stacked quant now uses one batch-parallel region per layer (init + run + accumulate), avoiding extra `at::parallel_for` overhead; rechecked with `verify_stacked_int8_sanity.py --preset small_l2 --batch-size 2 --seq-len 64`.
- Phase 18 perf spotcheck (stacked quant): `run_embed_bench.py` on `small_l2` (ascii_english_4096, B=64, L=4096) shows 3276.63 emb/s @ threads=1 and 41469.48 emb/s @ threads=16 (~12.6x).
- Fast-math sanity: `MONOID_CPU_FAST_MATH=1 MONOID_CPU_FAST_TANH=1 scripts/verify_kernel_correctness.py --presets small --batch-sizes 2 --seq-lens 128 --modes float --pool-strategies mean --normalize-output 1 --use-exchange 1 --use-second-activation 0` (passes with relaxed tolerances).
- Determinism gate: `scripts/verify_kernel_correctness.py --presets small --batch-sizes 2 --seq-lens 128 --modes float quant quant_int8 --pool-strategies mean --normalize-output 1 --use-exchange 1 --use-second-activation 0 --determinism --determinism-runs 3 --determinism-tol 0` (all pass).

### TODO
- Investigate multi-core regression (verify thread affinity, batching strategy, and memory bandwidth).
- Regression harness now gates on p50 to avoid avg outliers on `small` (B=64, threads=16).
- Projection intrinsics toggle: set `MONOID_CPU_FORCE_PROJ_INTRINSICS=1` to force the dot-product intrinsics path (default keeps `#pragma omp simd` for perf).
- False-sharing mitigation: shared exchange buffers and per-thread scratch use cacheline padding (`cacheline_pad_elems`, `AlignedVector`).
- Nested threading guard: kernel skips `at::parallel_for` when already in a parallel region; use `MONOID_CPU_THREADS`/`MonoidCpuConfig(threads=N)` to set `torch.set_num_threads`.
