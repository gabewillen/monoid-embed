## 2026-01-14 14:30:34 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.26 | 0.26 | 0.26 | 15479.42 | 3869.86 |
## 2026-01-14 14:30:36 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | quant |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16738.04 | 4184.51 |
## 2026-01-14 14:30:38 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | quant_int8 |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16621.73 | 4155.43 |
## 2026-01-14 14:30:40 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 64 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 1.75 | 1.74 | 1.75 | 146075.73 | 36518.93 |
## 2026-01-14 14:30:42 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode quant --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | quant |
| device | cpu |
| threads | 16 |
| batch_size | 64 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 1.60 | 1.58 | 1.60 | 160405.28 | 40101.32 |
## 2026-01-14 14:30:44 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode quant_int8 --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_kernel_modes_20260114_0430.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | quant_int8 |
| device | cpu |
| threads | 16 |
| batch_size | 64 |
| max_bytes | 4096 |
| doc_max_bytes | 0 |
| chunk_stride | 0 |
| chunk_aggregate | mean |
| chunk_count | 1 |
| chunk_bytes_total | 4096 |
| doc_bytes | 52228 |
| seq_len | 4096 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 1.64 | 1.63 | 1.65 | 156108.34 | 39027.08 |
