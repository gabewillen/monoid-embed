## 2026-01-14 03:29:46 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 64 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | mean |
| chunk_count | 817 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 20.00 | 19.98 | 20.01 | 2550.32 | 50.00 |
## 2026-01-14 03:29:49 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 64 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | max |
| chunk_count | 817 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 20.21 | 20.18 | 20.26 | 2523.67 | 49.48 |
## 2026-01-14 03:29:51 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 64 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | last |
| chunk_count | 817 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 18.98 | 18.97 | 19.00 | 2686.96 | 52.68 |
## 2026-01-14 03:29:53 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 48 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 48 |
| chunk_aggregate | mean |
| chunk_count | 1088 |
| chunk_bytes_total | 69620 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 26.26 | 26.23 | 26.27 | 1942.61 | 38.09 |
## 2026-01-14 03:29:55 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 48 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 48 |
| chunk_aggregate | max |
| chunk_count | 1088 |
| chunk_bytes_total | 69620 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 26.29 | 26.25 | 26.28 | 1939.73 | 38.03 |
## 2026-01-14 03:29:58 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 48 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 48 |
| chunk_aggregate | last |
| chunk_count | 1088 |
| chunk_bytes_total | 69620 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 25.25 | 25.25 | 25.26 | 2019.71 | 39.60 |
## 2026-01-14 03:30:00 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 32 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 32 |
| chunk_aggregate | mean |
| chunk_count | 1632 |
| chunk_bytes_total | 104420 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 40.57 | 40.53 | 40.57 | 1257.22 | 24.65 |
## 2026-01-14 03:30:02 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 32 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 32 |
| chunk_aggregate | max |
| chunk_count | 1632 |
| chunk_bytes_total | 104420 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 39.06 | 38.86 | 38.88 | 1305.88 | 25.60 |
## 2026-01-14 03:30:05 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 64 --chunk-stride 32 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 64 |
| doc_max_bytes | 0 |
| chunk_stride | 32 |
| chunk_aggregate | last |
| chunk_count | 1632 |
| chunk_bytes_total | 104420 |
| doc_bytes | 52228 |
| seq_len | 64 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 37.26 | 37.21 | 37.29 | 1369.02 | 26.84 |
## 2026-01-14 03:30:07 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 128 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | mean |
| chunk_count | 409 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.84 | 12.76 | 12.80 | 3971.40 | 77.86 |
## 2026-01-14 03:30:09 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 128 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | max |
| chunk_count | 409 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.96 | 11.94 | 11.95 | 4265.41 | 83.63 |
## 2026-01-14 03:30:11 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 128 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | last |
| chunk_count | 409 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.30 | 11.27 | 11.30 | 4511.96 | 88.46 |
## 2026-01-14 03:30:14 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 96 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 96 |
| chunk_aggregate | mean |
| chunk_count | 544 |
| chunk_bytes_total | 69604 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 16.52 | 16.50 | 16.53 | 3088.26 | 60.55 |
## 2026-01-14 03:30:16 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 96 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 96 |
| chunk_aggregate | max |
| chunk_count | 544 |
| chunk_bytes_total | 69604 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 15.84 | 15.82 | 15.87 | 3219.03 | 63.11 |
## 2026-01-14 03:30:18 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 96 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 96 |
| chunk_aggregate | last |
| chunk_count | 544 |
| chunk_bytes_total | 69604 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 16.02 | 16.01 | 16.03 | 3184.05 | 62.43 |
## 2026-01-14 03:30:20 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 64 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | mean |
| chunk_count | 816 |
| chunk_bytes_total | 104388 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 24.51 | 24.37 | 24.42 | 2081.25 | 40.81 |
## 2026-01-14 03:30:23 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 64 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | max |
| chunk_count | 816 |
| chunk_bytes_total | 104388 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 23.62 | 23.58 | 23.66 | 2159.55 | 42.34 |
## 2026-01-14 03:30:25 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 128 --chunk-stride 64 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 128 |
| doc_max_bytes | 0 |
| chunk_stride | 64 |
| chunk_aggregate | last |
| chunk_count | 816 |
| chunk_bytes_total | 104388 |
| doc_bytes | 52228 |
| seq_len | 128 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 22.81 | 22.78 | 22.83 | 2236.25 | 43.84 |
## 2026-01-14 03:30:27 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 256 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | mean |
| chunk_count | 205 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.85 | 7.81 | 7.88 | 6496.07 | 127.36 |
## 2026-01-14 03:30:29 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 256 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | max |
| chunk_count | 205 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.35 | 8.35 | 8.36 | 6105.87 | 119.71 |
## 2026-01-14 03:30:31 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 256 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | last |
| chunk_count | 205 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.75 | 7.72 | 7.78 | 6582.76 | 129.06 |
## 2026-01-14 03:30:33 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 192 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 192 |
| chunk_aggregate | mean |
| chunk_count | 272 |
| chunk_bytes_total | 69572 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 10.64 | 10.63 | 10.63 | 4795.01 | 94.01 |
## 2026-01-14 03:30:36 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 192 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 192 |
| chunk_aggregate | max |
| chunk_count | 272 |
| chunk_bytes_total | 69572 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 10.96 | 10.67 | 11.35 | 4651.90 | 91.21 |
## 2026-01-14 03:30:38 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 192 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 192 |
| chunk_aggregate | last |
| chunk_count | 272 |
| chunk_bytes_total | 69572 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 10.60 | 10.54 | 10.64 | 4813.76 | 94.38 |
## 2026-01-14 03:30:40 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 128 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | mean |
| chunk_count | 408 |
| chunk_bytes_total | 104324 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 16.04 | 16.00 | 16.06 | 3180.18 | 62.35 |
## 2026-01-14 03:30:42 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 128 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | max |
| chunk_count | 408 |
| chunk_bytes_total | 104324 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 16.11 | 16.09 | 16.12 | 3166.54 | 62.08 |
## 2026-01-14 03:30:45 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 256 --chunk-stride 128 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 256 |
| doc_max_bytes | 0 |
| chunk_stride | 128 |
| chunk_aggregate | last |
| chunk_count | 408 |
| chunk_bytes_total | 104324 |
| doc_bytes | 52228 |
| seq_len | 256 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 15.72 | 15.69 | 15.73 | 3243.68 | 63.60 |
## 2026-01-14 03:30:47 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 512 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | mean |
| chunk_count | 103 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.36 | 7.30 | 7.43 | 6931.32 | 135.90 |
## 2026-01-14 03:30:49 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 512 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | max |
| chunk_count | 103 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.23 | 7.22 | 7.24 | 7054.51 | 138.31 |
## 2026-01-14 03:30:51 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 512 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | last |
| chunk_count | 103 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.98 | 6.98 | 6.99 | 7304.55 | 143.22 |
## 2026-01-14 03:30:53 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 384 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 384 |
| chunk_aggregate | mean |
| chunk_count | 136 |
| chunk_bytes_total | 69508 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 9.68 | 9.67 | 9.68 | 5270.52 | 103.34 |
## 2026-01-14 03:30:55 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 384 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 384 |
| chunk_aggregate | max |
| chunk_count | 136 |
| chunk_bytes_total | 69508 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 9.72 | 9.67 | 9.76 | 5247.50 | 102.88 |
## 2026-01-14 03:30:58 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 384 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 384 |
| chunk_aggregate | last |
| chunk_count | 136 |
| chunk_bytes_total | 69508 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 9.64 | 9.61 | 9.71 | 5288.25 | 103.68 |
## 2026-01-14 03:31:00 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 256 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | mean |
| chunk_count | 204 |
| chunk_bytes_total | 104196 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 14.81 | 14.41 | 15.11 | 3443.67 | 67.52 |
## 2026-01-14 03:31:02 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 256 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | max |
| chunk_count | 204 |
| chunk_bytes_total | 104196 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 14.54 | 14.49 | 14.56 | 3508.95 | 68.80 |
## 2026-01-14 03:31:04 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 512 --chunk-stride 256 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 512 |
| doc_max_bytes | 0 |
| chunk_stride | 256 |
| chunk_aggregate | last |
| chunk_count | 204 |
| chunk_bytes_total | 104196 |
| doc_bytes | 52228 |
| seq_len | 512 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 14.20 | 14.14 | 14.18 | 3591.27 | 70.41 |
## 2026-01-14 03:31:06 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 1024 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | mean |
| chunk_count | 52 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.66 | 6.65 | 6.67 | 7657.98 | 150.14 |
## 2026-01-14 03:31:08 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 1024 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | max |
| chunk_count | 52 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.05 | 7.06 | 7.07 | 7234.71 | 141.85 |
## 2026-01-14 03:31:10 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 1024 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | last |
| chunk_count | 52 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.79 | 6.78 | 6.78 | 7512.14 | 147.29 |
## 2026-01-14 03:31:13 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 768 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 768 |
| chunk_aggregate | mean |
| chunk_count | 68 |
| chunk_bytes_total | 69380 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 9.00 | 8.89 | 9.10 | 5665.84 | 111.09 |
## 2026-01-14 03:31:15 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 768 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 768 |
| chunk_aggregate | max |
| chunk_count | 68 |
| chunk_bytes_total | 69380 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 9.07 | 8.99 | 9.10 | 5620.54 | 110.20 |
## 2026-01-14 03:31:17 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 768 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 768 |
| chunk_aggregate | last |
| chunk_count | 68 |
| chunk_bytes_total | 69380 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.99 | 8.96 | 9.02 | 5672.66 | 111.22 |
## 2026-01-14 03:31:19 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 512 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | mean |
| chunk_count | 102 |
| chunk_bytes_total | 103940 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 13.59 | 13.52 | 13.57 | 3753.89 | 73.60 |
## 2026-01-14 03:31:21 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 512 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | max |
| chunk_count | 102 |
| chunk_bytes_total | 103940 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 13.93 | 13.88 | 13.91 | 3660.25 | 71.76 |
## 2026-01-14 03:31:23 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 1024 --chunk-stride 512 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 1024 |
| doc_max_bytes | 0 |
| chunk_stride | 512 |
| chunk_aggregate | last |
| chunk_count | 102 |
| chunk_bytes_total | 103940 |
| doc_bytes | 52228 |
| seq_len | 1024 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 13.15 | 13.14 | 13.16 | 3879.37 | 76.06 |
## 2026-01-14 03:31:25 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 2048 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 2048 |
| chunk_aggregate | mean |
| chunk_count | 26 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.46 | 6.45 | 6.47 | 7897.21 | 154.84 |
## 2026-01-14 03:31:28 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 2048 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 2048 |
| chunk_aggregate | max |
| chunk_count | 26 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.65 | 6.64 | 6.65 | 7671.25 | 150.41 |
## 2026-01-14 03:31:30 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 2048 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 2048 |
| chunk_aggregate | last |
| chunk_count | 26 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.72 | 6.69 | 6.74 | 7593.09 | 148.87 |
## 2026-01-14 03:31:32 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1536 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1536 |
| chunk_aggregate | mean |
| chunk_count | 34 |
| chunk_bytes_total | 69124 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.82 | 8.77 | 8.92 | 5784.77 | 113.42 |
## 2026-01-14 03:31:34 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1536 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1536 |
| chunk_aggregate | max |
| chunk_count | 34 |
| chunk_bytes_total | 69124 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.71 | 8.69 | 8.71 | 5852.86 | 114.75 |
## 2026-01-14 03:31:36 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1536 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1536 |
| chunk_aggregate | last |
| chunk_count | 34 |
| chunk_bytes_total | 69124 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.66 | 8.65 | 8.67 | 5886.33 | 115.41 |
## 2026-01-14 03:31:38 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1024 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | mean |
| chunk_count | 51 |
| chunk_bytes_total | 103428 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 13.03 | 12.99 | 13.01 | 3914.03 | 76.74 |
## 2026-01-14 03:31:41 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1024 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | max |
| chunk_count | 51 |
| chunk_bytes_total | 103428 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 13.02 | 12.94 | 13.08 | 3916.14 | 76.78 |
## 2026-01-14 03:31:43 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 2048 --chunk-stride 1024 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 2048 |
| doc_max_bytes | 0 |
| chunk_stride | 1024 |
| chunk_aggregate | last |
| chunk_count | 51 |
| chunk_bytes_total | 103428 |
| doc_bytes | 52228 |
| seq_len | 2048 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.86 | 12.82 | 12.85 | 3966.98 | 77.78 |
## 2026-01-14 03:31:45 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 4096 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 4096 |
| chunk_aggregate | mean |
| chunk_count | 13 |
| chunk_bytes_total | 52228 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.32 | 6.32 | 6.33 | 8064.28 | 158.11 |
## 2026-01-14 03:31:47 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 4096 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 4096 |
| chunk_aggregate | max |
| chunk_count | 13 |
| chunk_bytes_total | 52228 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.25 | 6.25 | 6.25 | 8155.86 | 159.91 |
## 2026-01-14 03:31:49 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 4096 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 4096 |
| chunk_aggregate | last |
| chunk_count | 13 |
| chunk_bytes_total | 52228 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.30 | 6.29 | 6.30 | 8101.21 | 158.84 |
## 2026-01-14 03:31:51 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 3072 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 3072 |
| chunk_aggregate | mean |
| chunk_count | 17 |
| chunk_bytes_total | 68612 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.47 | 8.45 | 8.50 | 6018.76 | 118.01 |
## 2026-01-14 03:31:53 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 3072 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 3072 |
| chunk_aggregate | max |
| chunk_count | 17 |
| chunk_bytes_total | 68612 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.56 | 8.55 | 8.57 | 5959.57 | 116.85 |
## 2026-01-14 03:31:56 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 3072 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 3072 |
| chunk_aggregate | last |
| chunk_count | 17 |
| chunk_bytes_total | 68612 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.50 | 8.49 | 8.51 | 6003.57 | 117.71 |
## 2026-01-14 03:31:58 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 2048 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 2048 |
| chunk_aggregate | mean |
| chunk_count | 25 |
| chunk_bytes_total | 101380 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.78 | 12.66 | 12.74 | 3991.24 | 78.25 |
## 2026-01-14 03:32:00 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 2048 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 2048 |
| chunk_aggregate | max |
| chunk_count | 25 |
| chunk_bytes_total | 101380 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.91 | 12.89 | 12.92 | 3950.94 | 77.46 |
## 2026-01-14 03:32:02 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --chunk-stride 2048 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

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
| chunk_stride | 2048 |
| chunk_aggregate | last |
| chunk_count | 25 |
| chunk_bytes_total | 101380 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.41 | 12.35 | 12.45 | 4109.21 | 80.57 |
## 2026-01-14 03:32:04 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 8192 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | mean |
| chunk_count | 7 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.78 | 6.76 | 6.76 | 7519.72 | 147.43 |
## 2026-01-14 03:32:06 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 8192 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | max |
| chunk_count | 7 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.39 | 6.35 | 6.40 | 7980.86 | 156.48 |
## 2026-01-14 03:32:08 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 8192 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | last |
| chunk_count | 7 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.54 | 6.53 | 6.56 | 7795.50 | 152.84 |
## 2026-01-14 03:32:10 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 6144 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 6144 |
| chunk_aggregate | mean |
| chunk_count | 9 |
| chunk_bytes_total | 68612 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.17 | 8.14 | 8.15 | 6244.74 | 122.44 |
## 2026-01-14 03:32:13 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 6144 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 6144 |
| chunk_aggregate | max |
| chunk_count | 9 |
| chunk_bytes_total | 68612 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.36 | 8.33 | 8.39 | 6102.46 | 119.65 |
## 2026-01-14 03:32:15 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 6144 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 6144 |
| chunk_aggregate | last |
| chunk_count | 9 |
| chunk_bytes_total | 68612 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 8.76 | 8.72 | 8.81 | 5819.67 | 114.10 |
## 2026-01-14 03:32:17 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 4096 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 4096 |
| chunk_aggregate | mean |
| chunk_count | 12 |
| chunk_bytes_total | 97284 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.72 | 11.69 | 11.73 | 4351.64 | 85.32 |
## 2026-01-14 03:32:19 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 4096 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 4096 |
| chunk_aggregate | max |
| chunk_count | 12 |
| chunk_bytes_total | 97284 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.47 | 11.47 | 11.47 | 4447.51 | 87.20 |
## 2026-01-14 03:32:21 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 8192 --chunk-stride 4096 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 8192 |
| doc_max_bytes | 0 |
| chunk_stride | 4096 |
| chunk_aggregate | last |
| chunk_count | 12 |
| chunk_bytes_total | 97284 |
| doc_bytes | 52228 |
| seq_len | 8192 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.69 | 11.69 | 11.70 | 4363.22 | 85.55 |
## 2026-01-14 03:32:24 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 16384 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 16384 |
| chunk_aggregate | mean |
| chunk_count | 4 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.12 | 6.11 | 6.13 | 8334.01 | 163.40 |
## 2026-01-14 03:32:26 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 16384 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 16384 |
| chunk_aggregate | max |
| chunk_count | 4 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.57 | 6.48 | 6.65 | 7764.29 | 152.23 |
## 2026-01-14 03:32:28 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 16384 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 16384 |
| chunk_aggregate | last |
| chunk_count | 4 |
| chunk_bytes_total | 52228 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.40 | 6.40 | 6.42 | 7965.58 | 156.18 |
## 2026-01-14 03:32:30 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 12288 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 12288 |
| chunk_aggregate | mean |
| chunk_count | 4 |
| chunk_bytes_total | 64516 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.95 | 7.95 | 8.00 | 6413.00 | 125.74 |
## 2026-01-14 03:32:32 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 12288 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 12288 |
| chunk_aggregate | max |
| chunk_count | 4 |
| chunk_bytes_total | 64516 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.81 | 7.74 | 7.83 | 6533.06 | 128.09 |
## 2026-01-14 03:32:34 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 12288 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 12288 |
| chunk_aggregate | last |
| chunk_count | 4 |
| chunk_bytes_total | 64516 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.73 | 7.72 | 7.73 | 6601.10 | 129.42 |
## 2026-01-14 03:32:36 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 8192 --chunk-aggregate mean --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | mean |
| chunk_count | 6 |
| chunk_bytes_total | 93188 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.07 | 11.07 | 11.08 | 4605.61 | 90.30 |
## 2026-01-14 03:32:39 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 8192 --chunk-aggregate max --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | max |
| chunk_count | 6 |
| chunk_bytes_total | 93188 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 11.69 | 11.35 | 11.71 | 4364.62 | 85.57 |
## 2026-01-14 03:32:41 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 16384 --chunk-stride 8192 --chunk-aggregate last --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_len_chunk_20260114_0335.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 16 |
| batch_size | 1 |
| max_bytes | 16384 |
| doc_max_bytes | 0 |
| chunk_stride | 8192 |
| chunk_aggregate | last |
| chunk_count | 6 |
| chunk_bytes_total | 93188 |
| doc_bytes | 52228 |
| seq_len | 16384 |
| input | docs/embed.md |
| pool_strategy | mean |
| normalize_output | 1 |
| use_exchange | 1 |
| use_second_activation | 0 |
| warmup | 3 |
| repeat | 5 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 12.32 | 12.24 | 12.25 | 4140.55 | 81.18 |
