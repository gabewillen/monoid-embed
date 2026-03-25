## 2026-01-14 14:29:43 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 2.61 | 2.60 | 2.62 | 98092.94 | 24523.24 |
## 2026-01-14 14:29:45 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 32 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 32 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 32 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 32 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 2.52 | 2.46 | 2.51 | 101651.77 | 25412.94 |
## 2026-01-14 14:29:52 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 1 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 1 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 1 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 1 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 21.60 | 21.04 | 22.30 | 11849.70 | 2962.42 |
## 2026-01-14 14:29:55 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 2 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 2 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 2 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 2 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 10.49 | 10.47 | 10.48 | 24400.73 | 6100.18 |
## 2026-01-14 14:29:57 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 4 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 4 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 4 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 4 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 7.16 | 5.32 | 8.82 | 35777.97 | 8944.49 |
## 2026-01-14 14:29:59 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 8 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 8 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 8 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 8 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 6.13 | 6.07 | 6.58 | 41740.99 | 10435.25 |
## 2026-01-14 14:30:01 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 2.51 | 2.33 | 2.77 | 102102.45 | 25525.61 |
## 2026-01-14 14:30:09 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 1 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 1 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 1 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 1 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 20.87 | 20.84 | 20.89 | 12267.55 | 3066.89 |
## 2026-01-14 14:30:11 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 2 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 2 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 2 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 2 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 10.95 | 10.54 | 11.52 | 23369.69 | 5842.42 |
## 2026-01-14 14:30:13 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 4 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 4 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 4 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 4 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 5.43 | 5.38 | 5.49 | 47118.28 | 11779.57 |
## 2026-01-14 14:30:16 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 8 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 8 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 8 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 8 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 4.87 | 4.65 | 5.20 | 52512.99 | 13128.25 |
## 2026-01-14 14:30:18 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 2.58 | 2.58 | 2.59 | 99116.33 | 24779.08 |
## 2026-01-14 14:30:20 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 24 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 24 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 24 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 24 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 3.36 | 3.35 | 3.36 | 76291.29 | 19072.82 |
## 2026-01-14 14:30:22 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 32 --batch-size 64 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_thread_matrix_20260114_0415.md --append`

### Parameters
| Field | Value |
| --- | --- |
| engine | kernel |
| mode | float |
| device | cpu |
| threads | 32 |
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
| presets | small_l2 |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 32 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 32 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 2.53 | 2.51 | 2.56 | 101015.63 | 25253.91 |
