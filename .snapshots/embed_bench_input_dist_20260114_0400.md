## 2026-01-14 14:29:19 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/ascii_english_4096.txt --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/ascii_english_4096.txt |
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
| small_l2 | OK | 0.50 | 0.50 | 0.50 | 7922.91 | 1980.73 |
## 2026-01-14 14:29:21 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/random_bytes_4096.bin --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/random_bytes_4096.bin |
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
| small_l2 | OK | 0.59 | 0.59 | 0.59 | 6759.13 | 1689.78 |
## 2026-01-14 14:29:23 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/scifact_style_4096.txt --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/scifact_style_4096.txt |
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
| small_l2 | OK | 0.51 | 0.51 | 0.51 | 7850.49 | 1962.62 |
## 2026-01-14 14:29:25 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/token_dense_4096.txt --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/token_dense_4096.txt |
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
| small_l2 | OK | 0.54 | 0.54 | 0.54 | 7406.21 | 1851.55 |
## 2026-01-14 14:29:27 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/utf8_heavy_4096.txt --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/utf8_heavy_4096.txt |
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
| small_l2 | OK | 0.49 | 0.49 | 0.49 | 8131.11 | 2032.78 |
## 2026-01-14 14:29:29 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/whitespace_dense_4096.txt --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small_l2 --output .snapshots/embed_bench_input_dist_20260114_0400.md --append`

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
| doc_bytes | 4096 |
| seq_len | 4096 |
| input | tmp/bench_inputs/whitespace_dense_4096.txt |
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
| small_l2 | OK | 0.45 | 0.44 | 0.45 | 8973.97 | 2243.49 |
