## 2026-01-14 15:06:26 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/ascii_english_4096.txt --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.23 | 0.23 | 0.23 | 17117.81 | 4279.45 |
## 2026-01-14 15:06:28 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/random_bytes_4096.bin --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.25 | 0.25 | 0.25 | 16062.62 | 4015.65 |
## 2026-01-14 15:06:30 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/scifact_style_4096.txt --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.23 | 0.23 | 0.24 | 17066.33 | 4266.58 |
## 2026-01-14 15:06:32 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/token_dense_4096.txt --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.23 | 0.23 | 0.23 | 17040.41 | 4260.10 |
## 2026-01-14 15:06:34 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/utf8_heavy_4096.txt --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.25 | 0.24 | 0.25 | 16249.53 | 4062.38 |
## 2026-01-14 15:06:36 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/whitespace_dense_4096.txt --engine kernel --mode quant --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16911.53 | 4227.88 |
## 2026-01-14 15:06:38 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/ascii_english_4096.txt --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16658.16 | 4164.54 |
## 2026-01-14 15:06:40 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/random_bytes_4096.bin --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16624.78 | 4156.20 |
## 2026-01-14 15:06:42 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/scifact_style_4096.txt --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.23 | 0.23 | 0.23 | 17609.91 | 4402.48 |
## 2026-01-14 15:06:45 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/token_dense_4096.txt --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.23 | 0.23 | 0.23 | 17121.18 | 4280.29 |
## 2026-01-14 15:06:47 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/utf8_heavy_4096.txt --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.23 | 0.24 | 16716.19 | 4179.05 |
## 2026-01-14 15:06:49 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input tmp/bench_inputs/whitespace_dense_4096.txt --engine kernel --mode quant_int8 --threads 16 --batch-size 1 --max-bytes 4096 --warmup 3 --repeat 5 --presets small --output .snapshots/embed_bench_input_dist_quant_20260114_0615.md --append`

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
| presets | small |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.24 | 0.24 | 0.24 | 16933.91 | 4233.48 |
