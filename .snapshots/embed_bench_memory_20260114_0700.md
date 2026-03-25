## 2026-01-14 15:43:43 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 1 --max-bytes 4096 --warmup 1 --repeat 3 --presets small small_l2 medium base --memory-profile --memory-warm-reps 5 --cache-evict-bytes 268435456 --output .snapshots/embed_bench_memory_20260114_0700.md --append`

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
| warmup | 1 |
| repeat | 3 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small, small_l2, medium, base |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |
| memory_profile | 1 |
| memory_warm_reps | 5 |
| cache_evict_bytes | 268435456 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 0.26 | 0.25 | 0.26 | 15343.88 | 3835.97 |
| small_l2 | OK | 0.50 | 0.50 | 0.50 | 7930.92 | 1982.73 |
| medium | OK | 1.30 | 1.28 | 1.28 | 3085.78 | 771.44 |
| base | OK | 13.88 | 13.84 | 13.84 | 288.23 | 72.06 |

### Memory
| Preset | Status | RSS before (KB) | RSS after (KB) | RSS delta (KB) | Warm RSS min (KB) | Warm RSS max (KB) | Warm RSS spread (KB) | Cold (ms) | Warm avg (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | OK | 567868 | 567868 | 0 | 567868 | 567868 | 0 | 0.41 | 0.26 |
| small_l2 | OK | 578444 | 578444 | 0 | 578444 | 578444 | 0 | 0.67 | 0.51 |
| medium | OK | 602524 | 602524 | 0 | 602524 | 602524 | 0 | 1.68 | 1.31 |
| base | OK | 837300 | 837300 | 0 | 837300 | 837300 | 0 | 14.32 | 13.98 |
## 2026-01-14 15:43:46 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 1 --repeat 3 --presets small small_l2 medium base --memory-profile --memory-warm-reps 5 --cache-evict-bytes 268435456 --output .snapshots/embed_bench_memory_20260114_0700.md --append`

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
| warmup | 1 |
| repeat | 3 |
| checkpoint_dir | tmp/bench_ckpts |
| presets | small, small_l2, medium, base |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 0 |
| memory_profile | 1 |
| memory_warm_reps | 5 |
| cache_evict_bytes | 268435456 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small | OK | 12.59 | 12.11 | 12.13 | 20336.53 | 5084.13 |
| small_l2 | OK | 2.73 | 2.29 | 2.59 | 93783.06 | 23445.77 |
| medium | OK | 6.24 | 5.87 | 6.38 | 41008.44 | 10252.11 |
| base | OK | 48.31 | 48.14 | 48.14 | 5299.46 | 1324.86 |

### Memory
| Preset | Status | RSS before (KB) | RSS after (KB) | RSS delta (KB) | Warm RSS min (KB) | Warm RSS max (KB) | Warm RSS spread (KB) | Cold (ms) | Warm avg (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | OK | 568804 | 568804 | 0 | 568804 | 568804 | 0 | 2.89 | 1.17 |
| small_l2 | OK | 577100 | 577100 | 0 | 577100 | 577100 | 0 | 2.77 | 2.45 |
| medium | OK | 596840 | 596840 | 0 | 596840 | 596840 | 0 | 4.72 | 3.88 |
| base | OK | 818272 | 818272 | 0 | 818272 | 818272 | 0 | 48.37 | 47.83 |
