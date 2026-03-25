## 2026-01-14 15:05:45 UTC: Embed benchmark snapshot

### Setup
- host: shadecloud
- cpu: AMD EPYC 9124 16-Core Processor
- os: Linux-5.15.0-164-generic-x86_64-with-glibc2.35
- python: 3.10.12
- torch: 2.9.1+cu128
- git: unknown (unknown)
- command: `scripts/run_embed_bench.py --input docs/embed.md --engine kernel --mode float --threads 16 --batch-size 64 --max-bytes 4096 --warmup 1 --repeat 3 --presets small_l2 --profile-breakdown --output .snapshots/embed_bench_breakdown_20260114_0600.md`

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
| presets | small_l2 |
| shapes | none |
| fast_math | unset |
| fast_tanh | unset |
| OMP_NUM_THREADS | 16 |
| MKL_NUM_THREADS | 1 |
| MONOID_CPU_THREADS | 16 |
| profile_breakdown | 1 |

### Results
| Preset | Status | Avg (ms) | p50 (ms) | p95 (ms) | Throughput (KB/s) | Embeddings/s |
| --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 3.16 | 3.16 | 3.16 | 80927.99 | 20232.00 |

### Breakdown (ms)
| Preset | Status | setup | recurrence | butterfly | activation | exchange | pooling | layer_norm | proj | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_l2 | OK | 0.04 | 28.57 | 1.15 | 0.49 | 0.85 | 0.18 | 0.06 | 0.00 | 31.34 |
