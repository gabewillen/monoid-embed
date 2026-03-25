import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig, MonoidCpuConfig, MonoidCpuKernel


def load_model(checkpoint: str, device: torch.device, quantized: bool) -> MonoidEmbed:
    config = MonoidEmbedConfig()
    config.use_quantized = quantized
    model = MonoidEmbed(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    model.eval()
    return model


def _parse_pin_list(pin: str) -> set[int]:
    pin = pin.strip()
    if not pin or pin.lower() == "none":
        return set()
    cpus: set[int] = set()
    for part in pin.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def _apply_affinity(pin: str) -> None:
    cpus = _parse_pin_list(pin)
    if not cpus:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    else:
        print("WARNING: os.sched_setaffinity unavailable; pinning skipped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max_bytes", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pin", type=str, default="0-15")
    parser.add_argument("--fast-math-values", nargs="*", default=["0", "1"])
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--emit-int8", action="store_true")
    parser.add_argument("--cpu_kernel", action="store_true", help="Use the C++ CPU kernel (full precision).")
    args = parser.parse_args()

    _apply_affinity(args.pin)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, args.threads))

    fast_math_values = [int(v) for v in args.fast_math_values if v in {"0", "1"} or str(v).isdigit()]
    fast_math_values = [v for v in fast_math_values if v in (0, 1)]
    if not fast_math_values:
        fast_math_values = [0, 1]

    if args.cpu_kernel:
        if device.type != "cpu":
            raise SystemExit("--cpu_kernel requires --device cpu")
        if args.quantized:
            raise SystemExit("--cpu_kernel runs full precision; drop --quantized")
        if args.emit_int8:
            raise SystemExit("--cpu_kernel full precision does not support --emit-int8")
        model = None
        kernel = None
    else:
        model = load_model(args.checkpoint, device=device, quantized=args.quantized)
        kernel = None
        if args.emit_int8:
            model.config.emit_int8 = True

    with open(args.input, "rb") as f:
        data = f.read()
    if args.max_bytes > 0:
        data = data[: args.max_bytes]

    x_single = torch.tensor(list(data), dtype=torch.long, device=device).unsqueeze(0)
    lengths_single = torch.tensor([x_single.size(1)], device=device)
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.batch_size == 1:
        x = x_single
        lengths = lengths_single
    else:
        x = x_single.repeat(args.batch_size, 1)
        lengths = lengths_single.repeat(args.batch_size)

    fast_math_loop = fast_math_values if args.cpu_kernel else [None]

    for fast_math in fast_math_loop:
        if args.cpu_kernel:
            cfg = MonoidCpuConfig(fast_math=bool(fast_math))
            kernel = MonoidCpuKernel.from_checkpoint(args.checkpoint, config=cfg)
            print(f"=== fast_math={fast_math} ===")

        def _run_once() -> float:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            if kernel is not None:
                _ = kernel.forward_full_precision(x, lengths=lengths)
            else:
                _ = model(x, lengths=lengths)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.time() - start

        with torch.inference_mode():
            for _ in range(max(0, args.warmup)):
                _run_once()

            times = []
            for _ in range(max(1, args.repeat)):
                times.append(_run_once())
            times_sorted = sorted(times)
            elapsed = sum(times) / len(times)
            p50 = times_sorted[len(times_sorted) // 2]
            p95 = times_sorted[max(0, int(len(times_sorted) * 0.95) - 1)]
            if kernel is not None:
                out = kernel.forward_full_precision(x, lengths=lengths)
            else:
                out = model(x, lengths=lengths)

        emb = out["embeddings"] if kernel is None else out
        print(f"Embedding shape: {tuple(emb.shape)}")
        print(f"Embedding sample: {emb[0, :10].tolist()}")
        if kernel is None and args.emit_int8:
            emb_int8 = out.get("embeddings_int8")
            scale_q15 = out.get("embeddings_scale_q15")
            if emb_int8 is not None and scale_q15 is not None:
                print(f"Embedding int8 shape: {tuple(emb_int8.shape)}")
                print(f"Embedding scale_q15 sample: {scale_q15[:5].tolist()}")
        total_bytes = int(lengths.sum().item())
        kb_per_s = (total_bytes / 1024.0) / max(elapsed, 1e-9)
        emb_per_s = (int(lengths.size(0)) / max(elapsed, 1e-9))
        print(f"Elapsed avg: {elapsed * 1000:.2f} ms")
        print(f"Elapsed p50: {p50 * 1000:.2f} ms")
        print(f"Elapsed p95: {p95 * 1000:.2f} ms")
        print(f"Throughput: {kb_per_s:.2f} KB/s ({total_bytes} bytes)")
        print(f"Embeddings/s: {emb_per_s:.2f}")
        print("")


if __name__ == "__main__":
    main()
