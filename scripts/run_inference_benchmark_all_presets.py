#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig, MonoidCpuConfig, MonoidCpuKernel


def _make_random_checkpoint(preset: str, out_dir: Path) -> Path:
    config = MonoidEmbedConfig.from_preset(preset)
    model = MonoidEmbed(config)
    path = out_dir / f"{preset}.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, path)
    return path


def _run_once_cpu_kernel(
    kernel: MonoidCpuKernel,
    x: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return kernel.forward_full_precision(x, lengths=lengths)


def _run_once_torch(
    model: MonoidEmbed,
    x: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return model(x, lengths=lengths)["embeddings"]


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
    parser = argparse.ArgumentParser(description="Run inference benchmarks across all presets.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max_bytes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pin", type=str, default="0-15")
    parser.add_argument("--fast-math-values", nargs="*", default=["0", "1"])
    parser.add_argument("--cpu_kernel", action="store_true", help="Use the C++ CPU kernel (full precision).")
    parser.add_argument("--checkpoint_dir", type=str, default="tmp/random_ckpts")
    parser.add_argument("--presets", type=str, nargs="*", default=None)
    args = parser.parse_args()

    _apply_affinity(args.pin)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, args.threads))

    if args.cpu_kernel and device.type != "cpu":
        raise SystemExit("--cpu_kernel requires --device cpu")

    presets = args.presets or sorted(MonoidEmbedConfig._PRESET_SPECS.keys())
    fast_math_values = [int(v) for v in args.fast_math_values if v in {"0", "1"} or str(v).isdigit()]
    fast_math_values = [v for v in fast_math_values if v in (0, 1)]
    if not fast_math_values:
        fast_math_values = [0, 1]
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "rb") as f:
        data = f.read()
    if args.max_bytes > 0:
        data = data[: args.max_bytes]

    x = torch.tensor(list(data), dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([x.size(1)], device=device)

    fast_math_loop = fast_math_values if args.cpu_kernel else [None]

    for fast_math in fast_math_loop:
        if args.cpu_kernel:
            print(f"=== fast_math={fast_math} ===")
        for preset in presets:
            print(f"=== {preset} ===")
            ckpt_path = out_dir / f"{preset}.pt"
            if not ckpt_path.exists():
                ckpt_path = _make_random_checkpoint(preset, out_dir)

            if args.cpu_kernel:
                cfg = MonoidCpuConfig(fast_math=bool(fast_math))
                kernel = MonoidCpuKernel.from_checkpoint(str(ckpt_path), config=cfg)
                run_once = lambda: _run_once_cpu_kernel(kernel, x, lengths)
            else:
                config = MonoidEmbedConfig.from_preset(preset)
                model = MonoidEmbed(config).to(device)
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"WARNING: Missing keys: {missing}")
                if unexpected:
                    print(f"WARNING: Unexpected keys: {unexpected}")
                model.eval()
                run_once = lambda: _run_once_torch(model, x, lengths)

            with torch.inference_mode():
                for _ in range(max(0, args.warmup)):
                    run_once()
                times = []
                for _ in range(max(1, args.repeat)):
                    start = time.time()
                    _ = run_once()
                    times.append(time.time() - start)

            elapsed = sum(times) / len(times)
            kb_per_s = (int(lengths.item()) / 1024.0) / max(elapsed, 1e-9)
            print(f"Elapsed: {elapsed * 1000:.2f} ms")
            print(f"Throughput: {kb_per_s:.2f} KB/s ({int(lengths.item())} bytes)")
            print()


if __name__ == "__main__":
    main()
