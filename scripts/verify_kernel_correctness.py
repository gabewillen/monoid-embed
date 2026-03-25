import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig, MonoidCpuConfig, MonoidCpuKernel


def _parse_bool_list(values: list[str]) -> list[bool]:
    parsed: list[bool] = []
    for value in values:
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            parsed.append(True)
        elif lowered in {"0", "false", "f", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Invalid bool value: {value}")
    return parsed


def _compute_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict:
    diff = (ref - out).abs()
    max_abs_err = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs_err = float(diff.mean().item()) if diff.numel() else 0.0
    cosine = float(F.cosine_similarity(ref, out, dim=-1).mean().item()) if ref.numel() else 0.0
    nan_or_inf = int((~torch.isfinite(out)).sum().item())
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "cosine": cosine,
        "nan_or_inf": nan_or_inf,
    }


def _format_case(
    preset: str,
    mode: str,
    batch_size: int,
    seq_len: int,
    pool_strategy: str,
    normalize_output: bool,
    use_exchange: bool,
    use_second_activation: bool,
) -> str:
    return (
        f"preset={preset} mode={mode} B={batch_size} L={seq_len} "
        f"pool={pool_strategy} norm={int(normalize_output)} "
        f"exchange={int(use_exchange)} second_act={int(use_second_activation)}"
    )


def _make_checkpoint(preset: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = MonoidEmbedConfig.from_preset(preset)
    model = MonoidEmbed(config)
    path = out_dir / f"{preset}.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, path)
    return path


def _load_state_dict(checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _build_model(
    preset: str,
    state_dict: dict,
    normalize_output: bool,
    pool_strategy: str,
    use_exchange: bool,
    use_second_activation: bool,
    emit_int8: bool,
) -> MonoidEmbed:
    config = MonoidEmbedConfig.from_preset(preset)
    config.normalize_output = normalize_output
    config.pool_strategy = pool_strategy
    config.use_exchange = use_exchange
    config.use_second_activation = use_second_activation
    config.emit_int8 = emit_int8
    model = MonoidEmbed(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _build_kernel(
    checkpoint: Path,
    normalize_output: bool,
    pool_strategy: str,
    use_exchange: bool,
    use_second_activation: bool,
    emit_int8: bool,
) -> MonoidCpuKernel:
    config = MonoidCpuConfig(
        normalize_output=normalize_output,
        pool_strategy=pool_strategy,
        use_exchange=use_exchange,
        use_second_activation=use_second_activation,
        emit_int8=emit_int8,
    )
    return MonoidCpuKernel.from_checkpoint(str(checkpoint), config=config)


def _run_float(
    model: MonoidEmbed,
    kernel: MonoidCpuKernel,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    normalize_output: bool,
    pool_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        torch_out = model(
            tokens,
            lengths=lengths,
            quantized=False,
            normalize_output=normalize_output,
            pool_strategy=pool_strategy,
        )["embeddings"]
        kernel_out = kernel.forward_full_precision(tokens, lengths=lengths)
    return torch_out, kernel_out


def _run_quant(
    model: MonoidEmbed,
    kernel: MonoidCpuKernel,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    normalize_output: bool,
    pool_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        torch_out = model(
            tokens,
            lengths=lengths,
            quantized=True,
            normalize_output=normalize_output,
            pool_strategy=pool_strategy,
        )["embeddings"]
        kernel_out = kernel.forward(tokens, lengths=lengths)
    return torch_out, kernel_out


def _run_quant_int8(
    model: MonoidEmbed,
    kernel: MonoidCpuKernel,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    normalize_output: bool,
    pool_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        torch_out = model(
            tokens,
            lengths=lengths,
            quantized=True,
            normalize_output=normalize_output,
            pool_strategy=pool_strategy,
        )
        kernel_out = kernel.forward(tokens, lengths=lengths)
    torch_float = torch_out["embeddings"]
    torch_int8 = torch_out.get("embeddings_int8")
    torch_scale = torch_out.get("embeddings_scale_q15")
    if torch_int8 is None or torch_scale is None:
        raise RuntimeError("emit_int8 was requested but torch outputs are missing.")
    kernel_float, kernel_int8, kernel_scale = kernel_out
    return torch_float, kernel_float, torch_int8, kernel_int8, torch_scale, kernel_scale


def _check_determinism(
    kernel: MonoidCpuKernel,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    mode: str,
    runs: int,
    tol: float,
) -> tuple[float, bool]:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(runs):
            if mode == "float":
                outputs.append(kernel.forward_full_precision(tokens, lengths=lengths))
            elif mode == "quant":
                outputs.append(kernel.forward(tokens, lengths=lengths))
            else:
                outputs.append(kernel.forward(tokens, lengths=lengths)[0])
    max_diff = 0.0
    for idx in range(1, len(outputs)):
        diff = (outputs[0] - outputs[idx]).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff, max_diff <= tol


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness + determinism checks for Monoid CPU kernel.")
    parser.add_argument("--presets", nargs="*", default=["small", "base"])
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[2])
    parser.add_argument("--seq-lens", nargs="*", type=int, default=[512])
    parser.add_argument("--modes", nargs="*", default=["float"])
    parser.add_argument("--pool-strategies", nargs="*", default=["mean"])
    parser.add_argument("--normalize-output", nargs="*", default=["1"])
    parser.add_argument("--use-exchange", nargs="*", default=["1"])
    parser.add_argument("--use-second-activation", nargs="*", default=["0"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--determinism", action="store_true")
    parser.add_argument("--determinism-runs", type=int, default=2)
    parser.add_argument("--determinism-tol", type=float, default=0.0)
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--quant-atol", type=float, default=None)
    parser.add_argument("--quant-rtol", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="tmp/correctness_ckpts")
    parser.add_argument("--edge-tests", action="store_true", help="Run edge/saturation tests for quantized mode.")
    args = parser.parse_args()

    normalize_output_vals = _parse_bool_list(args.normalize_output)
    use_exchange_vals = _parse_bool_list(args.use_exchange)
    use_second_activation_vals = _parse_bool_list(args.use_second_activation)

    fast_math_env = os.getenv("MONOID_CPU_FAST_MATH")
    if fast_math_env is None:
        fast_math = True
    else:
        fast_math = fast_math_env == "1"
    fast_tanh_env = os.getenv("MONOID_CPU_FAST_TANH")
    if fast_tanh_env is None:
        fast_tanh = fast_math
    else:
        fast_tanh = fast_tanh_env == "1"
    relaxed = fast_tanh or fast_math
    atol = 2e-3 if relaxed else 5e-5
    rtol = 2e-3 if relaxed else 1e-4
    quant_atol = 2e-3
    quant_rtol = 2e-3
    if args.atol is not None:
        atol = args.atol
    if args.rtol is not None:
        rtol = args.rtol
    if args.quant_atol is not None:
        quant_atol = args.quant_atol
    if args.quant_rtol is not None:
        quant_rtol = args.quant_rtol

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    out_dir = Path(args.checkpoint_dir)

    failures = 0
    total = 0

    for preset in args.presets:
        ckpt_path = _make_checkpoint(preset, out_dir)
        state_dict = _load_state_dict(ckpt_path)
        base_config = MonoidEmbedConfig.from_preset(preset)
        has_proj = base_config.d_state != max(base_config.matryoshka_dims)

        for (
            mode,
            batch_size,
            seq_len,
            pool_strategy,
            normalize_output,
            use_exchange,
            use_second_activation,
        ) in itertools.product(
            args.modes,
            args.batch_sizes,
            args.seq_lens,
            args.pool_strategies,
            normalize_output_vals,
            use_exchange_vals,
            use_second_activation_vals,
        ):
            mode = mode.lower()
            if mode not in {"float", "quant", "quant_int8"}:
                raise ValueError(f"Unknown mode: {mode}")
            if mode != "float" and base_config.n_layers > 1:
                print(f"SKIP: preset={preset} mode={mode} (quantized only supports 1 layer)")
                continue
            if mode in {"quant", "quant_int8"} and has_proj:
                print(f"SKIP: preset={preset} mode={mode} (quantized kernel does not apply projection)")
                continue

            case = _format_case(
                preset,
                mode,
                batch_size,
                seq_len,
                pool_strategy,
                normalize_output,
                use_exchange,
                use_second_activation,
            )
            total += 1

            tokens = torch.randint(
                0,
                256,
                (batch_size, seq_len),
                device=device,
                dtype=torch.long,
            )
            lengths = torch.randint(
                1,
                seq_len + 1,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            emit_int8 = mode == "quant_int8"
            model = _build_model(
                preset,
                state_dict,
                normalize_output,
                pool_strategy,
                use_exchange,
                use_second_activation,
                emit_int8,
            )
            kernel = _build_kernel(
                ckpt_path,
                normalize_output,
                pool_strategy,
                use_exchange,
                use_second_activation,
                emit_int8,
            )

            try:
                if mode == "float":
                    case_atol = atol
                    case_rtol = rtol
                    if base_config.n_layers > 1:
                        case_atol = max(case_atol, 5e-3)
                        case_rtol = max(case_rtol, 1e-3)
                    torch_out, kernel_out = _run_float(
                        model,
                        kernel,
                        tokens,
                        lengths,
                        normalize_output,
                        pool_strategy,
                    )
                    metrics = _compute_metrics(torch_out, kernel_out)
                    ok = torch.allclose(torch_out, kernel_out, atol=case_atol, rtol=case_rtol)
                elif mode == "quant":
                    torch_out, kernel_out = _run_quant(
                        model,
                        kernel,
                        tokens,
                        lengths,
                        normalize_output,
                        pool_strategy,
                    )
                    metrics = _compute_metrics(torch_out, kernel_out)
                    ok = torch.allclose(torch_out, kernel_out, atol=quant_atol, rtol=quant_rtol)
                else:
                    (
                        torch_float,
                        kernel_float,
                        torch_int8,
                        kernel_int8,
                        torch_scale,
                        kernel_scale,
                    ) = _run_quant_int8(
                        model,
                        kernel,
                        tokens,
                        lengths,
                        normalize_output,
                        pool_strategy,
                    )
                    metrics = _compute_metrics(torch_float, kernel_float)
                    ok = torch.allclose(torch_float, kernel_float, atol=quant_atol, rtol=quant_rtol)
                    int8_match = torch.equal(torch_int8, kernel_int8)
                    scale_match = torch.equal(torch_scale, kernel_scale)
                    ok = ok and int8_match and scale_match
                    metrics["int8_match"] = int(int8_match)
                    metrics["scale_match"] = int(scale_match)
            except Exception as exc:
                failures += 1
                print(f"ERROR: {case} -> {exc}")
                if not args.continue_on_fail:
                    sys.exit(1)
                continue

            if args.determinism:
                max_diff, det_ok = _check_determinism(
                    kernel,
                    tokens,
                    lengths,
                    mode,
                    runs=max(2, args.determinism_runs),
                    tol=args.determinism_tol,
                )
                metrics["determinism_max_diff"] = max_diff
                metrics["determinism_ok"] = int(det_ok)
                ok = ok and det_ok

            verdict = "PASS" if ok else "FAIL"
            print(f"{verdict}: {case} -> {metrics}")

            if not ok:
                failures += 1
                if not args.continue_on_fail:
                    sys.exit(1)

    print(f"Done. Total cases: {total}. Failures: {failures}.")
    if failures:
        sys.exit(1)

    if not args.edge_tests:
        return

    edge_cases = [
        {
            "name": "hi_gain",
            "activation_shift": 0,
            "activation_T_q15": 32767,
            "b_shift": 15,
            "inj_shift": 0,
            "use_second_activation": True,
        },
        {
            "name": "hi_shift",
            "activation_shift": 12,
            "activation_T_q15": 16384,
            "b_shift": 0,
            "inj_shift": 8,
            "use_second_activation": False,
        },
    ]

    edge_failures = 0
    edge_total = 0
    for preset in args.presets:
        base_config = MonoidEmbedConfig.from_preset(preset)
        has_proj = base_config.d_state != max(base_config.matryoshka_dims)
        if base_config.n_layers > 1 or has_proj:
            print(f"EDGE SKIP: preset={preset} (quant edge tests require 1 layer without projection)")
            continue

        for case in edge_cases:
            for mode in ("quant", "quant_int8"):
                edge_total += 1
                config = MonoidEmbedConfig.from_preset(preset)
                config.activation_shift = case["activation_shift"]
                config.activation_T_q15 = case["activation_T_q15"]
                config.b_shift = case["b_shift"]
                config.inj_shift = case["inj_shift"]
                config.use_exchange = True
                config.use_second_activation = case["use_second_activation"]
                config.normalize_output = True
                config.pool_strategy = "mean"
                config.emit_int8 = mode == "quant_int8"

                model = MonoidEmbed(config)
                model.eval()

                edge_ckpt = out_dir / f"{preset}_{case['name']}.pt"
                torch.save({"model": model.state_dict(), "model_config": config}, edge_ckpt)

                kernel = MonoidCpuKernel.from_checkpoint(
                    str(edge_ckpt),
                    config=MonoidCpuConfig(
                        normalize_output=True,
                        pool_strategy="mean",
                        use_exchange=True,
                        use_second_activation=case["use_second_activation"],
                        emit_int8=mode == "quant_int8",
                        activation_shift=case["activation_shift"],
                        activation_T_q15=case["activation_T_q15"],
                        b_shift=case["b_shift"],
                        inj_shift=case["inj_shift"],
                    ),
                )

                tokens = torch.randint(0, 256, (2, 256), dtype=torch.long)
                lengths = torch.randint(1, 257, (2,), dtype=torch.long)

                case_label = (
                    f"preset={preset} mode={mode} case={case['name']} "
                    f"act_shift={case['activation_shift']} act_T_q15={case['activation_T_q15']} "
                    f"b_shift={case['b_shift']} inj_shift={case['inj_shift']} "
                    f"second_act={int(case['use_second_activation'])}"
                )

                try:
                    if mode == "quant":
                        torch_out, kernel_out = _run_quant(
                            model,
                            kernel,
                            tokens,
                            lengths,
                            True,
                            "mean",
                        )
                        metrics = _compute_metrics(torch_out, kernel_out)
                        ok = torch.allclose(torch_out, kernel_out, atol=quant_atol, rtol=quant_rtol)
                    else:
                        (
                            torch_float,
                            kernel_float,
                            torch_int8,
                            kernel_int8,
                            torch_scale,
                            kernel_scale,
                        ) = _run_quant_int8(
                            model,
                            kernel,
                            tokens,
                            lengths,
                            True,
                            "mean",
                        )
                        metrics = _compute_metrics(torch_float, kernel_float)
                        ok = torch.allclose(torch_float, kernel_float, atol=quant_atol, rtol=quant_rtol)
                        ok = ok and torch.equal(torch_int8, kernel_int8) and torch.equal(torch_scale, kernel_scale)
                        metrics["int8_match"] = int(torch.equal(torch_int8, kernel_int8))
                        metrics["scale_match"] = int(torch.equal(torch_scale, kernel_scale))
                except Exception as exc:
                    edge_failures += 1
                    print(f"EDGE ERROR: {case_label} -> {exc}")
                    if not args.continue_on_fail:
                        sys.exit(1)
                    continue

                verdict = "EDGE PASS" if ok else "EDGE FAIL"
                print(f"{verdict}: {case_label} -> {metrics}")
                if not ok:
                    edge_failures += 1
                    if not args.continue_on_fail:
                        sys.exit(1)

    print(f"Edge tests done. Total cases: {edge_total}. Failures: {edge_failures}.")
    if edge_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
