import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from monoid.embed import MonoidEmbed, MonoidEmbedConfig, MonoidCpuKernel, MonoidCpuConfig


def _compute_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict:
    diff = (ref - out).abs()
    max_abs_err = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs_err = float(diff.mean().item()) if diff.numel() else 0.0
    cosine = float(F.cosine_similarity(ref, out, dim=-1).mean().item()) if ref.numel() else 0.0
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "cosine": cosine,
    }


def _build_checkpoint(preset: str, out_dir: Path) -> tuple[MonoidEmbedConfig, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = MonoidEmbedConfig.from_preset(preset)
    model = MonoidEmbed(config)
    path = out_dir / f"{preset}_stacked_int8.pt"
    torch.save({"model": model.state_dict(), "model_config": config}, path)
    return config, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for stacked quant int8 output.")
    parser.add_argument("--preset", default="small_l2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="tmp/stacked_int8_sanity")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config, ckpt_path = _build_checkpoint(args.preset, Path(args.out_dir))
    kernel_quant = MonoidCpuKernel.from_checkpoint(str(ckpt_path), config=MonoidCpuConfig(emit_int8=False))
    kernel_int8 = MonoidCpuKernel.from_checkpoint(str(ckpt_path), config=MonoidCpuConfig(emit_int8=True))

    tokens = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
    lengths = torch.full((args.batch_size,), args.seq_len, dtype=torch.long)

    out_quant = kernel_quant.forward(tokens, lengths)
    out_int8 = kernel_int8.forward(tokens, lengths)
    if not isinstance(out_int8, (list, tuple)) or len(out_int8) != 3:
        raise RuntimeError("Expected (float, int8, scale_q15) from stacked int8 path.")

    float_out, int8_out, scale_q15 = out_int8
    float_match = _compute_metrics(out_quant, float_out)
    print(f"float_match: {float_match}")

    scale = scale_q15.float() / 32768.0
    mask = scale > 0
    if mask.any():
        dequant = int8_out.float() / scale.unsqueeze(1)
        dequant_metrics = _compute_metrics(float_out[mask], dequant[mask])
        print(f"dequant_metrics: {dequant_metrics}")
    else:
        print("dequant_metrics: skipped (all scales are 0)")

    float_full = kernel_quant.forward_full_precision(tokens, lengths)
    float_vs_quant = _compute_metrics(float_full, out_quant)
    print(f"float_vs_quant: {float_vs_quant}")

    if config.n_layers == 1:
        pool_strategy = 0 if kernel_quant.config.pool_strategy == "mean" else 1
        ext = kernel_quant._ext
        stacked = ext.monoid_forward_quantized_stacked(
            tokens,
            lengths,
            kernel_quant.a_q15,
            kernel_quant.b_int8,
            kernel_quant.tanh_lut,
            kernel_quant.exchange_weight,
            kernel_quant.ln_weight,
            kernel_quant.ln_bias,
            kernel_quant.proj_weight,
            kernel_quant.exchange_shift,
            kernel_quant.config.microblock_size,
            kernel_quant.config.n_tiles,
            kernel_quant.config.tile_dim,
            kernel_quant.config.activation_shift,
            kernel_quant.config.activation_T_q15,
            kernel_quant.config.b_shift,
            kernel_quant.config.exchange_every,
            kernel_quant.config.inj_shift,
            int(kernel_quant.config.use_exchange),
            int(kernel_quant.config.use_second_activation),
            pool_strategy,
            int(kernel_quant.config.normalize_output),
        )
        stacked_match = _compute_metrics(out_quant, stacked)
        print(f"stacked_vs_single: {stacked_match}")


if __name__ == "__main__":
    main()
