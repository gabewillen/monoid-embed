from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional
import math
import time

import torch

from ..model import MonoidEmbed, MonoidEmbedConfig
from .extension import load_monoid_cpu_ext


@dataclass
class MonoidCpuConfig:
    microblock_size: int = 256
    n_tiles: int = 8
    tile_dim: int = 64
    activation_shift: int = 8
    activation_T_q15: int = 24576
    activation_T: float = 1.0
    b_shift: int = 0
    auto_b_shift: bool = False
    exchange_every: int = 1
    inj_shift: int = 3
    use_exchange: bool = True
    use_second_activation: bool = False
    pool_strategy: str = "mean"
    normalize_output: bool = True
    emit_int8: bool = False
    fast_math: Optional[bool] = None
    threads: Optional[int] = None


class MonoidCpuKernel:
    def __init__(
        self,
        a_q15: torch.Tensor,
        b_int8: torch.Tensor,
        tanh_lut: torch.Tensor,
        exchange_weight: torch.Tensor,
        a_f32: torch.Tensor,
        b_f32: torch.Tensor,
        exchange_weight_f32: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        config: MonoidCpuConfig,
    ):
        self.a_q15 = a_q15.contiguous()
        self.b_int8 = b_int8.contiguous()
        self.tanh_lut = tanh_lut.contiguous()
        self.exchange_weight = exchange_weight.contiguous()
        self.a_f32 = a_f32.contiguous()
        self.b_f32 = b_f32.contiguous()
        self.exchange_weight_f32 = exchange_weight_f32.contiguous()
        self.ln_weight = ln_weight.contiguous()
        self.ln_bias = ln_bias.contiguous()
        self.proj_weight = proj_weight.contiguous()
        self.config = config
        threads = config.threads
        if threads is None:
            env_threads = os.getenv("MONOID_CPU_THREADS")
            if env_threads:
                try:
                    threads = int(env_threads)
                except ValueError:
                    threads = None
        if threads is not None:
            torch.set_num_threads(threads)
        if self.a_f32.dim() == 3:
            self.n_layers = int(self.a_f32.size(0))
        else:
            self.n_layers = 1
        fast_math = config.fast_math
        self._ext = load_monoid_cpu_ext(
            fast_math=fast_math,
            fast_tanh=fast_math,
        )
        self._quant_shapes_validated = False

    def _validate_quant_shapes(self) -> None:
        if self._quant_shapes_validated:
            return
        d_state = self.config.n_tiles * self.config.tile_dim
        if self.a_q15.dim() not in {2, 3}:
            raise ValueError("a_q15 must be 2D or 3D.")
        if self.b_int8.dim() not in {2, 3}:
            raise ValueError("b_int8 must be 2D or 3D.")
        if self.a_q15.shape[-1] != d_state:
            raise ValueError("a_q15 last dim must match n_tiles * tile_dim.")
        if self.b_int8.shape[-1] != d_state:
            raise ValueError("b_int8 last dim must match n_tiles * tile_dim.")
        if self.n_layers > 1:
            if self.a_q15.dim() != 3 or self.b_int8.dim() != 3:
                raise ValueError("Stacked quant requires 3D a_q15 and b_int8.")
            if self.a_q15.size(0) != self.n_layers:
                raise ValueError("a_q15 layer count must match n_layers.")
            if self.b_int8.size(0) != self.n_layers:
                raise ValueError("b_int8 layer count must match n_layers.")
            if self.b_int8.size(1) != self.a_q15.size(1):
                raise ValueError("b_int8 vocab dim must match a_q15.")
        else:
            if self.a_q15.dim() != 2 or self.b_int8.dim() != 2:
                raise ValueError("Single-layer quant requires 2D a_q15 and b_int8.")
            if self.b_int8.size(0) != self.a_q15.size(0):
                raise ValueError("b_int8 vocab dim must match a_q15.")

        if self.exchange_weight.numel() > 0:
            if self.n_layers > 1:
                if self.exchange_weight.dim() != 3 or self.exchange_weight.size(0) != self.n_layers:
                    raise ValueError("exchange_weight must be (n_layers, dim, dim).")
                if self.exchange_weight.size(1) != self.exchange_weight.size(2):
                    raise ValueError("exchange_weight must be square.")
                exchange_dim = self.exchange_weight.size(1)
            else:
                if self.exchange_weight.dim() != 2:
                    raise ValueError("exchange_weight must be 2D for single-layer.")
                if self.exchange_weight.size(0) != self.exchange_weight.size(1):
                    raise ValueError("exchange_weight must be square.")
                exchange_dim = self.exchange_weight.size(0)
            if self.exchange_shift.numel() > 0:
                if self.n_layers > 1:
                    if self.exchange_shift.dim() != 2 or self.exchange_shift.size(0) != self.n_layers:
                        raise ValueError("exchange_shift must be (n_layers, exchange_dim).")
                    if self.exchange_shift.size(1) != exchange_dim:
                        raise ValueError("exchange_shift dim must match exchange_weight.")
                else:
                    if self.exchange_shift.dim() != 1 or self.exchange_shift.size(0) != exchange_dim:
                        raise ValueError("exchange_shift dim must match exchange_weight.")
        if self.ln_weight.numel() > 0:
            if self.ln_weight.dim() != 2:
                raise ValueError("ln_weight must be (n_layers, d_state).")
            if self.ln_weight.size(0) != self.n_layers or self.ln_weight.size(1) != d_state:
                raise ValueError("ln_weight shape mismatch.")
        if self.ln_bias.numel() > 0:
            if self.ln_bias.dim() != 2:
                raise ValueError("ln_bias must be (n_layers, d_state).")
            if self.ln_bias.size(0) != self.n_layers or self.ln_bias.size(1) != d_state:
                raise ValueError("ln_bias shape mismatch.")
        if self.proj_weight.numel() > 0:
            if self.proj_weight.dim() != 2 or self.proj_weight.size(1) != d_state:
                raise ValueError("proj_weight must be (out_dim, d_state).")

        self._quant_shapes_validated = True

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[MonoidCpuConfig] = None,
    ) -> "MonoidCpuKernel":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = state
        if "model_state_dict" in state:
            model_state = state["model_state_dict"]
        elif "model" in state:
            model_state = state["model"]
        if isinstance(model_state, dict):
            has_orig_mod = any(key.startswith("_orig_mod.") or "._orig_mod." in key for key in model_state)
            if has_orig_mod:
                remapped: dict = {}
                for key, value in model_state.items():
                    new_key = key.replace("._orig_mod.", ".")
                    if new_key.startswith("_orig_mod."):
                        new_key = new_key[len("_orig_mod.") :]
                    remapped[new_key] = value
                model_state = remapped

        model_config = _load_model_config(state, model_state)
        model = MonoidEmbed(model_config)
        model.load_state_dict(model_state, strict=False)

        a_list = [block._compute_a().detach().cpu() for block in model.blocks]
        b_list = [block.b.detach().cpu() for block in model.blocks]
        a = torch.stack(a_list, dim=0)
        b = torch.stack(b_list, dim=0)

        a_q15_list = [block._compute_a_q15().detach().cpu() for block in model.blocks]
        b_shift = model_config.b_shift
        if config is not None:
            if config.auto_b_shift:
                max_abs = 0.0
                for block in model.blocks:
                    max_abs = max(max_abs, float(block.b.detach().abs().max().item()))
                if max_abs > 0.0:
                    b_shift = int(math.floor(math.log2(127.0 / max_abs)))
                    b_shift = max(0, min(15, b_shift))
            else:
                b_shift = int(config.b_shift)
        scale = float(2 ** b_shift)
        b_int8_list = [
            torch.clamp((block.b.detach().cpu() / scale).round(), -127, 127).to(torch.int8)
            for block in model.blocks
        ]
        if model.config.n_layers > 1:
            a_q15 = torch.stack(a_q15_list, dim=0)
            b_int8 = torch.stack(b_int8_list, dim=0)
        else:
            a_q15 = a_q15_list[0]
            b_int8 = b_int8_list[0]
        tanh_lut = model.tanh_lut.detach().cpu()

        exchange_weight = torch.zeros(0, dtype=torch.int8)
        exchange_weight_f32 = torch.zeros(0, dtype=torch.float32)
        exchange_shift = torch.zeros(0, dtype=torch.int8)
        if model.blocks[0].exchange is not None:
            weights = []
            for block in model.blocks:
                weight_orig = None
                if (
                    hasattr(block.exchange, "parametrizations")
                    and hasattr(block.exchange.parametrizations, "weight")
                    and hasattr(block.exchange.parametrizations.weight, "original")
                ):
                    weight_orig = block.exchange.parametrizations.weight.original
                elif hasattr(block.exchange, "weight_orig"):
                    weight_orig = block.exchange.weight_orig
                else:
                    weight_orig = block.exchange.weight
                weight = weight_orig.detach().cpu().float()
                if weight.numel():
                    sigma = float(torch.linalg.svdvals(weight).max().item())
                    scale = max(1.0, sigma)
                    weight = weight / scale
                weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
                weights.append(weight)
            exchange_weight_f32 = torch.stack(weights, dim=0).contiguous()
            q_layers = []
            shift_layers = []
            for layer_weight in exchange_weight_f32:
                rows = layer_weight.size(0)
                q = torch.zeros_like(layer_weight, dtype=torch.int8)
                shifts = torch.zeros(rows, dtype=torch.int8)
                for r in range(rows):
                    row = layer_weight[r]
                    max_abs = row.abs().max().item()
                    if not math.isfinite(max_abs) or max_abs <= 0.0:
                        shift = 0
                    else:
                        shift = int(math.floor(math.log2(127.0 / max_abs)))
                        shift = max(0, min(15, shift))
                    if not math.isfinite(max_abs):
                        q[r] = torch.zeros_like(q[r])
                    else:
                        q[r] = torch.clamp((row * (2 ** shift)).round(), -127, 127).to(torch.int8)
                    shifts[r] = shift
                q_layers.append(q)
                shift_layers.append(shifts)
            if model.config.n_layers > 1:
                exchange_weight = torch.stack(q_layers, dim=0)
                exchange_shift = torch.stack(shift_layers, dim=0).contiguous()
            else:
                exchange_weight = q_layers[0]
                exchange_shift = shift_layers[0].contiguous()

        ln_weight = torch.zeros(0, dtype=torch.float32)
        ln_bias = torch.zeros(0, dtype=torch.float32)
        if model.config.n_layers > 1:
            ln_weights = []
            ln_biases = []
            for ln in model.norms:
                if ln.weight is None:
                    ln_weights.append(torch.ones(model.config.d_state, dtype=torch.float32))
                else:
                    ln_weights.append(ln.weight.detach().cpu().float())
                if ln.bias is None:
                    ln_biases.append(torch.zeros(model.config.d_state, dtype=torch.float32))
                else:
                    ln_biases.append(ln.bias.detach().cpu().float())
            ln_weight = torch.stack(ln_weights, dim=0).contiguous()
            ln_bias = torch.stack(ln_biases, dim=0).contiguous()

        proj_weight = torch.zeros(0, dtype=torch.float32)
        if model.proj is not None:
            proj_weight = model.proj.weight.detach().cpu().float().contiguous()

        cfg = MonoidCpuConfig(
            microblock_size=model_config.microblock_size,
            n_tiles=model_config.n_tiles,
            tile_dim=model_config.tile_dim,
            activation_shift=model_config.activation_shift,
            activation_T_q15=model_config.activation_T_q15,
            activation_T=model_config.activation_T,
            b_shift=b_shift,
            exchange_every=model_config.exchange_every,
            inj_shift=model_config.inj_shift,
            use_exchange=model_config.use_exchange,
            use_second_activation=model_config.use_second_activation,
            pool_strategy=model_config.pool_strategy,
            normalize_output=model_config.normalize_output,
            emit_int8=model_config.emit_int8,
        )
        if config is not None:
            cfg.pool_strategy = config.pool_strategy
            cfg.normalize_output = config.normalize_output
            cfg.emit_int8 = config.emit_int8
            cfg.use_exchange = config.use_exchange
            cfg.use_second_activation = config.use_second_activation
            cfg.activation_shift = config.activation_shift
            cfg.activation_T_q15 = config.activation_T_q15
            cfg.activation_T = config.activation_T
            cfg.b_shift = b_shift
            cfg.auto_b_shift = config.auto_b_shift
            cfg.exchange_every = config.exchange_every
            cfg.inj_shift = config.inj_shift
            cfg.fast_math = config.fast_math
            cfg.threads = config.threads

        kernel = cls(
            a_q15=a_q15,
            b_int8=b_int8,
            tanh_lut=tanh_lut,
            exchange_weight=exchange_weight,
            a_f32=a.float(),
            b_f32=b.float(),
            exchange_weight_f32=exchange_weight_f32,
            ln_weight=ln_weight,
            ln_bias=ln_bias,
            proj_weight=proj_weight,
            config=cfg,
        )
        kernel.exchange_shift = exchange_shift
        return kernel

    def forward(self, byte_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        byte_tokens = byte_tokens.contiguous()
        if lengths is None:
            lengths = torch.full(
                (byte_tokens.size(0),),
                byte_tokens.size(1),
                dtype=torch.long,
            )
        lengths = lengths.contiguous()

        pool_strategy = 0 if self.config.pool_strategy == "mean" else 1
        if not hasattr(self, "exchange_shift"):
            self.exchange_shift = torch.zeros(0, dtype=torch.int8)
        self._validate_quant_shapes()
        if self.n_layers > 1:
            if self.config.emit_int8:
                return self._ext.monoid_forward_quantized_stacked_int8(
                    byte_tokens,
                    lengths,
                    self.a_q15,
                    self.b_int8,
                    self.tanh_lut,
                    self.exchange_weight,
                    self.ln_weight,
                    self.ln_bias,
                    self.proj_weight,
                    self.exchange_shift,
                    self.config.microblock_size,
                    self.config.n_tiles,
                    self.config.tile_dim,
                    self.config.activation_shift,
                    self.config.activation_T_q15,
                    self.config.b_shift,
                    self.config.exchange_every,
                    self.config.inj_shift,
                    int(self.config.use_exchange),
                    int(self.config.use_second_activation),
                    pool_strategy,
                    int(self.config.normalize_output),
                )
            return self._ext.monoid_forward_quantized_stacked(
                byte_tokens,
                lengths,
                self.a_q15,
                self.b_int8,
                self.tanh_lut,
                self.exchange_weight,
                self.ln_weight,
                self.ln_bias,
                self.proj_weight,
                self.exchange_shift,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                self.config.activation_shift,
                self.config.activation_T_q15,
                self.config.b_shift,
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )
        if self.config.emit_int8:
            return self._ext.monoid_forward_quantized_int8(
                byte_tokens,
                lengths,
                self.a_q15,
                self.b_int8,
                self.tanh_lut,
                self.exchange_weight,
                self.exchange_shift,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                self.config.activation_shift,
                self.config.activation_T_q15,
                self.config.b_shift,
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )
        return self._ext.monoid_forward_quantized(
            byte_tokens,
            lengths,
            self.a_q15,
            self.b_int8,
            self.tanh_lut,
            self.exchange_weight,
            self.exchange_shift,
            self.config.microblock_size,
            self.config.n_tiles,
            self.config.tile_dim,
            self.config.activation_shift,
            self.config.activation_T_q15,
            self.config.b_shift,
            self.config.exchange_every,
            self.config.inj_shift,
            int(self.config.use_exchange),
            int(self.config.use_second_activation),
            pool_strategy,
            int(self.config.normalize_output),
        )

    def forward_quantized_stacked(
        self, byte_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        byte_tokens = byte_tokens.contiguous()
        if lengths is None:
            lengths = torch.full(
                (byte_tokens.size(0),),
                byte_tokens.size(1),
                dtype=torch.long,
            )
        lengths = lengths.contiguous()

        if self.config.emit_int8:
            return self._ext.monoid_forward_quantized_stacked_int8(
                byte_tokens,
                lengths,
                self.a_q15,
                self.b_int8,
                self.tanh_lut,
                self.exchange_weight,
                self.ln_weight,
                self.ln_bias,
                self.proj_weight,
                self.exchange_shift,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                self.config.activation_shift,
                self.config.activation_T_q15,
                self.config.b_shift,
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )

        pool_strategy = 0 if self.config.pool_strategy == "mean" else 1
        if not hasattr(self, "exchange_shift"):
            self.exchange_shift = torch.zeros(0, dtype=torch.int8)
        self._validate_quant_shapes()
        return self._ext.monoid_forward_quantized_stacked(
            byte_tokens,
            lengths,
            self.a_q15,
            self.b_int8,
            self.tanh_lut,
            self.exchange_weight,
            self.ln_weight,
            self.ln_bias,
            self.proj_weight,
            self.exchange_shift,
            self.config.microblock_size,
            self.config.n_tiles,
            self.config.tile_dim,
            self.config.activation_shift,
            self.config.activation_T_q15,
            self.config.b_shift,
            self.config.exchange_every,
            self.config.inj_shift,
            int(self.config.use_exchange),
            int(self.config.use_second_activation),
            pool_strategy,
            int(self.config.normalize_output),
        )

    def forward_quantized_stacked_int8(
        self, byte_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        byte_tokens = byte_tokens.contiguous()
        if lengths is None:
            lengths = torch.full(
                (byte_tokens.size(0),),
                byte_tokens.size(1),
                dtype=torch.long,
            )
        lengths = lengths.contiguous()

        pool_strategy = 0 if self.config.pool_strategy == "mean" else 1
        if not hasattr(self, "exchange_shift"):
            self.exchange_shift = torch.zeros(0, dtype=torch.int8)
        self._validate_quant_shapes()
        return self._ext.monoid_forward_quantized_stacked_int8(
            byte_tokens,
            lengths,
            self.a_q15,
            self.b_int8,
            self.tanh_lut,
            self.exchange_weight,
            self.ln_weight,
            self.ln_bias,
            self.proj_weight,
            self.exchange_shift,
            self.config.microblock_size,
            self.config.n_tiles,
            self.config.tile_dim,
            self.config.activation_shift,
            self.config.activation_T_q15,
            self.config.b_shift,
            self.config.exchange_every,
            self.config.inj_shift,
            int(self.config.use_exchange),
            int(self.config.use_second_activation),
            pool_strategy,
            int(self.config.normalize_output),
        )

    def forward_full_precision(self, byte_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        byte_tokens = byte_tokens.contiguous()
        if lengths is None:
            lengths = torch.full(
                (byte_tokens.size(0),),
                byte_tokens.size(1),
                dtype=torch.long,
            )
        lengths = lengths.contiguous()

        pool_strategy = 0 if self.config.pool_strategy == "mean" else 1
        if (
            self.n_layers == 1
            and self.ln_weight.numel() == 0
            and self.ln_bias.numel() == 0
            and self.proj_weight.numel() > 0
        ):
            a_f32 = self.a_f32
            b_f32 = self.b_f32
            exchange_weight_f32 = self.exchange_weight_f32
            if a_f32.dim() == 3:
                a_f32 = a_f32[0].contiguous()
            if b_f32.dim() == 3:
                b_f32 = b_f32[0].contiguous()
            if exchange_weight_f32.dim() == 3:
                exchange_weight_f32 = exchange_weight_f32[0].contiguous()
            out = self._ext.monoid_forward_float(
                byte_tokens,
                lengths,
                a_f32,
                b_f32,
                exchange_weight_f32,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                float(self.config.activation_T),
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )
            proj = out @ self.proj_weight.t()
            if self.config.normalize_output:
                proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
            return proj

        if self.n_layers > 1 or self.proj_weight.numel() > 0 or self.ln_weight.numel() > 0:
            return self._ext.monoid_forward_float_stacked(
                byte_tokens,
                lengths,
                self.a_f32,
                self.b_f32,
                self.exchange_weight_f32,
                self.ln_weight,
                self.ln_bias,
                self.proj_weight,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                float(self.config.activation_T),
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )

        a_f32 = self.a_f32
        b_f32 = self.b_f32
        exchange_weight_f32 = self.exchange_weight_f32
        if a_f32.dim() == 3:
            a_f32 = a_f32[0].contiguous()
        if b_f32.dim() == 3:
            b_f32 = b_f32[0].contiguous()
        if exchange_weight_f32.dim() == 3:
            exchange_weight_f32 = exchange_weight_f32[0].contiguous()
        return self._ext.monoid_forward_float(
            byte_tokens,
            lengths,
            a_f32,
            b_f32,
            exchange_weight_f32,
            self.config.microblock_size,
            self.config.n_tiles,
            self.config.tile_dim,
            float(self.config.activation_T),
            self.config.exchange_every,
            self.config.inj_shift,
            int(self.config.use_exchange),
            int(self.config.use_second_activation),
            pool_strategy,
            int(self.config.normalize_output),
        )

    def forward_full_precision_profile(
        self,
        byte_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        byte_tokens = byte_tokens.contiguous()
        if lengths is None:
            lengths = torch.full(
                (byte_tokens.size(0),),
                byte_tokens.size(1),
                dtype=torch.long,
            )
        lengths = lengths.contiguous()

        pool_strategy = 0 if self.config.pool_strategy == "mean" else 1
        if (
            self.n_layers == 1
            and self.ln_weight.numel() == 0
            and self.ln_bias.numel() == 0
            and self.proj_weight.numel() > 0
        ):
            a_f32 = self.a_f32
            b_f32 = self.b_f32
            exchange_weight_f32 = self.exchange_weight_f32
            if a_f32.dim() == 3:
                a_f32 = a_f32[0].contiguous()
            if b_f32.dim() == 3:
                b_f32 = b_f32[0].contiguous()
            if exchange_weight_f32.dim() == 3:
                exchange_weight_f32 = exchange_weight_f32[0].contiguous()
            out, timing = self._ext.monoid_forward_float_profile(
                byte_tokens,
                lengths,
                a_f32,
                b_f32,
                exchange_weight_f32,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                float(self.config.activation_T),
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )
            proj_start = time.perf_counter()
            proj = out @ self.proj_weight.t()
            if self.config.normalize_output:
                proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
            proj_ms = (time.perf_counter() - proj_start) * 1000.0
            timing = dict(timing)
            timing["proj_ms"] = timing.get("proj_ms", 0.0) + proj_ms
            timing["total_ms"] = timing.get("total_ms", 0.0) + proj_ms
            return proj, timing

        if self.n_layers > 1 or self.proj_weight.numel() > 0 or self.ln_weight.numel() > 0:
            return self._ext.monoid_forward_float_stacked_profile(
                byte_tokens,
                lengths,
                self.a_f32,
                self.b_f32,
                self.exchange_weight_f32,
                self.ln_weight,
                self.ln_bias,
                self.proj_weight,
                self.config.microblock_size,
                self.config.n_tiles,
                self.config.tile_dim,
                float(self.config.activation_T),
                self.config.exchange_every,
                self.config.inj_shift,
                int(self.config.use_exchange),
                int(self.config.use_second_activation),
                pool_strategy,
                int(self.config.normalize_output),
            )

        a_f32 = self.a_f32
        b_f32 = self.b_f32
        exchange_weight_f32 = self.exchange_weight_f32
        if a_f32.dim() == 3:
            a_f32 = a_f32[0].contiguous()
        if b_f32.dim() == 3:
            b_f32 = b_f32[0].contiguous()
        if exchange_weight_f32.dim() == 3:
            exchange_weight_f32 = exchange_weight_f32[0].contiguous()
        return self._ext.monoid_forward_float_profile(
            byte_tokens,
            lengths,
            a_f32,
            b_f32,
            exchange_weight_f32,
            self.config.microblock_size,
            self.config.n_tiles,
            self.config.tile_dim,
            float(self.config.activation_T),
            self.config.exchange_every,
            self.config.inj_shift,
            int(self.config.use_exchange),
            int(self.config.use_second_activation),
            pool_strategy,
            int(self.config.normalize_output),
        )


def _build_config_from_dict(cfg_dict: dict) -> MonoidEmbedConfig:
    preset_env = os.environ.pop("MONOID_PRESET", None)
    try:
        return MonoidEmbedConfig(**cfg_dict)
    finally:
        if preset_env is not None:
            os.environ["MONOID_PRESET"] = preset_env


def _infer_model_config_from_state(model_state: dict) -> MonoidEmbedConfig:
    block_ids = set()
    for key in model_state:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    n_layers = max(block_ids) + 1 if block_ids else 1

    d_state = None
    tile_dim = None
    if "proj.weight" in model_state:
        d_state = int(model_state["proj.weight"].shape[1])
    for key, tensor in model_state.items():
        if key.endswith("a_raw"):
            tile_dim = int(tensor.shape[0])
            if d_state is None:
                d_state = int(tensor.shape[1])
            break
    if d_state is None:
        raise ValueError("Unable to infer d_state from checkpoint.")

    exchange_dim = None
    use_exchange = False
    for key, tensor in model_state.items():
        if key.endswith("exchange.parametrizations.weight.original") or key.endswith("exchange.weight"):
            exchange_dim = int(tensor.shape[0])
            use_exchange = True
            break
    if exchange_dim is None:
        exchange_dim = d_state // 16

    microblock_size = 256 if d_state <= 512 else 128
    preset_env = os.environ.pop("MONOID_PRESET", None)
    try:
        model_config = MonoidEmbedConfig(
            n_layers=n_layers,
            d_state=d_state,
            microblock_size=microblock_size,
            exchange_dim=exchange_dim,
        )
    finally:
        if preset_env is not None:
            os.environ["MONOID_PRESET"] = preset_env

    if tile_dim is not None and tile_dim != model_config.tile_dim:
        raise ValueError(
            f"Checkpoint tile_dim={tile_dim} does not match config tile_dim={model_config.tile_dim}."
        )
    model_config.use_exchange = use_exchange
    return model_config


def _load_model_config(state: dict, model_state: dict) -> MonoidEmbedConfig:
    cfg = state.get("model_config")
    if isinstance(cfg, MonoidEmbedConfig):
        return cfg
    if isinstance(cfg, dict):
        return _build_config_from_dict(cfg)
    return _infer_model_config_from_state(model_state)


__all__ = ["MonoidCpuKernel", "MonoidCpuConfig"]
