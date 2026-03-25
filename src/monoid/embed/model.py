import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MonoidEmbedConfig:
    _PRESET_SPECS = {
        "small": {"n_layers": 1, "d_state": 512, "microblock_size": 256},
        "small_2l": {"n_layers": 2, "d_state": 512, "microblock_size": 256},
        "small_l2": {"n_layers": 2, "d_state": 512, "microblock_size": 256},
        "small_l3": {"n_layers": 3, "d_state": 512, "microblock_size": 256},
        "small_l4": {"n_layers": 4, "d_state": 512, "microblock_size": 256},
        "small_l5": {"n_layers": 5, "d_state": 512, "microblock_size": 256},
        "mbe_30m": {"n_layers": 2, "d_state": 16384, "microblock_size": 512},
        "mbe_6m": {"n_layers": 2, "d_state": 4096, "microblock_size": 512},
        "medium": {"n_layers": 1, "d_state": 2048, "microblock_size": 64},
        "medium_deep": {"n_layers": 8, "d_state": 512, "microblock_size": 128},
        "base": {"n_layers": 15, "d_state": 2048, "microblock_size": 128},
        "large": {"n_layers": 27, "d_state": 2048, "microblock_size": 128},
        "xlarge": {"n_layers": 216, "d_state": 2048, "microblock_size": 128},
    }
    n_layers: int = 1
    vocab_size: int = 256
    d_state: int = 512
    n_tiles: int = 8
    tile_dim: int = 64
    microblock_size: int = 256
    exchange_dim: int = 32
    matryoshka_dims: Tuple[int, ...] = (512, 256, 128)

    use_quantized: bool = False
    activation_shift: int = 8
    activation_T_q15: int = 24576
    b_shift: int = 0
    pool_strategy: str = "mean"
    normalize_output: bool = True
    emit_int8: bool = False

    a_min: float = 0.90
    a_max: float = 0.999

    activation_T: float = 1.0
    use_exchange: bool = True
    exchange_every: int = 1
    use_second_activation: bool = False
    inj_shift: int = 3

    def __post_init__(self) -> None:
        preset = os.getenv("MONOID_PRESET")
        if preset:
            preset = preset.lower()
            spec = self._PRESET_SPECS.get(preset)
            if spec is None:
                raise ValueError(f"Unknown MONOID_PRESET: {preset}")
            self._apply_preset(
                spec["n_layers"],
                spec["d_state"],
                spec["microblock_size"],
            )
        self._validate_config()

    def _apply_preset(
        self,
        n_layers: int,
        d_state: int,
        microblock_size: int,
    ) -> None:
        self.n_layers = n_layers
        self.d_state = d_state
        self.microblock_size = microblock_size
        self.n_tiles = 8
        self.tile_dim = d_state // self.n_tiles
        self.exchange_dim = d_state // 16

    def _validate_config(self) -> None:
        if self.n_tiles != 8:
            raise ValueError("MonoidEmbedConfig requires n_tiles=8 for v1.2.2 presets.")
        if self.d_state % self.n_tiles != 0:
            raise ValueError("d_state must be divisible by n_tiles.")
        self.tile_dim = self.d_state // self.n_tiles
        if self.tile_dim <= 0 or (self.tile_dim & (self.tile_dim - 1)) != 0:
            raise ValueError("tile_dim must be a power of two.")
        if self.tile_dim % 4 != 0:
            raise ValueError("tile_dim must be divisible by 4.")
        expected_exchange = self.d_state // 16
        if self.exchange_dim != expected_exchange:
            raise ValueError(f"exchange_dim must be d_state/16 ({expected_exchange}).")
        if self.microblock_size < 64 or (self.microblock_size & (self.microblock_size - 1)) != 0:
            raise ValueError("microblock_size must be a power of two and >= 64.")
            
    @classmethod
    def small(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("small")

    @classmethod
    def small_2l(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("small_2l")

    @classmethod
    def medium(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("medium")

    @classmethod
    def medium_deep(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("medium_deep")


    @classmethod
    def base(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("base")

    @classmethod
    def large(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("large")

    @classmethod
    def xlarge(cls) -> "MonoidEmbedConfig":
        return cls.from_preset("xlarge")

    @classmethod
    def from_preset(cls, name: str) -> "MonoidEmbedConfig":
        spec = cls._PRESET_SPECS.get(name)
        if spec is None:
            raise ValueError(f"Unknown preset: {name}")
        return cls(
            n_layers=spec["n_layers"],
            d_state=spec["d_state"],
            microblock_size=spec["microblock_size"],
            exchange_dim=spec["d_state"] // 16,
        )


def _tanh_squash(x: torch.Tensor, T: float) -> torch.Tensor:
    if T <= 0.0:
        return x
    return T * torch.tanh(x / T)


def _update_activation_saturation(stats: dict, state_tiles: torch.Tensor, threshold: float) -> None:
    sat = (state_tiles.abs() > threshold).sum().item()
    total = state_tiles.numel()
    stats["activation_sat_count"] = stats.get("activation_sat_count", 0.0) + sat
    stats["activation_elem_count"] = stats.get("activation_elem_count", 0.0) + total


def _butterfly_mix(tile: torch.Tensor) -> torch.Tensor:
    out = tile.clone()
    tile_dim = out.size(-1)
    if tile_dim <= 0 or (tile_dim & (tile_dim - 1)) != 0:
        raise ValueError("tile_dim must be a power of two.")
    step = 1
    stages = int(math.log2(tile_dim))
    for _ in range(stages):
        for start in range(0, out.size(-1), step * 2):
            a = out[..., start:start + step].clone()
            b = out[..., start + step:start + step * 2].clone()
            out[..., start:start + step] = a + b
            out[..., start + step:start + step * 2] = a - b
        step *= 2
    return out


def _butterfly_mix_int16(tile: torch.Tensor) -> torch.Tensor:
    out = tile.clone()
    tile_dim = out.size(-1)
    if tile_dim <= 0 or (tile_dim & (tile_dim - 1)) != 0:
        raise ValueError("tile_dim must be a power of two.")
    step = 1
    stages = int(math.log2(tile_dim))
    for _ in range(stages):
        for start in range(0, out.size(-1), step * 2):
            a = out[..., start:start + step].to(torch.int32)
            b = out[..., start + step:start + step * 2].to(torch.int32)
            s = a + b
            d = a - b
            s = s.clamp(-32768, 32767).to(torch.int16)
            d = d.clamp(-32768, 32767).to(torch.int16)
            out[..., start:start + step] = s
            out[..., start + step:start + step * 2] = d
        step *= 2
    return out


class MonoidBlock(nn.Module):
    def __init__(self, config: MonoidEmbedConfig):
        super().__init__()
        if config.n_tiles * config.tile_dim != config.d_state:
            raise ValueError("d_state must equal n_tiles * tile_dim")
        self.config = config

        d_state = config.d_state
        self.a_raw = nn.Parameter(torch.zeros(config.vocab_size, d_state))
        self.b = nn.Parameter(torch.zeros(config.vocab_size, d_state))

        a_min = torch.full((d_state,), config.a_min)
        a_max = torch.full((d_state,), config.a_max)
        self.register_buffer("a_min", a_min)
        self.register_buffer("a_max", a_max)

        self.exchange = None
        if config.use_exchange:
            exchange_dim = config.exchange_dim
            exchange = nn.Linear(exchange_dim, exchange_dim, bias=False)
            exchange.weight.data.normal_(mean=0.0, std=1e-3)
            self.exchange = nn.utils.parametrizations.spectral_norm(exchange)

        tanh_lut = torch.zeros(256, dtype=torch.int16)
        for i in range(256):
            x = (i - 128) / 128.0
            val = math.tanh(x)
            tanh_lut[i] = int(round(val * 32767.0))
        self.register_buffer("tanh_lut", tanh_lut)

        self._init_a_raw()
        self.b.data.normal_(mean=0.0, std=1e-3)

    def _init_a_raw(self) -> None:
        d_state = self.config.d_state
        half_life = torch.logspace(math.log10(8.0), math.log10(8192.0), d_state)
        a_base = torch.pow(0.5, 1.0 / half_life)
        a_min = self.a_min
        a_max = self.a_max
        ratio = (a_base - a_min) / (a_max - a_min)
        ratio = ratio.clamp(1e-4, 1.0 - 1e-4)
        a_raw_init = torch.log(ratio / (1.0 - ratio))
        self.a_raw.data.copy_(a_raw_init.unsqueeze(0).repeat(self.config.vocab_size, 1))

    def _compute_a(self) -> torch.Tensor:
        return self.a_min + (self.a_max - self.a_min) * torch.sigmoid(self.a_raw)

    def _compute_a_q15(self) -> torch.Tensor:
        a = self._compute_a()
        return torch.clamp((a * 32768.0).round(), -32768, 32767).to(torch.int16)

    def _compute_b_int8(self) -> torch.Tensor:
        scale = float(2 ** self.config.b_shift)
        return torch.clamp((self.b / scale).round(), -127, 127).to(torch.int8)

    def _apply_exchange(self, state_tiles: torch.Tensor, phase: int, stats: Optional[dict] = None) -> torch.Tensor:
        if self.exchange is None:
            return state_tiles
        if getattr(self, "exchange_disabled", False):
            return state_tiles
        batch = state_tiles.size(0)
        tile_dim = self.config.tile_dim
        n_tiles = self.config.n_tiles
        groups = n_tiles // 4
        exchange_dim = self.config.exchange_dim
        if n_tiles % 4 != 0:
            raise ValueError("n_tiles must be divisible by 4 for exchange.")
        if exchange_dim % groups != 0:
            raise ValueError("exchange_dim must be divisible by n_tiles/4.")
        per_group = exchange_dim // groups

        u = torch.zeros(batch, exchange_dim, device=state_tiles.device, dtype=state_tiles.dtype)
        for g in range(groups):
            tiles = state_tiles[:, :, g * 4:(g + 1) * 4]
            local = tiles.permute(0, 2, 1).reshape(batch, -1)
            u_group = torch.zeros(batch, per_group, device=state_tiles.device, dtype=state_tiles.dtype)
            for j in range(per_group):
                start = (4 * j + phase) % (4 * tile_dim)
                idx = torch.tensor([(start + k) % (4 * tile_dim) for k in range(4)], device=state_tiles.device)
                u_group[:, j] = local[:, idx].sum(dim=1)
            u[:, g * per_group:(g + 1) * per_group] = u_group

        weight_orig = None
        if (
            hasattr(self.exchange, "parametrizations")
            and hasattr(self.exchange.parametrizations, "weight")
            and hasattr(self.exchange.parametrizations.weight, "original")
        ):
            weight_orig = self.exchange.parametrizations.weight.original
        elif hasattr(self.exchange, "weight_orig"):
            weight_orig = self.exchange.weight_orig
        else:
            weight_orig = self.exchange.weight
        weight = weight_orig
        if weight_orig is not None and weight_orig.numel():
            with torch.no_grad():
                sigma = float(torch.linalg.svdvals(weight_orig.float()).max().item())
                scale = max(1.0, sigma)
            weight = weight_orig / scale
        if not torch.isfinite(weight).all():
            weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        if stats is not None:
            stats["exchange_weight_used"] = weight
            if weight.requires_grad:
                weight.retain_grad()
        v = F.linear(u, weight)
        inj_scale = 1.0 / float(2 ** self.config.inj_shift)
        exchange_scale = float(getattr(self, "exchange_scale", 1.0))
        exchange_inj_norm_max = float(getattr(self, "exchange_inj_norm_max", 0.0))
        if exchange_scale <= 0.0:
            if stats is not None:
                stats["exchange_scale_sum"] = stats.get("exchange_scale_sum", 0.0) + exchange_scale
                stats["exchange_scale_count"] = stats.get("exchange_scale_count", 0.0) + 1.0
                stats["exchange_inj_norm_max_sum"] = stats.get("exchange_inj_norm_max_sum", 0.0) + exchange_inj_norm_max
                stats["exchange_inj_norm_max_count"] = stats.get("exchange_inj_norm_max_count", 0.0) + 1.0
            return state_tiles
        clamp_scale = 1.0
        inj_norm_raw = 0.0
        if stats is not None or exchange_inj_norm_max > 0.0:
            with torch.no_grad():
                u_norm = float(u.detach().float().norm(dim=1).mean().item()) if u.numel() else 0.0
                v_norm = float(v.detach().float().norm(dim=1).mean().item()) if v.numel() else 0.0
                inj = v.detach().float() * float(inj_scale * exchange_scale)
                inj_norm_raw = float(inj.norm(dim=1).mean().item()) if inj.numel() else 0.0
                if exchange_inj_norm_max > 0.0 and inj_norm_raw > 0.0:
                    clamp_scale = min(1.0, exchange_inj_norm_max / max(inj_norm_raw, 1e-12))
                if stats is not None:
                    inj_norm = inj_norm_raw * clamp_scale
                    stats["exchange_exec_count"] = stats.get("exchange_exec_count", 0.0) + 1.0
                    stats["exchange_u_norm_sum"] = stats.get("exchange_u_norm_sum", 0.0) + u_norm
                    stats["exchange_v_norm_sum"] = stats.get("exchange_v_norm_sum", 0.0) + v_norm
                    stats["exchange_inj_norm_sum"] = stats.get("exchange_inj_norm_sum", 0.0) + inj_norm
                    stats["exchange_inj_norm_raw_sum"] = stats.get("exchange_inj_norm_raw_sum", 0.0) + inj_norm_raw
                    stats["exchange_inj_norm_clamped_sum"] = stats.get("exchange_inj_norm_clamped_sum", 0.0) + inj_norm
                    stats["exchange_scale_sum"] = stats.get("exchange_scale_sum", 0.0) + exchange_scale
                    stats["exchange_scale_count"] = stats.get("exchange_scale_count", 0.0) + 1.0
                    stats["exchange_inj_norm_max_sum"] = stats.get("exchange_inj_norm_max_sum", 0.0) + exchange_inj_norm_max
                    stats["exchange_inj_norm_max_count"] = stats.get("exchange_inj_norm_max_count", 0.0) + 1.0

        for g in range(groups):
            tiles = state_tiles[:, :, g * 4:(g + 1) * 4]
            local = tiles.permute(0, 2, 1).reshape(batch, -1)
            v_group = v[:, g * per_group:(g + 1) * per_group]
            for j in range(per_group):
                start = (4 * j + phase) % (4 * tile_dim)
                idx = [(start + k) % (4 * tile_dim) for k in range(4)]
                inj = v_group[:, j] * inj_scale * exchange_scale * clamp_scale
                local[:, idx[0]] += inj
                local[:, idx[1]] += inj
                local[:, idx[2]] -= inj
                local[:, idx[3]] -= inj
            state_tiles[:, :, g * 4:(g + 1) * 4] = local.view(batch, 4, tile_dim).permute(0, 2, 1)
        return state_tiles

    def _emit_embedding(self, state_tiles: torch.Tensor, normalize: bool) -> torch.Tensor:
        flat = state_tiles.contiguous().view(state_tiles.size(0), -1)
        if normalize:
            return F.normalize(flat, p=2, dim=-1)
        return flat

    def _forward_float(
        self,
        x_bytes: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[torch.Tensor],
        normalize_output: bool,
        return_outputs: bool,
        stats: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], int]:
        if x_bytes.dim() != 2:
            raise ValueError("x_bytes must be (B, L)")
        device = x_bytes.device
        dtype = torch.float32
        x_bytes = x_bytes.to(device=device, dtype=torch.long)

        batch, seq_len = x_bytes.shape
        if lengths is None:
            lengths = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        microblock = self.config.microblock_size
        num_blocks = (seq_len + microblock - 1) // microblock
        if state is None:
            state = torch.zeros(batch, self.config.d_state, device=device, dtype=dtype)
        else:
            state = state.to(device=device, dtype=dtype)

        a_table = self._compute_a()
        b_table = self.b
        ones = torch.ones(1, self.config.d_state, device=device, dtype=dtype)
        zeros = torch.zeros(1, self.config.d_state, device=device, dtype=dtype)

        outputs = [] if return_outputs else None
        for block_idx in range(num_blocks):
            start = block_idx * microblock
            end = min(start + microblock, seq_len)
            block = x_bytes[:, start:end]
            for offset in range(block.size(1)):
                codes = block[:, offset]
                a = F.embedding(codes, a_table)
                b = F.embedding(codes, b_table)
                pos = start + offset
                mask = pos >= lengths
                if mask.any():
                    mask = mask.unsqueeze(-1)
                    a = torch.where(mask, ones, a)
                    b = torch.where(mask, zeros, b)
                state = a * state + b

            state_tiles = state.view(batch, self.config.tile_dim, self.config.n_tiles)
            for tile in range(self.config.n_tiles):
                state_tiles[:, :, tile] = _butterfly_mix(state_tiles[:, :, tile])
            if stats is not None:
                with torch.no_grad():
                    pre_max = float(state_tiles.abs().max().item())
                    stats["activation_pre_clip_max"] = max(stats.get("activation_pre_clip_max", 0.0), pre_max)
            state_tiles = _tanh_squash(state_tiles, self.config.activation_T)
            if stats is not None:
                with torch.no_grad():
                    post_max = float(state_tiles.abs().max().item())
                    stats["activation_post_clip_max"] = max(stats.get("activation_post_clip_max", 0.0), post_max)
                    exchange_due = (
                        self.config.use_exchange
                        and self.config.exchange_every > 0
                        and (block_idx + 1) % self.config.exchange_every == 0
                    )
                    if not (exchange_due and self.config.use_second_activation):
                        _update_activation_saturation(stats, state_tiles, int(0.99 * 32768))
                    exchange_due = (
                        self.config.use_exchange
                        and self.config.exchange_every > 0
                        and (block_idx + 1) % self.config.exchange_every == 0
                    )
                    if not (exchange_due and self.config.use_second_activation):
                        _update_activation_saturation(stats, state_tiles, 0.99)

            if self.config.use_exchange and self.config.exchange_every > 0:
                if (block_idx + 1) % self.config.exchange_every == 0:
                    phase = block_idx % 4
                    state_tiles = self._apply_exchange(state_tiles, phase, stats=stats)
                    if self.config.use_second_activation:
                        state_tiles = _tanh_squash(state_tiles, self.config.activation_T)
                        if stats is not None:
                            with torch.no_grad():
                                post_max = float(state_tiles.abs().max().item())
                                stats["activation_post_clip_max"] = max(
                                    stats.get("activation_post_clip_max", 0.0),
                                    post_max,
                                )
                                _update_activation_saturation(stats, state_tiles, 0.99)

            state = state_tiles.reshape(batch, self.config.d_state)
            if return_outputs:
                outputs.append(self._emit_embedding(state_tiles, normalize_output))

        if stats is not None:
            sat_count = stats.get("activation_sat_count", 0.0)
            elem_count = stats.get("activation_elem_count", 0.0)
            if elem_count:
                stats["activation_sat_frac_gt_0p99"] = float(sat_count) / float(elem_count)

        return state, outputs, num_blocks

    def _forward_quantized(
        self,
        x_bytes: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[torch.Tensor],
        normalize_output: bool,
        return_outputs: bool,
        stats: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], int]:
        if x_bytes.dim() != 2:
            raise ValueError("x_bytes must be (B, L)")
        device = x_bytes.device
        x_bytes = x_bytes.to(device=device, dtype=torch.long)

        batch, seq_len = x_bytes.shape
        microblock = self.config.microblock_size
        num_blocks = (seq_len + microblock - 1) // microblock

        if state is None:
            state = torch.zeros(batch, self.config.d_state, device=device, dtype=torch.int32)
        else:
            state = state.to(device=device, dtype=torch.int32)

        a_table = self._compute_a_q15()
        b_table = self._compute_b_int8()
        ones = torch.full((1, self.config.d_state), 32767, device=device, dtype=torch.int16)
        zeros = torch.zeros(1, self.config.d_state, device=device, dtype=torch.int8)

        outputs = [] if return_outputs else None
        for block_idx in range(num_blocks):
            start = block_idx * microblock
            end = min(start + microblock, seq_len)
            block = x_bytes[:, start:end]
            for offset in range(block.size(1)):
                codes = block[:, offset]
                a = F.embedding(codes, a_table)
                b = F.embedding(codes, b_table)
                pos = start + offset
                mask = pos >= lengths
                if mask.any():
                    mask = mask.unsqueeze(-1)
                    a = torch.where(mask, ones, a)
                    b = torch.where(mask, zeros, b)
                mul = (state * a.to(torch.int32) + 16384) >> 15
                b_add = b.to(torch.int32) << self.config.b_shift
                state = mul + b_add

            state = state.clamp(-32768, 32767).to(torch.int16)

            state_tiles = state.view(batch, self.config.tile_dim, self.config.n_tiles)
            for tile in range(self.config.n_tiles):
                state_tiles[:, :, tile] = _butterfly_mix_int16(state_tiles[:, :, tile])
            if stats is not None:
                with torch.no_grad():
                    pre_max = float(state_tiles.abs().max().item())
                    stats["activation_pre_clip_max"] = max(stats.get("activation_pre_clip_max", 0.0), pre_max)

            x8 = (state_tiles.to(torch.int32) >> self.config.activation_shift).clamp(-128, 127) + 128
            t = self.tanh_lut[x8.to(torch.long)]
            T = int(self.config.activation_T_q15)
            state_tiles = ((t.to(torch.int32) * T) >> 15).clamp(-32768, 32767).to(torch.int16)
            if stats is not None:
                with torch.no_grad():
                    post_max = float(state_tiles.abs().max().item())
                    stats["activation_post_clip_max"] = max(stats.get("activation_post_clip_max", 0.0), post_max)

            if self.config.use_exchange and self.config.exchange_every > 0:
                if (block_idx + 1) % self.config.exchange_every == 0:
                    phase = block_idx % 4
                    state_tiles = self._apply_exchange(state_tiles.float(), phase, stats=stats).to(torch.int16)
                    if self.config.use_second_activation:
                        x8 = (state_tiles.to(torch.int32) >> self.config.activation_shift).clamp(-128, 127) + 128
                        t = self.tanh_lut[x8.to(torch.long)]
                        state_tiles = ((t.to(torch.int32) * T) >> 15).clamp(-32768, 32767).to(torch.int16)
                        if stats is not None:
                            with torch.no_grad():
                                post_max = float(state_tiles.abs().max().item())
                                stats["activation_post_clip_max"] = max(
                                    stats.get("activation_post_clip_max", 0.0),
                                    post_max,
                                )
                                _update_activation_saturation(stats, state_tiles, int(0.99 * 32768))

            state = state_tiles.reshape(batch, self.config.d_state).to(torch.int32)

            if return_outputs:
                emb_float = state_tiles.contiguous().view(batch, -1).float() / 32768.0
                if normalize_output:
                    emb_float = F.normalize(emb_float, p=2, dim=-1)
                outputs.append(emb_float)

        if stats is not None:
            sat_count = stats.get("activation_sat_count", 0.0)
            elem_count = stats.get("activation_elem_count", 0.0)
            if elem_count:
                stats["activation_sat_frac_gt_0p99"] = float(sat_count) / float(elem_count)

        return state, outputs, num_blocks

    def forward(
        self,
        x_bytes: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        quantized: bool = False,
        normalize_output: bool = True,
        return_outputs: bool = False,
        stats: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], int]:
        if lengths is None:
            lengths = torch.full((x_bytes.size(0),), x_bytes.size(1), device=x_bytes.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=x_bytes.device, dtype=torch.long)
        if quantized:
            return self._forward_quantized(x_bytes, lengths, state, normalize_output, return_outputs, stats)
        return self._forward_float(x_bytes, lengths, state, normalize_output, return_outputs, stats)


class MonoidEmbed(nn.Module):
    def __init__(self, config: MonoidEmbedConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([MonoidBlock(config) for _ in range(config.n_layers)])
        use_affine = config.n_layers > 1
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.d_state, elementwise_affine=use_affine) for _ in range(config.n_layers)]
        )
        embed_dim = max(config.matryoshka_dims)
        self.proj = None
        if config.d_state != embed_dim:
            self.proj = nn.Linear(config.d_state, embed_dim, bias=False)
        self._print_param_count()

    def _print_param_count(self) -> None:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MonoidEmbed params: {total}")

    @property
    def b(self) -> torch.nn.Parameter:
        return self.blocks[0].b

    @property
    def tanh_lut(self) -> torch.Tensor:
        return self.blocks[0].tanh_lut

    @property
    def exchange(self) -> Optional[torch.nn.Module]:
        return self.blocks[0].exchange

    def _compute_a(self) -> torch.Tensor:
        return self.blocks[0]._compute_a()

    def _compute_a_q15(self) -> torch.Tensor:
        return self.blocks[0]._compute_a_q15()

    def _compute_b_int8(self) -> torch.Tensor:
        return self.blocks[0]._compute_b_int8()

    def _pool_outputs(
        self,
        outputs: list[torch.Tensor],
        lengths: torch.Tensor,
        num_blocks: int,
        pool_strategy: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if outputs:
            if pool_strategy == "last":
                pooled = outputs[-1]
            else:
                stacked = torch.stack(outputs, dim=1)
                block_lengths = (lengths + self.config.microblock_size - 1) // self.config.microblock_size
                block_lengths = block_lengths.clamp(min=1)
                mask = torch.arange(num_blocks, device=device).unsqueeze(0) < block_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1)
                pooled = (stacked * mask).sum(dim=1) / block_lengths.unsqueeze(-1)
        else:
            pooled = torch.zeros(lengths.size(0), self.config.d_state, device=device, dtype=dtype)
        return pooled

    def load_state_dict(self, state_dict, strict: bool = True):
        if not any(k.startswith("blocks.") for k in state_dict.keys()):
            remapped = {}
            for key, value in state_dict.items():
                remapped[f"blocks.0.{key}"] = value
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        x_bytes: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        quantized: Optional[bool] = None,
        pool_strategy: Optional[str] = None,
        normalize_output: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        return_stats = bool(kwargs.pop("return_stats", False))
        if lengths is None:
            lengths = torch.full((x_bytes.size(0),), x_bytes.size(1), device=x_bytes.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=x_bytes.device, dtype=torch.long)
        if quantized is None:
            quantized = self.config.use_quantized
        if pool_strategy is None:
            pool_strategy = self.config.pool_strategy
        if normalize_output is None:
            normalize_output = self.config.normalize_output
        if quantized and self.config.n_layers > 1:
            quantized = False
        state = torch.zeros(x_bytes.size(0), self.config.d_state, device=x_bytes.device, dtype=torch.float32)
        outputs = None
        num_blocks = 0
        stats = {} if return_stats else None
        for idx, block in enumerate(self.blocks):
            state_in = state
            if self.config.n_layers > 1:
                state_in = self.norms[idx](state_in)
            return_outputs = idx == (self.config.n_layers - 1)
            state_block, outputs, num_blocks = block(
                x_bytes,
                lengths=lengths,
                state=state_in,
                quantized=quantized,
                normalize_output=normalize_output,
                return_outputs=return_outputs,
                stats=stats,
            )
            state = state + state_block

        pooled = self._pool_outputs(
            outputs or [],
            lengths,
            num_blocks,
            pool_strategy,
            x_bytes.device,
            torch.float32,
        )
        if self.proj is not None:
            pooled = self.proj(pooled)
            if normalize_output:
                pooled = F.normalize(pooled, p=2, dim=-1)

        out = {
            "embeddings": pooled,
            "causal_emb": pooled,
            "bidirectional_emb": None,
        }
        if stats is not None:
            out["activation_pre_clip_max"] = stats.get("activation_pre_clip_max", 0.0)
            out["activation_post_clip_max"] = stats.get("activation_post_clip_max", 0.0)
            out["activation_sat_frac_gt_0p99"] = stats.get("activation_sat_frac_gt_0p99", 0.0)
            if "exchange_weight_used" in stats:
                out["exchange_weight_used"] = stats["exchange_weight_used"]
            exec_count = stats.get("exchange_exec_count", 0.0)
            if exec_count:
                out["exchange_executed"] = 1.0
                out["exchange_u_norm"] = float(stats.get("exchange_u_norm_sum", 0.0)) / float(exec_count)
                out["exchange_v_norm"] = float(stats.get("exchange_v_norm_sum", 0.0)) / float(exec_count)
                out["exchange_inj_norm"] = float(stats.get("exchange_inj_norm_sum", 0.0)) / float(exec_count)
                out["exchange_inj_norm_raw"] = float(stats.get("exchange_inj_norm_raw_sum", 0.0)) / float(exec_count)
                out["exchange_inj_norm_clamped"] = float(
                    stats.get("exchange_inj_norm_clamped_sum", 0.0)
                ) / float(exec_count)
                max_count = stats.get("exchange_inj_norm_max_count", 0.0)
                if max_count:
                    out["exchange_inj_norm_max"] = float(stats.get("exchange_inj_norm_max_sum", 0.0)) / float(max_count)
                else:
                    out["exchange_inj_norm_max"] = 0.0
            else:
                out["exchange_executed"] = 0.0
                out["exchange_u_norm"] = 0.0
                out["exchange_v_norm"] = 0.0
                out["exchange_inj_norm"] = 0.0
                out["exchange_inj_norm_raw"] = 0.0
                out["exchange_inj_norm_clamped"] = 0.0
                out["exchange_inj_norm_max"] = 0.0
            scale_count = stats.get("exchange_scale_count", 0.0)
            if scale_count:
                out["exchange_scale"] = float(stats.get("exchange_scale_sum", 0.0)) / float(scale_count)
            else:
                out["exchange_scale"] = 0.0
        if self.config.emit_int8:
            with torch.no_grad():
                norm = pooled.norm(dim=1).clamp(min=1e-6)
                scale = (127.0 / norm).clamp(max=1.0)
                scale_q15 = torch.clamp((scale * 32768.0).round(), 0, 32767).to(torch.int16)
                emb_int8 = torch.clamp((pooled * scale.unsqueeze(1)).round(), -127, 127).to(torch.int8)
            out["embeddings_int8"] = emb_int8
            out["embeddings_scale_q15"] = scale_q15
        return out

    def get_matryoshka_embeddings(self, embeddings: torch.Tensor) -> dict:
        results = {}
        for dim in self.config.matryoshka_dims:
            truncated = embeddings[:, :dim]
            truncated = F.normalize(truncated, p=2, dim=-1)
            results[dim] = truncated
        return results
