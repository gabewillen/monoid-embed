
import json
import math
import os
import queue
import random
import re
import sys
import threading
import subprocess
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, interleave_datasets
import argparse
import logging
from tqdm import tqdm
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig
from monoid.training.embed.data import (
    MonoidDataset,
    MonoidIterableDataset,
    MonoidPrecomputedTeacherDataset,
    RetrievalPairsDataset,
    PairedAudioTextDataset,
    PairedAudioTextIterableDataset,
    collate_fn,
    collate_paired_fn,
    MonoidDatasetProcessor,
)
from monoid.training.embed.loss import (
    GeometricDistillationLoss,
    SpreadOutRegularizer,
    HardnessWeightedContrastiveLoss,
    ConsistencyLoss,
    PairwiseCosineDistillationLoss,
    SimilarityPreservingKDLoss,
    RKDDistanceLoss,
    RKDAngleLoss,
    VICRegVarianceLoss,
    NeighborhoodDistillationLoss,
)
from monoid.training.embed.teacher import TeacherModelHandler, Gemma3nHiddenStateTeacher, M2DClapTeacher
import wandb

# H100/A100 optimization: Enable TF32
torch.set_float32_matmul_precision('high')

def should_log(step: int, every: int) -> bool:
    if every <= 0:
        return False
    return step % every == 0

def update_ema(state: dict, key: str, value: float, alpha: float = 0.98) -> float:
    prev = state.get(key, value)
    ema = (prev * alpha) + (value * (1.0 - alpha))
    state[key] = ema
    return ema

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        val = p.grad.detach().float().norm().item()
        total += val * val
    return total ** 0.5

def grads_finite(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True

def compute_spectral_norm_cpu(
    weight: torch.Tensor,
    step: int,
    label: str,
    raise_on_invalid: bool = True,
) -> tuple[float, bool]:
    def _compute(clean_nan: bool) -> float:
        w_cpu = weight.detach().float().cpu()
        if w_cpu.numel() == 0:
            return 0.0
        if clean_nan and not torch.isfinite(w_cpu).all():
            w_cpu = torch.nan_to_num(w_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        svals = torch.linalg.svdvals(w_cpu)
        return float(svals.max().item())

    try:
        sn = _compute(clean_nan=False)
    except Exception:
        sn = float("nan")
    if not math.isfinite(sn):
        try:
            sn_retry = _compute(clean_nan=True)
        except Exception:
            sn_retry = float("nan")
        if not math.isfinite(sn_retry):
            if raise_on_invalid:
                raise RuntimeError(
                    "spectral norm measurement invalid: "
                    f"sn={sn_retry} step={step} device={weight.device} label={label}"
                )
            return 0.0, False
        sn = sn_retry
    return sn, True

def collect_monoid_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    a_params = []
    b_params = []
    m_params = []
    if not hasattr(model, "blocks"):
        return a_params, b_params, m_params
    for block in model.blocks:
        if hasattr(block, "a_raw"):
            a_params.append(block.a_raw)
        if hasattr(block, "b"):
            b_params.append(block.b)
        if hasattr(block, "exchange") and block.exchange is not None:
            exchange = block.exchange
            param = None
            if (
                hasattr(exchange, "parametrizations")
                and hasattr(exchange.parametrizations, "weight")
                and hasattr(exchange.parametrizations.weight, "original")
            ):
                param = exchange.parametrizations.weight.original
            elif hasattr(exchange, "weight_orig"):
                param = exchange.weight_orig
            elif hasattr(exchange, "weight"):
                param = exchange.weight
            if param is not None:
                m_params.append(param)
    return a_params, b_params, m_params

def get_first_exchange_module(model: nn.Module) -> Optional[nn.Module]:
    if not hasattr(model, "blocks"):
        return None
    for block in model.blocks:
        exchange = getattr(block, "exchange", None)
        if exchange is not None:
            return exchange
    return None

def find_exchange_leaf_param(
    model: nn.Module,
    exchange_dim: Optional[int],
) -> tuple[Optional[str], Optional[nn.Parameter]]:
    candidates = []
    for name, param in model.named_parameters():
        if "exchange" not in name:
            continue
        if exchange_dim is not None and tuple(param.shape) != (exchange_dim, exchange_dim):
            continue
        candidates.append((name, param))
    if not candidates:
        return None, None
    for name, param in candidates:
        if "parametrizations.weight.original" in name:
            return name, param
    for name, param in candidates:
        if "weight_orig" in name:
            return name, param
    return candidates[0]

def _git_value(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"

def set_exchange_trainable(model: nn.Module, enabled: bool) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        if hasattr(block, "exchange") and block.exchange is not None:
            for param in block.exchange.parameters():
                param.requires_grad = enabled
            updated = True
    return updated

def set_exchange_disabled(model: nn.Module, disabled: bool) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        if hasattr(block, "exchange") and block.exchange is not None:
            block.exchange_disabled = disabled
            updated = True
    return updated

def set_exchange_scale(model: nn.Module, scale: float) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        if hasattr(block, "exchange") and block.exchange is not None:
            block.exchange_scale = float(scale)
            updated = True
    return updated

def set_exchange_inj_norm_max(model: nn.Module, value: float) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        if hasattr(block, "exchange") and block.exchange is not None:
            block.exchange_inj_norm_max = float(value)
            updated = True
    return updated

def reset_exchange_spectral_norm(model: nn.Module) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        exchange = getattr(block, "exchange", None)
        if exchange is None:
            continue
        if hasattr(exchange, "parametrizations") and hasattr(exchange.parametrizations, "weight"):
            params = exchange.parametrizations.weight
            if params:
                spectral = params[0]
                if hasattr(spectral, "_u") and hasattr(spectral, "_v"):
                    with torch.no_grad():
                        spectral._u.copy_(torch.randn_like(spectral._u))
                        spectral._v.copy_(torch.randn_like(spectral._v))
                        spectral._u.copy_(spectral._u / (spectral._u.norm() + 1e-12))
                        spectral._v.copy_(spectral._v / (spectral._v.norm() + 1e-12))
                    updated = True
    return updated

def zero_exchange_weights(model: nn.Module) -> bool:
    updated = False
    if not hasattr(model, "blocks"):
        return False
    for block in model.blocks:
        if hasattr(block, "exchange") and block.exchange is not None:
            exchange = block.exchange
            param = None
            if (
                hasattr(exchange, "parametrizations")
                and hasattr(exchange.parametrizations, "weight")
                and hasattr(exchange.parametrizations.weight, "original")
            ):
                param = exchange.parametrizations.weight.original
            elif hasattr(exchange, "weight_orig"):
                param = exchange.weight_orig
            elif hasattr(exchange, "weight"):
                param = exchange.weight
            if param is not None:
                param.data.zero_()
            updated = True
    return updated

def compute_lr_schedule(
    step: int,
    max_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    if schedule == "constant" or max_steps <= 0:
        return float(peak_lr)
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak_lr) * float(step + 1) / float(max(1, warmup_steps))
    denom = max(1, max_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(peak_lr) - float(min_lr)) * cosine

def cross_modal_contrastive_loss(
    text_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    temperature: float,
    return_samples: bool = False,
    sample_k: int = 5,
) -> tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[list], Optional[list], Optional[dict]]:
    if text_emb.numel() == 0 or audio_emb.numel() == 0:
        return torch.tensor(0.0, device=text_emb.device), None, None, None, None, None, None
    if text_emb.size(0) != audio_emb.size(0):
        n = min(text_emb.size(0), audio_emb.size(0))
        text_emb = text_emb[:n]
        audio_emb = audio_emb[:n]
    text_norm = F.normalize(text_emb, p=2, dim=-1)
    audio_norm = F.normalize(audio_emb, p=2, dim=-1)
    sim = text_norm @ audio_norm.t()
    labels = torch.arange(sim.size(0), device=sim.device)
    logits = sim / max(1e-6, float(temperature))
    loss_t2a = F.cross_entropy(logits, labels)
    loss_a2t = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_t2a + loss_a2t)
    diag = sim.diagonal()
    diag_mean = diag.mean().item() if diag.numel() else None
    off_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    off_vals = sim[off_mask]
    off_mean = off_vals.mean().item() if off_vals.numel() else None
    gap = None
    if diag_mean is not None and off_mean is not None:
        gap = diag_mean - off_mean
    diag_sample = None
    off_sample = None
    sim_stats = None
    if return_samples:
        k = max(0, int(sample_k))
        diag_sample = diag[:k].detach().float().cpu().tolist() if k and diag.numel() else []
        off_sample = off_vals[:k].detach().float().cpu().tolist() if k and off_vals.numel() else []
        sim_stats = {
            "min": sim.min().item() if sim.numel() else 0.0,
            "mean": sim.mean().item() if sim.numel() else 0.0,
            "max": sim.max().item() if sim.numel() else 0.0,
        }
    return loss, diag_mean, off_mean, gap, diag_sample, off_sample, sim_stats


def cross_modal_sim_matrix_distill(
    student_text: torch.Tensor,
    student_audio: torch.Tensor,
    teacher_text: torch.Tensor,
    teacher_audio: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_text.numel() == 0 or student_audio.numel() == 0:
        return torch.tensor(0.0, device=student_text.device)
    n = min(student_text.size(0), student_audio.size(0), teacher_text.size(0), teacher_audio.size(0))
    if n <= 0:
        return torch.tensor(0.0, device=student_text.device)
    student_text = F.normalize(student_text[:n], p=2, dim=-1)
    student_audio = F.normalize(student_audio[:n], p=2, dim=-1)
    teacher_text = F.normalize(teacher_text[:n].detach(), p=2, dim=-1)
    teacher_audio = F.normalize(teacher_audio[:n].detach(), p=2, dim=-1)
    scale = max(1e-6, float(temperature))
    s_logits = (student_text @ student_audio.t()) / scale
    t_logits = (teacher_text @ teacher_audio.t()) / scale
    t_row = F.softmax(t_logits, dim=1)
    t_col = F.softmax(t_logits, dim=0)
    s_row = F.log_softmax(s_logits, dim=1)
    s_col = F.log_softmax(s_logits, dim=0)
    loss_row = F.kl_div(s_row, t_row, reduction="batchmean")
    loss_col = F.kl_div(s_col, t_col, reduction="batchmean")
    return 0.5 * (loss_row + loss_col)


def _bytes_to_float_audio(byte_tensor: torch.Tensor) -> np.ndarray:
    byte_arr = byte_tensor.detach().cpu().numpy().astype(np.uint8)
    pcm = np.frombuffer(byte_arr.tobytes(), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32767.0
    return audio


class PairedTeacherPrefetcher:
    def __init__(
        self,
        loader,
        teacher,
        cache,
        device: torch.device,
        prefetch_batches: int = 2,
        cache_key_extra: str | None = None,
        text_prompt_name: str | None = None,
        use_thread: bool = True,
    ):
        self.loader = iter(loader)
        self.teacher = teacher
        self.cache = cache
        self.device = device
        self.cache_key_extra = cache_key_extra
        self.text_prompt_name = text_prompt_name
        self.use_thread = bool(use_thread)
        self.queue = None
        if self.use_thread:
            self.queue = queue.Queue(maxsize=max(1, prefetch_batches))
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _embed_teacher_batch(self, inputs, prompts, modalities):
        if isinstance(modalities, str):
            modalities_list = [modalities] * len(inputs)
        else:
            modalities_list = list(modalities)
        prompts_list = list(prompts) if prompts is not None else [None] * len(inputs)
        grouped = {}
        for idx, modality in enumerate(modalities_list):
            prompt = prompts_list[idx] if idx < len(prompts_list) else None
            prompt_key = prompt or getattr(self.teacher, "text_prompt_name", None)
            if modality != "text":
                prompt_key = None
            grouped.setdefault((modality, prompt_key), []).append(idx)

        out = [None] * len(inputs)
        for (modality, prompt_name), indices in grouped.items():
            batch_inputs = [inputs[i] for i in indices]
            emb = self.teacher.get_embedding(batch_inputs, modality, prompt_name=prompt_name)
            emb = emb.clone().to(self.device, non_blocking=True)
            for offset, idx in enumerate(indices):
                out[idx] = emb[offset]
        return torch.stack(out, dim=0)

    def _cache_embeddings(self, inputs, prompts, modalities, cache_inputs):
        from monoid.training.embed.teacher_cache import hash_batch

        hashes = hash_batch(cache_inputs, prompts, modalities, extra=self.cache_key_extra)
        cached, missing = self.cache.get_many(hashes)
        if missing:
            if self.teacher is None:
                raise RuntimeError("Teacher cache miss and no teacher available.")
            missing_inputs = [inputs[i] for i in missing]
            missing_prompts = [prompts[i] for i in missing] if prompts else None
            missing_modalities = modalities
            if not isinstance(missing_modalities, str):
                missing_modalities = [missing_modalities[i] for i in missing]
            with torch.no_grad():
                emb = self._embed_teacher_batch(missing_inputs, missing_prompts, missing_modalities)
            emb_cpu = emb.detach().to("cpu", dtype=torch.float16).numpy()
            missing_hashes = [hashes[i] for i in missing]
            self.cache.put_many(missing_hashes, emb_cpu)
            for offset, idx in enumerate(missing):
                cached[idx] = emb_cpu[offset]
        return torch.from_numpy(np.stack(cached)).float()

    def _process_batch(self, batch):
        captions = batch.get("text_captions") or []
        audio_bytes = batch.get("audio_bytes")
        if audio_bytes is None:
            raise RuntimeError("Paired batch missing audio_bytes.")
        text_prompt_name = self.text_prompt_name or getattr(self.teacher, "text_prompt_name", None)
        text_prompts = [text_prompt_name] * len(captions) if captions else None
        text_modalities = ["text"] * len(captions)
        text_cache_inputs = captions
        text_emb = self._cache_embeddings(captions, text_prompts, text_modalities, text_cache_inputs) if captions else None

        audio_inputs = []
        audio_cache_inputs = []
        for idx in range(audio_bytes.size(0)):
            byte_tensor = audio_bytes[idx]
            audio_inputs.append(_bytes_to_float_audio(byte_tensor))
            audio_cache_inputs.append(byte_tensor.detach().cpu().numpy().astype(np.uint8))
        audio_modalities = ["audio"] * len(audio_inputs)
        audio_emb = self._cache_embeddings(audio_inputs, None, audio_modalities, audio_cache_inputs)

        batch["teacher_mm_text_emb"] = text_emb
        batch["teacher_mm_audio_emb"] = audio_emb
        return batch

    def _worker(self) -> None:
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device.index or 0)
            for batch in self.loader:
                batch = self._process_batch(batch)
                self.queue.put(batch)
        except Exception as exc:
            self.queue.put(exc)
        finally:
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_thread:
            item = self.queue.get()
            if item is None:
                raise StopIteration
            if isinstance(item, Exception):
                raise item
            return item
        batch = next(self.loader)
        return self._process_batch(batch)

    def take_cache_stats(self) -> Tuple[int, int]:
        return self.cache.take_stats()

def per_dim_correlation(student_emb: torch.Tensor, teacher_emb: torch.Tensor, eps: float = 1e-8) -> Optional[torch.Tensor]:
    if student_emb.size(0) < 2:
        return None
    s = student_emb.float()
    t = teacher_emb.float()
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    denom = s.std(dim=0, unbiased=False) * t.std(dim=0, unbiased=False)
    denom = denom.clamp_min(eps)
    corr = (s * t).mean(dim=0) / denom
    return corr

def _iter_tail_lines(path: str, max_bytes: int = 2 * 1024 * 1024) -> Tuple[str, ...]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            data = f.read()
    except FileNotFoundError:
        return tuple()
    text = data.decode("utf-8", errors="replace")
    return tuple(text.splitlines())

def _infer_resume_from_log(path: str) -> Tuple[Optional[int], Optional[float]]:
    if not path:
        return (None, None)
    lines = _iter_tail_lines(path)
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        step = obj.get("step", obj.get("_step"))
        if isinstance(step, float) and step.is_integer():
            step = int(step)
        if not isinstance(step, int):
            continue
        contrast_scale = obj.get("train/contrast_scale")
        if isinstance(contrast_scale, (int, float)):
            contrast_scale = float(contrast_scale)
        else:
            contrast_scale = None
        return (step, contrast_scale)
    return (None, None)

def _parse_step_from_checkpoint(path: str) -> Optional[int]:
    base = os.path.basename(path)
    match = re.search(r"(?:checkpoint_step)?(\d+)\.pt$", base)
    if match:
        return int(match.group(1))
    return None

def _extract_resume_state(
    state: object,
) -> Tuple[
    dict,
    Optional[dict],
    Optional[int],
    Optional[int],
    Optional[dict],
    Optional[torch.Tensor],
    Optional[dict],
    Optional[torch.Tensor],
]:
    if isinstance(state, dict) and "model" in state:
        model_state = state.get("model", {})
        if isinstance(model_state, dict) and any("._orig_mod." in key or key.startswith("_orig_mod.") for key in model_state):
            remapped = {}
            for key, value in model_state.items():
                new_key = key.replace("._orig_mod.", ".")
                if new_key.startswith("_orig_mod."):
                    new_key = new_key[len("_orig_mod.") :]
                remapped[new_key] = value
            model_state = remapped
        optimizer_state = state.get("optimizer")
        step = state.get("step")
        ramp_start = state.get("contrast_ramp_start_step")
        teacher_proj_state = state.get("teacher_proj")
        layer_mix_state = state.get("layer_mix")
        teacher_proj_ema_state = state.get("teacher_proj_ema")
        layer_mix_ema_state = state.get("layer_mix_ema")
        if isinstance(step, float) and step.is_integer():
            step = int(step)
        if isinstance(ramp_start, float) and ramp_start.is_integer():
            ramp_start = int(ramp_start)
        if not isinstance(step, int):
            step = None
        if not isinstance(ramp_start, int):
            ramp_start = None
        return (
            model_state,
            optimizer_state,
            step,
            ramp_start,
            teacher_proj_state,
            layer_mix_state,
            teacher_proj_ema_state,
            layer_mix_ema_state,
        )
    if isinstance(state, dict):
        model_state = state
        if any("._orig_mod." in key or key.startswith("_orig_mod.") for key in model_state):
            remapped = {}
            for key, value in model_state.items():
                new_key = key.replace("._orig_mod.", ".")
                if new_key.startswith("_orig_mod."):
                    new_key = new_key[len("_orig_mod.") :]
                remapped[new_key] = value
            model_state = remapped
        return (model_state, None, None, None, None, None, None, None)
    raise ValueError("Unsupported checkpoint format for resume.")

def compute_matryoshka_weights(
    dims: Tuple[int, ...],
    base_weights: dict,
    cos_t_s_512: Optional[float],
    ramp_threshold: float,
) -> Tuple[list, float]:
    if cos_t_s_512 is None:
        ramp = 0.0
    else:
        ramp = max(0.0, (cos_t_s_512 - ramp_threshold) / max(1e-6, 1.0 - ramp_threshold))
        ramp = min(1.0, ramp)
    max_dim = max(dims)
    weights = []
    for dim in dims:
        base = base_weights.get(dim, dim / max_dim)
        if dim == max_dim:
            weight = base
        else:
            weight = base + ramp * (1.0 - base)
        weights.append(weight)
    weight_sum = float(sum(weights))
    return weights, weight_sum

def knn_overlap_metrics(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
    ks: Tuple[int, ...] = (5, 10),
) -> dict:
    n = teacher_emb.size(0)
    if n <= 1:
        return {f"geom/knn_overlap@{k}": 0.0 for k in ks} | {
            f"geom/knn_overlap_rand@{k}": 0.0 for k in ks
        } | {f"geom/knn_overlap_lift@{k}": 0.0 for k in ks}

    teacher = l2_normalize(teacher_emb)
    student = l2_normalize(student_emb)
    tt = teacher @ teacher.t()
    ss = student @ student.t()
    tt.fill_diagonal_(-float("inf"))
    ss.fill_diagonal_(-float("inf"))

    metrics = {}
    device = teacher.device
    for k in ks:
        k_eff = min(k, n - 1)
        if k_eff <= 0:
            metrics[f"geom/knn_overlap@{k}"] = 0.0
            metrics[f"geom/knn_overlap_rand@{k}"] = 0.0
            metrics[f"geom/knn_overlap_lift@{k}"] = 0.0
            continue
        idx_t = tt.topk(k_eff, dim=1).indices
        idx_s = ss.topk(k_eff, dim=1).indices
        overlaps = []
        overlaps_rand = []
        for i in range(n):
            t_i = idx_t[i]
            s_i = idx_s[i]
            match = (t_i.unsqueeze(1) == s_i.unsqueeze(0)).any(dim=1).float().sum()
            overlaps.append(match / float(k_eff))

            candidates = torch.cat(
                [torch.arange(0, i, device=device), torch.arange(i + 1, n, device=device)]
            )
            rand_idx = candidates[torch.randperm(n - 1, device=device)[:k_eff]]
            match_rand = (t_i.unsqueeze(1) == rand_idx.unsqueeze(0)).any(dim=1).float().sum()
            overlaps_rand.append(match_rand / float(k_eff))

        overlap_mean = torch.stack(overlaps).mean().item()
        overlap_rand_mean = torch.stack(overlaps_rand).mean().item()
        metrics[f"geom/knn_overlap@{k}"] = overlap_mean
        metrics[f"geom/knn_overlap_rand@{k}"] = overlap_rand_mean
        metrics[f"geom/knn_overlap_lift@{k}"] = overlap_mean - overlap_rand_mean
    return metrics

def spearman_rowwise_mean(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
) -> float:
    n = teacher_emb.size(0)
    if n <= 2:
        return 0.0
    teacher = l2_normalize(teacher_emb)
    student = l2_normalize(student_emb)
    tt = teacher @ teacher.t()
    ss = student @ student.t()
    corrs = []
    eps = 1e-8
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool, device=teacher.device)
        mask[i] = False
        t_vals = tt[i][mask]
        s_vals = ss[i][mask]
        t_ranks = torch.argsort(torch.argsort(t_vals))
        s_ranks = torch.argsort(torch.argsort(s_vals))
        t_r = t_ranks.float()
        s_r = s_ranks.float()
        t_r = t_r - t_r.mean()
        s_r = s_r - s_r.mean()
        denom = (t_r.std(unbiased=False) * s_r.std(unbiased=False)) + eps
        corr = (t_r * s_r).mean() / denom
        corrs.append(corr)
    return torch.stack(corrs).mean().item()

def spearman_rowwise_mean_from_sim(tt: torch.Tensor, ss: torch.Tensor) -> float:
    n = tt.size(0)
    if n <= 2:
        return 0.0
    corrs = []
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool, device=tt.device)
        mask[i] = False
        t_vals = tt[i][mask]
        s_vals = ss[i][mask]
        t_ranks = torch.argsort(torch.argsort(t_vals))
        s_ranks = torch.argsort(torch.argsort(s_vals))
        t_r = t_ranks.float()
        s_r = s_ranks.float()
        t_r = t_r - t_r.mean()
        s_r = s_r - s_r.mean()
        denom = (t_r.std(unbiased=False) * s_r.std(unbiased=False)) + 1e-8
        corr = (t_r * s_r).mean() / denom
        corrs.append(corr)
    return torch.stack(corrs).mean().item()

def _offdiag_stats(sim: torch.Tensor) -> Tuple[float, float]:
    n = sim.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return vals.mean().item(), vals.std(unbiased=False).item()

def _hist_kl(teacher_vals: torch.Tensor, student_vals: torch.Tensor, bins: int = 50) -> float:
    if teacher_vals.numel() == 0 or student_vals.numel() == 0:
        return 0.0
    t_hist = torch.histc(teacher_vals, bins=bins, min=-1.0, max=1.0)
    s_hist = torch.histc(student_vals, bins=bins, min=-1.0, max=1.0)
    eps = 1e-8
    t_prob = t_hist / (t_hist.sum() + eps)
    s_prob = s_hist / (s_hist.sum() + eps)
    kl = (t_prob * (t_prob.add(eps).log() - s_prob.add(eps).log())).sum()
    return kl.item()

def pos_neg_margin_metrics(
    student_emb: torch.Tensor,
    batch: dict,
    indices: Optional[torch.Tensor] = None,
) -> dict:
    if student_emb.dim() == 3:
        if student_emb.size(1) == 2:
            view0 = student_emb[:, 0]
            view1 = student_emb[:, 1]
        elif student_emb.size(0) == 2:
            view0 = student_emb[0]
            view1 = student_emb[1]
        else:
            return {}
        view0 = l2_normalize(view0)
        view1 = l2_normalize(view1)
        pos = (view0 * view1).sum(dim=-1)
        sim = view0 @ view1.t()
        sim.fill_diagonal_(-float("inf"))
        neg = sim.mean(dim=1)
        return {
            "geom/cos_pos_mean": pos.mean().item(),
            "geom/cos_neg_mean": neg.mean().item(),
            "geom/margin_mean": (pos - neg).mean().item(),
        }

    pair_ids = batch.get("pair_ids") if isinstance(batch, dict) else None
    view_ids = batch.get("view_ids") if isinstance(batch, dict) else None
    ids = pair_ids if pair_ids is not None else view_ids
    if ids is None:
        return {}
    if indices is not None:
        if indices.device != ids.device:
            indices = indices.to(ids.device)
        ids = ids[indices]
    elif ids.numel() != student_emb.size(0):
        return {}
    ids = ids.to(student_emb.device)
    emb = l2_normalize(student_emb)
    sim = emb @ emb.t()
    sim.fill_diagonal_(-float("inf"))
    pos_vals = []
    neg_vals = []
    for i in range(sim.size(0)):
        same = ids == ids[i]
        same[i] = False
        if same.any():
            pos_vals.append(sim[i][same].mean())
        diff = ~same
        diff[i] = False
        if diff.any():
            neg_vals.append(sim[i][diff].mean())
    if not pos_vals or not neg_vals:
        return {}
    pos = torch.stack(pos_vals)
    neg = torch.stack(neg_vals)
    return {
        "geom/cos_pos_mean": pos.mean().item(),
        "geom/cos_neg_mean": neg.mean().item(),
        "geom/margin_mean": (pos - neg).mean().item(),
    }

def _covariance_metrics(emb: torch.Tensor) -> dict:
    n = emb.size(0)
    if n <= 1:
        return {"geom/cov_trace": 0.0, "geom/eig_top1_frac": 0.0}
    x = emb - emb.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False)
    cov_trace = var.sum()
    v = torch.randn(emb.size(1), device=emb.device)
    v = v / (v.norm() + 1e-8)
    for _ in range(5):
        mv = x.t().mv(x.mv(v)) / max(1, n - 1)
        v = mv / (mv.norm() + 1e-8)
    top1 = (v * (x.t().mv(x.mv(v)) / max(1, n - 1))).sum()
    top1_frac = (top1 / (cov_trace + 1e-8)).item()
    return {
        "geom/cov_trace": cov_trace.item(),
        "geom/eig_top1_frac": top1_frac,
    }

def _dim_std_percentiles(emb: torch.Tensor) -> dict:
    stds = emb.std(dim=0, unbiased=False)
    p10, p50, p90 = torch.quantile(stds, torch.tensor([0.1, 0.5, 0.9], device=stds.device))
    return {
        "geom/dim_std_p10": p10.item(),
        "geom/dim_std_p50": p50.item(),
        "geom/dim_std_p90": p90.item(),
    }

def compute_geom_metrics(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
    batch: dict,
    ks: Tuple[int, ...] = (5, 10),
) -> dict:
    if teacher_emb is None or student_emb is None:
        return {}
    if teacher_emb.size(0) <= 1:
        return {}
    teacher_emb = teacher_emb.float()
    student_emb = student_emb.float()
    n = teacher_emb.size(0)
    if n > 256:
        idx = torch.randperm(n, device=teacher_emb.device)[:256]
        teacher_emb = teacher_emb[idx]
        student_emb = student_emb[idx]
    metrics = {}
    metrics.update(knn_overlap_metrics(teacher_emb, student_emb, ks=ks))
    metrics["geom/spearman_mean"] = spearman_rowwise_mean(teacher_emb, student_emb)
    metrics.update(pos_neg_margin_metrics(student_emb, batch))
    teacher = l2_normalize(teacher_emb)
    student = l2_normalize(student_emb)
    tt = teacher @ teacher.t()
    ss = student @ student.t()
    tt.fill_diagonal_(-float("inf"))
    ss.fill_diagonal_(-float("inf"))
    t_mean, t_std = _offdiag_stats(tt)
    s_mean, s_std = _offdiag_stats(ss)
    metrics["geom/teacher_cos_offdiag_mean"] = t_mean
    metrics["geom/teacher_cos_offdiag_std"] = t_std
    metrics["geom/student_cos_offdiag_mean"] = s_mean
    metrics["geom/student_cos_offdiag_std"] = s_std
    metrics["geom/cos_mean_gap"] = s_mean - t_mean
    metrics["geom/cos_std_gap"] = s_std - t_std
    mask = ~torch.eye(tt.size(0), dtype=torch.bool, device=tt.device)
    metrics["geom/cos_hist_kl"] = _hist_kl(tt[mask], ss[mask])

    n = tt.size(0)
    k_t = min(5, n - 1)
    k_s = min(10, n - 1)
    if k_t > 0 and k_s > 0:
        idx_t5 = tt.topk(k_t, dim=1).indices
        idx_s10 = ss.topk(k_s, dim=1).indices
        overlaps = []
        for i in range(n):
            match = (idx_t5[i].unsqueeze(1) == idx_s10[i].unsqueeze(0)).any(dim=1).float().sum()
            overlaps.append(match / float(k_t))
        metrics["geom/recall_T5_in_S10"] = torch.stack(overlaps).mean().item()
    else:
        metrics["geom/recall_T5_in_S10"] = 0.0

    metrics.update(_covariance_metrics(student))
    metrics.update(_dim_std_percentiles(student))

    for dim in (128, 256, 512):
        if teacher_emb.size(1) >= dim and student_emb.size(1) >= dim:
            t_dim = l2_normalize(teacher_emb[:, :dim])
            s_dim = l2_normalize(student_emb[:, :dim])
            metrics[f"geom/cos_T_S_{dim}"] = (t_dim * s_dim).sum(dim=1).mean().item()

    if teacher_emb.size(1) >= 10 and student_emb.size(1) >= 10:
        def _knn_overlap_at_dim(dim: int) -> float:
            t_dim = l2_normalize(teacher_emb[:, :dim])
            s_dim = l2_normalize(student_emb[:, :dim])
            tt_dim = t_dim @ t_dim.t()
            ss_dim = s_dim @ s_dim.t()
            tt_dim.fill_diagonal_(-float("inf"))
            ss_dim.fill_diagonal_(-float("inf"))
            k_eff = min(10, tt_dim.size(0) - 1)
            if k_eff <= 0:
                return 0.0
            idx_t = tt_dim.topk(k_eff, dim=1).indices
            idx_s = ss_dim.topk(k_eff, dim=1).indices
            overlaps = []
            for i in range(tt_dim.size(0)):
                match = (idx_t[i].unsqueeze(1) == idx_s[i].unsqueeze(0)).any(dim=1).float().sum()
                overlaps.append(match / float(k_eff))
            return torch.stack(overlaps).mean().item()

        metrics["geom/knn_overlap@10_128"] = _knn_overlap_at_dim(128)
        metrics["geom/knn_overlap@10_512"] = _knn_overlap_at_dim(512)
    return metrics

def _embed_teacher_batch(teacher, inputs, prompts, modalities, device):
    if isinstance(modalities, str):
        modalities_list = [modalities] * len(inputs)
    else:
        modalities_list = list(modalities)
    prompts_list = list(prompts) if prompts is not None else [None] * len(inputs)

    grouped = {}
    for idx, modality in enumerate(modalities_list):
        prompt = prompts_list[idx] if idx < len(prompts_list) else None
        prompt_key = prompt or teacher.text_prompt_name
        if modality != "text":
            prompt_key = None
        grouped.setdefault((modality, prompt_key), []).append(idx)

    out = [None] * len(inputs)
    for (modality, prompt_name), indices in grouped.items():
        batch_inputs = [inputs[i] for i in indices]
        emb = teacher.get_embedding(batch_inputs, modality, prompt_name=prompt_name)
        emb = emb.clone().to(device, non_blocking=True)
        for offset, idx in enumerate(indices):
            out[idx] = emb[offset]

    return torch.stack(out, dim=0)

def _build_retrieval_pairs(args, logger):
    rng = random.Random(args.retrieval_train_seed)
    qrels = None
    try:
        qrels = load_dataset(
            args.retrieval_train_dataset,
            split=args.retrieval_train_split,
            trust_remote_code=args.datasets_trust_remote_code,
        )
    except ValueError as exc:
        msg = str(exc)
        if "multiple" in msg and "configurations" in msg:
            logger.warning(
                "Multiple configs found for %s in cache; falling back to config 'default'.",
                args.retrieval_train_dataset,
            )
            qrels = load_dataset(
                args.retrieval_train_dataset,
                "default",
                split=args.retrieval_train_split,
                trust_remote_code=args.datasets_trust_remote_code,
            )
        elif "Config name is missing" in msg or "Please pick one among the available configs" in msg:
            logger.info(
                "Retrieval dataset %s requires a config; falling back to pair/triplet loader.",
                args.retrieval_train_dataset,
            )
            qrels = None
        else:
            raise
    if qrels is not None and "query-id" in qrels.column_names and "corpus-id" in qrels.column_names:
        qrels_by_query = {}
        for row in qrels:
            if row.get("score", 0) <= 0:
                continue
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            qrels_by_query.setdefault(qid, set()).add(cid)

        query_ids = list(qrels_by_query.keys())
        rng.shuffle(query_ids)
        if args.retrieval_train_pairs:
            query_ids = query_ids[: args.retrieval_train_pairs]

        queries_ds = load_dataset(
            args.retrieval_train_dataset,
            "queries",
            split="queries",
            trust_remote_code=args.datasets_trust_remote_code,
        )
        query_texts = {str(row["_id"]): row["text"] for row in queries_ds}
        query_ids = [qid for qid in query_ids if qid in query_texts]

        corpus_ds = load_dataset(
            args.retrieval_train_dataset,
            "corpus",
            split="corpus",
            trust_remote_code=args.datasets_trust_remote_code,
        )
        doc_texts = {}
        for row in corpus_ds:
            doc_id = str(row["_id"])
            title = row.get("title") or ""
            text = row.get("text") or ""
            doc_texts[doc_id] = (title + "\n" + text).strip()

        pairs = []
        for qid in query_ids:
            pos_ids = list(qrels_by_query.get(qid, []))
            if not pos_ids:
                continue
            rng.shuffle(pos_ids)
            doc_id = pos_ids[0]
            doc_text = doc_texts.get(doc_id)
            if not doc_text:
                continue
            pairs.append((query_texts[qid], doc_text))
    else:
        pairs_ds = qrels
        if pairs_ds is None:
            config = None
            if args.retrieval_train_dataset == args.dataset_name and args.dataset_config:
                config = args.dataset_config
            try:
                if config:
                    pairs_ds = load_dataset(
                        args.retrieval_train_dataset,
                        config,
                        split=args.retrieval_train_split,
                        trust_remote_code=args.datasets_trust_remote_code,
                    )
                else:
                    pairs_ds = load_dataset(
                        args.retrieval_train_dataset,
                        split=args.retrieval_train_split,
                        trust_remote_code=args.datasets_trust_remote_code,
                    )
            except ValueError as exc:
                msg = str(exc)
                if "multiple" in msg and "configurations" in msg:
                    logger.warning(
                        "Multiple configs found for %s in cache; falling back to config 'default'.",
                        args.retrieval_train_dataset,
                    )
                    pairs_ds = load_dataset(
                        args.retrieval_train_dataset,
                        "default",
                        split=args.retrieval_train_split,
                        trust_remote_code=args.datasets_trust_remote_code,
                    )
                else:
                    raise

        columns = set(pairs_ds.column_names)
        pair_cols = None
        for left, right in (
            ("anchor", "positive"),
            ("sentence1", "sentence2"),
            ("premise", "hypothesis"),
        ):
            if left in columns and right in columns:
                pair_cols = (left, right)
                break
        if pair_cols is None:
            raise ValueError(
                "Retrieval train dataset does not have supported pair columns. "
                "Expected one of: anchor/positive, sentence1/sentence2, premise/hypothesis."
            )

        if args.retrieval_train_pairs:
            if hasattr(pairs_ds, "shuffle"):
                pairs_ds = pairs_ds.shuffle(seed=args.retrieval_train_seed)
            if hasattr(pairs_ds, "select"):
                pairs_ds = pairs_ds.select(range(min(args.retrieval_train_pairs, len(pairs_ds))))

        pairs = []
        for row in pairs_ds:
            anchor = row.get(pair_cols[0])
            positive = row.get(pair_cols[1])
            if not anchor or not positive:
                continue
            pairs.append((anchor, positive))

    if not pairs:
        raise ValueError("No retrieval pairs found.")

    logger.info("Built %d retrieval pairs from %s/%s", len(pairs), args.retrieval_train_dataset, args.retrieval_train_split)
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128) # Increased for H100
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--peak_lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=2e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.02)
    parser.add_argument("--freeze_exchange_steps", type=int, default=10000)
    parser.add_argument("--enable_saturation_penalty", dest="enable_saturation_penalty", action="store_true")
    parser.add_argument("--disable_saturation_penalty", dest="enable_saturation_penalty", action="store_false")
    parser.add_argument("--saturation_penalty_weight", type=float, default=1.0)
    parser.add_argument("--log_grad_stats", dest="log_grad_stats", action="store_true")
    parser.add_argument("--no_log_grad_stats", dest="log_grad_stats", action="store_false")
    parser.add_argument("--log_embedding_stats", dest="log_embedding_stats", action="store_true")
    parser.add_argument("--no_log_embedding_stats", dest="log_embedding_stats", action="store_false")
    parser.add_argument("--log_activation_stats", dest="log_activation_stats", action="store_true")
    parser.add_argument("--no_log_activation_stats", dest="log_activation_stats", action="store_false")
    parser.add_argument("--log_jsonl", type=str, default=None, help="Write JSONL logs to this path.")
    parser.add_argument("--log_every", type=int, default=10, help="Log JSON every N steps.")
    parser.add_argument("--spec_header_every", type=int, default=200, help="Emit spec header JSON every N steps.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="allenai/c4") 
    parser.add_argument("--dataset_config", type=str, default="en")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_name2", type=str, default=None, help="Optional secondary dataset for mixing")
    parser.add_argument("--dataset_config2", type=str, default=None, help="Optional config for secondary dataset")
    parser.add_argument("--dataset_split2", type=str, default=None, help="Optional split for secondary dataset")
    parser.add_argument("--dataset_mix_ratio", type=float, default=1.0, help="Probability for primary dataset in mix")
    parser.add_argument("--dataset_mix_seed", type=int, default=1234, help="Seed for dataset mixing")
    parser.add_argument(
        "--datasets_trust_remote_code",
        action="store_true",
        help="Allow datasets that require remote code (HF dataset scripts).",
    )
    parser.add_argument(
        "--dataset_mix",
        nargs="*",
        default=None,
        help="Optional dataset mix entries name[:config][:split][:weight]. Overrides dataset_name/dataset_name2.",
    )
    parser.add_argument("--modality", type=str, default="text")
    parser.add_argument("--audio_dataset_name", type=str, default="OpenSound/AudioCaps")
    parser.add_argument("--audio_dataset_config", type=str, default=None)
    parser.add_argument("--audio_dataset_split", type=str, default="train")
    parser.add_argument("--audio_dataset_name2", type=str, default=None, help="Optional secondary audio dataset for mixing")
    parser.add_argument("--audio_dataset_config2", type=str, default=None, help="Optional config for secondary audio dataset")
    parser.add_argument("--audio_dataset_split2", type=str, default=None, help="Optional split for secondary audio dataset")
    parser.add_argument("--audio_dataset_mix_ratio", type=float, default=1.0, help="Probability for primary audio dataset in mix")
    parser.add_argument("--text_audio_ratio", type=float, default=0.5, help="Probability for text samples in multimodal batches")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--use_real_teacher", action="store_true", help="Use real teacher models")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--no_use_wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm/progress bars")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "text_only_finisher"])
    parser.add_argument("--finisher_steps", type=int, default=2000)
    parser.add_argument("--finisher_lr_mult", type=float, default=0.5)
    parser.add_argument("--finisher_lr_schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--text_prompt_name", type=str, default="document", help="EmbeddingGemma prompt name for text teacher")
    parser.add_argument("--teacher_backend", type=str, default=None, choices=["embeddinggemma", "gemma3n_hidden"])
    parser.add_argument("--teacher_text_backend", type=str, default=None, choices=["embeddinggemma", "gemma3n_hidden"])
    parser.add_argument("--teacher_mm_backend", type=str, default=None, choices=["m2d_clap", "gemma3n_hidden"])
    parser.add_argument("--teacher_mm_checkpoint", type=str, default=None, help="Checkpoint for M2D-CLAP teacher")
    parser.add_argument("--teacher_mm_repo", type=str, default=None, help="Path to M2D repo for portable runtime")
    parser.add_argument("--teacher_model_id", type=str, default="google/gemma-3n-E4B")
    parser.add_argument("--teacher_layer", type=int, default=-1, help="Hidden state layer for Gemma-3n teacher")
    parser.add_argument(
        "--teacher_audio_source",
        type=str,
        default="llm_hidden",
        choices=["llm_hidden", "audio_tower"],
        help="Audio embedding source for Gemma-3n hidden teacher.",
    )
    parser.add_argument(
        "--teacher_text_mode",
        type=str,
        default="templated",
        choices=["raw", "templated"],
        help="Text prompt handling for Gemma-3n hidden teacher.",
    )
    parser.add_argument(
        "--teacher_stream_idx_text",
        type=int,
        default=None,
        help="Stream index to select for text hidden states when Gemma-3n returns streams.",
    )
    parser.add_argument(
        "--teacher_stream_idx_audio",
        type=int,
        default=None,
        help="Stream index to select for audio hidden states when Gemma-3n returns streams.",
    )
    parser.add_argument("--teacher_lr", type=float, default=1e-5, help="Learning rate for teacher projection/layer mix")
    parser.add_argument("--teacher_ema_decay", type=float, default=0.999, help="EMA decay for teacher projection/layer mix")
    parser.add_argument("--teacher_text_weight", type=float, default=1.0, help="Scale for text-teacher losses")
    parser.add_argument("--mm_distill_weight", type=float, default=0.0, help="Weight for cross-modal sim-matrix distill")
    parser.add_argument("--mm_distill_temp", type=float, default=0.07, help="Temperature for cross-modal sim-matrix distill")
    parser.add_argument(
        "--text_prompt_mix",
        type=str,
        default="query,document",
        help="Comma-separated prompt names to mix for text distillation (e.g. query,document).",
    )
    parser.add_argument(
        "--text_prompt_mix_mode",
        type=str,
        default="alternate",
        choices=["random", "alternate"],
        help="How to mix prompts when --text_prompt_mix is set.",
    )
    parser.add_argument("--audio_sample_rate", type=int, default=16000, help="Target sample rate for audio inputs")
    parser.add_argument(
        "--paired_audio_max_seconds",
        type=float,
        default=10.0,
        help="Max seconds for paired audio examples (set <=0 to disable cap).",
    )
    parser.add_argument("--audio_random_crop", dest="audio_random_crop", action="store_true", help="Randomly crop audio to max_bytes")
    parser.add_argument("--no_audio_random_crop", dest="audio_random_crop", action="store_false", help="Disable random audio crop")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset (infinite)")
    parser.add_argument("--max_steps", type=int, default=50000, help="Max steps per epoch for streaming")
    parser.add_argument("--bptt_chunk_size", type=int, default=256, help="Chunk size for truncated BPTT (0 disables)")
    parser.add_argument("--bptt_detach_every", type=int, default=4, help="Detach cached state every N chunks")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_from", type=str, default=None, help="Alias for --resume")
    parser.add_argument("--resume_step", type=int, default=None, help="Override step when resuming")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Monoid preset override (sets MONOID_PRESET).",
    )
    parser.add_argument(
        "--resume_log",
        type=str,
        default=os.path.join("wandb", "latest-run", "files", "output.log"),
        help="Output log to infer step/contrast_scale when resuming",
    )
    parser.add_argument("--w_pairwise", type=float, default=0.1, help="Weight for pairwise cosine distillation")
    parser.add_argument("--w_spread", type=float, default=1.0, help="Weight for spread-out regularizer")
    parser.add_argument("--distill_weight", type=float, default=1.0, help="Weight for teacher distillation loss")
    parser.add_argument("--spkd_weight", type=float, default=1.0, help="Weight for SPKD similarity loss")
    parser.add_argument("--rkd_distance_weight", type=float, default=1.0, help="Weight for RKD distance loss")
    parser.add_argument("--rkd_angle_weight", type=float, default=0.5, help="Weight for RKD angle loss")
    parser.add_argument("--rkd_d_weight", type=float, default=None, help="Alias for --rkd_distance_weight")
    parser.add_argument("--rkd_a_weight", type=float, default=None, help="Alias for --rkd_angle_weight")
    parser.add_argument(
        "--loss_weight_rkd_distance",
        type=float,
        default=None,
        help="Alias for --rkd_distance_weight",
    )
    parser.add_argument(
        "--loss_weight_rkd_angle",
        type=float,
        default=None,
        help="Alias for --rkd_angle_weight",
    )
    parser.add_argument(
        "--loss_weight_neighborhood",
        type=float,
        default=None,
        help="Alias for --neighborhood_weight",
    )
    parser.add_argument("--var_weight", type=float, default=0.1, help="Weight for VICReg variance floor loss")
    parser.add_argument(
        "--alpha_hardness",
        type=float,
        default=5.0,
        help="Deprecated used for ramp start if alpha_hardness_fixed not set",
    )
    parser.add_argument("--alpha_hardness_fixed", type=float, default=5.0, help="Fixed alpha hardness (overrides ramp if set)")
    parser.add_argument("--neighborhood_temp", type=float, default=1.0, help="Temperature for neighborhood distillation")
    parser.add_argument("--neighborhood_weight", type=float, default=1.0, help="Weight for neighborhood distillation loss")
    parser.add_argument("--no_matryoshka", action="store_true", help="Disable Matryoshka multi-dim losses")
    parser.add_argument("--matryoshka_weights", type=float, nargs="*", default=None, help="Optional weights for Matryoshka dims")
    parser.add_argument("--matryoshka_ramp_cos", type=float, default=0.5, help="Cosine threshold to ramp Matryoshka weights")
    parser.add_argument("--contrast_warmup_steps", type=int, default=2000, help="Steps to keep contrast loss at 0")
    parser.add_argument("--contrast_weight", type=float, default=0.05, help="Weight for contrast loss (pre-warmup)")
    parser.add_argument("--contrast_ramp_steps", type=int, default=2000, help="Steps to ramp contrast loss to full weight")
    parser.add_argument("--contrast_ramp_power", type=float, default=3.0, help="Power for contrast ramp (default 3 for cubic)")
    parser.add_argument(
        "--contrast_scale_cap",
        "--contrast_scale_max",
        dest="contrast_scale_cap",
        type=float,
        default=0.05,
        help="Cap for contrast scale",
    )
    parser.add_argument("--contrast_start_cos", type=float, default=0.2, help="Cosine threshold to start contrast ramp")
    parser.add_argument(
        "--retrieval_ramp_start",
        type=int,
        default=None,
        help="Step to start align/retrieval multiplier ramp (defaults to freeze_exchange_steps + 500).",
    )
    parser.add_argument(
        "--retrieval_ramp_steps",
        type=int,
        default=5000,
        help="Steps for align/retrieval multiplier ramp.",
    )
    parser.add_argument(
        "--align_mult_start",
        type=float,
        default=1.0,
        help="Starting multiplier for alignment losses.",
    )
    parser.add_argument(
        "--align_mult_target",
        type=float,
        default=0.25,
        help="Target multiplier for alignment losses after ramp.",
    )
    parser.add_argument(
        "--retrieval_mult_start",
        type=float,
        default=1.0,
        help="Starting multiplier for retrieval losses.",
    )
    parser.add_argument(
        "--retrieval_mult_target",
        type=float,
        default=3.0,
        help="Target multiplier for retrieval losses after ramp.",
    )
    parser.add_argument("--m_ramp_lr_mult", type=float, default=0.1, help="Scale factor for M LR during retrieval ramp window")
    parser.add_argument("--m_ramp_lr_start_mult", type=float, default=0.01, help="Starting M LR scale at ramp start")
    parser.add_argument("--m_ramp_lr_end_mult", type=float, default=0.1, help="Ending M LR scale at ramp end")
    parser.add_argument("--m_ramp_hold_steps", type=int, default=1000, help="Hold M LR at ramp end scale after ramp")
    parser.add_argument("--m_ramp_full_lr_threshold", type=float, default=1.2, help="Max multiple over ramp baseline to allow full M LR")
    parser.add_argument("--m_spectral_norm_max", type=float, default=1.0, help="Max spectral norm for exchange M projection")
    parser.add_argument(
        "--exchange_scale_ramp_steps",
        type=int,
        default=2000,
        help="Steps to linearly ramp exchange injection from 0 to 1 after unfreeze/resume",
    )
    parser.add_argument(
        "--exchange_inj_norm_max",
        type=float,
        default=None,
        help="Clamp exchange injection norm to this value (L2 mean per batch)",
    )
    parser.add_argument(
        "--exchange_inj_norm_ramp_steps",
        type=int,
        default=2000,
        help="Steps to ramp exchange injection clamp from 0 to max after unfreeze/resume",
    )
    parser.add_argument(
        "--disable_m_brake",
        action="store_true",
        help="Disable exchange/M brake on geometry drops",
    )
    parser.add_argument(
        "--m_grad_clip",
        type=float,
        default=0.005,
        help="Clip grad norm for exchange M during ramp/hold window",
    )
    parser.add_argument("--retrieval_eval", action="store_true", help="Run lightweight retrieval eval")
    parser.add_argument("--retrieval_dataset", type=str, default="mteb/scifact")
    parser.add_argument("--retrieval_split", type=str, default="test")
    parser.add_argument("--retrieval_eval_every", type=int, default=1000)
    parser.add_argument("--retrieval_queries", type=int, default=200)
    parser.add_argument("--retrieval_docs", type=int, default=2000)
    parser.add_argument("--retrieval_k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--retrieval_batch_size", type=int, default=256)
    parser.add_argument("--retrieval_seed", type=int, default=1234)
    parser.add_argument("--retrieval_max_corpus", type=int, default=200000)
    parser.add_argument("--retrieval_positives_per_query", type=int, default=1)
    parser.add_argument("--cross_modal_eval", action="store_true", help="Run AudioCaps text->audio retrieval eval")
    parser.add_argument("--cross_modal_dataset", type=str, default="audiocaps/audiocaps")
    parser.add_argument("--cross_modal_split", type=str, default="test")
    parser.add_argument("--cross_modal_eval_every", type=int, default=100)
    parser.add_argument("--cross_modal_queries", type=int, default=200)
    parser.add_argument("--cross_modal_audios", type=int, default=2000)
    parser.add_argument("--cross_modal_k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--cross_modal_seed", type=int, default=1234)
    parser.add_argument("--paired_cross_modal", action="store_true", help="Use paired audio/text batches for cross-modal loss")
    parser.add_argument("--cross_modal_weight", type=float, default=1.0, help="Weight for cross-modal contrastive loss")
    parser.add_argument("--cross_modal_temp", type=float, default=0.07, help="Temperature for cross-modal InfoNCE loss")
    parser.add_argument("--max_grad_norm", type=float, default=50.0, help="Gradient clipping norm")
    parser.add_argument(
        "--disable_grad_clip_guard",
        action="store_true",
        help="Disable gradient clipping guard that aborts after prolonged clipping.",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--debug_cross_modal_pairs", action="store_true", help="Assert/log paired text/audio alignment for cross-modal loss")
    parser.add_argument("--train_retrieval", dest="train_retrieval", action="store_true", help="Train on retrieval query/doc pairs")
    parser.add_argument("--no_train_retrieval", dest="train_retrieval", action="store_false", help="Disable retrieval pair training")
    parser.add_argument("--retrieval_train_dataset", type=str, default="mteb/scifact")
    parser.add_argument("--retrieval_train_split", type=str, default="train")
    parser.add_argument("--retrieval_train_pairs", type=int, default=2000)
    parser.add_argument("--retrieval_train_seed", type=int, default=1234)
    parser.add_argument("--geom_log_b_every", type=int, default=500, help="Log heavier geom metrics every N steps (0 disables).")
    parser.add_argument("--geom_log_c_every", type=int, default=500, help="Log heaviest geom metrics every N steps (0 disables).")
    parser.add_argument("--geom_ema_decay", type=float, default=0.98, help="EMA decay for geom metrics.")
    parser.add_argument("--debug_geom_ema", action="store_true", help="Log extra details for geom EMA metrics.")
    parser.add_argument(
        "--neighborhood_log_every",
        type=int,
        default=50,
        help="Log neighborhood overlap/spearman every N steps (0 disables).",
    )
    parser.add_argument("--max_bytes", type=int, default=1024, help="Max bytes per sample")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (default: 0 with teacher)")
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs (requires num_workers > 0).",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor per worker (requires num_workers > 0).",
    )
    parser.add_argument("--precomputed_teacher_path", type=str, default=None, help="Path to precomputed teacher memmap")
    parser.add_argument("--precomputed_teacher_meta", type=str, default=None, help="Path to teacher metadata json")
    parser.add_argument("--teacher_cache_dir", type=str, default=None, help="Lazy teacher embedding cache directory")
    parser.add_argument("--teacher_cache_prefetch", type=int, default=2, help="Batches to prefetch in background")
    parser.add_argument("--teacher_cache_dir_text", type=str, default=None, help="Cache dir for text teacher")
    parser.add_argument("--teacher_cache_dir_mm", type=str, default=None, help="Cache dir for multimodal teacher")
    parser.add_argument("--teacher_cache_prefetch_text", type=int, default=None, help="Prefetch batches for text teacher")
    parser.add_argument("--teacher_cache_prefetch_mm", type=int, default=None, help="Prefetch batches for multimodal teacher")
    parser.add_argument("--teacher_cache_dtype", type=str, default="float16")
    parser.add_argument("--teacher_cache_embed_dim", type=int, default=None)
    parser.add_argument("--teacher_cache_embed_dim_text", type=int, default=None)
    parser.add_argument("--teacher_cache_embed_dim_mm", type=int, default=None)
    parser.set_defaults(
        train_retrieval=True,
        audio_random_crop=True,
        enable_saturation_penalty=True,
        log_grad_stats=True,
        log_embedding_stats=True,
        log_activation_stats=True,
        use_wandb=True,
    )
    args = parser.parse_args()
    if args.resume is None and args.resume_from:
        args.resume = args.resume_from
    if args.preset:
        os.environ["MONOID_PRESET"] = args.preset.strip().lower()
    original_modality = args.modality
    original_paired_cross_modal = args.paired_cross_modal
    original_cross_modal_eval = args.cross_modal_eval
    original_text_audio_ratio = args.text_audio_ratio
    if args.mode == "text_only_finisher":
        if not args.resume:
            raise ValueError("Finisher mode requires resuming from a checkpoint.")
        args.modality = "text"
        args.paired_cross_modal = False
        args.cross_modal_eval = False
        args.text_audio_ratio = 1.0
    if args.rkd_d_weight is not None:
        args.rkd_distance_weight = args.rkd_d_weight
    if args.rkd_a_weight is not None:
        args.rkd_angle_weight = args.rkd_a_weight
    if args.loss_weight_rkd_distance is not None:
        args.rkd_distance_weight = args.loss_weight_rkd_distance
    if args.loss_weight_rkd_angle is not None:
        args.rkd_angle_weight = args.loss_weight_rkd_angle
    if args.loss_weight_neighborhood is not None:
        args.neighborhood_weight = args.loss_weight_neighborhood
    def _force_text_teacher_only() -> bool:
        if args.modality != "multimodal":
            return False
        try:
            ratio = float(args.text_audio_ratio)
        except Exception:
            ratio = 1.0
        if ratio < 1.0:
            return False
        if args.paired_cross_modal or args.cross_modal_eval:
            return False
        if args.teacher_mm_backend is not None or args.teacher_mm_checkpoint is not None:
            return False
        return True

    force_text_teacher_only = _force_text_teacher_only()
    if args.teacher_backend is None:
        if force_text_teacher_only:
            args.teacher_backend = "embeddinggemma"
        else:
            args.teacher_backend = "gemma3n_hidden" if args.modality == "multimodal" else "embeddinggemma"
    if args.teacher_text_backend is None:
        args.teacher_text_backend = args.teacher_backend
    if args.teacher_cache_dir_text is None:
        args.teacher_cache_dir_text = args.teacher_cache_dir
    if args.teacher_cache_prefetch_text is None:
        args.teacher_cache_prefetch_text = args.teacher_cache_prefetch
    if args.teacher_cache_embed_dim_text is None:
        args.teacher_cache_embed_dim_text = args.teacher_cache_embed_dim
    if args.teacher_mm_repo is None:
        repo_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp", "m2d_repo"))
        if os.path.exists(repo_candidate):
            args.teacher_mm_repo = repo_candidate
    use_dual_teachers = args.teacher_mm_backend is not None or args.teacher_mm_checkpoint is not None
    if args.teacher_cache_prefetch_mm is None:
        args.teacher_cache_prefetch_mm = args.teacher_cache_prefetch
    if args.teacher_cache_embed_dim is None:
        args.teacher_cache_embed_dim = 512 if args.teacher_backend == "gemma3n_hidden" else 512
    if args.teacher_cache_embed_dim_text is None:
        args.teacher_cache_embed_dim_text = args.teacher_cache_embed_dim
    if args.exchange_inj_norm_max is None:
        monoid_preset = os.getenv("MONOID_PRESET", "").lower()
        if monoid_preset in ("medium", "medium_deep"):
            args.exchange_inj_norm_max = 0.0025
        else:
            args.exchange_inj_norm_max = 0.005

    if args.disable_tqdm:
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    spec_version = "v1.2.2"
    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    git_status = _git_value(["git", "status", "--porcelain"])

    def _log_run_summary() -> None:
        monoid_preset = os.getenv("MONOID_PRESET", "")
        logger.info(
            "Run summary: run_name=%s mode=%s monoid_preset=%s output_dir=%s resume=%s use_wandb=%s",
            args.run_name,
            args.mode,
            monoid_preset or "none",
            args.output_dir,
            args.resume,
            args.use_wandb,
        )
        if force_text_teacher_only:
            logger.info("Text-only ratio=1.0; skipping audio teacher loading.")
        logger.info(
            "Compute summary: device=%s compile=%s bptt_chunk=%s grad_accum=%s max_grad_norm=%s",
            device,
            args.compile,
            args.bptt_chunk_size,
            args.grad_accum_steps,
            args.max_grad_norm,
        )
        logger.info(
            "Data summary: dataset=%s/%s:%s dataset2=%s/%s:%s streaming=%s max_steps=%s batch_size=%s max_bytes=%s modality=%s text_audio_ratio=%.2f",
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            args.dataset_name2,
            args.dataset_config2,
            args.dataset_split2,
            args.streaming,
            args.max_steps,
            args.batch_size,
            args.max_bytes,
            args.modality,
            args.text_audio_ratio,
        )
        if args.dataset_mix:
            logger.info("Data mix summary: dataset_mix=%s", args.dataset_mix)
        logger.info(
            "Optimizer summary: optimizer=%s lr_schedule=%s lr=%s peak_lr=%s min_lr=%s warmup_frac=%.4f teacher_lr=%s freeze_exchange_steps=%s",
            args.optimizer,
            args.lr_schedule,
            args.lr,
            args.peak_lr,
            args.min_lr,
            args.warmup_frac,
            args.teacher_lr,
            args.freeze_exchange_steps,
        )
        logger.info(
            "Audio data summary: audio_dataset=%s/%s:%s audio_dataset2=%s/%s:%s mix_ratio=%.2f",
            args.audio_dataset_name,
            args.audio_dataset_config,
            args.audio_dataset_split,
            args.audio_dataset_name2,
            args.audio_dataset_config2,
            args.audio_dataset_split2,
            args.audio_dataset_mix_ratio,
        )
        logger.info(
            "Teacher summary: use_real=%s precomputed=%s backend=%s text_backend=%s mm_backend=%s model_id=%s layer=%s text_prompt=%s prompt_mix=%s mix_mode=%s",
            args.use_real_teacher,
            args.precomputed_teacher_path,
            args.teacher_backend,
            args.teacher_text_backend,
            args.teacher_mm_backend,
            args.teacher_model_id,
            args.teacher_layer,
            args.text_prompt_name,
            args.text_prompt_mix,
            args.text_prompt_mix_mode,
        )
        logger.info(
            "Teacher cache summary: cache=%s cache_text=%s cache_mm=%s prefetch=%s prefetch_text=%s prefetch_mm=%s embed_dim=%s embed_dim_text=%s embed_dim_mm=%s",
            args.teacher_cache_dir,
            args.teacher_cache_dir_text,
            args.teacher_cache_dir_mm,
            args.teacher_cache_prefetch,
            args.teacher_cache_prefetch_text,
            args.teacher_cache_prefetch_mm,
            args.teacher_cache_embed_dim,
            args.teacher_cache_embed_dim_text,
            args.teacher_cache_embed_dim_mm,
        )
        logger.info(
            "Retrieval train summary: train_retrieval=%s dataset=%s split=%s pairs=%s",
            args.train_retrieval,
            args.retrieval_train_dataset,
            args.retrieval_train_split,
            args.retrieval_train_pairs,
        )
        logger.info(
            "Retrieval eval summary: enabled=%s dataset=%s split=%s every=%s queries=%s docs=%s k=%s batch=%s max_corpus=%s",
            args.retrieval_eval,
            args.retrieval_dataset,
            args.retrieval_split,
            args.retrieval_eval_every,
            args.retrieval_queries,
            args.retrieval_docs,
            args.retrieval_k,
            args.retrieval_batch_size,
            args.retrieval_max_corpus,
        )
        logger.info(
            "Finisher summary: steps=%s lr_mult=%s lr_schedule=%s",
            args.finisher_steps,
            args.finisher_lr_mult,
            args.finisher_lr_schedule,
        )

    if args.mode == "text_only_finisher":
        if original_modality != args.modality:
            logger.info("Finisher mode: forcing modality from %s -> %s", original_modality, args.modality)
        if original_paired_cross_modal and not args.paired_cross_modal:
            logger.info("Finisher mode: disabling paired_cross_modal batches.")
        if original_cross_modal_eval and not args.cross_modal_eval:
            logger.info("Finisher mode: disabling cross_modal_eval.")
        if original_text_audio_ratio != args.text_audio_ratio:
            logger.info("Finisher mode: forcing text_audio_ratio=%.2f", args.text_audio_ratio)

    if args.run_name and args.output_dir == "checkpoints":
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        logger.info("Using run-scoped checkpoint directory: %s", args.output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    _log_run_summary()
    if args.modality == "multimodal":
        use_audio_stream = bool(args.paired_cross_modal or args.cross_modal_eval or (args.text_audio_ratio < 1.0))
        if use_audio_stream and args.max_bytes == 1024:
            args.max_bytes = 32768
            logger.info("Multimodal mode: setting max_bytes to %d for audio.", args.max_bytes)
        elif not use_audio_stream:
            logger.info("Multimodal mode: text-only stream enabled; keeping max_bytes=%d.", args.max_bytes)
        if args.train_retrieval:
            logger.info("Multimodal + retrieval: alternating retrieval pairs with mixed batches.")
        if use_dual_teachers and args.text_audio_ratio < 1.0:
            logger.warning(
                "Dual-teacher setup uses text teacher for the main stream; set --text_audio_ratio 1.0 to avoid "
                "audio-only samples without an audio teacher."
            )
    if args.modality in ("audio", "multimodal") and args.audio_random_crop:
        if args.teacher_cache_dir_mm or args.teacher_cache_dir:
            logger.warning(
                "Teacher cache with audio_random_crop may reduce cache hits; consider --no_audio_random_crop."
            )
    resume_log_step = None
    resume_log_contrast_scale = None
    if args.resume:
        resume_log_step, resume_log_contrast_scale = _infer_resume_from_log(args.resume_log)

    # 0. W&B Init
    if args.use_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "monoid"),
            name=args.run_name,
            config=vars(args)
        )

    # 1. Teacher Model
    teacher = None
    teacher_output_dim = None
    teacher_cache_key_extra = None
    teacher_layer_count = None
    teacher_base_dim = None
    text_teacher = None
    text_teacher_output_dim = None
    text_teacher_cache_key_extra = None
    mm_teacher = None
    mm_teacher_output_dim = None
    mm_teacher_cache_key_extra = None
    if args.precomputed_teacher_path:
        if args.use_real_teacher:
            logger.info("Ignoring --use_real_teacher because precomputed embeddings are provided.")
        teacher = None
    elif args.use_real_teacher:
        logger.info("Initializing real teacher models...")
        if use_dual_teachers:
            if args.teacher_text_backend == "gemma3n_hidden":
                text_teacher = Gemma3nHiddenStateTeacher(
                    device=device,
                    model_id=args.teacher_model_id,
                    layer=args.teacher_layer,
                    text_prompt_name=args.text_prompt_name,
                    audio_sample_rate=args.audio_sample_rate,
                    audio_source=args.teacher_audio_source,
                    text_mode=args.teacher_text_mode,
                    stream_idx_text=args.teacher_stream_idx_text,
                    stream_idx_audio=args.teacher_stream_idx_audio,
                    max_bytes=args.max_bytes,
                )
                text_teacher_output_dim = text_teacher.output_dim
                text_teacher_cache_key_extra = text_teacher.cache_key
                teacher_layer_count = text_teacher.layer_count
                teacher_base_dim = text_teacher.base_dim
            else:
                text_teacher = TeacherModelHandler(
                    device=device,
                    modalities=["text"],
                    text_prompt_name=args.text_prompt_name,
                )
                text_teacher_output_dim = 512

            if args.teacher_mm_backend == "gemma3n_hidden":
                mm_teacher = Gemma3nHiddenStateTeacher(
                    device=device,
                    model_id=args.teacher_model_id,
                    layer=args.teacher_layer,
                    text_prompt_name=args.text_prompt_name,
                    audio_sample_rate=args.audio_sample_rate,
                    audio_source=args.teacher_audio_source,
                    text_mode=args.teacher_text_mode,
                    stream_idx_text=args.teacher_stream_idx_text,
                    stream_idx_audio=args.teacher_stream_idx_audio,
                    max_bytes=args.max_bytes,
                )
                mm_teacher_output_dim = mm_teacher.output_dim
                mm_teacher_cache_key_extra = mm_teacher.cache_key
            else:
                if not args.teacher_mm_checkpoint:
                    raise ValueError("M2D-CLAP teacher requires --teacher_mm_checkpoint.")
                mm_teacher = M2DClapTeacher(
                    checkpoint=args.teacher_mm_checkpoint,
                    device=device,
                    repo_path=args.teacher_mm_repo,
                )
                mm_teacher_output_dim = mm_teacher.output_dim
                mm_teacher_cache_key_extra = mm_teacher.cache_key
            text_teacher.eval()
            mm_teacher.eval()
            teacher = text_teacher
            teacher_output_dim = text_teacher_output_dim
        else:
            if args.teacher_backend == "gemma3n_hidden":
                teacher = Gemma3nHiddenStateTeacher(
                    device=device,
                    model_id=args.teacher_model_id,
                    layer=args.teacher_layer,
                    text_prompt_name=args.text_prompt_name,
                    audio_sample_rate=args.audio_sample_rate,
                    audio_source=args.teacher_audio_source,
                    text_mode=args.teacher_text_mode,
                    stream_idx_text=args.teacher_stream_idx_text,
                    stream_idx_audio=args.teacher_stream_idx_audio,
                    max_bytes=args.max_bytes,
                )
                teacher_output_dim = teacher.output_dim
                teacher_cache_key_extra = teacher.cache_key
                teacher_layer_count = teacher.layer_count
                teacher_base_dim = teacher.base_dim
            else:
                if force_text_teacher_only:
                    modalities = ["text"]
                else:
                    modalities = ["text", "audio"] if args.modality == "multimodal" else [args.modality]
                teacher = TeacherModelHandler(
                    device=device,
                    modalities=modalities,
                    text_prompt_name=args.text_prompt_name,
                )
                teacher_output_dim = 512
            teacher.eval()
    else:
        logger.info("Using dummy teacher (random embeddings)...")
        teacher = None

    if args.teacher_cache_dir_text and text_teacher is None and teacher is None and not args.precomputed_teacher_path:
        logger.warning("Text teacher cache enabled without a teacher; cache misses will raise.")
    if args.teacher_cache_dir_mm and mm_teacher is None and not args.precomputed_teacher_path:
        logger.warning("MM teacher cache enabled without a teacher; cache misses will raise.")
    if text_teacher_output_dim is None:
        text_teacher_output_dim = teacher_output_dim
    if text_teacher_cache_key_extra is None:
        text_teacher_cache_key_extra = teacher_cache_key_extra
    if text_teacher_output_dim is not None and args.teacher_cache_embed_dim_text != text_teacher_output_dim:
        logger.info(
            "Adjusting teacher_cache_embed_dim_text from %s to %s to match teacher output.",
            args.teacher_cache_embed_dim_text,
            text_teacher_output_dim,
        )
        args.teacher_cache_embed_dim_text = text_teacher_output_dim
    if mm_teacher_output_dim is not None and args.teacher_cache_embed_dim_mm != mm_teacher_output_dim:
        logger.info(
            "Adjusting teacher_cache_embed_dim_mm from %s to %s to match teacher output.",
            args.teacher_cache_embed_dim_mm,
            mm_teacher_output_dim,
        )
        args.teacher_cache_embed_dim_mm = mm_teacher_output_dim

    def _parse_dataset_mix():
        if not args.dataset_mix:
            return []
        specs = []
        for raw in args.dataset_mix:
            parts = [part.strip() for part in raw.split(":")]
            if not parts or not parts[0]:
                raise ValueError(f"Invalid dataset_mix entry: {raw}")
            name = parts[0]
            config = parts[1] if len(parts) > 1 and parts[1] else None
            split = parts[2] if len(parts) > 2 and parts[2] else None
            weight = 1.0
            if len(parts) > 3 and parts[3]:
                weight = float(parts[3])
            if len(parts) > 4:
                raise ValueError(f"Invalid dataset_mix entry (too many fields): {raw}")
            if weight <= 0.0:
                raise ValueError(f"dataset_mix weight must be > 0: {raw}")
            specs.append(
                {
                    "name": name,
                    "config": config,
                    "split": split,
                    "weight": weight,
                }
            )
        return specs

    dataset_mix_specs = _parse_dataset_mix()

    if args.precomputed_teacher_path and (args.dataset_name2 or dataset_mix_specs):
        raise ValueError("Dataset mixing is not supported with precomputed teacher embeddings.")
    
    if dataset_mix_specs and args.dataset_name2:
        logger.warning("dataset_mix provided; ignoring dataset_name2 mix settings.")

    def _format_dataset_mix_label(specs):
        parts = []
        for spec in specs:
            name = spec["name"]
            config = spec["config"]
            split = spec["split"] if spec["split"] is not None else args.dataset_split
            label = name
            if config:
                label = f"{label}/{config}"
            if split:
                label = f"{label}:{split}"
            label = f"{label}@{spec['weight']:.2f}"
            parts.append(label)
        return "+".join(parts)

    if args.modality == "multimodal":
        text_label = args.dataset_name
        if args.dataset_config:
            text_label = f"{text_label}/{args.dataset_config}"
        if args.dataset_split:
            text_label = f"{text_label}:{args.dataset_split}"
        if dataset_mix_specs:
            text_label = _format_dataset_mix_label(dataset_mix_specs)
        elif args.dataset_name2:
            ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
            ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
            ds2_label = args.dataset_name2
            if ds2_config:
                ds2_label = f"{ds2_label}/{ds2_config}"
            if ds2_split:
                ds2_label = f"{ds2_label}:{ds2_split}"
            text_label = f"{text_label}+{ds2_label}@{args.dataset_mix_ratio:.2f}"

        audio_label = args.audio_dataset_name
        if args.audio_dataset_config:
            audio_label = f"{audio_label}/{args.audio_dataset_config}"
        if args.audio_dataset_split:
            audio_label = f"{audio_label}:{args.audio_dataset_split}"
        if args.audio_dataset_name2:
            ds2_config = args.audio_dataset_config2 if args.audio_dataset_config2 is not None else args.audio_dataset_config
            ds2_split = args.audio_dataset_split2 if args.audio_dataset_split2 is not None else args.audio_dataset_split
            ds2_label = args.audio_dataset_name2
            if ds2_config:
                ds2_label = f"{ds2_label}/{ds2_config}"
            if ds2_split:
                ds2_label = f"{ds2_label}:{ds2_split}"
            audio_label = f"{audio_label}+{ds2_label}@{args.audio_dataset_mix_ratio:.2f}"

        dataset_label = f"text={text_label} audio={audio_label} mix={args.text_audio_ratio:.2f}"
    else:
        dataset_label = args.dataset_name
        if args.dataset_config:
            dataset_label = f"{dataset_label}/{args.dataset_config}"
        if args.dataset_split:
            dataset_label = f"{dataset_label}:{args.dataset_split}"
        if dataset_mix_specs:
            dataset_label = _format_dataset_mix_label(dataset_mix_specs)
        elif args.dataset_name2:
            ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
            ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
            ds2_label = args.dataset_name2
            if ds2_config:
                ds2_label = f"{ds2_label}/{ds2_config}"
            if ds2_split:
                ds2_label = f"{ds2_label}:{ds2_split}"
            dataset_label = f"{dataset_label}+{ds2_label}@{args.dataset_mix_ratio:.2f}"
    if args.train_retrieval:
        dataset_label = f"{args.retrieval_train_dataset}/{args.retrieval_train_split}"
    logger.info(f"Loading dataset {dataset_label} (Streaming: {args.streaming or args.train_retrieval})...")

    prompt_mix = [p.strip() for p in args.text_prompt_mix.split(",") if p.strip()]
    prompt_mix = prompt_mix or None

    if args.modality == "text" and args.bptt_chunk_size:
        logger.info("Disabling BPTT for text to avoid padding-only chunks.")
        args.bptt_chunk_size = 0

    def _load_base_dataset(name, config, split, streaming, expect_audio: bool = False):
        ds = load_dataset(
            name,
            config,
            split=split,
            streaming=streaming,
            trust_remote_code=args.datasets_trust_remote_code,
        )
        if expect_audio:
            try:
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as exc:
                logger.warning("Audio decode override failed for %s: %s", name, exc)
        return ds

    def _load_text_mix_dataset(streaming):
        if not dataset_mix_specs:
            return None
        datasets = []
        weights = []
        for spec in dataset_mix_specs:
            config = spec["config"]
            split = spec["split"] if spec["split"] is not None else args.dataset_split
            ds = _load_base_dataset(spec["name"], config, split, streaming)
            datasets.append(ds)
            weights.append(float(spec["weight"]))
        if len(datasets) == 1:
            return datasets[0]
        total = sum(weights)
        if total <= 0:
            raise ValueError("dataset_mix weights must sum to > 0.")
        probabilities = [w / total for w in weights]
        return interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=args.dataset_mix_seed,
            stopping_strategy="first_exhausted",
        )

    def _maybe_mix_datasets(primary, secondary):
        if secondary is None:
            return primary
        prob = min(1.0, max(0.0, float(args.dataset_mix_ratio)))
        if prob in (0.0, 1.0):
            return secondary if prob == 0.0 else primary
        return interleave_datasets(
            [primary, secondary],
            probabilities=[prob, 1.0 - prob],
            seed=args.dataset_mix_seed,
            stopping_strategy="first_exhausted",
        )

    def _maybe_mix_audio_datasets(primary, secondary):
        if secondary is None:
            return primary
        prob = min(1.0, max(0.0, float(args.audio_dataset_mix_ratio)))
        if prob in (0.0, 1.0):
            return secondary if prob == 0.0 else primary
        return interleave_datasets(
            [primary, secondary],
            probabilities=[prob, 1.0 - prob],
            seed=args.dataset_mix_seed,
            stopping_strategy="first_exhausted",
        )

    def _tag_dataset_modality(ds, modality: str):
        def _annotate(example):
            update = {"modality": modality}
            if "text" not in example:
                update["text"] = None
            if "audio" not in example:
                update["audio"] = None
            return update
        return ds.map(_annotate)

    retrieval = None
    if args.retrieval_eval:
        retrieval = _build_retrieval_eval(args, logger)
    cross_modal = None
    if args.cross_modal_eval:
        cross_modal = _build_cross_modal_eval(args, logger)
    
    if args.precomputed_teacher_path and (args.streaming or args.train_retrieval):
        raise ValueError("Precomputed teacher embeddings require non-streaming, non-retrieval training.")
    if args.precomputed_teacher_path and args.modality == "multimodal":
        raise ValueError("Precomputed teacher embeddings are not supported for multimodal mode.")

    dataset_teacher = None  # Dataset returns raw inputs; avoid pickling GPU teacher in workers.

    train_ds = None
    retrieval_train_ds = None
    paired_train_ds = None
    if args.train_retrieval:
        pairs = _build_retrieval_pairs(args, logger)
        retrieval_train_ds = RetrievalPairsDataset(
            pairs,
            modality=args.modality,
            max_bytes=args.max_bytes,
            teacher=dataset_teacher,
            text_prompt_mix=prompt_mix,
            text_prompt_mix_mode=args.text_prompt_mix_mode,
            audio_sample_rate=args.audio_sample_rate,
            audio_random_crop=args.audio_random_crop,
        )
        if args.modality != "multimodal":
            train_ds = retrieval_train_ds
            shuffle = True
    if args.paired_cross_modal:
        paired_base = _load_base_dataset(
            args.audio_dataset_name,
            args.audio_dataset_config,
            args.audio_dataset_split,
            streaming=args.streaming,
            expect_audio=True,
        )
        if args.streaming:
            paired_train_ds = PairedAudioTextIterableDataset(
                paired_base,
                max_bytes=args.max_bytes,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
                audio_max_seconds=args.paired_audio_max_seconds,
            )
        else:
            paired_train_ds = PairedAudioTextDataset(
                paired_base,
                max_bytes=args.max_bytes,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
                audio_max_seconds=args.paired_audio_max_seconds,
            )
        logger.info(
            "Paired cross-modal dataset: %s (streaming=%s)",
            args.audio_dataset_name,
            args.streaming,
        )
    if args.modality == "multimodal" and (not args.train_retrieval or train_ds is None):
        from monoid.training.embed.data import MonoidIterableDataset
        prob_text = min(1.0, max(0.0, float(args.text_audio_ratio)))
        if args.streaming:
            text_ds = None
            audio_ds = None
            if prob_text > 0.0:
                text_ds = _load_text_mix_dataset(streaming=True)
                if text_ds is None:
                    text_ds = _load_base_dataset(
                        args.dataset_name,
                        args.dataset_config,
                        args.dataset_split,
                        streaming=True,
                        expect_audio=False,
                    )
                    if args.dataset_name2:
                        ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
                        ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
                        text_ds2 = _load_base_dataset(
                            args.dataset_name2,
                            ds2_config,
                            ds2_split,
                            streaming=True,
                            expect_audio=False,
                        )
                        text_ds = _maybe_mix_datasets(text_ds, text_ds2)
            if prob_text < 1.0:
                audio_ds = _load_base_dataset(
                    args.audio_dataset_name,
                    args.audio_dataset_config,
                    args.audio_dataset_split,
                    streaming=True,
                    expect_audio=True,
                )
                if args.audio_dataset_name2:
                    ds2_config = args.audio_dataset_config2 if args.audio_dataset_config2 is not None else args.audio_dataset_config
                    ds2_split = args.audio_dataset_split2 if args.audio_dataset_split2 is not None else args.audio_dataset_split
                    audio_ds2 = _load_base_dataset(
                        args.audio_dataset_name2,
                        ds2_config,
                        ds2_split,
                        streaming=True,
                        expect_audio=True,
                    )
                    audio_ds = _maybe_mix_audio_datasets(audio_ds, audio_ds2)
            if prob_text == 1.0:
                mixed_ds = text_ds
            elif prob_text == 0.0:
                mixed_ds = audio_ds
            else:
                mixed_ds = interleave_datasets(
                    [text_ds, audio_ds],
                    probabilities=[prob_text, 1.0 - prob_text],
                    seed=args.dataset_mix_seed,
                    stopping_strategy="first_exhausted",
                )
            train_ds = MonoidIterableDataset(
                mixed_ds,
                modality="multimodal",
                max_bytes=args.max_bytes,
                teacher=dataset_teacher,
                text_prompt_mix=prompt_mix,
                text_prompt_mix_mode=args.text_prompt_mix_mode,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
            )
            shuffle = False
        else:
            text_ds = None
            audio_ds = None
            if prob_text > 0.0:
                text_ds = _load_text_mix_dataset(streaming=False)
                if text_ds is None:
                    text_ds = _load_base_dataset(
                        args.dataset_name,
                        args.dataset_config,
                        args.dataset_split,
                        streaming=False,
                        expect_audio=False,
                    )
                    if args.dataset_name2:
                        ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
                        ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
                        text_ds2 = _load_base_dataset(
                            args.dataset_name2,
                            ds2_config,
                            ds2_split,
                            streaming=False,
                            expect_audio=False,
                        )
                        text_ds = _maybe_mix_datasets(text_ds, text_ds2)
                text_ds = _tag_dataset_modality(text_ds, "text")
            if prob_text < 1.0:
                audio_ds = _load_base_dataset(
                    args.audio_dataset_name,
                    args.audio_dataset_config,
                    args.audio_dataset_split,
                    streaming=False,
                    expect_audio=True,
                )
                if args.audio_dataset_name2:
                    ds2_config = args.audio_dataset_config2 if args.audio_dataset_config2 is not None else args.audio_dataset_config
                    ds2_split = args.audio_dataset_split2 if args.audio_dataset_split2 is not None else args.audio_dataset_split
                    audio_ds2 = _load_base_dataset(
                        args.audio_dataset_name2,
                        ds2_config,
                        ds2_split,
                        streaming=False,
                        expect_audio=True,
                    )
                    audio_ds = _maybe_mix_audio_datasets(audio_ds, audio_ds2)
                audio_ds = _tag_dataset_modality(audio_ds, "audio")
            if prob_text == 1.0:
                mixed_ds = text_ds
            elif prob_text == 0.0:
                mixed_ds = audio_ds
            else:
                mixed_ds = interleave_datasets(
                    [text_ds, audio_ds],
                    probabilities=[prob_text, 1.0 - prob_text],
                    seed=args.dataset_mix_seed,
                    stopping_strategy="first_exhausted",
                )
            train_ds = MonoidDataset(
                mixed_ds,
                modality="multimodal",
                max_bytes=args.max_bytes,
                teacher=dataset_teacher,
                text_prompt_mix=prompt_mix,
                text_prompt_mix_mode=args.text_prompt_mix_mode,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
            )
            shuffle = True
    elif train_ds is None and args.streaming:
        # Load as streaming IterableDataset
        from monoid.training.embed.data import MonoidIterableDataset
        hf_ds = _load_text_mix_dataset(streaming=True)
        if hf_ds is None:
            hf_ds = _load_base_dataset(
                args.dataset_name,
                args.dataset_config,
                args.dataset_split,
                streaming=True,
                expect_audio=(args.modality == "audio"),
            )
            if args.dataset_name2:
                ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
                ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
                hf_ds2 = _load_base_dataset(
                    args.dataset_name2,
                    ds2_config,
                    ds2_split,
                    streaming=True,
                    expect_audio=(args.modality == "audio"),
                )
                hf_ds = _maybe_mix_datasets(hf_ds, hf_ds2)
        # Using iterable dataset
        train_ds = MonoidIterableDataset(
            hf_ds,
            modality=args.modality,
            max_bytes=args.max_bytes,
            teacher=dataset_teacher,
            text_prompt_mix=prompt_mix,
            text_prompt_mix_mode=args.text_prompt_mix_mode,
            audio_sample_rate=args.audio_sample_rate,
            audio_random_crop=args.audio_random_crop,
        )
        shuffle = False # Cannot shuffle iterable easy, shuffling buffer inside dataset (not impl yet) or HF shuffle
    elif train_ds is None:
        # Map-style (default behavior kept for small datasets/tests)
        try:
            if args.precomputed_teacher_path:
                hf_ds = _load_base_dataset(
                    args.dataset_name,
                    args.dataset_config,
                    args.dataset_split,
                    streaming=False,
                    expect_audio=(args.modality == "audio"),
                )
                if args.dataset_name2:
                    ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
                    ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
                    hf_ds2 = _load_base_dataset(
                        args.dataset_name2,
                        ds2_config,
                        ds2_split,
                        streaming=False,
                        expect_audio=(args.modality == "audio"),
                    )
                    hf_ds = _maybe_mix_datasets(hf_ds, hf_ds2)
                hf_subset = hf_ds
            else:
                hf_ds = _load_text_mix_dataset(streaming=True)
                if hf_ds is None:
                    hf_ds = _load_base_dataset(
                        args.dataset_name,
                        args.dataset_config,
                        args.dataset_split,
                        streaming=True,
                        expect_audio=(args.modality == "audio"),
                    )
                    if args.dataset_name2:
                        ds2_config = args.dataset_config2 if args.dataset_config2 is not None else args.dataset_config
                        ds2_split = args.dataset_split2 if args.dataset_split2 is not None else args.dataset_split
                        hf_ds2 = _load_base_dataset(
                            args.dataset_name2,
                            ds2_config,
                            ds2_split,
                            streaming=True,
                            expect_audio=(args.modality == "audio"),
                        )
                        hf_ds = _maybe_mix_datasets(hf_ds, hf_ds2)
                hf_subset = list(hf_ds.take(1000))
        except Exception as e:
            logger.warning(f"Fallback to dummy: {e}")
            hf_subset = [{'text': 'Hello world ' * 10}] * 100

        if args.precomputed_teacher_path:
            import json as json_lib
            from monoid.training.embed.data import MonoidPrecomputedTeacherDataset
            embed_dim = 512
            dtype = "float16"
            max_samples = None
            if args.precomputed_teacher_meta:
                with open(args.precomputed_teacher_meta, "r", encoding="utf-8") as f:
                    meta = json_lib.load(f)
                embed_dim = int(meta.get("embed_dim", embed_dim))
                dtype = meta.get("dtype", dtype)
                max_samples = meta.get("num_samples")
            if max_samples is not None and hasattr(hf_subset, "select"):
                hf_subset = hf_subset.select(range(int(max_samples)))
            train_ds = MonoidPrecomputedTeacherDataset(
                hf_subset,
                embeddings_path=args.precomputed_teacher_path,
                embed_dim=embed_dim,
                dtype=dtype,
                modality=args.modality,
                max_bytes=args.max_bytes,
                teacher=dataset_teacher,
                text_prompt_mix=prompt_mix,
                text_prompt_mix_mode=args.text_prompt_mix_mode,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
            )
        else:
            train_ds = MonoidDataset(
                hf_subset,
                modality=args.modality,
                max_bytes=args.max_bytes,
                teacher=dataset_teacher,
                text_prompt_mix=prompt_mix,
                text_prompt_mix_mode=args.text_prompt_mix_mode,
                audio_sample_rate=args.audio_sample_rate,
                audio_random_crop=args.audio_random_crop,
            )
        shuffle = True

    num_workers = args.num_workers
    if num_workers is None:
        if args.precomputed_teacher_path:
            num_workers = 4
        else:
            num_workers = 0 if args.use_real_teacher else 4
    use_cache_threads = True
    if (args.teacher_cache_dir_text or args.teacher_cache_dir_mm or args.teacher_cache_dir) and num_workers and num_workers > 0:
        use_cache_threads = False
        logger.info(
            "Teacher cache enabled with num_workers>0; disabling cache prefetch threads to avoid deadlocks."
        )
    drop_last = not args.train_retrieval
    def _loader_kwargs_for(workers: int):
        if workers and workers > 0:
            kwargs = {
                "multiprocessing_context": "spawn",
                "persistent_workers": bool(args.persistent_workers),
            }
            if args.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(args.prefetch_factor)
            return kwargs
        return {}

    loader_kwargs = _loader_kwargs_for(num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last, # Good for static shapes optimization
        **loader_kwargs,
    )
    retrieval_train_loader = None
    if retrieval_train_ds is not None and args.modality == "multimodal":
        retrieval_train_loader = DataLoader(
            retrieval_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            **_loader_kwargs_for(num_workers),
        )
    paired_train_loader = None
    if paired_train_ds is not None:
        paired_num_workers = num_workers
        paired_train_loader = DataLoader(
            paired_train_ds,
            batch_size=args.batch_size,
            shuffle=not args.streaming,
            num_workers=paired_num_workers,
            pin_memory=True,
            collate_fn=collate_paired_fn,
            drop_last=True,
            **_loader_kwargs_for(paired_num_workers),
        )

    # 2. Model (Monoid only)
    config = MonoidEmbedConfig()
    logger.info(
        "Monoid config: n_layers=%s d_state=%s n_tiles=%s tile_dim=%s microblock=%s exchange_dim=%s matryoshka_dims=%s use_exchange=%s exchange_every=%s second_activation=%s activation_shift=%s activation_T=%s b_shift=%s pool=%s normalize_output=%s emit_int8=%s a_min=%.4f a_max=%.4f",
        config.n_layers,
        config.d_state,
        config.n_tiles,
        config.tile_dim,
        config.microblock_size,
        config.exchange_dim,
        config.matryoshka_dims,
        config.use_exchange,
        config.exchange_every,
        config.use_second_activation,
        config.activation_shift,
        config.activation_T,
        config.b_shift,
        config.pool_strategy,
        config.normalize_output,
        config.emit_int8,
        config.a_min,
        config.a_max,
    )
    model = MonoidEmbed(config).to(device)
    if args.bptt_chunk_size:
        logger.info("Disabling BPTT for MonoidEmbed.")
        args.bptt_chunk_size = 0

    use_compile = bool(args.compile)

    param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    m_leaf_name = None
    m_leaf_param = None
    m_leaf_shape = None
    exchange_dim = getattr(config, "exchange_dim", None)
    if exchange_dim:
        exchange_dim = int(exchange_dim)
    else:
        exchange_dim = None
    m_leaf_name, m_leaf_param = find_exchange_leaf_param(model, exchange_dim)
    if m_leaf_param is None:
        logger.warning("Could not locate exchange leaf parameter for monoid.")
    else:
        m_leaf_shape = tuple(m_leaf_param.shape)
    
    layer_mix = None
    layer_mix_ema = None
    teacher_proj = None
    teacher_proj_ema = None
    teacher_target_dim = max(model.config.matryoshka_dims)
    teacher_mixed_dim = teacher_output_dim
    if teacher is not None and teacher_layer_count and teacher_base_dim:
        if teacher_layer_count >= 2:
            layer_mix = nn.Parameter(torch.zeros(teacher_layer_count, device=device))
            layer_mix_ema = layer_mix.detach().clone()
            teacher_mixed_dim = teacher_base_dim
    if teacher is not None and teacher_mixed_dim is not None and teacher_mixed_dim != teacher_target_dim:
        teacher_proj = nn.Linear(teacher_mixed_dim, teacher_target_dim).to(device)
        teacher_proj_ema = nn.Linear(teacher_mixed_dim, teacher_target_dim).to(device)
        teacher_proj_ema.load_state_dict(teacher_proj.state_dict())
        for p in teacher_proj_ema.parameters():
            p.requires_grad = False
        logger.info(
            "Training teacher projection: %d -> %d",
            teacher_mixed_dim,
            teacher_target_dim,
        )
    teacher_raw_dim = teacher_output_dim if teacher_output_dim is not None else teacher_target_dim
    train_teacher = bool(teacher_proj is not None or layer_mix is not None)

    # 3. Optim & Loss
    if args.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {args.optimizer} (only adamw)")
    m_param_group_index = None
    if m_leaf_param is not None:
        m_leaf_id = id(m_leaf_param)
        non_m_params = [param for param in model.parameters() if id(param) != m_leaf_id]
        param_groups = [{"params": non_m_params, "lr": args.lr}]
        param_groups.append({"params": [m_leaf_param], "lr": args.lr})
        m_param_group_index = 1
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]
    if teacher_proj is not None:
        param_groups.append({"params": teacher_proj.parameters(), "lr": args.teacher_lr})
    if layer_mix is not None:
        param_groups.append({"params": [layer_mix], "lr": args.teacher_lr})
    optimizer = optim.AdamW(param_groups)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    schedule_base_lrs = [float(group.get("lr", args.lr)) for group in optimizer.param_groups]
    m_leaf_in_optimizer = None
    if m_leaf_param is not None:
        m_leaf_in_optimizer = id(m_leaf_param) in optimizer_param_ids
        if not m_leaf_in_optimizer:
            raise AssertionError("M.leaf not found in optimizer param groups.")
    if args.lr_schedule != "constant":
        schedule_base_lrs[0] = float(args.peak_lr)
        optimizer.param_groups[0]["lr"] = schedule_base_lrs[0]

    resume_step = None
    resume_ramp_start = None
    resume_contrast_scale = None
    if args.resume:
        logger.info("Loading checkpoint for resume: %s", args.resume)
        state = torch.load(args.resume, map_location=device)
        (
            model_state,
            optimizer_state,
            ckpt_step,
            ckpt_ramp_start,
            teacher_proj_state,
            layer_mix_state,
            teacher_proj_ema_state,
            layer_mix_ema_state,
        ) = _extract_resume_state(state)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            logger.warning("Resume checkpoint missing keys: %s", missing)
        if unexpected:
            logger.warning("Resume checkpoint unexpected keys: %s", unexpected)
        if teacher_proj is not None and teacher_proj_state is not None:
            missing_proj, unexpected_proj = teacher_proj.load_state_dict(teacher_proj_state, strict=False)
            if missing_proj:
                logger.warning("Resume checkpoint missing teacher_proj keys: %s", missing_proj)
            if unexpected_proj:
                logger.warning("Resume checkpoint unexpected teacher_proj keys: %s", unexpected_proj)
        if teacher_proj_ema is not None and teacher_proj_ema_state is not None:
            missing_proj, unexpected_proj = teacher_proj_ema.load_state_dict(teacher_proj_ema_state, strict=False)
            if missing_proj:
                logger.warning("Resume checkpoint missing teacher_proj_ema keys: %s", missing_proj)
            if unexpected_proj:
                logger.warning("Resume checkpoint unexpected teacher_proj_ema keys: %s", unexpected_proj)
        if layer_mix is not None and layer_mix_state is not None:
            try:
                layer_mix.data.copy_(layer_mix_state.to(layer_mix.device))
            except Exception as exc:
                logger.warning("Failed to load layer_mix: %s", exc)
        if layer_mix_ema is not None and layer_mix_ema_state is not None:
            try:
                layer_mix_ema.data.copy_(layer_mix_ema_state.to(layer_mix_ema.device))
            except Exception as exc:
                logger.warning("Failed to load layer_mix_ema: %s", exc)
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                logger.info("Loaded optimizer state from resume checkpoint.")
            except Exception as exc:
                logger.warning("Failed to load optimizer state: %s", exc)
        else:
            logger.info("Resume checkpoint has no optimizer state; optimizer will reset.")

        resume_step = args.resume_step
        if resume_step is None:
            resume_step = ckpt_step
        if resume_step is None:
            resume_step = _parse_step_from_checkpoint(args.resume)

        resume_contrast_scale = resume_log_contrast_scale
        if resume_log_step is not None:
            if resume_step is None:
                resume_step = resume_log_step
            elif resume_step != resume_log_step:
                logger.warning(
                    "Resume step %s does not match log step %s from %s",
                    resume_step,
                    resume_log_step,
                    args.resume_log,
                )

        resume_ramp_start = ckpt_ramp_start
        if resume_ramp_start is None and resume_contrast_scale is not None and resume_step is not None:
            ramp_steps = max(1, args.contrast_ramp_steps)
            scale = min(1.0, max(0.0, resume_contrast_scale))
            elapsed = int(round(scale * ramp_steps))
            resume_ramp_start = max(0, resume_step - elapsed)

        if resume_step is None:
            logger.warning("Resume step could not be inferred; defaulting to 0.")
            resume_step = 0
        logger.info("Resuming training from step %d", resume_step)

    if use_compile:
        logger.info("Compiling model (torch.compile)...")
        try:
            import torch._inductor.config as inductor_config

            disabled = False
            if hasattr(inductor_config, "cudagraphs"):
                inductor_config.cudagraphs = False
                disabled = True
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
                disabled = True
            if disabled:
                logger.info("torch.compile: disabled cudagraphs for stability.")
            else:
                logger.warning("torch.compile: cudagraphs config not available to disable.")
        except Exception as exc:
            logger.warning("torch.compile: failed to disable cudagraphs (%s).", exc)
        # Use no-cudagraphs mode to avoid capture-unsafe ops (e.g., SVD).
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    exchange_frozen = False
    exchange_freeze_steps = max(0, int(args.freeze_exchange_steps))
    exchange_scale_start = exchange_freeze_steps
    if exchange_freeze_steps > 0 and getattr(config, "use_exchange", False):
        if resume_step is None or resume_step < exchange_freeze_steps:
            zero_exchange_weights(model)
            set_exchange_trainable(model, False)
            set_exchange_disabled(model, True)
            exchange_frozen = True
            logger.info("Freezing exchange weights for %d steps.", exchange_freeze_steps)
            if resume_step is not None:
                logger.info("Resume step %d < freeze_exchange_steps; zeroed exchange weights.", resume_step)
        else:
            set_exchange_trainable(model, True)
            set_exchange_disabled(model, False)

    def _save_checkpoint(step: int) -> None:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, f"{step}.pt")
        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "contrast_ramp_start_step": contrast_ramp_start_step,
        }
        if teacher_proj is not None:
            ckpt_state["teacher_proj"] = teacher_proj.state_dict()
        if teacher_proj_ema is not None:
            ckpt_state["teacher_proj_ema"] = teacher_proj_ema.state_dict()
        if layer_mix is not None:
            ckpt_state["layer_mix"] = layer_mix.detach().cpu()
        if layer_mix_ema is not None:
            ckpt_state["layer_mix_ema"] = layer_mix_ema.detach().cpu()
        torch.save(ckpt_state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)

        latest_path = os.path.join(args.output_dir, "monoid_embed_latest.pt")
        torch.save(model.state_dict(), latest_path)
        latest_full_path = os.path.join(args.output_dir, "monoid_embed_latest_full.pt")
        torch.save(ckpt_state, latest_full_path)

    quit_event = None
    quit_state = {"save": True}
    if sys.stdin is not None and sys.stdin.isatty():
        quit_event = threading.Event()

        def _watch_keyboard():
            while True:
                ch = sys.stdin.read(1)
                if not ch:
                    return
                if ch.lower() == "q":
                    print("\nQuit requested. Save checkpoint before quitting? [Y/n]: ", end="", flush=True)
                    try:
                        response = sys.stdin.readline()
                    except Exception:
                        response = ""
                    response = (response or "").strip().lower()
                    quit_state["save"] = not response.startswith("n")
                    quit_event.set()
                    return

        threading.Thread(target=_watch_keyboard, daemon=True).start()
    else:
        logger.info("Keyboard watcher disabled (stdin not a TTY).")
    
    criterion_distill = GeometricDistillationLoss().to(device)
    criterion_spread = SpreadOutRegularizer().to(device)
    criterion_contrast = HardnessWeightedContrastiveLoss(temperature=0.05, alpha_hardness=5.0).to(device)
    criterion_consistency = ConsistencyLoss().to(device)
    criterion_pairwise = PairwiseCosineDistillationLoss().to(device)
    criterion_spkd = SimilarityPreservingKDLoss().to(device)
    criterion_rkd_distance = RKDDistanceLoss().to(device)
    criterion_rkd_angle = RKDAngleLoss().to(device)
    criterion_var = VICRegVarianceLoss(variance_floor=1.0).to(device)
    criterion_neighborhood = NeighborhoodDistillationLoss(temperature=args.neighborhood_temp).to(device)
    
    # Weights for losses
    finisher_mode = args.mode == "text_only_finisher"
    text_teacher_scale = float(args.teacher_text_weight)
    w_distill = args.distill_weight * text_teacher_scale
    w_spread = args.w_spread
    w_contrast = args.contrast_weight * text_teacher_scale
    w_consistency = 1.0
    w_pairwise = args.w_pairwise * text_teacher_scale
    w_spkd = args.spkd_weight * text_teacher_scale
    w_rkd_distance = args.rkd_distance_weight * text_teacher_scale
    w_rkd_angle = args.rkd_angle_weight * text_teacher_scale
    w_var = args.var_weight
    w_neighborhood = args.neighborhood_weight * text_teacher_scale
    w_cross_modal = args.cross_modal_weight
    w_mm_distill = args.mm_distill_weight
    w_var_audio = 1.0
    if finisher_mode:
        w_cross_modal = 0.0
        w_mm_distill = 0.0
        w_var_audio = 0.0

    use_matryoshka = not args.no_matryoshka
    matryoshka_dims = model.config.matryoshka_dims if use_matryoshka else (max(model.config.matryoshka_dims),)
    if args.matryoshka_weights:
        if len(args.matryoshka_weights) != len(matryoshka_dims):
            raise ValueError("matryoshka_weights must match matryoshka_dims length")
        matryoshka_weights = list(args.matryoshka_weights)
        matryoshka_weight_sum = float(sum(matryoshka_weights))
    else:
        matryoshka_weights = None
        matryoshka_weight_sum = 1.0
    matryoshka_base_weights = {512: 1.0, 256: 0.7, 128: 0.5}
    if spec_version == "v1.2.2" and not finisher_mode:
        if args.matryoshka_weights is not None:
            weight_map = {dim: weight for dim, weight in zip(matryoshka_dims, args.matryoshka_weights)}
            w256 = weight_map.get(256, None)
            w128 = weight_map.get(128, None)
            w512 = weight_map.get(512, None)
            if w512 != 1.0 or w256 != 0.7 or w128 != 0.5:
                raise ValueError(
                    "Spec v1.2.2 requires matryoshka weights w512=1.0 w256=0.7 w128=0.5; "
                    f"got w512={w512} w256={w256} w128={w128}."
                )
        else:
            if (
                matryoshka_base_weights.get(512) != 1.0
                or matryoshka_base_weights.get(256) != 0.7
                or matryoshka_base_weights.get(128) != 0.5
            ):
                raise ValueError(
                    "Spec v1.2.2 requires matryoshka base weights w512=1.0 w256=0.7 w128=0.5."
                )
    contrast_ramp_start_step = None
    finisher_start_step = None
    finisher_end_step = None
    finisher_base_lrs = None
    finisher_lr_start = None
    finisher_debug_logged = False
    run_header_logged = False

    # 4. Loop
    model.train()
    if teacher_proj is not None:
        teacher_proj.train()
    if teacher_proj_ema is not None:
        teacher_proj_ema.eval()
    
    total_steps = 0
    if resume_step is not None:
        total_steps = int(resume_step)
    if resume_ramp_start is not None:
        contrast_ramp_start_step = min(int(resume_ramp_start), total_steps)
    if finisher_mode:
        finisher_start_step = total_steps
        finisher_end_step = total_steps + int(args.finisher_steps)
        logger.info(
            "Finisher mode: start_step=%d end_step=%d (finisher_steps=%d)",
            finisher_start_step,
            finisher_end_step,
            args.finisher_steps,
        )
        args.max_steps = finisher_end_step
        finisher_base_lrs = []
        for group in optimizer.param_groups:
            base_lr = float(group.get("lr", args.lr))
            new_lr = base_lr * float(args.finisher_lr_mult)
            group["lr"] = new_lr
            finisher_base_lrs.append(new_lr)
        if finisher_base_lrs:
            finisher_lr_start = finisher_base_lrs[0]
    geom_ema_state = {}
    streaming_mode = args.streaming or args.train_retrieval
    max_steps = args.max_steps if streaming_mode else len(train_loader) * args.epochs
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    warmup_steps = 0
    if not finisher_mode and args.lr_schedule != "constant":
        warmup_steps = int(round(float(args.warmup_frac) * float(max_steps)))
        warmup_steps = max(1, warmup_steps) if max_steps > 0 else 0
    unroll_microblocks = 0
    if args.bptt_chunk_size and getattr(config, "microblock_size", 0) > 0:
        unroll_microblocks = int(args.bptt_chunk_size // config.microblock_size)
    monoid_preset = os.getenv("MONOID_PRESET", "") or "none"
    log_jsonl_path = args.log_jsonl
    if log_jsonl_path:
        log_dir = os.path.dirname(log_jsonl_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    log_jsonl_fh = open(log_jsonl_path, "a", encoding="utf-8") if log_jsonl_path else None
    stat_buffers = {
        "loss.total": [],
        "grad.norm.global": [],
        "loss.saturation_penalty": [],
        "activation.pre_clip_max": [],
        "activation.post_clip_max": [],
        "grad.nan_or_inf": [],
    }
    last_metrics = {}
    ramp_buffer_steps = 500
    ramp_steps = int(args.retrieval_ramp_steps)
    if args.retrieval_ramp_start is not None:
        ramp_start_step = int(args.retrieval_ramp_start)
    else:
        ramp_start_step = exchange_freeze_steps + ramp_buffer_steps if exchange_freeze_steps else None
    ramp_end_step = (ramp_start_step + ramp_steps) if ramp_start_step is not None else None
    grad_clipped_streak = 0
    disable_grad_clip_guard = bool(args.disable_grad_clip_guard)
    sat_baseline = None
    sat_spike_streak = 0
    m_spectral_last = None
    m_spectral_monotonic = 0
    m_spectral_max = None
    m_spectral_baseline = None
    m_spectral_full_lr_allowed = False
    m_brake_steps_left = 0
    m_brake_reason = ""
    m_spectral_skip_until = None
    enable_m_spectral_monotonic_guard = False
    if resume_step is not None:
        m_spectral_skip_until = resume_step + 500
    retrieval_r10_baseline = None

    def _emit_json(payload: dict) -> None:
        line = json.dumps(payload)
        print(line, file=sys.stdout, flush=True)
        if log_jsonl_fh is not None:
            log_jsonl_fh.write(line + "\n")
            log_jsonl_fh.flush()
    if resume_step is not None and total_steps >= max_steps:
        logger.warning("Resume step %d is >= max_steps %d; training will exit immediately.", total_steps, max_steps)

    if getattr(config, "use_exchange", False):
        exchange_scale_at_resume = 0.0
        if not exchange_frozen:
            ramp_steps = max(0, int(args.exchange_scale_ramp_steps))
            if ramp_steps > 0:
                if total_steps < exchange_scale_start:
                    exchange_scale_at_resume = 0.0
                elif total_steps < exchange_scale_start + ramp_steps:
                    exchange_scale_at_resume = float(total_steps - exchange_scale_start) / float(ramp_steps)
                else:
                    exchange_scale_at_resume = 1.0
            else:
                exchange_scale_at_resume = 1.0
        logger.info(
            "Exchange resume: step=%d scale_at_resume=%.6f frozen=%s",
            total_steps,
            exchange_scale_at_resume,
            exchange_frozen,
        )
    
    # For streaming, epochs conceptually are just chunks of steps or one infinite loop
    # We will wrap in a single loop or 'epochs' that just reset stats
    
    text_cache_prefetcher = None
    mm_cache_prefetcher = None
    retrieval_iter = None
    paired_iter = None
    def _build_train_iter():
        nonlocal text_cache_prefetcher
        cache_dir = args.teacher_cache_dir_text or args.teacher_cache_dir
        if cache_dir:
            from monoid.training.embed.teacher_cache import TeacherEmbeddingCache, TeacherPrefetcher
            cache = TeacherEmbeddingCache(
                cache_dir,
                embed_dim=args.teacher_cache_embed_dim_text,
                dtype=args.teacher_cache_dtype,
            )
            text_cache_prefetcher = TeacherPrefetcher(
                train_loader,
                teacher,
                cache,
                device,
                args.modality,
                prefetch_batches=args.teacher_cache_prefetch_text,
                cache_key_extra=text_teacher_cache_key_extra,
                use_thread=use_cache_threads,
            )
            return text_cache_prefetcher
        return iter(train_loader)
    def _build_retrieval_iter():
        if retrieval_train_loader is None:
            return None
        return iter(retrieval_train_loader)
    def _build_paired_iter():
        if paired_train_loader is None:
            return None
        nonlocal mm_cache_prefetcher
        if mm_teacher is not None and args.teacher_cache_dir_mm:
            from monoid.training.embed.teacher_cache import TeacherEmbeddingCache
            mm_cache = TeacherEmbeddingCache(
                args.teacher_cache_dir_mm,
                embed_dim=args.teacher_cache_embed_dim_mm,
                dtype=args.teacher_cache_dtype,
            )
            mm_cache_prefetcher = PairedTeacherPrefetcher(
                paired_train_loader,
                mm_teacher,
                mm_cache,
                device,
                prefetch_batches=args.teacher_cache_prefetch_mm,
                cache_key_extra=mm_teacher_cache_key_extra,
                text_prompt_name=None,
                use_thread=use_cache_threads,
            )
            return mm_cache_prefetcher
        return iter(paired_train_loader)

    quit_now = False
    m_grad_probe_window = 20
    m_probe_assert_after_step = exchange_freeze_steps + 10
    m_grad_probe_deadline = None
    m_grad_probe_start = None
    for epoch in range(args.epochs):
        if streaming_mode and total_steps >= max_steps:
             break
             
        # Tqdm total handling
        params = {}
        if not streaming_mode:
            params['total'] = len(train_loader)
        else:
            # If streaming, maybe one epoch is 'max_steps'
            # Or we iterate until max_steps global
            params['total'] = max_steps
            
        # JSONL / TQDM Toggle
        # User requested clean JSONL logs
        # Removed tqdm wrapper
        iterator = range(args.max_steps)
        if not streaming_mode:
            iterator = range(len(train_loader)) # For map-style
            
        train_iter = _build_train_iter()
        retrieval_iter = _build_retrieval_iter()
        paired_iter = _build_paired_iter()
        epoch_loss = 0
        steps_in_epoch = 0
        
        for step_i in iterator:
            finisher_step_in_phase = None
            if finisher_mode and finisher_start_step is not None:
                finisher_step_in_phase = max(0, total_steps - finisher_start_step)
                if args.finisher_lr_schedule == "cosine" and finisher_base_lrs:
                    progress = min(1.0, finisher_step_in_phase / max(1, args.finisher_steps))
                    decay = 0.5 * (1.0 - math.cos(progress * math.pi))
                    for idx, start_lr in enumerate(finisher_base_lrs):
                        target_lr = start_lr * 0.2
                        optimizer.param_groups[idx]["lr"] = start_lr + (target_lr - start_lr) * decay
            if not finisher_mode and args.lr_schedule != "constant":
                lr_main = compute_lr_schedule(
                    total_steps,
                    max_steps,
                    warmup_steps,
                    float(args.peak_lr),
                    float(args.min_lr),
                    args.lr_schedule,
                )
                scale = lr_main / max(1e-12, float(args.peak_lr))
                for idx, base_lr in enumerate(schedule_base_lrs):
                    optimizer.param_groups[idx]["lr"] = base_lr * scale
            align_mult = 1.0
            retrieval_mult = 1.0
            ramp_active = False
            if ramp_start_step is not None and ramp_steps > 0:
                t_raw = (float(total_steps) - float(ramp_start_step)) / float(ramp_steps)
                t = max(0.0, min(1.0, t_raw))
                smooth = t * t * (3.0 - 2.0 * t)
                align_mult = float(args.align_mult_start) + (
                    float(args.align_mult_target) - float(args.align_mult_start)
                ) * smooth
                retrieval_mult = float(args.retrieval_mult_start) + (
                    float(args.retrieval_mult_target) - float(args.retrieval_mult_start)
                ) * smooth
                ramp_active = 0.0 < t < 1.0
            m_lr_mult = 1.0
            if m_param_group_index is not None and ramp_start_step is not None and ramp_steps > 0:
                if total_steps < ramp_start_step:
                    m_lr_mult = 1.0
                elif ramp_end_step is not None and total_steps < ramp_end_step:
                    t = max(0.0, min(1.0, (float(total_steps) - float(ramp_start_step)) / float(ramp_steps)))
                    m_lr_mult = float(args.m_ramp_lr_start_mult) + (
                        float(args.m_ramp_lr_end_mult) - float(args.m_ramp_lr_start_mult)
                    ) * t
                else:
                    hold_end = (ramp_end_step or 0) + int(args.m_ramp_hold_steps)
                    if total_steps < hold_end:
                        m_lr_mult = float(args.m_ramp_lr_end_mult)
                    else:
                        m_lr_mult = 1.0 if m_spectral_full_lr_allowed else float(args.m_ramp_lr_end_mult)
            if m_brake_steps_left > 0:
                if not args.disable_m_brake:
                    m_lr_mult = 0.0
            if m_param_group_index is not None:
                optimizer.param_groups[m_param_group_index]["lr"] *= m_lr_mult
            if getattr(config, "use_exchange", False):
                exchange_scale = 1.0
                if exchange_frozen:
                    exchange_scale = 0.0
                else:
                    ramp_steps = int(args.exchange_scale_ramp_steps)
                    if ramp_steps > 0:
                        if total_steps < exchange_scale_start:
                            exchange_scale = 0.0
                        elif total_steps < exchange_scale_start + ramp_steps:
                            exchange_scale = float(total_steps - exchange_scale_start) / float(ramp_steps)
                        else:
                            exchange_scale = 1.0
                if m_brake_steps_left > 0 and not args.disable_m_brake:
                    exchange_scale = 0.0
                set_exchange_scale(model, exchange_scale)
                exchange_inj_norm_max = float(args.exchange_inj_norm_max or 0.0)
                if exchange_scale <= 0.0:
                    exchange_inj_norm_max = 0.0
                else:
                    inj_ramp_steps = max(0, int(args.exchange_inj_norm_ramp_steps))
                    if inj_ramp_steps > 0:
                        if total_steps < exchange_scale_start:
                            exchange_inj_norm_max = 0.0
                        elif total_steps < exchange_scale_start + inj_ramp_steps:
                            frac = float(total_steps - exchange_scale_start) / float(inj_ramp_steps)
                            exchange_inj_norm_max *= max(0.0, min(1.0, frac))
                set_exchange_inj_norm_max(model, exchange_inj_norm_max)
            if exchange_frozen and exchange_freeze_steps and total_steps >= exchange_freeze_steps:
                if set_exchange_trainable(model, True):
                    logger.info("Unfreezing exchange weights at step %d.", total_steps)
                if reset_exchange_spectral_norm(model):
                    logger.info("Resetting exchange spectral norm vectors at step %d.", total_steps)
                if m_leaf_param is not None:
                    state = optimizer.state.get(m_leaf_param)
                    if state is not None:
                        exp_avg = state.get("exp_avg")
                        exp_avg_sq = state.get("exp_avg_sq")
                        if exp_avg is not None:
                            exp_avg.zero_()
                        if exp_avg_sq is not None:
                            exp_avg_sq.zero_()
                        if "step" in state:
                            state["step"] = 0
                        logger.info("Resetting exchange optimizer moments at step %d.", total_steps)
                exchange_frozen = False
                set_exchange_disabled(model, False)
            use_retrieval = retrieval_iter is not None and args.modality == "multimodal" and (step_i % 2 == 0)
            if use_retrieval:
                try:
                    batch = next(retrieval_iter)
                except StopIteration:
                    retrieval_iter = _build_retrieval_iter()
                    if retrieval_iter is None:
                        batch = next(train_iter)
                    else:
                        batch = next(retrieval_iter)
            else:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    if streaming_mode:
                        train_iter = _build_train_iter()
                        batch = next(train_iter)
                    else:
                        break
            if finisher_mode:
                audio_keys = [key for key in batch.keys() if "audio" in key]
                if audio_keys:
                    raise ValueError(
                        f"Finisher mode batch contains audio keys: {sorted(audio_keys)}"
                    )

            paired_batch = None
            if paired_iter is not None:
                try:
                    paired_batch = next(paired_iter)
                except StopIteration:
                    paired_iter = _build_paired_iter()
                    if paired_iter is not None:
                        try:
                            paired_batch = next(paired_iter)
                        except StopIteration:
                            paired_batch = None
            if finisher_mode and paired_batch is not None:
                raise ValueError("Finisher mode received paired audio batch.")

            # Move to device non-blocking
            bytes_in = batch['bytes'].to(device, non_blocking=True)
            lengths = batch.get('lengths')
            if lengths is not None:
                lengths = lengths.to(device, non_blocking=True)

            paired_text_bytes = None
            paired_text_lengths = None
            paired_text_captions = None
            paired_text_pair_ids = None
            paired_audio_bytes = None
            paired_audio_lengths = None
            paired_pair_ids = None
            paired_audio_pair_ids = None
            paired_audio_crop = None
            paired_mm_text_emb = None
            paired_mm_audio_emb = None
            if paired_batch is not None:
                paired_text_bytes = paired_batch["text_bytes"].to(device, non_blocking=True)
                paired_text_lengths = paired_batch["text_lengths"].to(device, non_blocking=True)
                paired_text_captions = paired_batch.get("text_captions")
                paired_text_pair_ids = paired_batch.get("text_pair_ids")
                paired_audio_bytes = paired_batch["audio_bytes"].to(device, non_blocking=True)
                paired_audio_lengths = paired_batch["audio_lengths"].to(device, non_blocking=True)
                paired_pair_ids = paired_batch.get("pair_id")
                paired_audio_pair_ids = paired_batch.get("audio_pair_ids")
                paired_audio_crop = paired_batch.get("audio_crop")
                paired_mm_text_emb = paired_batch.get("teacher_mm_text_emb")
                paired_mm_audio_emb = paired_batch.get("teacher_mm_audio_emb")
                if paired_mm_text_emb is not None:
                    paired_mm_text_emb = paired_mm_text_emb.to(device, non_blocking=True)
                if paired_mm_audio_emb is not None:
                    paired_mm_audio_emb = paired_mm_audio_emb.to(device, non_blocking=True)
            
            # Teacher Inference (Main Process)
            teacher_input = batch['teacher_input']
            teacher_prompts = batch.get('teacher_prompt')
            modalities = batch.get('modality', args.modality)
            pair_ids = batch.get('pair_id')
            teacher_emb = batch.get('teacher_emb')
            batch_text_count = None
            batch_audio_count = None
            if finisher_mode or args.modality == "text":
                if isinstance(modalities, str):
                    modality_list = [modalities] * bytes_in.size(0)
                else:
                    modality_list = list(modalities) if modalities is not None else []
                batch_audio_count = sum(1 for m in modality_list if m == "audio")
                batch_text_count = sum(1 for m in modality_list if m == "text")
                if finisher_mode and batch_audio_count:
                    raise ValueError(
                        f"Finisher mode received audio batch (audio_count={batch_audio_count})."
                    )
                if args.modality == "text" and batch_audio_count:
                    raise ValueError(
                        f"Text-only run received audio batch (audio_count={batch_audio_count})."
                    )

            if teacher_emb is not None:
                teacher_emb = teacher_emb.to(device, non_blocking=True)
            elif teacher is not None:
                try:
                    with torch.no_grad():
                        teacher_emb = _embed_teacher_batch(teacher, teacher_input, teacher_prompts, modalities, device)
                except Exception as e:
                    logger.error(f"Teacher Batch Inference failed: {e}")
                    teacher_emb = torch.randn(bytes_in.shape[0], teacher_raw_dim, device=device)
            else:
                teacher_emb = torch.randn(bytes_in.shape[0], teacher_raw_dim, device=device)

            teacher_mixed_live = teacher_emb
            teacher_mixed_ema = teacher_emb.detach()
            layer_weights_live = None
            layer_weights_ema = None
            if layer_mix is not None and teacher_layer_count and teacher_base_dim:
                if teacher_emb.size(1) == teacher_layer_count * teacher_base_dim:
                    teacher_layers = teacher_emb.view(
                        teacher_emb.size(0),
                        teacher_layer_count,
                        teacher_base_dim,
                    )
                    layer_weights_live = F.softmax(layer_mix, dim=0)
                    layer_weights_ema = F.softmax(layer_mix_ema, dim=0) if layer_mix_ema is not None else layer_weights_live.detach()
                    teacher_mixed_live = (teacher_layers * layer_weights_live.view(1, -1, 1)).sum(dim=1)
                    teacher_mixed_ema = (teacher_layers * layer_weights_ema.view(1, -1, 1)).sum(dim=1)

            teacher_live = teacher_mixed_live
            teacher_ema = teacher_mixed_ema
            if teacher_proj is not None and teacher_proj_ema is not None and teacher_mixed_live.size(1) == teacher_proj.in_features:
                teacher_live = teacher_proj(teacher_mixed_live.float())
                with torch.no_grad():
                    teacher_ema = teacher_proj_ema(teacher_mixed_ema.float())

            teacher_live = F.normalize(teacher_live, p=2, dim=-1)
            teacher_ema = F.normalize(teacher_ema, p=2, dim=-1)
            teacher_emb = teacher_ema + (teacher_live - teacher_live.detach())
            
            accum_index = total_steps % grad_accum_steps
            if accum_index == 0:
                optimizer.zero_grad(set_to_none=True) 
            
            amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
            amp_dtype = torch.bfloat16
            return_stats = args.log_activation_stats
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                if return_stats:
                    outputs = model(bytes_in, lengths=lengths, return_stats=True)
                else:
                    outputs = model(bytes_in, lengths=lengths)

                exchange_executed_flag = None
                m_used_weight = None
                if isinstance(outputs, dict):
                    exchange_executed_flag = float(outputs.get("exchange_executed", 0.0))
                    m_used_weight = outputs.get("exchange_weight_used")
                    if m_used_weight is None and exchange_executed_flag >= 1.0:
                        exchange_module = get_first_exchange_module(model)
                        if exchange_module is not None:
                            m_used_weight = exchange_module.weight

                student_emb = outputs['embeddings']
                alpha_hardness = 0.0
                l_distill = torch.tensor(0.0, device=device)
                l_spread = torch.tensor(0.0, device=device)
                l_spkd = torch.tensor(0.0, device=device)
                l_var = torch.tensor(0.0, device=device)
                l_var_text = torch.tensor(0.0, device=device)
                l_var_audio = torch.tensor(0.0, device=device)
                l_rkd_distance = torch.tensor(0.0, device=device)
                l_rkd_angle = torch.tensor(0.0, device=device)
                l_contrast = torch.tensor(0.0, device=device)
                l_pairwise = torch.tensor(0.0, device=device)
                l_consistency = torch.tensor(0.0, device=device)
                l_neighborhood = torch.tensor(0.0, device=device)
                l_cross_modal = torch.tensor(0.0, device=device)
                l_mm_distill = torch.tensor(0.0, device=device)
                cm_cos_mean = None
                cm_cos_gap = None
                cm_off_mean = None
                cm_diag_sample = None
                cm_off_sample = None
                cm_sim_stats = None
                cm_text_requires_grad = None
                cm_audio_requires_grad = None
                cm_text_dtype = None
                cm_audio_dtype = None
                contrast_scale = 0.0
                grad_model = None
                grad_last_block = None
                grad_proj = None
                teacher_cm_diag = None
                teacher_cm_off = None
                teacher_cm_gap = None
                mm_teacher_diag = None
                mm_teacher_off = None
                mm_teacher_gap = None

                student_emb_float = student_emb.float()
                teacher_emb_float = teacher_emb.float()
                if finisher_mode and teacher_emb_float.shape != student_emb_float.shape:
                    raise ValueError(
                        "Finisher mode embedding shape mismatch before cosine: "
                        f"teacher={tuple(teacher_emb_float.shape)} "
                        f"student={tuple(student_emb_float.shape)}"
                    )
                max_dim = max(matryoshka_dims)

                if (
                    paired_text_bytes is not None
                    and paired_audio_bytes is not None
                    and args.paired_cross_modal
                    and (args.cross_modal_weight > 0 or args.mm_distill_weight > 0)
                    and not finisher_mode
                ):
                    log_cross_modal_debug = should_log(total_steps, args.neighborhood_log_every)
                    if args.debug_cross_modal_pairs and paired_text_pair_ids is not None and paired_audio_pair_ids is not None:
                        if len(paired_text_pair_ids) != len(paired_audio_pair_ids):
                            logger.error(
                                "Cross-modal pair_id length mismatch: text=%d audio=%d",
                                len(paired_text_pair_ids),
                                len(paired_audio_pair_ids),
                            )
                            raise AssertionError("Cross-modal pair_id length mismatch")
                        mismatches = [
                            (i, paired_text_pair_ids[i], paired_audio_pair_ids[i])
                            for i in range(len(paired_text_pair_ids))
                            if paired_text_pair_ids[i] != paired_audio_pair_ids[i]
                        ]
                        if mismatches:
                            logger.error("Cross-modal pair_id mismatches: %s", mismatches[:5])
                            raise AssertionError("Cross-modal pair_id mismatch")
                        if log_cross_modal_debug:
                            logger.info(
                                "Cross-modal pair_id lists (text/audio): %s / %s",
                                paired_text_pair_ids[: min(6, len(paired_text_pair_ids))],
                                paired_audio_pair_ids[: min(6, len(paired_audio_pair_ids))],
                            )
                    paired_text_out = model(paired_text_bytes, lengths=paired_text_lengths)
                    paired_audio_out = model(paired_audio_bytes, lengths=paired_audio_lengths)
                    l_cross_modal, cm_cos_mean, cm_off_mean, cm_cos_gap, cm_diag_sample, cm_off_sample, cm_sim_stats = cross_modal_contrastive_loss(
                        paired_text_out["embeddings"],
                        paired_audio_out["embeddings"],
                        args.cross_modal_temp,
                        return_samples=log_cross_modal_debug,
                    )
                    if mm_teacher is not None and args.mm_distill_weight > 0:
                        if paired_mm_text_emb is None or paired_mm_audio_emb is None:
                            captions = paired_text_captions or [""] * paired_text_bytes.size(0)
                            audio_wave = []
                            for idx in range(paired_audio_bytes.size(0)):
                                audio_wave.append(_bytes_to_float_audio(paired_audio_bytes[idx]))
                            try:
                                with torch.no_grad():
                                    paired_mm_text_emb = mm_teacher.get_text_embedding(captions)
                                    paired_mm_audio_emb = mm_teacher.get_audio_embedding(
                                        audio_wave,
                                        sample_rate=args.audio_sample_rate,
                                    )
                                paired_mm_text_emb = paired_mm_text_emb.to(device, non_blocking=True)
                                paired_mm_audio_emb = paired_mm_audio_emb.to(device, non_blocking=True)
                            except Exception as exc:
                                logger.warning("M2D-CLAP teacher embed failed: %s", exc)
                                paired_mm_text_emb = None
                                paired_mm_audio_emb = None
                        if paired_mm_text_emb is not None and paired_mm_audio_emb is not None:
                            l_mm_distill = cross_modal_sim_matrix_distill(
                                paired_text_out["embeddings"],
                                paired_audio_out["embeddings"],
                                paired_mm_text_emb,
                                paired_mm_audio_emb,
                                args.mm_distill_temp,
                            )
                            t_text = F.normalize(paired_mm_text_emb.float(), p=2, dim=-1)
                            t_audio = F.normalize(paired_mm_audio_emb.float(), p=2, dim=-1)
                            t_sim = t_text @ t_audio.t()
                            mm_teacher_diag = t_sim.diagonal().mean().item() if t_sim.numel() else None
                            off_mask = ~torch.eye(t_sim.size(0), dtype=torch.bool, device=t_sim.device)
                            t_off = t_sim[off_mask]
                            mm_teacher_off = t_off.mean().item() if t_off.numel() else None
                            if mm_teacher_diag is not None and mm_teacher_off is not None:
                                mm_teacher_gap = mm_teacher_diag - mm_teacher_off
                    cm_text_requires_grad = float(paired_text_out["embeddings"].requires_grad)
                    cm_audio_requires_grad = float(paired_audio_out["embeddings"].requires_grad)
                    cm_text_dtype = str(paired_text_out["embeddings"].dtype)
                    cm_audio_dtype = str(paired_audio_out["embeddings"].dtype)
                    if log_cross_modal_debug and paired_text_captions is not None:
                        text_norm = F.normalize(paired_text_out["embeddings"].float(), p=2, dim=-1)
                        audio_norm = F.normalize(paired_audio_out["embeddings"].float(), p=2, dim=-1)
                        diag_cos = (text_norm * audio_norm).sum(dim=-1).detach().cpu().tolist()
                        for i in range(min(3, len(diag_cos))):
                            caption = paired_text_captions[i] if i < len(paired_text_captions) else None
                            if isinstance(caption, str):
                                caption = " ".join(caption.strip().split())
                                if len(caption) > 120:
                                    caption = caption[:117] + "..."
                            pid = None
                            if paired_pair_ids is not None and i < len(paired_pair_ids):
                                pid = paired_pair_ids[i]
                            logger.info("Cross-modal pair %d: cos=%.4f pair_id=%s caption=%s", i, diag_cos[i], pid, caption)
                        if len(diag_cos) > 1:
                            t0_a0 = diag_cos[0]
                            t0_a1 = float((text_norm[0] * audio_norm[1]).sum().item())
                            logger.info("Cross-modal sanity: cos(text0,audio0)=%.4f cos(text0,audio1)=%.4f", t0_a0, t0_a1)
                        logger.info("Cross-modal dtype: text=%s audio=%s", cm_text_dtype, cm_audio_dtype)
                        if paired_audio_crop is not None:
                            sample_crops = paired_audio_crop[: min(3, len(paired_audio_crop))]
                            logger.info("Audio crop samples: %s", sample_crops)
                    if log_cross_modal_debug and args.debug_cross_modal_pairs and mm_teacher is not None and paired_text_captions is not None:
                        captions = [c if isinstance(c, str) else "" for c in paired_text_captions]
                        audio_wave = []
                        for idx in range(paired_audio_bytes.size(0)):
                            audio_wave.append(_bytes_to_float_audio(paired_audio_bytes[idx]))
                        try:
                            with torch.no_grad():
                                prompt_name = getattr(mm_teacher, "text_prompt_name", None)
                                teacher_text = mm_teacher.get_text_embedding(captions, prompt_name=prompt_name)
                                teacher_audio = mm_teacher.get_audio_embedding(audio_wave, sample_rate=args.audio_sample_rate)
                            t_text = F.normalize(teacher_text.float(), p=2, dim=-1)
                            t_audio = F.normalize(teacher_audio.float(), p=2, dim=-1)
                            t_sim = t_text @ t_audio.t()
                            t_diag = t_sim.diagonal()
                            t_off = t_sim[~torch.eye(t_sim.size(0), dtype=torch.bool, device=t_sim.device)]
                            teacher_cm_diag = t_diag.mean().item() if t_diag.numel() else None
                            teacher_cm_off = t_off.mean().item() if t_off.numel() else None
                            if teacher_cm_diag is not None and teacher_cm_off is not None:
                                teacher_cm_gap = teacher_cm_diag - teacher_cm_off
                            logger.info(
                                "Teacher cross-modal (cropped) diag=%.4f off=%.4f gap=%.4f",
                                teacher_cm_diag if teacher_cm_diag is not None else 0.0,
                                teacher_cm_off if teacher_cm_off is not None else 0.0,
                                teacher_cm_gap if teacher_cm_gap is not None else 0.0,
                            )
                        except Exception as exc:
                            logger.warning("Teacher cross-modal debug failed: %s", exc)
                    if not l_cross_modal.requires_grad:
                        logger.warning("Cross-modal loss has no grad at step %d.", total_steps)

                with torch.no_grad():
                    cos_t_s_512 = None
                    if student_emb_float.size(1) >= max_dim and teacher_emb_float.size(1) >= max_dim:
                        cos_t_s_512 = F.cosine_similarity(
                            F.normalize(student_emb_float[:, :max_dim], p=2, dim=-1),
                            F.normalize(teacher_emb_float[:, :max_dim], p=2, dim=-1),
                            dim=1,
                        ).mean().item()

                    if contrast_ramp_start_step is None:
                        if total_steps >= args.contrast_warmup_steps or (
                            cos_t_s_512 is not None and cos_t_s_512 >= args.contrast_start_cos
                        ):
                            contrast_ramp_start_step = total_steps

                    if args.matryoshka_weights is None:
                        matryoshka_weights, matryoshka_weight_sum = compute_matryoshka_weights(
                            matryoshka_dims,
                            matryoshka_base_weights,
                            cos_t_s_512,
                            args.matryoshka_ramp_cos,
                        )

                if contrast_ramp_start_step is None:
                    contrast_scale = 0.0
                else:
                    elapsed = max(0, total_steps - contrast_ramp_start_step)
                    t = min(1.0, elapsed / max(1, args.contrast_ramp_steps))
                    # Power ramp with cap
                    raw = t ** args.contrast_ramp_power
                    contrast_scale = min(raw, args.contrast_scale_cap)

                if total_steps == 0 or should_log(total_steps, args.spec_header_every):
                    weight_map = {dim: weight for dim, weight in zip(matryoshka_dims, matryoshka_weights)}
                    base_weight_map = {dim: matryoshka_base_weights.get(dim, 0.0) for dim in matryoshka_dims}
                    m_param_abs_max = 0.0
                    m_used_abs_max = 0.0
                    m_param_abs_means = []
                    m_used_abs_means = []
                    with torch.no_grad():
                        for block in getattr(model, "blocks", []):
                            exchange = getattr(block, "exchange", None)
                            if exchange is None:
                                continue
                            if (
                                hasattr(exchange, "parametrizations")
                                and hasattr(exchange.parametrizations, "weight")
                                and hasattr(exchange.parametrizations.weight, "original")
                            ):
                                param_weight = exchange.parametrizations.weight.original.float()
                            elif hasattr(exchange, "weight_orig"):
                                param_weight = exchange.weight_orig.float()
                            else:
                                param_weight = exchange.weight.float()
                            used_weight = param_weight
                            if param_weight.numel():
                                sigma, _ = compute_spectral_norm_cpu(
                                    param_weight.float(),
                                    total_steps,
                                    "spec_header.param",
                                    raise_on_invalid=False,
                                )
                                scale = max(1.0, sigma)
                                used_weight = param_weight / scale
                            if not torch.isfinite(used_weight).all():
                                used_weight = torch.nan_to_num(used_weight, nan=0.0, posinf=0.0, neginf=0.0)
                            if param_weight.numel():
                                m_param_abs_max = max(m_param_abs_max, float(param_weight.abs().max().item()))
                                m_param_abs_means.append(float(param_weight.abs().mean().item()))
                            if used_weight.numel():
                                m_used_abs_max = max(m_used_abs_max, float(used_weight.abs().max().item()))
                                m_used_abs_means.append(float(used_weight.abs().mean().item()))
                    m_param_abs_mean = (
                        float(sum(m_param_abs_means) / len(m_param_abs_means)) if m_param_abs_means else 0.0
                    )
                    m_used_abs_mean = (
                        float(sum(m_used_abs_means) / len(m_used_abs_means)) if m_used_abs_means else 0.0
                    )
                    if exchange_frozen:
                        m_used_abs_mean = 0.0
                        m_used_abs_max = 0.0
                    spec_header = {
                        "event": "spec_header",
                        "step": total_steps,
                        "epoch": epoch,
                        "spec.version": spec_version,
                        "model.preset": monoid_preset,
                        "model.n_layers": int(getattr(config, "n_layers", 0)),
                        "model.d_state": int(getattr(config, "d_state", 0)),
                        "model.microblock_size": int(getattr(config, "microblock_size", 0)),
                        "model.exchange_dim": int(getattr(config, "exchange_dim", 0)),
                        "train.batch_size": int(args.batch_size),
                        "train.max_bytes": int(args.max_bytes),
                        "train.unroll_microblocks": int(unroll_microblocks),
                        "opt.name": args.optimizer,
                        "lr.schedule": args.lr_schedule,
                        "lr.peak": float(args.peak_lr),
                        "lr.min": float(args.min_lr),
                        "lr.warmup_frac": float(args.warmup_frac),
                        "lr.warmup_steps": int(warmup_steps),
                        "lr.value": float(optimizer.param_groups[0]["lr"]),
                        "exchange.freeze_steps": int(exchange_freeze_steps),
                        "saturation_penalty.enabled": float(args.enable_saturation_penalty),
                        "loss.weights.w512": float(base_weight_map.get(512, 0.0)),
                        "loss.weights.w256": float(base_weight_map.get(256, 0.0)),
                        "loss.weights.w128": float(base_weight_map.get(128, 0.0)),
                        "loss.weights.base.w512": float(base_weight_map.get(512, 0.0)),
                        "loss.weights.base.w256": float(base_weight_map.get(256, 0.0)),
                        "loss.weights.base.w128": float(base_weight_map.get(128, 0.0)),
                        "loss.weights.effective.w512": float(weight_map.get(512, 0.0)),
                        "loss.weights.effective.w256": float(weight_map.get(256, 0.0)),
                        "loss.weights.effective.w128": float(weight_map.get(128, 0.0)),
                        "M.abs_mean": m_used_abs_mean,
                        "M.abs_max": m_used_abs_max,
                        "M.param.abs_mean": m_param_abs_mean,
                        "M.param.abs_max": m_param_abs_max,
                        "M.used.abs_mean": m_used_abs_mean,
                        "M.used.abs_max": m_used_abs_max,
                        "M.leaf.name": m_leaf_name or "",
                        "M.leaf.requires_grad": float(m_leaf_param.requires_grad) if m_leaf_param is not None else 0.0,
                        "M.leaf.in_optimizer": float(m_leaf_in_optimizer) if m_leaf_in_optimizer is not None else 0.0,
                        "git.commit": git_commit,
                        "git.status_porcelain": git_status,
                    }
                    if total_steps == 0:
                        spec_header["M.param.abs_max_init"] = m_param_abs_max
                        spec_header["M.used.abs_max_init"] = m_used_abs_max
                        spec_header["M.leaf.shape"] = list(m_leaf_shape) if m_leaf_shape is not None else []
                    if args.use_wandb:
                        wandb.log(spec_header, step=total_steps)
                    _emit_json(spec_header)

                # Alpha Hardness
                if args.alpha_hardness_fixed is not None:
                    alpha_hardness = float(args.alpha_hardness_fixed)
                elif total_steps < args.contrast_ramp_steps:
                    alpha_hardness = 5.0
                else:
                    remaining = max(1, max_steps - args.contrast_ramp_steps)
                    progress = min(1.0, float(total_steps - args.contrast_ramp_steps) / float(remaining))
                    alpha_hardness = 5.0 + (5.0 * progress)
                criterion_contrast.alpha_hardness = alpha_hardness

                l_distill_512 = None
                l_distill_256 = None
                l_distill_128 = None
                for dim, weight in zip(matryoshka_dims, matryoshka_weights):
                    student_dim = F.normalize(student_emb_float[:, :dim], p=2, dim=-1)
                    teacher_dim = F.normalize(teacher_emb_float[:, :dim], p=2, dim=-1)
                    dim_distill = criterion_distill(
                        student_dim,
                        teacher_dim,
                        detach_teacher=not train_teacher,
                    )
                    l_distill = l_distill + weight * dim_distill
                    l_spread = l_spread + weight * criterion_spread(student_dim)
                    if contrast_scale > 0:
                        l_contrast = l_contrast + weight * criterion_contrast.forward_distillation(
                            student_dim,
                            teacher_dim,
                            assume_normalized=True,
                        )
                    if w_pairwise > 0:
                        l_pairwise = l_pairwise + weight * criterion_pairwise(
                            student_dim,
                            teacher_dim,
                            assume_normalized=True,
                            detach_teacher=not train_teacher,
                        )
                    if dim == 512:
                        l_distill_512 = dim_distill
                    elif dim == 256:
                        l_distill_256 = dim_distill
                    elif dim == 128:
                        l_distill_128 = dim_distill

                l_distill = l_distill / matryoshka_weight_sum
                l_spread = l_spread / matryoshka_weight_sum
                l_contrast = l_contrast / matryoshka_weight_sum
                l_pairwise = l_pairwise / matryoshka_weight_sum

                if w_spkd > 0:
                    student_spkd = F.normalize(student_emb_float[:, :max_dim], p=2, dim=-1)
                    teacher_spkd = F.normalize(teacher_emb_float[:, :max_dim], p=2, dim=-1)
                    l_spkd = criterion_spkd(
                        student_spkd,
                        teacher_spkd,
                        assume_normalized=True,
                        detach_teacher=not train_teacher,
                    )
                if w_var > 0:
                    if isinstance(modalities, str):
                        l_var = criterion_var(student_emb_float[:, :max_dim])
                    else:
                        modality_list = list(modalities) if modalities is not None else []
                        losses = []
                        for label, holder in (("text", "l_var_text"), ("audio", "l_var_audio")):
                            if label == "audio" and finisher_mode:
                                continue
                            idx = [i for i, m in enumerate(modality_list) if m == label]
                            if idx:
                                idx_tensor = torch.tensor(idx, device=device)
                                loss_val = criterion_var(student_emb_float[idx_tensor, :max_dim])
                                losses.append(loss_val)
                                if label == "text":
                                    l_var_text = loss_val
                                else:
                                    l_var_audio = loss_val
                        if losses:
                            l_var = sum(losses) / float(len(losses))
                        else:
                            l_var = criterion_var(student_emb_float[:, :max_dim])
                if w_rkd_distance > 0:
                    l_rkd_distance = criterion_rkd_distance(
                        student_emb_float[:, :max_dim],
                        teacher_emb_float[:, :max_dim],
                        detach_teacher=not train_teacher,
                    )
                if w_rkd_angle > 0:
                    l_rkd_angle = criterion_rkd_angle(
                        student_emb_float[:, :max_dim],
                        teacher_emb_float[:, :max_dim],
                        detach_teacher=not train_teacher,
                    )

                if outputs['bidirectional_emb'] is not None:
                    l_consistency = criterion_consistency(outputs['causal_emb'], outputs['bidirectional_emb'])

                if args.train_retrieval and w_neighborhood > 0 and teacher_prompts is not None:
                    query_idx = [i for i, prompt in enumerate(teacher_prompts) if prompt == "query"]
                    doc_idx = [i for i, prompt in enumerate(teacher_prompts) if prompt == "document"]
                    if query_idx and doc_idx:
                        query_idx = torch.tensor(query_idx, device=device, dtype=torch.long)
                        doc_idx = torch.tensor(doc_idx, device=device, dtype=torch.long)

                        student_q = student_emb[query_idx]
                        student_d = student_emb[doc_idx]
                        teacher_q = teacher_emb[query_idx]
                        teacher_d = teacher_emb[doc_idx]
                        l_neighborhood = criterion_neighborhood(student_q, student_d, teacher_q, teacher_d)

                align_term = w_distill * l_distill
                retrieval_term = (
                    (w_contrast * contrast_scale) * l_contrast
                    + (w_pairwise * l_pairwise)
                    + (w_neighborhood * l_neighborhood)
                )
                loss = (
                    (align_mult * align_term)
                    + (w_spread * l_spread)
                    + (w_spkd * l_spkd)
                    + (w_var * l_var)
                    + (w_rkd_distance * l_rkd_distance)
                    + (w_rkd_angle * l_rkd_angle)
                    + (retrieval_mult * retrieval_term)
                    + (w_consistency * l_consistency)
                    + (w_cross_modal * l_cross_modal)
                    + (w_mm_distill * l_mm_distill)
                )
                l_saturation_penalty = torch.tensor(0.0, device=device)
                saturation_term = torch.tensor(0.0, device=device)
                if args.enable_saturation_penalty:
                    penalties = []
                    scale = float(2 ** config.b_shift)
                    for block in model.blocks:
                        b_over = torch.relu((block.b.abs() / scale) - 127.0)
                        penalties.append(b_over.mean())
                    if penalties:
                        l_saturation_penalty = torch.stack(penalties).mean()
                    saturation_term = float(args.saturation_penalty_weight) * l_saturation_penalty
                    loss = loss + saturation_term

                align_weighted = None
                if l_distill_512 is not None and l_distill_256 is not None and l_distill_128 is not None:
                    align_weighted = float(
                        (1.0 * l_distill_512 + 0.7 * l_distill_256 + 0.5 * l_distill_128).item()
                    )
                with torch.no_grad():
                    loss_recomputed = (
                        (align_mult * align_term)
                        + (w_spread * l_spread)
                        + (w_spkd * l_spkd)
                        + (w_var * l_var)
                        + (w_rkd_distance * l_rkd_distance)
                        + (w_rkd_angle * l_rkd_angle)
                        + (retrieval_mult * retrieval_term)
                        + (w_consistency * l_consistency)
                        + (w_cross_modal * l_cross_modal)
                        + (w_mm_distill * l_mm_distill)
                        + saturation_term
                    )
                    loss_recomputed_val = float(loss_recomputed.item())
                    loss_unscaled_val = float(loss.item())
                    loss_recomputed_err = abs(loss_recomputed_val - loss_unscaled_val)
                    if loss_recomputed_err > 1e-5:
                        raise ValueError(
                            "Loss recompute mismatch: loss=%.6f recomputed=%.6f err=%.6f"
                            % (loss_unscaled_val, loss_recomputed_val, loss_recomputed_err)
                        )

            loss_unscaled = loss
            if grad_accum_steps > 1:
                loss = loss / float(grad_accum_steps)
            loss.backward()
            grad_a = None
            grad_b = None
            grad_m = None
            m_param = None
            m_param_name = None
            m_param_requires_grad = None
            m_param_in_optimizer = None
            m_param_grad_norm = None
            m_param_grad_is_none = None
            m_leaf_grad_norm = None
            m_leaf_grad_is_none = None
            m_leaf_requires_grad = None
            m_used_grad_norm = None
            m_used_grad_is_none = None
            m_param_delta_l2 = None
            m_leaf_delta_l2 = None
            all_params = [p for group in optimizer.param_groups for p in group["params"]]
            grads_ok = grads_finite(all_params)
            grad_nan_or_inf = 0.0 if grads_ok else 1.0
            if not grads_ok:
                raise ValueError("Non-finite gradient detected (NaN/Inf).")
            if args.log_grad_stats:
                a_params, b_params, m_params = collect_monoid_params(model)
                grad_a = grad_norm(a_params) if a_params else 0.0
                grad_b = grad_norm(b_params) if b_params else 0.0
                grad_m = grad_norm(m_params) if m_params else 0.0
                if m_leaf_param is not None:
                    m_leaf_requires_grad = float(m_leaf_param.requires_grad)
                    if m_leaf_param.grad is None:
                        m_leaf_grad_is_none = 1.0
                        m_leaf_grad_norm = 0.0
                    else:
                        m_leaf_grad_is_none = 0.0
                        m_leaf_grad_norm = float(m_leaf_param.grad.detach().float().norm().item())
                if m_leaf_param is not None:
                    m_param = m_leaf_param
                    m_param_name = m_leaf_name
                    m_param_requires_grad = m_leaf_requires_grad
                    m_param_in_optimizer = float(m_leaf_in_optimizer) if m_leaf_in_optimizer is not None else 0.0
                    m_param_grad_is_none = m_leaf_grad_is_none
                    m_param_grad_norm = m_leaf_grad_norm
                elif m_params:
                    m_param = m_params[0]
                    m_param_name = param_name_by_id.get(id(m_param))
                    if m_param_name is None:
                        for name, param in model.named_parameters():
                            if param is m_param:
                                m_param_name = name
                                break
                    m_param_requires_grad = float(m_param.requires_grad)
                    m_param_in_optimizer = float(id(m_param) in optimizer_param_ids)
                    if m_param.grad is None:
                        m_param_grad_is_none = 1.0
                        m_param_grad_norm = 0.0
                    else:
                        m_param_grad_is_none = 0.0
                        m_param_grad_norm = float(m_param.grad.detach().float().norm().item())
                if m_used_weight is not None:
                    if m_used_weight.grad is None:
                        m_used_grad_is_none = 1.0
                        m_used_grad_norm = 0.0
                    else:
                        m_used_grad_is_none = 0.0
                        m_used_grad_norm = float(m_used_weight.grad.detach().float().norm().item())
            if args.debug_cross_modal_pairs and should_log(total_steps, args.neighborhood_log_every):
                grad_model = grad_norm(model.parameters())
                if hasattr(model, "blocks") and model.blocks:
                    grad_last_block = grad_norm(model.blocks[-1].parameters())
                if hasattr(model, "proj") and model.proj is not None:
                    grad_proj = grad_norm(model.proj.parameters())
            do_step = (accum_index == grad_accum_steps - 1)
            if not do_step:
                last_step = len(iterator) - 1
                do_step = step_i == last_step
            grad_norm_clipped = None
            grad_norm_post = None
            grad_clipped = 0.0
            m_param_pre_step = None
            m_leaf_pre_step = None
            m_proj_sn_before = None
            m_proj_sn_after = None
            m_proj_sn_isfinite = None
            if do_step:
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if grad_norm_clipped is not None:
                    grad_clipped = float(grad_norm_clipped > args.max_grad_norm)
                grad_norm_post = grad_norm(model.parameters())
                if (
                    m_leaf_param is not None
                    and ramp_start_step is not None
                    and ramp_steps > 0
                    and (exchange_freeze_steps is None or total_steps >= exchange_freeze_steps)
                ):
                    hold_end = (ramp_end_step or 0) + int(args.m_ramp_hold_steps)
                    if total_steps < hold_end:
                        m_grad_clip = float(args.m_grad_clip)
                        if m_grad_clip > 0.0 and m_leaf_param.grad is not None:
                            grad_norm_m = float(m_leaf_param.grad.detach().float().norm().item())
                            if grad_norm_m > m_grad_clip:
                                scale = m_grad_clip / max(grad_norm_m, 1e-12)
                                m_leaf_param.grad.mul_(scale)
                if args.log_grad_stats:
                    if m_param is not None and total_steps >= exchange_freeze_steps:
                        m_param_pre_step = m_param.detach().float().clone()
                    if m_leaf_param is not None and total_steps >= exchange_freeze_steps:
                        m_leaf_pre_step = m_leaf_param.detach().float().clone()
                optimizer.step()
                if (
                    m_leaf_param is not None
                    and (exchange_freeze_steps is None or total_steps >= exchange_freeze_steps)
                ):
                    with torch.no_grad():
                        weight = m_leaf_param.detach().float()
                        if weight.numel():
                            sn, sn_isfinite = compute_spectral_norm_cpu(
                                weight, total_steps, "proj.before"
                            )
                            m_proj_sn_before = sn
                            m_proj_sn_isfinite = sn_isfinite
                            sn_max = float(args.m_spectral_norm_max)
                            if sn_max > 0.0 and sn > sn_max:
                                scale = sn_max / max(sn, 1e-12)
                                m_leaf_param.data.mul_(scale)
                                weight_after = m_leaf_param.detach().float()
                                if weight_after.numel():
                                    m_proj_sn_after, sn_after_isfinite = compute_spectral_norm_cpu(
                                        weight_after, total_steps, "proj.after"
                                    )
                                    m_proj_sn_isfinite = m_proj_sn_isfinite and sn_after_isfinite
                            else:
                                m_proj_sn_after = sn
                if m_param_pre_step is not None:
                    m_param_delta_l2 = float((m_param.detach().float() - m_param_pre_step).norm().item())
                if m_leaf_pre_step is not None:
                    m_leaf_delta_l2 = float((m_leaf_param.detach().float() - m_leaf_pre_step).norm().item())
                if teacher_proj_ema is not None or layer_mix_ema is not None:
                    decay = float(args.teacher_ema_decay)
                    with torch.no_grad():
                        if teacher_proj_ema is not None and teacher_proj is not None:
                            for ema_param, live_param in zip(teacher_proj_ema.parameters(), teacher_proj.parameters()):
                                ema_param.data.mul_(decay).add_(live_param.data, alpha=1.0 - decay)
                        if layer_mix_ema is not None and layer_mix is not None:
                            layer_mix_ema.mul_(decay).add_(layer_mix.detach(), alpha=1.0 - decay)
                optimizer.zero_grad(set_to_none=True)
            
            # W&B + JSONL Logging (tiered)
            loss_scalar = float(loss_unscaled.item())
            current_lr = optimizer.param_groups[0]['lr']
            grad_norm_value = float(grad_norm_clipped) if grad_norm_clipped is not None else 0.0
            grad_norm_post_value = float(grad_norm_post) if grad_norm_post is not None else 0.0
            align_block = float((align_mult * align_term).item())
            retrieval_block = float((retrieval_mult * retrieval_term).item())

            logA = {
                "loss/total": loss_scalar,
                "loss/distill": l_distill.item(),
                "loss/spread": l_spread.item(),
                "loss/spkd": l_spkd.item(),
                "loss/var": l_var.item(),
                "loss/var_text": l_var_text.item(),
                "loss/var_audio": l_var_audio.item(),
                "loss/rkd_distance": l_rkd_distance.item(),
                "loss/rkd_angle": l_rkd_angle.item(),
                "loss/contrast": l_contrast.item(),
                "loss/pairwise": l_pairwise.item(),
                "loss/consistency": l_consistency.item(),
                "loss/neighborhood": l_neighborhood.item(),
                "loss/cross_modal": l_cross_modal.item(),
                "loss/mm_distill": l_mm_distill.item(),
                "loss.total": loss_scalar,
                "loss.distill": l_distill.item(),
                "loss.contrast": l_contrast.item(),
                "loss.neighborhood": l_neighborhood.item(),
                "loss.rkd_d": l_rkd_distance.item(),
                "loss.rkd_a": l_rkd_angle.item(),
                "loss.saturation_penalty": float(saturation_term.item()),
                "train/lr": current_lr,
                "lr.value": current_lr,
                "sched.align_mult": align_mult,
                "sched.retrieval_mult": retrieval_mult,
                "loss.align_block": align_block,
                "loss.retrieval_block": retrieval_block,
                "train/grad_norm": grad_norm_value,
                "grad.norm.global": grad_norm_value,
                "grad.norm.global_post_clip": grad_norm_post_value,
                "grad.clipped": grad_clipped,
                "grad.nan_or_inf": grad_nan_or_inf,
                "train/alpha_hardness": alpha_hardness,
                "train/contrast_scale": contrast_scale,
            }
            if not disable_grad_clip_guard:
                if grad_clipped > 0.5:
                    grad_clipped_streak += 1
                else:
                    grad_clipped_streak = 0
                if grad_clipped_streak > 200:
                    raise RuntimeError("grad.clipped > 0.5 for >200 steps.")
            if l_distill_512 is not None:
                logA["loss/L512"] = l_distill_512.item()
                logA["loss.L512"] = l_distill_512.item()
            if l_distill_256 is not None:
                logA["loss/L256"] = l_distill_256.item()
                logA["loss.L256"] = l_distill_256.item()
            if l_distill_128 is not None:
                logA["loss/L128"] = l_distill_128.item()
                logA["loss.L128"] = l_distill_128.item()
            if align_weighted is not None:
                logA["loss/align_weighted"] = align_weighted
                logA["loss.align_weighted"] = align_weighted
            logA["loss/total_recomputed"] = loss_recomputed_val
            logA["loss/total_recompute_err"] = loss_recomputed_err
            logA["loss/tau"] = float(getattr(criterion_contrast, "temperature", 0.0))
            if grad_a is not None:
                logA["grad.norm.a"] = grad_a
            if grad_b is not None:
                logA["grad.norm.b"] = grad_b
            m_probe_assert = None
            if grad_m is not None:
                logA["grad.norm.M"] = grad_m
                if exchange_frozen and grad_m > 1e-6:
                    raise ValueError(
                        f"Exchange frozen but grad.norm.M={grad_m:.6f} (expected ~0)."
                    )
            if m_param_grad_is_none is not None:
                logA["grad.is_none.M_param"] = m_param_grad_is_none
                logA["grad.norm.M_param"] = float(m_param_grad_norm or 0.0)
            m_grad_norm_log = None
            if m_leaf_grad_norm is not None:
                m_grad_norm_log = m_leaf_grad_norm
            elif m_param_grad_norm is not None:
                m_grad_norm_log = m_param_grad_norm
            elif grad_m is not None:
                m_grad_norm_log = grad_m
            if m_grad_norm_log is not None:
                logA["grad_norm_M"] = float(m_grad_norm_log)
            logA["M.lr_mult"] = float(m_lr_mult)
            logA["param.name.M_param"] = m_param_name or ""
            if exchange_freeze_steps:
                logA["exchange.frozen"] = float(exchange_frozen)
            exchange_executed = None
            if exchange_executed_flag is not None:
                exchange_executed = exchange_executed_flag
            elif isinstance(outputs, dict):
                exchange_executed = float(outputs.get("exchange_executed", 0.0))
            if not exchange_frozen and exchange_executed is not None:
                logA["exchange.executed"] = exchange_executed
                logA["exchange.u_norm"] = float(outputs.get("exchange_u_norm", 0.0))
                logA["exchange.v_norm"] = float(outputs.get("exchange_v_norm", 0.0))
                logA["exchange.inj_norm"] = float(outputs.get("exchange_inj_norm", 0.0))
                logA["exchange.inj_norm_raw"] = float(outputs.get("exchange_inj_norm_raw", 0.0))
                logA["exchange.inj_norm_clamped"] = float(outputs.get("exchange_inj_norm_clamped", 0.0))
                logA["exchange.scale"] = float(outputs.get("exchange_scale", 0.0))
                logA["exchange.inj_norm_max"] = float(outputs.get("exchange_inj_norm_max", 0.0))
            if exchange_executed is not None and exchange_executed >= 1.0:
                logA["M_param.name"] = m_param_name or ""
                logA["M_param.requires_grad"] = float(m_param_requires_grad or 0.0)
                logA["M_param.in_optimizer"] = float(m_param_in_optimizer or 0.0)
                logA["grad.is_none.M_param"] = float(m_param_grad_is_none if m_param_grad_is_none is not None else 1.0)
                logA["grad.norm.M_param"] = float(m_param_grad_norm or 0.0)
                if m_param_delta_l2 is not None:
                    logA["M_param.delta_l2"] = m_param_delta_l2
                logA["grad.is_none.M_leaf"] = float(m_leaf_grad_is_none if m_leaf_grad_is_none is not None else 1.0)
                logA["grad.norm.M_leaf"] = float(m_leaf_grad_norm or 0.0)
                if m_leaf_delta_l2 is not None:
                    logA["M_leaf.delta_l2"] = m_leaf_delta_l2
                logA["grad.is_none.M_used"] = float(m_used_grad_is_none if m_used_grad_is_none is not None else 1.0)
                logA["grad.norm.M_used"] = float(m_used_grad_norm or 0.0)
            if args.log_grad_stats and exchange_freeze_steps:
                if exchange_executed is not None and exchange_executed >= 1.0 and total_steps >= exchange_freeze_steps:
                    if m_leaf_in_optimizer is not None:
                        in_optimizer = float(m_leaf_in_optimizer)
                    else:
                        in_optimizer = float(m_param_in_optimizer or 0.0)
                    if m_leaf_requires_grad is not None:
                        requires_grad = float(m_leaf_requires_grad)
                    else:
                        requires_grad = float(m_param_requires_grad or 0.0)
                    if total_steps <= exchange_freeze_steps + m_grad_probe_window:
                        if in_optimizer < 1.0:
                            if m_probe_assert is None:
                                m_probe_assert = (
                                    "M_leaf.in_optimizer=0 after unfreeze at step %d." % total_steps
                                )
                        if requires_grad < 1.0:
                            if m_probe_assert is None:
                                m_probe_assert = (
                                    "M_leaf.requires_grad=0 after unfreeze at step %d." % total_steps
                                )
                    if m_grad_probe_deadline is None:
                        m_grad_probe_deadline = exchange_freeze_steps + m_grad_probe_window
                        m_grad_probe_start = total_steps
                    if m_leaf_grad_is_none is not None:
                        grad_present = m_leaf_grad_is_none == 0.0
                    else:
                        grad_present = (m_param_grad_is_none == 0.0) if m_param_grad_is_none is not None else False
                    if m_leaf_delta_l2 is not None:
                        delta_moved = m_leaf_delta_l2 > 0.0
                    else:
                        delta_moved = (m_param_delta_l2 is not None and m_param_delta_l2 > 0.0)
                    if grad_present or delta_moved:
                        m_grad_probe_deadline = None
                        m_grad_probe_start = None
                    elif total_steps >= m_grad_probe_deadline:
                        if m_probe_assert is None:
                            m_probe_assert = (
                                "M grad/update missing after exchange executed at step %d (window=%d)."
                                % (m_grad_probe_start, m_grad_probe_window)
                            )
            if finisher_mode and finisher_step_in_phase is not None:
                logA["finisher/is_active"] = 1.0
                logA["finisher/step_in_phase"] = finisher_step_in_phase
                logA["finisher/steps_total"] = float(args.finisher_steps)
                logA["finisher/lr"] = current_lr
                if finisher_start_step is not None:
                    logA["finisher/start_step"] = float(finisher_start_step)
                if finisher_end_step is not None:
                    logA["finisher/end_step"] = float(finisher_end_step)
                if finisher_step_in_phase == 0:
                    logA["finisher/selected_checkpoint_path"] = args.resume or ""
                    if resume_step is not None:
                        logA["finisher/resumed_step"] = float(resume_step)
                    if finisher_start_step is not None:
                        logA["finisher/start_step"] = float(finisher_start_step)
                    if finisher_end_step is not None:
                        logA["finisher/end_step"] = float(finisher_end_step)
                    if finisher_lr_start is not None:
                        logA["finisher/lr_start"] = float(finisher_lr_start)
            if not run_header_logged:
                header = {}
                if args.modality == "text":
                    training_scope = "text-only"
                    enabled_banks = ["BANK_RAW256"]
                elif args.modality == "audio":
                    training_scope = "audio-only"
                    enabled_banks = ["BANK_AUDIO_CFF256"]
                else:
                    training_scope = "multimodal"
                    enabled_banks = ["BANK_RAW256", "BANK_AUDIO_CFF256"]
                header["training_scope"] = training_scope
                header["enabled_banks"] = enabled_banks
                header["bank_audio_cff256_disabled"] = float(training_scope == "text-only")
                header["optimizer"] = args.optimizer
                header["lr_schedule"] = args.lr_schedule
                header["peak_lr"] = float(args.peak_lr)
                header["min_lr"] = float(args.min_lr)
                header["warmup_frac"] = float(args.warmup_frac)
                header["warmup_steps"] = int(warmup_steps)
                header["freeze_exchange_steps"] = int(exchange_freeze_steps)
                header["saturation_penalty_weight"] = float(args.saturation_penalty_weight)
                header["saturation_penalty_enabled"] = float(args.enable_saturation_penalty)
                header["spec_version"] = spec_version
                header["n_layers"] = int(config.n_layers)
                header["d_state"] = int(config.d_state)
                header["microblock_size"] = int(config.microblock_size)
                header["exchange_dim"] = int(config.exchange_dim)
                header["activation_shift"] = int(config.activation_shift)
                header["activation_T_q15"] = int(config.activation_T_q15)
                header["b_shift"] = int(config.b_shift)
                tile_dim_expected = config.d_state // 8
                exchange_dim_expected = config.d_state // 16
                tile_dim_power2 = 1 if (config.tile_dim & (config.tile_dim - 1)) == 0 else 0
                microblock_power2 = 1 if (config.microblock_size & (config.microblock_size - 1)) == 0 else 0
                header["tile_dim"] = int(config.tile_dim)
                header["tile_dim_expected"] = int(tile_dim_expected)
                header["tile_dim_power2"] = float(tile_dim_power2)
                header["exchange_dim_expected"] = int(exchange_dim_expected)
                header["exchange_dim_matches"] = float(config.exchange_dim == exchange_dim_expected)
                header["microblock_power2"] = float(microblock_power2)
                if config.tile_dim != tile_dim_expected:
                    raise ValueError(
                        f"tile_dim mismatch: tile_dim={config.tile_dim} expected={tile_dim_expected}"
                    )
                if config.exchange_dim != exchange_dim_expected:
                    raise ValueError(
                        f"exchange_dim mismatch: exchange_dim={config.exchange_dim} expected={exchange_dim_expected}"
                    )
                    if not tile_dim_power2:
                        raise ValueError(f"tile_dim must be power of two: tile_dim={config.tile_dim}")
                    if config.microblock_size < 64 or not microblock_power2:
                        raise ValueError(
                            f"microblock_size must be power of two >=64: microblock_size={config.microblock_size}"
                        )
                    if "BANK_RAW256" in enabled_banks:
                        if config.activation_T_q15 != 24576:
                            raise ValueError(
                                f"BANK_RAW256 activation_T_q15 mismatch: {config.activation_T_q15} != 24576"
                            )
                        if config.activation_shift != 8:
                            raise ValueError(
                                f"BANK_RAW256 activation_shift mismatch: {config.activation_shift} != 8"
                            )
                        if config.b_shift != 0:
                            raise ValueError(
                                f"BANK_RAW256 b_shift mismatch: {config.b_shift} != 0"
                            )

                    if args.modality == "text":
                        if not config.use_exchange or config.exchange_every != 1:
                            raise ValueError(
                                "Text-only exchange cadence mismatch: use_exchange=%s exchange_every=%s expected=1"
                                % (config.use_exchange, config.exchange_every)
                            )
                        lengths_local = lengths
                        if lengths_local is None:
                            lengths_local = torch.full(
                                (bytes_in.size(0),),
                                bytes_in.size(1),
                                device=bytes_in.device,
                                dtype=torch.long,
                            )
                        num_blocks = (lengths_local + config.microblock_size - 1) // config.microblock_size
                        tick_count = int(num_blocks.max().item()) if num_blocks.numel() else 0
                        phase = int((tick_count - 1) % 4) if tick_count > 0 else 0
                        header["tick_accum_q16"] = 0
                        header["tick_count"] = tick_count
                        header["phase"] = phase
                        header["exchange_scheduled"] = 1.0

                    with torch.no_grad():
                        a_max_abs = 0.0
                        a_violation_count = 0
                        a_q15_clamp_rates = []
                        b_int8_clamp_rates = []
                        m_spectral = []
                        for idx, block in enumerate(model.blocks):
                            a_vals = block._compute_a()
                            a_max_abs = max(a_max_abs, float(a_vals.abs().max().item()))
                            a_violation_count += int((a_vals.abs() > 1.0 + 1e-6).sum().item())
                            a_q15 = block._compute_a_q15()
                            a_q15_clamp = (
                                ((a_q15 == -32768) | (a_q15 == 32767)).float().mean().item()
                            )
                            a_q15_clamp_rates.append(float(a_q15_clamp))
                            b_int8 = block._compute_b_int8()
                            b_clamp = (
                                ((b_int8 == -127) | (b_int8 == 127)).float().mean().item()
                            )
                            b_int8_clamp_rates.append(float(b_clamp))
                            if block.exchange is not None:
                                weight = block.exchange.weight.float()
                                if weight.numel():
                                    sn, _ = compute_spectral_norm_cpu(
                                        weight,
                                        total_steps,
                                        f"header.block{idx}",
                                        raise_on_invalid=False,
                                    )
                                    m_spectral.append(float(sn))
                        if a_violation_count > 0:
                            raise ValueError(
                                f"Monoid a bounds violated: a_max_abs={a_max_abs:.6f} violations={a_violation_count}"
                            )
                        header["a_max_abs"] = float(a_max_abs)
                        header["a_violation_count"] = float(a_violation_count)
                        header["a_q15_clamp_rate"] = float(sum(a_q15_clamp_rates) / max(1, len(a_q15_clamp_rates)))
                        header["b_int8_clamp_rate"] = float(sum(b_int8_clamp_rates) / max(1, len(b_int8_clamp_rates)))
                        if m_spectral:
                            header["M.spectral_norm_before"] = float(max(m_spectral))

                        compile_checked = 0.0
                        compile_max_abs = 0.0
                        compile_mean_abs = 0.0
                        compile_reason = ""
                        if config.n_layers == 1:
                            sample_n = min(2, bytes_in.size(0))
                            if sample_n > 0:
                                sample_bytes = bytes_in[:sample_n]
                                sample_lengths = None
                                if lengths is not None:
                                    sample_lengths = lengths[:sample_n]
                                with torch.amp.autocast(device_type=amp_device, enabled=False):
                                    out_float = model(sample_bytes, lengths=sample_lengths, quantized=False)
                                    out_quant = model(sample_bytes, lengths=sample_lengths, quantized=True)
                                diff = (out_float["embeddings"] - out_quant["embeddings"]).abs()
                                compile_max_abs = float(diff.max().item())
                                compile_mean_abs = float(diff.mean().item())
                                compile_checked = 1.0
                        else:
                            compile_reason = "quantized_equiv_disabled_n_layers>1"
                        header["monoid/compile_equiv_checked"] = compile_checked
                        header["monoid/compile_equiv_max_abs_diff"] = compile_max_abs
                        header["monoid/compile_equiv_mean_abs_diff"] = compile_mean_abs
                        if compile_reason:
                            header["monoid/compile_equiv_reason"] = compile_reason
                logA.update(header)
                run_header_logged = True
            if grad_model is not None:
                logA["grad/model"] = grad_model
            if grad_last_block is not None:
                logA["grad/last_block"] = grad_last_block
            if grad_proj is not None:
                logA["grad/proj"] = grad_proj
            if layer_weights_live is not None:
                for idx, val in enumerate(layer_weights_live):
                    logA[f"teacher/layer_mix_w{idx}"] = val.item()
            if layer_weights_ema is not None:
                for idx, val in enumerate(layer_weights_ema):
                    logA[f"teacher/layer_mix_ema_w{idx}"] = val.item()
            logB = {}
            logC = {}

            with torch.no_grad():
                student_emb_float = student_emb.float()
                teacher_emb_float = teacher_emb.float()

                if total_steps == 0 and step_i == 0:
                    sample_n = min(4, student_emb_float.size(0))
                    logger.info(
                        "Dim check sample student[0][:8]=%s teacher[0][:8]=%s",
                        student_emb_float[0, :8].tolist(),
                        teacher_emb_float[0, :8].tolist(),
                    )
                    corr = per_dim_correlation(
                        student_emb_float[:sample_n],
                        teacher_emb_float[:sample_n],
                    )
                    if corr is not None:
                        logger.info(
                            "Dim corr stats: mean=%.4f min=%.4f max=%.4f first8=%s",
                            corr.mean().item(),
                            corr.min().item(),
                            corr.max().item(),
                            corr[:8].tolist(),
                        )

                student_norm_pre = student_emb_float.norm(dim=1)
                teacher_norm_pre = teacher_emb_float.norm(dim=1)
                student_normed = F.normalize(student_emb_float, p=2, dim=-1)
                teacher_normed = F.normalize(teacher_emb_float, p=2, dim=-1)
                student_norm_post = student_normed.norm(dim=1)
                teacher_norm_post = teacher_normed.norm(dim=1)
                logA["embed/student_norm_mean"] = student_norm_pre.mean().item()
                logA["embed/student_norm_std"] = student_norm_pre.std(unbiased=False).item()
                logA["embed/teacher_norm_mean"] = teacher_norm_pre.mean().item()
                logA["embed/teacher_norm_std"] = teacher_norm_pre.std(unbiased=False).item()
                logA["embed/student_norm_pre_mean"] = student_norm_pre.mean().item()
                logA["embed/student_norm_pre_std"] = student_norm_pre.std(unbiased=False).item()
                logA["embed/student_norm_post_mean"] = student_norm_post.mean().item()
                logA["embed/student_norm_post_std"] = student_norm_post.std(unbiased=False).item()
                logA["embed/teacher_norm_pre_mean"] = teacher_norm_pre.mean().item()
                logA["embed/teacher_norm_pre_std"] = teacher_norm_pre.std(unbiased=False).item()
                logA["embed/teacher_norm_post_mean"] = teacher_norm_post.mean().item()
                logA["embed/teacher_norm_post_std"] = teacher_norm_post.std(unbiased=False).item()
                if args.log_embedding_stats:
                    logA["embed.l2_norm_mean"] = student_norm_pre.mean().item()
                    logA["embed.l2_norm_p95"] = torch.quantile(student_norm_pre, 0.95).item()
                    scale = (127.0 / student_norm_pre.clamp(min=1e-6)).clamp(max=1.0)
                    scale_q15 = torch.clamp((scale * 32768.0).round(), 0, 32767).float()
                    logA["embed.scale_q15_mean"] = scale_q15.mean().item()
                    logA["embed.scale_q15_p95"] = torch.quantile(scale_q15, 0.95).item()
                a_max_abs = 0.0
                a_sum_abs = 0.0
                a_count = 0
                m_spectral_before = []
                m_spectral_after = []
                for idx, block in enumerate(model.blocks):
                    a_vals = block._compute_a()
                    a_max_abs = max(a_max_abs, float(a_vals.abs().max().item()))
                    a_sum_abs += float(a_vals.abs().mean().item())
                    a_count += 1
                    if block.exchange is not None:
                        weight_after = None
                        weight_orig = None
                        if (
                            hasattr(block.exchange, "parametrizations")
                            and hasattr(block.exchange.parametrizations, "weight")
                            and hasattr(block.exchange.parametrizations.weight, "original")
                        ):
                            weight_orig = block.exchange.parametrizations.weight.original.float()
                        elif hasattr(block.exchange, "weight_orig"):
                            weight_orig = block.exchange.weight_orig.float()
                        else:
                            weight_orig = block.exchange.weight.float()
                        weight_after = weight_orig
                        if weight_orig is not None and weight_orig.numel():
                            sigma, _ = compute_spectral_norm_cpu(
                                weight_orig.float(), total_steps, f"log.orig.block{idx}"
                            )
                            scale = max(1.0, sigma)
                            weight_after = weight_orig / scale
                        if weight_after.numel():
                            if not torch.isfinite(weight_after).all():
                                weight_after = torch.nan_to_num(weight_after, nan=0.0, posinf=0.0, neginf=0.0)
                            sn_after, _ = compute_spectral_norm_cpu(
                                weight_after, total_steps, f"log.after.block{idx}"
                            )
                            m_spectral_after.append(float(sn_after))
                        if hasattr(block.exchange, "weight_orig"):
                            weight_before = block.exchange.weight_orig.float()
                            if weight_before.numel():
                                if not torch.isfinite(weight_before).all():
                                    weight_before = torch.nan_to_num(weight_before, nan=0.0, posinf=0.0, neginf=0.0)
                                sn_before, _ = compute_spectral_norm_cpu(
                                    weight_before, total_steps, f"log.before.block{idx}"
                                )
                                m_spectral_before.append(float(sn_before))
                if a_count:
                    logA["a.max_abs"] = float(a_max_abs)
                    logA["a.mean_abs"] = float(a_sum_abs / float(a_count))
                if m_spectral_before:
                    logA["M.spectral_norm_before"] = float(max(m_spectral_before))
                if m_spectral_after:
                    logA["M.spectral_norm_after"] = float(max(m_spectral_after))
                if m_proj_sn_before is not None:
                    logA["M.spectral_norm_before"] = float(m_proj_sn_before)
                if m_proj_sn_after is not None:
                    logA["M.spectral_norm_after"] = float(m_proj_sn_after)
                sn_isfinite_val = 1.0 if m_proj_sn_isfinite is None else float(m_proj_sn_isfinite)
                logA["M.sn_isfinite"] = sn_isfinite_val
                logA["M.sn_max"] = float(args.m_spectral_norm_max)
                activation_pre = outputs.get("activation_pre_clip_max") if isinstance(outputs, dict) else None
                activation_post = outputs.get("activation_post_clip_max") if isinstance(outputs, dict) else None
                activation_sat = outputs.get("activation_sat_frac_gt_0p99") if isinstance(outputs, dict) else None
                if activation_pre is not None:
                    logA["activation.pre_clip_max"] = float(activation_pre)
                if activation_post is not None:
                    logA["activation.post_clip_max"] = float(activation_post)
                if activation_sat is not None:
                    logA["activation.sat_frac_gt_0p99"] = float(activation_sat)

                if ramp_start_step is not None:
                    in_ramp_window = ramp_start_step <= total_steps < (ramp_end_step or 0)
                else:
                    in_ramp_window = False
                if "activation.sat_frac_gt_0p99" in logA and sat_baseline is None and ramp_start_step is not None:
                    if total_steps < ramp_start_step:
                        sat_baseline = float(logA["activation.sat_frac_gt_0p99"])
                if in_ramp_window and "activation.sat_frac_gt_0p99" in logA and sat_baseline is not None:
                    if float(logA["activation.sat_frac_gt_0p99"]) > sat_baseline + 0.1:
                        sat_spike_streak += 1
                    else:
                        sat_spike_streak = 0
                    if sat_spike_streak > 200:
                        raise RuntimeError("activation.sat_frac_gt_0p99 spiked during ramp window.")
                if in_ramp_window and "M.spectral_norm_after" in logA:
                    m_spec = float(logA["M.spectral_norm_after"])
                    if not math.isfinite(m_spec):
                        logA["M.sn_isfinite"] = 0.0
                    else:
                        if m_spectral_baseline is None:
                            m_spectral_baseline = m_spec
                        if m_spectral_max is None:
                            m_spectral_max = m_spec
                        else:
                            if m_spectral_max >= 1e-3 and m_spec > m_spectral_max * 1.5:
                                raise RuntimeError("M.spectral_norm_after spiked during ramp window.")
                            m_spectral_max = max(m_spectral_max, m_spec)
                        if enable_m_spectral_monotonic_guard and (
                            m_spectral_skip_until is None or total_steps >= m_spectral_skip_until
                        ):
                            if m_spectral_last is not None and m_spec >= m_spectral_last:
                                m_spectral_monotonic += 1
                            else:
                                m_spectral_monotonic = 0
                            m_spectral_last = m_spec
                            if m_spectral_monotonic > 200:
                                raise RuntimeError("M.spectral_norm_after is monotonically increasing during ramp window.")
                if m_param_group_index is not None:
                    logA["sched.m_lr_mult"] = float(m_lr_mult)
                    if m_brake_steps_left > 0:
                        logA["M_freeze_brake_active"] = 0.0 if args.disable_m_brake else 1.0
                        logA["M_freeze_brake_steps_left"] = float(m_brake_steps_left)
                        if m_brake_reason:
                            logA["M_freeze_brake_reason"] = m_brake_reason
                    else:
                        logA["M_freeze_brake_active"] = 0.0
                if (
                    not m_spectral_full_lr_allowed
                    and ramp_end_step is not None
                    and total_steps >= (ramp_end_step + int(args.m_ramp_hold_steps))
                    and m_spectral_baseline is not None
                    and m_spectral_max is not None
                ):
                    threshold = float(args.m_ramp_full_lr_threshold)
                    if threshold <= 0.0:
                        threshold = 1.0
                    if m_spectral_max <= m_spectral_baseline * threshold and m_spectral_monotonic == 0:
                        m_spectral_full_lr_allowed = True

                if finisher_mode and finisher_step_in_phase == 0 and not finisher_debug_logged:
                    proj_dim = 512
                    if student_emb_float.size(1) < proj_dim or teacher_emb_float.size(1) < proj_dim:
                        raise ValueError(
                            "Finisher debug requires projection dim 512 but got "
                            f"student_dim={student_emb_float.size(1)} "
                            f"teacher_dim={teacher_emb_float.size(1)}"
                        )
                    student_proj = F.normalize(student_emb_float[:, :proj_dim], p=2, dim=-1)
                    teacher_proj_norm = F.normalize(teacher_emb_float[:, :proj_dim], p=2, dim=-1)
                    cos_vals = F.cosine_similarity(student_proj, teacher_proj_norm, dim=1)
                    logA["finisher/debug_cos_T_S_512_mean"] = cos_vals.mean().item()
                    logA["finisher/debug_cos_T_S_512_min"] = cos_vals.min().item()
                    logA["finisher/debug_cos_T_S_512_max"] = cos_vals.max().item()
                    logA["finisher/debug_student_norm_pre_mean"] = student_norm_pre.mean().item()
                    logA["finisher/debug_teacher_norm_pre_mean"] = teacher_norm_pre.mean().item()
                    logA["finisher/debug_student_norm_post_mean"] = student_norm_post.mean().item()
                    logA["finisher/debug_teacher_norm_post_mean"] = teacher_norm_post.mean().item()
                    logA["finisher/debug_batch_text_count"] = float(batch_text_count or 0)
                    logA["finisher/debug_batch_audio_count"] = float(batch_audio_count or 0)
                    logA["finisher/debug_proj"] = str(proj_dim)
                    logger.info(
                        "Finisher debug step0: resume_step=%s finisher_start=%s finisher_end=%s lr_start=%s step_in_phase=%s",
                        resume_step,
                        finisher_start_step,
                        finisher_end_step,
                        finisher_lr_start,
                        finisher_step_in_phase,
                    )
                    finisher_debug_logged = True

                for dim in (128, 256, 512):
                    if student_emb_float.size(1) >= dim and teacher_emb_float.size(1) >= dim:
                        student_dim = F.normalize(student_emb_float[:, :dim], p=2, dim=-1)
                        teacher_dim = F.normalize(teacher_emb_float[:, :dim], p=2, dim=-1)
                        cos_val = F.cosine_similarity(student_dim, teacher_dim, dim=1).mean().item()
                        key = f"geom/cos_T_S_{dim}"
                        logA[key] = cos_val
                        ema_val = update_ema(geom_ema_state, key, cos_val, alpha=args.geom_ema_decay)
                        logA[f"geom_ema/{key.split('geom/')[1]}"] = ema_val
                        if (
                            dim == 512
                            and m_param_group_index is not None
                            and (exchange_freeze_steps is None or total_steps >= exchange_freeze_steps)
                        ):
                            if args.disable_m_brake:
                                logA["M_freeze_brake_triggered"] = 0.0
                            else:
                                drop_threshold = float(ema_val) - 0.03
                                if cos_val < drop_threshold:
                                    m_brake_steps_left = max(m_brake_steps_left, 50)
                                    m_brake_reason = "cos512_drop_vs_ema"
                                    logA["M_freeze_brake_triggered"] = 1.0
                                else:
                                    logA["M_freeze_brake_triggered"] = 0.0

                if student_emb_float.size(1) >= max_dim and teacher_emb_float.size(1) >= max_dim:
                    if isinstance(modalities, str):
                        modality_list = [modalities] * student_emb_float.size(0)
                    else:
                        modality_list = list(modalities) if modalities is not None else []
                    if modality_list:
                        if should_log(total_steps, args.neighborhood_log_every):
                            logA["geom/batch_text_count"] = sum(1 for m in modality_list if m == "text")
                            logA["geom/batch_audio_count"] = sum(1 for m in modality_list if m == "audio")
                        student_max = F.normalize(student_emb_float[:, :max_dim], p=2, dim=-1)
                        teacher_max = F.normalize(teacher_emb_float[:, :max_dim], p=2, dim=-1)
                        for label in ("text", "audio"):
                            idx = [i for i, m in enumerate(modality_list) if m == label]
                            if idx:
                                idx_tensor = torch.tensor(idx, device=student_emb_float.device)
                                cos_val = F.cosine_similarity(
                                    student_max[idx_tensor],
                                    teacher_max[idx_tensor],
                                    dim=1,
                                ).mean().item()
                                logA[f"geom/cos_T_S_{max_dim}_{label}"] = cos_val

                if (
                    teacher_emb_float.size(0) > 1
                    and should_log(total_steps, args.neighborhood_log_every)
                    and pair_ids
                    and not isinstance(modalities, str)
                ):
                    modality_list = list(modalities) if modalities is not None else []
                    if len(modality_list) == len(pair_ids):
                        text_idx = {}
                        audio_idx = {}
                        for idx, pid in enumerate(pair_ids):
                            if pid is None:
                                continue
                            if modality_list[idx] == "text" and pid not in text_idx:
                                text_idx[pid] = idx
                            elif modality_list[idx] == "audio" and pid not in audio_idx:
                                audio_idx[pid] = idx
                        matched = [(text_idx[pid], audio_idx[pid]) for pid in text_idx if pid in audio_idx]
                        logA["geom/text_audio_matched_count"] = len(matched)
                        if matched:
                            text_sel = torch.tensor([p[0] for p in matched], device=student_emb_float.device)
                            audio_sel = torch.tensor([p[1] for p in matched], device=student_emb_float.device)
                            text_emb = F.normalize(student_emb_float[text_sel, :max_dim], p=2, dim=-1)
                            audio_emb = F.normalize(student_emb_float[audio_sel, :max_dim], p=2, dim=-1)
                            cos_vals = (text_emb * audio_emb).sum(dim=-1)
                            logA["geom/text_audio_matched_cos_mean"] = cos_vals.mean().item()
                            logA["geom/text_audio_matched_cos_std"] = cos_vals.std(unbiased=False).item()

                if cm_cos_mean is not None and should_log(total_steps, args.neighborhood_log_every):
                    logA["geom/student_cross_modal_cos_mean"] = cm_cos_mean
                    if cm_off_mean is not None:
                        logA["geom/student_cross_modal_cos_off_mean"] = cm_off_mean
                    if cm_cos_gap is not None:
                        logA["geom/student_cross_modal_cos_gap"] = cm_cos_gap
                    if cm_sim_stats is not None:
                        logA["geom/student_cross_modal_sim_min"] = cm_sim_stats["min"]
                        logA["geom/student_cross_modal_sim_mean"] = cm_sim_stats["mean"]
                        logA["geom/student_cross_modal_sim_max"] = cm_sim_stats["max"]
                    if cm_text_requires_grad is not None:
                        logA["geom/paired_text_requires_grad"] = cm_text_requires_grad
                    if cm_audio_requires_grad is not None:
                        logA["geom/paired_audio_requires_grad"] = cm_audio_requires_grad
                    if teacher_cm_diag is not None:
                        logA["geom/teacher_cross_modal_cos_mean"] = teacher_cm_diag
                    if teacher_cm_off is not None:
                        logA["geom/teacher_cross_modal_cos_off_mean"] = teacher_cm_off
                    if teacher_cm_gap is not None:
                        logA["geom/teacher_cross_modal_cos_gap"] = teacher_cm_gap
                    if mm_teacher_diag is not None:
                        logA["geom/mm_teacher_cross_modal_cos_mean"] = mm_teacher_diag
                    if mm_teacher_off is not None:
                        logA["geom/mm_teacher_cross_modal_cos_off_mean"] = mm_teacher_off
                    if mm_teacher_gap is not None:
                        logA["geom/mm_teacher_cross_modal_cos_gap"] = mm_teacher_gap
                    if cm_diag_sample is not None or cm_off_sample is not None:
                        logger.info(
                            "Cross-modal sims step %d diag=%s off=%s",
                            total_steps,
                            cm_diag_sample,
                            cm_off_sample,
                        )

                if paired_text_bytes is not None and paired_audio_bytes is not None and should_log(total_steps, args.neighborhood_log_every):
                    logA["geom/paired_text_byte_std"] = paired_text_bytes.float().std(unbiased=False).item()
                    logA["geom/paired_audio_byte_std"] = paired_audio_bytes.float().std(unbiased=False).item()
                    logA["geom/paired_batch_size"] = int(paired_text_bytes.size(0))
                    logA["geom/paired_text_audio_matched_count"] = int(paired_text_bytes.size(0))
                    if paired_audio_crop is not None:
                        trunc = [
                            crop for crop in paired_audio_crop
                            if crop and crop.get("orig_len", 0) > crop.get("max_samples", 0)
                        ]
                        logA["geom/paired_audio_trunc_frac"] = len(trunc) / float(len(paired_audio_crop))
                        if trunc:
                            logA["geom/paired_audio_max_samples"] = max(
                                crop.get("max_samples", 0) for crop in trunc
                            )
                        max_samples = max(
                            (crop.get("max_samples", 0) for crop in paired_audio_crop if crop),
                            default=0,
                        )
                        logA["geom/paired_audio_sample_rate"] = float(args.audio_sample_rate)
                        logA["geom/paired_audio_max_seconds"] = (
                            float(max_samples) / float(args.audio_sample_rate) if args.audio_sample_rate else 0.0
                        )
                    if paired_pair_ids is not None:
                        pair_ids_list = list(paired_pair_ids)
                        valid_ids = [pid for pid in pair_ids_list if pid is not None]
                        unique_ids = len(set(valid_ids))
                        logA["geom/paired_pair_id_null_frac"] = (
                            1.0 - (len(valid_ids) / max(1, len(pair_ids_list)))
                        )
                        logA["geom/paired_pair_id_unique_frac"] = (
                            unique_ids / max(1, len(valid_ids))
                        )
                        logger.info(
                            "Paired batch IDs step %d: %s",
                            total_steps,
                            pair_ids_list[: min(6, len(pair_ids_list))],
                        )

                if m_brake_steps_left > 0:
                    m_brake_steps_left -= 1

                if teacher_emb_float.size(0) > 1 and should_log(total_steps, args.neighborhood_log_every):
                    n = teacher_emb_float.size(0)
                    m = min(64, n)
                    idx = torch.randperm(n, device=teacher_emb_float.device)[:m]
                    tm = F.normalize(teacher_emb_float[idx].float(), dim=-1)
                    sm = F.normalize(student_emb_float[idx].float(), dim=-1)
                    tt = tm @ tm.t()
                    ss = sm @ sm.t()
                    tt.fill_diagonal_(-float("inf"))
                    ss.fill_diagonal_(-float("inf"))
                    k5 = min(5, m - 1)
                    k10 = min(10, m - 1)

                    def _overlap_mean(idx_t, idx_s, k_eff):
                        overlaps = []
                        for i in range(m):
                            match = (idx_t[i].unsqueeze(1) == idx_s[i].unsqueeze(0)).any(dim=1).float().sum()
                            overlaps.append(match / float(k_eff))
                        return torch.stack(overlaps).mean().item()

                    if k5 > 0:
                        idx_t5 = tt.topk(k5, dim=1).indices
                        idx_s5 = ss.topk(k5, dim=1).indices
                        logA["geom/neighborhood_overlap@5"] = _overlap_mean(idx_t5, idx_s5, k5)
                    if k10 > 0:
                        idx_t10 = tt.topk(k10, dim=1).indices
                        idx_s10 = ss.topk(k10, dim=1).indices
                        logA["geom/neighborhood_overlap@10"] = _overlap_mean(idx_t10, idx_s10, k10)
                    logA["geom/rank_correlation"] = spearman_rowwise_mean_from_sim(tt, ss)

                if text_cache_prefetcher is not None:
                    hits, misses = text_cache_prefetcher.take_cache_stats()
                    total = hits + misses
                    hit_rate = (hits / total) if total > 0 else 0.0
                    logA.update({
                        "cache/text_hits": hits,
                        "cache/text_misses": misses,
                        "cache/text_hit_rate": hit_rate,
                    })
                    if total_steps < 300 and args.teacher_cache_dir_text:
                        logA.update({
                            "cache/text_dir": args.teacher_cache_dir_text,
                            "cache/text_dtype": str(args.teacher_cache_dtype),
                            "cache/text_embed_dim": float(args.teacher_cache_embed_dim_text),
                            "cache/text_prefetch": float(args.teacher_cache_prefetch_text),
                        })
                if mm_cache_prefetcher is not None:
                    hits, misses = mm_cache_prefetcher.take_cache_stats()
                    total = hits + misses
                    hit_rate = (hits / total) if total > 0 else 0.0
                    logA.update({
                        "cache/mm_hits": hits,
                        "cache/mm_misses": misses,
                        "cache/mm_hit_rate": hit_rate,
                    })

                do_log_b = should_log(total_steps, args.geom_log_b_every)
                do_log_c = should_log(total_steps, args.geom_log_c_every)
                if (do_log_b or do_log_c) and teacher_emb_float.size(0) > 1:
                    n = teacher_emb_float.size(0)
                    m = min(64, n)
                    idx = torch.randperm(n, device=teacher_emb_float.device)[:m]
                    tm = teacher_emb_float[idx].float()
                    sm = student_emb_float[idx].float()
                    tm = F.normalize(tm, dim=-1)
                    sm = F.normalize(sm, dim=-1)
                    tt = tm @ tm.t()
                    ss = sm @ sm.t()
                    tt.fill_diagonal_(-float("inf"))
                    ss.fill_diagonal_(-float("inf"))

                    if do_log_b:
                        ema_alpha_logb = args.geom_ema_decay ** max(1, args.geom_log_b_every)
                        t_mean, t_std = _offdiag_stats(tt)
                        s_mean, s_std = _offdiag_stats(ss)
                        logB.update({
                            "geom/teacher_cos_offdiag_mean": t_mean,
                            "geom/teacher_cos_offdiag_std": t_std,
                            "geom/student_cos_offdiag_mean": s_mean,
                            "geom/student_cos_offdiag_std": s_std,
                            "geom/cos_mean_gap": s_mean - t_mean,
                            "geom/cos_std_gap": s_std - t_std,
                            "embed/teacher_pairwise_cos_mean": t_mean,
                            "embed/teacher_pairwise_cos_std": t_std,
                            "embed/student_pairwise_cos_mean": s_mean,
                            "embed/student_pairwise_cos_std": s_std,
                        })
                        mask = ~torch.eye(tt.size(0), dtype=torch.bool, device=tt.device)
                        t_vals = tt[mask]
                        s_vals = ss[mask]
                        t_p10, t_p50, t_p90 = torch.quantile(t_vals, torch.tensor([0.1, 0.5, 0.9], device=tt.device))
                        s_p10, s_p50, s_p90 = torch.quantile(s_vals, torch.tensor([0.1, 0.5, 0.9], device=tt.device))
                        logB.update({
                            "geom/teacher_cos_offdiag_p10": t_p10.item(),
                            "geom/teacher_cos_offdiag_p50": t_p50.item(),
                            "geom/teacher_cos_offdiag_p90": t_p90.item(),
                            "geom/student_cos_offdiag_p10": s_p10.item(),
                            "geom/student_cos_offdiag_p50": s_p50.item(),
                            "geom/student_cos_offdiag_p90": s_p90.item(),
                            "geom/cos_p50_gap": (s_p50 - t_p50).item(),
                        })
                        ema_val = update_ema(
                            geom_ema_state,
                            "geom/student_cos_offdiag_mean",
                            s_mean,
                            alpha=ema_alpha_logb,
                        )
                        logB["geom_ema/student_cos_offdiag_mean"] = ema_val
                        ema_val = update_ema(
                            geom_ema_state,
                            "geom/cos_mean_gap",
                            s_mean - t_mean,
                            alpha=ema_alpha_logb,
                        )
                        logB["geom_ema/cos_mean_gap"] = ema_val
                        if args.debug_geom_ema:
                            student_raw = student_emb_float[idx].float()
                            teacher_raw = teacher_emb_float[idx].float()
                            s_norm = student_raw.norm(dim=1)
                            t_norm = teacher_raw.norm(dim=1)
                            prev_offdiag = geom_ema_state.get("geom/student_cos_offdiag_mean")
                            prev_gap = geom_ema_state.get("geom/cos_mean_gap")
                            logger.info(
                                "Geom EMA debug step %d: embeddings=student_emb_float/teacher_emb_float "
                                "(main batch), sample=%d, normalize=l2",
                                total_steps,
                                m,
                            )
                            logger.info(
                                "Geom EMA debug student_raw shape=%s norm_mean=%.4f norm_std=%.4f",
                                tuple(student_raw.shape),
                                s_norm.mean().item(),
                                s_norm.std(unbiased=False).item(),
                            )
                            logger.info(
                                "Geom EMA debug teacher_raw shape=%s norm_mean=%.4f norm_std=%.4f",
                                tuple(teacher_raw.shape),
                                t_norm.mean().item(),
                                t_norm.std(unbiased=False).item(),
                            )
                            logger.info(
                                "Geom EMA debug cos_offdiag_mean cur=%.4f prev=%.4f alpha=%.6f",
                                s_mean,
                                prev_offdiag if prev_offdiag is not None else float("nan"),
                                ema_alpha_logb,
                            )
                            logger.info(
                                "Geom EMA debug cos_mean_gap cur=%.4f prev=%.4f alpha=%.6f",
                                s_mean - t_mean,
                                prev_gap if prev_gap is not None else float("nan"),
                                ema_alpha_logb,
                            )

                    if do_log_c and m >= 8:
                        ema_alpha_logc = args.geom_ema_decay ** max(1, args.geom_log_c_every)
                        k5 = min(5, m - 1)
                        k10 = min(10, m - 1)
                        k20 = min(20, m - 1)

                        def _overlap_mean(idx_t, idx_s, k_eff):
                            overlaps = []
                            for i in range(m):
                                match = (idx_t[i].unsqueeze(1) == idx_s[i].unsqueeze(0)).any(dim=1).float().sum()
                                overlaps.append(match / float(k_eff))
                            return torch.stack(overlaps).mean().item()

                        def _rand_overlap_mean(idx_t, k_eff):
                            overlaps = []
                            for i in range(m):
                                candidates = torch.cat(
                                    [torch.arange(0, i, device=tt.device), torch.arange(i + 1, m, device=tt.device)]
                                )
                                rand_idx = candidates[torch.randperm(m - 1, device=tt.device)[:k_eff]]
                                match = (idx_t[i].unsqueeze(1) == rand_idx.unsqueeze(0)).any(dim=1).float().sum()
                                overlaps.append(match / float(k_eff))
                            return torch.stack(overlaps).mean().item()

                        if k5 > 0:
                            idx_t5 = tt.topk(k5, dim=1).indices
                            idx_s5 = ss.topk(k5, dim=1).indices
                            overlap5 = _overlap_mean(idx_t5, idx_s5, k5)
                            rand5 = _rand_overlap_mean(idx_t5, k5)
                            logC["geom/neighborhood_agreement@5"] = overlap5
                            logC["geom/knn_overlap_lift@5"] = overlap5 - rand5

                        if k10 > 0:
                            idx_t10 = tt.topk(k10, dim=1).indices
                            idx_s10 = ss.topk(k10, dim=1).indices
                            overlap10 = _overlap_mean(idx_t10, idx_s10, k10)
                            rand10 = _rand_overlap_mean(idx_t10, k10)
                            logC["geom/neighborhood_agreement@10"] = overlap10
                            logC["geom/knn_overlap_lift@10"] = overlap10 - rand10
                            ema_val = update_ema(
                                geom_ema_state,
                                "geom/knn_overlap_lift@10",
                                logC["geom/knn_overlap_lift@10"],
                                alpha=ema_alpha_logc,
                            )
                            logC["geom_ema/knn_overlap_lift@10"] = ema_val

                        if k5 > 0 and k10 > 0:
                            idx_t5 = tt.topk(k5, dim=1).indices
                            idx_s10 = ss.topk(k10, dim=1).indices
                            logC["geom/recall_T5_in_S10"] = _overlap_mean(idx_t5, idx_s10, k5)

                        if k10 > 0 and k20 > 0:
                            idx_t10 = tt.topk(k10, dim=1).indices
                            idx_s20 = ss.topk(k20, dim=1).indices
                            logC["geom/recall_T10_in_S20"] = _overlap_mean(idx_t10, idx_s20, k10)
                            if k5 > 0:
                                idx_t5 = tt.topk(k5, dim=1).indices
                                logC["geom/recall_T5_in_S20"] = _overlap_mean(idx_t5, idx_s20, k5)

                        logC["geom/spearman_mean"] = spearman_rowwise_mean_from_sim(tt, ss)
                        logC["geom/neighborhood_spearman"] = logC["geom/spearman_mean"]
                        ema_val = update_ema(
                            geom_ema_state,
                            "geom/spearman_mean",
                            logC["geom/spearman_mean"],
                            alpha=ema_alpha_logc,
                        )
                        logC["geom_ema/spearman_mean"] = ema_val

                        mask = ~torch.eye(tt.size(0), dtype=torch.bool, device=tt.device)
                        logC["geom/cos_hist_kl"] = _hist_kl(tt[mask], ss[mask])

                        x = sm - sm.mean(dim=0, keepdim=True)
                        svals = torch.linalg.svdvals(x)
                        eig_vals = (svals * svals) / max(1, (m - 1))
                        eig_sum = eig_vals.sum() + 1e-8
                        top1 = eig_vals[0]
                        top5 = eig_vals[: min(5, eig_vals.numel())].sum()
                        logC["geom/eig_top1_frac"] = (top1 / eig_sum).item()
                        logC["geom/eig_top5_frac"] = (top5 / eig_sum).item()
                        ema_val = update_ema(
                            geom_ema_state,
                            "geom/eig_top1_frac",
                            logC["geom/eig_top1_frac"],
                            alpha=ema_alpha_logc,
                        )
                        logC["geom_ema/eig_top1_frac"] = ema_val

                        logC.update(_dim_std_percentiles(sm))

                        def _prefix_knn_overlap(dim: int, k_eff: int) -> float:
                            if sm.size(1) < dim or k_eff <= 0:
                                return 0.0
                            s_dim = F.normalize(sm[:, :dim], dim=-1)
                            s_full = F.normalize(sm, dim=-1)
                            ss_dim = s_dim @ s_dim.t()
                            ss_full = s_full @ s_full.t()
                            ss_dim.fill_diagonal_(-float("inf"))
                            ss_full.fill_diagonal_(-float("inf"))
                            idx_a = ss_dim.topk(k_eff, dim=1).indices
                            idx_b = ss_full.topk(k_eff, dim=1).indices
                            overlaps = []
                            for i in range(m):
                                match = (idx_a[i].unsqueeze(1) == idx_b[i].unsqueeze(0)).any(dim=1).float().sum()
                                overlaps.append(match / float(k_eff))
                            return torch.stack(overlaps).mean().item()

                        if k10 > 0:
                            logC["geom/prefix_knn_overlap_S128_vs_S512@10"] = _prefix_knn_overlap(128, k10)
                        if k5 > 0:
                            logC["geom/prefix_knn_overlap_S128_vs_S512@5"] = _prefix_knn_overlap(128, k5)

                        logC.update(pos_neg_margin_metrics(sm, batch, indices=idx))

            if args.use_wandb:
                wandb.log(logA, step=total_steps)
                if logB and should_log(total_steps, args.geom_log_b_every):
                    wandb.log(logB, step=total_steps)
                if logC and should_log(total_steps, args.geom_log_c_every):
                    wandb.log(logC, step=total_steps)

            last_metrics = logA.copy()
            stat_buffers["loss.total"].append(logA["loss.total"])
            stat_buffers["grad.norm.global"].append(logA["grad.norm.global"])
            stat_buffers["loss.saturation_penalty"].append(logA["loss.saturation_penalty"])
            stat_buffers["grad.nan_or_inf"].append(logA["grad.nan_or_inf"])
            if "activation.pre_clip_max" in logA:
                stat_buffers["activation.pre_clip_max"].append(logA["activation.pre_clip_max"])
            if "activation.post_clip_max" in logA:
                stat_buffers["activation.post_clip_max"].append(logA["activation.post_clip_max"])

            if should_log(total_steps, args.log_every):
                json_metrics = {"step": total_steps, "epoch": epoch}
                json_metrics.update(logA)
                json_metrics.update(logB)
                json_metrics.update(logC)
                _emit_json(json_metrics)
            if m_probe_assert is not None and total_steps >= m_probe_assert_after_step:
                raise AssertionError(m_probe_assert)

            if retrieval and args.retrieval_eval_every and total_steps > 0 and total_steps % args.retrieval_eval_every == 0:
                eval_metrics = _run_retrieval_eval(model, device, retrieval, args.retrieval_batch_size)
                if eval_metrics:
                    eval_metrics["step"] = total_steps
                    if "retrieval/recall@10" in eval_metrics and ramp_start_step is not None:
                        r10 = float(eval_metrics["retrieval/recall@10"])
                        if total_steps < ramp_start_step and retrieval_r10_baseline is None:
                            retrieval_r10_baseline = r10
                        if total_steps >= ramp_start_step and retrieval_r10_baseline is not None:
                            if r10 < retrieval_r10_baseline - 0.05:
                                raise RuntimeError("retrieval R@10 dropped >5% vs pre-ramp baseline.")
                    if args.use_wandb:
                        wandb.log(eval_metrics)
                    _emit_json(eval_metrics)

            if cross_modal and args.cross_modal_eval_every and total_steps > 0 and total_steps % args.cross_modal_eval_every == 0:
                eval_metrics = _run_cross_modal_eval(model, device, cross_modal, args.retrieval_batch_size)
                if eval_metrics:
                    eval_metrics["step"] = total_steps
                    if args.use_wandb:
                        wandb.log(eval_metrics)
                    _emit_json(eval_metrics)

            # Checkpoint Every 500 Steps
            if total_steps % 500 == 0:
                _save_checkpoint(total_steps)
            
            epoch_loss += loss_scalar
            steps_in_epoch += 1
            total_steps += 1
            if quit_event is not None and quit_event.is_set():
                if quit_state.get("save", True):
                    _save_checkpoint(total_steps)
                logger.info("Quit requested by user. Exiting training loop.")
                quit_now = True
                break
            
            # Legacy tqdm removed
            
            if streaming_mode and total_steps >= max_steps:
                break
            
        avg_loss = epoch_loss / max(steps_in_epoch, 1)
        logger.info(f"Epoch {epoch+1} (or segment) Complete. Avg Loss: {avg_loss:.4f}")
        if quit_now:
            break
        
    if stat_buffers["loss.total"]:
        footer = {
            "event": "run_footer",
            "final.loss.total": last_metrics.get("loss.total", 0.0),
            "final.loss.distill": last_metrics.get("loss.distill", 0.0),
            "final.loss.contrast": last_metrics.get("loss.contrast", 0.0),
            "final.loss.neighborhood": last_metrics.get("loss.neighborhood", 0.0),
            "final.loss.rkd_d": last_metrics.get("loss.rkd_d", 0.0),
            "final.loss.rkd_a": last_metrics.get("loss.rkd_a", 0.0),
            "final.loss.saturation_penalty": last_metrics.get("loss.saturation_penalty", 0.0),
            "max.grad.nan_or_inf": float(max(stat_buffers["grad.nan_or_inf"])),
            "median.grad.norm.global": float(np.median(stat_buffers["grad.norm.global"])),
            "median.loss.saturation_penalty": float(np.median(stat_buffers["loss.saturation_penalty"])),
            "median.activation.pre_clip_max": float(
                np.median(stat_buffers["activation.pre_clip_max"])
            )
            if stat_buffers["activation.pre_clip_max"]
            else 0.0,
            "median.activation.post_clip_max": float(
                np.median(stat_buffers["activation.post_clip_max"])
            )
            if stat_buffers["activation.post_clip_max"]
            else 0.0,
        }
        if args.use_wandb:
            wandb.log(footer, step=total_steps)
        _emit_json(footer)

    if log_jsonl_fh is not None:
        log_jsonl_fh.close()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "monoid_embed_final.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")
    if teacher_proj is not None:
        full_path = os.path.join(args.output_dir, "monoid_embed_final_full.pt")
        state = {
            "model": model.state_dict(),
            "teacher_proj": teacher_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": total_steps,
            "contrast_ramp_start_step": contrast_ramp_start_step,
        }
        if teacher_proj_ema is not None:
            state["teacher_proj_ema"] = teacher_proj_ema.state_dict()
        if layer_mix is not None:
            state["layer_mix"] = layer_mix.detach().cpu()
        if layer_mix_ema is not None:
            state["layer_mix_ema"] = layer_mix_ema.detach().cpu()
        torch.save(state, full_path)
        logger.info("Saved full checkpoint to %s", full_path)

def _text_to_bytes(text, max_bytes):
    raw = (text or "").encode("utf-8")
    length = min(len(raw), max_bytes)
    byte_indices = list(raw[:max_bytes])
    if len(byte_indices) < max_bytes:
        byte_indices.extend([0] * (max_bytes - len(byte_indices)))
    return byte_indices, length


def _build_retrieval_eval(args, logger):
    try:
        qrels = load_dataset(
            args.retrieval_dataset,
            split=args.retrieval_split,
            trust_remote_code=args.datasets_trust_remote_code,
        )
    except Exception as exc:
        logger.warning(f"Retrieval eval disabled: failed to load qrels: {exc}")
        return None

    qrels_by_query = {}
    for row in qrels:
        if row.get("score", 0) <= 0:
            continue
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        qrels_by_query.setdefault(qid, set()).add(cid)

    if not qrels_by_query:
        logger.warning("Retrieval eval disabled: no positive qrels found.")
        return None

    rng = random.Random(args.retrieval_seed)
    query_ids = list(qrels_by_query.keys())
    rng.shuffle(query_ids)
    max_queries_by_docs = max(1, args.retrieval_docs // max(1, args.retrieval_positives_per_query))
    query_ids = query_ids[: min(args.retrieval_queries, max_queries_by_docs, len(query_ids))]

    query_texts = {}
    try:
        queries_ds = load_dataset(
            args.retrieval_dataset,
            "queries",
            split="queries",
            streaming=True,
            trust_remote_code=args.datasets_trust_remote_code,
        )
        query_set = set(query_ids)
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in query_set:
                query_texts[qid] = row["text"]
                if len(query_texts) == len(query_set):
                    break
    except Exception as exc:
        logger.warning(f"Retrieval eval disabled: failed to load queries: {exc}")
        return None

    query_ids = [qid for qid in query_ids if qid in query_texts]
    if not query_ids:
        logger.warning("Retrieval eval disabled: no query texts found.")
        return None

    pos_doc_ids = set()
    query_pos_docs = {}
    for qid in query_ids:
        pos_ids = list(qrels_by_query.get(qid, set()))
        if not pos_ids:
            continue
        rng.shuffle(pos_ids)
        pos_ids = pos_ids[: max(1, args.retrieval_positives_per_query)]
        query_pos_docs[qid] = pos_ids
        pos_doc_ids.update(pos_ids)

    if not query_pos_docs:
        logger.warning("Retrieval eval disabled: no queries with positives after sampling.")
        return None

    doc_texts = {}
    needed_pos = set(pos_doc_ids)
    distractor_ids = []
    try:
        corpus_ds = load_dataset(
            args.retrieval_dataset,
            "corpus",
            split="corpus",
            streaming=True,
            trust_remote_code=args.datasets_trust_remote_code,
        )
        scanned = 0
        for row in corpus_ds:
            doc_id = str(row["_id"])
            is_pos = doc_id in needed_pos
            include = is_pos or len(doc_texts) < args.retrieval_docs
            if include and doc_id not in doc_texts:
                title = row.get("title") or ""
                text = row.get("text") or ""
                doc_texts[doc_id] = (title + "\n" + text).strip()
                if is_pos:
                    needed_pos.discard(doc_id)
                else:
                    distractor_ids.append(doc_id)
                # Allow distractors while still searching; evict one if a late positive overfills.
                if len(doc_texts) > args.retrieval_docs and distractor_ids:
                    drop_id = distractor_ids.pop(0)
                    doc_texts.pop(drop_id, None)
            scanned += 1
            if len(doc_texts) >= args.retrieval_docs and not needed_pos:
                break
            if args.retrieval_max_corpus and scanned >= args.retrieval_max_corpus:
                break
    except Exception as exc:
        logger.warning(f"Retrieval eval disabled: failed to load corpus: {exc}")
        return None

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_texts.keys())}
    query_text_list = []
    relevances = []
    for qid in query_ids:
        pos_ids = [cid for cid in query_pos_docs.get(qid, []) if cid in doc_id_to_idx]
        if not pos_ids:
            continue
        query_text_list.append(query_texts[qid])
        relevances.append(set(doc_id_to_idx[cid] for cid in pos_ids))

    if not query_text_list:
        logger.warning("Retrieval eval disabled: no queries with positives in subset.")
        return None

    max_bytes = int(args.max_bytes) if args.max_bytes else 1024
    doc_bytes = []
    doc_lengths = []
    for text in doc_texts.values():
        byte_indices, length = _text_to_bytes(text, max_bytes)
        doc_bytes.append(byte_indices)
        doc_lengths.append(length)
    query_bytes = []
    query_lengths = []
    for text in query_text_list:
        byte_indices, length = _text_to_bytes(text, max_bytes)
        query_bytes.append(byte_indices)
        query_lengths.append(length)

    logger.info(
        "Retrieval eval ready: %d queries, %d docs, k=%s",
        len(query_text_list),
        len(doc_bytes),
        args.retrieval_k,
    )

    return {
        "query_bytes": query_bytes,
        "query_lengths": query_lengths,
        "doc_bytes": doc_bytes,
        "doc_lengths": doc_lengths,
        "relevances": relevances,
        "k_values": args.retrieval_k,
    }


def _embed_batches(model, device, batch_bytes, batch_lengths, batch_size):
    embeddings = []
    for idx in range(0, len(batch_bytes), batch_size):
        bytes_slice = torch.tensor(batch_bytes[idx:idx + batch_size], dtype=torch.long, device=device)
        lengths_slice = torch.tensor(batch_lengths[idx:idx + batch_size], dtype=torch.long, device=device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(bytes_slice, lengths=lengths_slice)
            else:
                out = model(bytes_slice, lengths=lengths_slice)
            embeddings.append(out["embeddings"])
    return torch.cat(embeddings, dim=0)


def _run_retrieval_eval(model, device, retrieval, batch_size):
    eval_model = getattr(model, "_orig_mod", model)
    model_was_training = eval_model.training
    logger = logging.getLogger("train")
    eval_model.eval()
    try:
        doc_emb = _embed_batches(eval_model, device, retrieval["doc_bytes"], retrieval["doc_lengths"], batch_size).float()
        query_emb = _embed_batches(eval_model, device, retrieval["query_bytes"], retrieval["query_lengths"], batch_size).float()
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        sim = query_emb @ doc_emb.t()

        if sim.numel() > 0:
            logger.info(
                "Retrieval eval sim stats: min=%.4f mean=%.4f max=%.4f",
                sim.min().item(),
                sim.mean().item(),
                sim.max().item(),
            )
        q_norm = query_emb.norm(dim=-1)
        d_norm = doc_emb.norm(dim=-1)
        logger.info(
            "Retrieval eval norms: query mean=%.4f std=%.4f | doc mean=%.4f std=%.4f",
            q_norm.mean().item(),
            q_norm.std(unbiased=False).item(),
            d_norm.mean().item(),
            d_norm.std(unbiased=False).item(),
        )
        pos_sims = []
        for idx, rel_set in enumerate(retrieval["relevances"]):
            if rel_set:
                rel_list = list(rel_set)
                pos_sims.append(sim[idx, rel_list].mean().item())
        if pos_sims:
            logger.info(
                "Retrieval eval positive sims: mean=%.4f std=%.4f (n=%d)",
                float(sum(pos_sims)) / max(1, len(pos_sims)),
                float(torch.tensor(pos_sims).std(unbiased=False).item()),
                len(pos_sims),
            )
        metrics = {}
        for k in retrieval["k_values"]:
            k = min(k, sim.size(1))
            topk = sim.topk(k, dim=1).indices
            hits = 0
            for idx, rel_set in enumerate(retrieval["relevances"]):
                if any(doc_idx in rel_set for doc_idx in topk[idx].tolist()):
                    hits += 1
            metrics[f"retrieval/recall@{k}"] = hits / len(retrieval["relevances"])
        return metrics
    finally:
        if model_was_training:
            eval_model.train()


def _get_caption(row: dict) -> Optional[str]:
    for key in ("caption", "text", "transcript", "sentence"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_cross_modal_eval(args, logger):
    try:
        ds = load_dataset(
            args.cross_modal_dataset,
            split=args.cross_modal_split,
            streaming=False,
            trust_remote_code=args.datasets_trust_remote_code,
        )
    except Exception as exc:
        logger.warning(f"Cross-modal eval disabled: failed to load dataset: {exc}")
        return None

    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        logger.warning("Cross-modal eval: audio decode override failed: %s", exc)

    try:
        ds = ds.shuffle(seed=args.cross_modal_seed)
    except Exception:
        pass

    max_pairs = min(args.cross_modal_queries, args.cross_modal_audios)
    audio_processor = MonoidDatasetProcessor(
        modality="audio",
        max_bytes=args.max_bytes,
        audio_sample_rate=args.audio_sample_rate,
        audio_random_crop=False,
    )
    logger.info(
        "Cross-modal eval config: dataset=%s split=%s max_bytes=%d audio_random_crop=%s",
        args.cross_modal_dataset,
        args.cross_modal_split,
        args.max_bytes,
        False,
    )
    query_bytes = []
    query_lengths = []
    audio_bytes = []
    audio_lengths = []
    relevances = []

    for row in ds:
        if len(query_bytes) >= max_pairs:
            break
        caption = _get_caption(row)
        if caption is None:
            continue
        if row.get("audio") is None:
            continue
        item = audio_processor.process_item(row)
        audio_bytes.append(item["bytes"].tolist())
        audio_lengths.append(item["length"])
        q_bytes, q_len = _text_to_bytes(caption, args.max_bytes)
        query_bytes.append(q_bytes)
        query_lengths.append(q_len)
        relevances.append({len(audio_bytes) - 1})

    if not query_bytes:
        logger.warning("Cross-modal eval disabled: no usable AudioCaps samples.")
        return None

    logger.info(
        "Cross-modal eval ready: %d queries, %d audios, k=%s",
        len(query_bytes),
        len(audio_bytes),
        args.cross_modal_k,
    )
    logger.info("Cross-modal eval direction: text queries -> audio corpus")

    return {
        "query_bytes": query_bytes,
        "query_lengths": query_lengths,
        "audio_bytes": audio_bytes,
        "audio_lengths": audio_lengths,
        "relevances": relevances,
        "k_values": args.cross_modal_k,
    }


def _run_cross_modal_eval(model, device, retrieval, batch_size):
    eval_model = getattr(model, "_orig_mod", model)
    model_was_training = eval_model.training
    logger = logging.getLogger("train")
    eval_model.eval()
    try:
        audio_emb = _embed_batches(eval_model, device, retrieval["audio_bytes"], retrieval["audio_lengths"], batch_size).float()
        query_emb = _embed_batches(eval_model, device, retrieval["query_bytes"], retrieval["query_lengths"], batch_size).float()
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        sim = query_emb @ audio_emb.t()

        if sim.numel() > 0:
            sim_min = sim.min().item()
            sim_mean = sim.mean().item()
            sim_max = sim.max().item()
            logger.info(
                "Cross-modal eval sim stats: min=%.4f mean=%.4f max=%.4f",
                sim_min,
                sim_mean,
                sim_max,
            )

        q_norm = query_emb.norm(dim=-1)
        a_norm = audio_emb.norm(dim=-1)
        logger.info(
            "Cross-modal eval norms: query mean=%.4f std=%.4f | audio mean=%.4f std=%.4f",
            q_norm.mean().item(),
            q_norm.std(unbiased=False).item(),
            a_norm.mean().item(),
            a_norm.std(unbiased=False).item(),
        )

        diag_count = min(sim.size(0), sim.size(1))
        if diag_count > 0:
            diag = sim.diagonal()[:diag_count]
            off_mask = torch.ones_like(sim, dtype=torch.bool)
            off_mask[:diag_count, :diag_count].fill_diagonal_(False)
            off_vals = sim[off_mask]
            logger.info(
                "Cross-modal eval diag vs offdiag: diag mean=%.4f std=%.4f | offdiag mean=%.4f std=%.4f",
                diag.mean().item(),
                diag.std(unbiased=False).item(),
                off_vals.mean().item() if off_vals.numel() else float("nan"),
                off_vals.std(unbiased=False).item() if off_vals.numel() else float("nan"),
            )
            sample = diag[: min(10, diag.numel())].tolist()
            logger.info("Cross-modal eval matched cosines (first %d): %s", len(sample), sample)

        metrics = {}
        for k in retrieval["k_values"]:
            k = min(k, sim.size(1))
            topk = sim.topk(k, dim=1).indices
            hits = 0
            for i, rel in enumerate(retrieval["relevances"]):
                if any(int(idx) in rel for idx in topk[i].tolist()):
                    hits += 1
            metrics[f"cross_modal/R@{k}"] = hits / max(1, len(retrieval["relevances"]))
        return metrics
    finally:
        if model_was_training:
            eval_model.train()


if __name__ == "__main__":
    main()
