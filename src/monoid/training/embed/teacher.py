import hashlib
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TeacherModelHandler(nn.Module):
    """
    Handles teacher model inference for text and audio.
    Unifies all teacher embeddings into a shared 512-dimensional space.
    """
    def __init__(self, device="cuda", unified_dim=512, modalities=None, text_prompt_name: str | None = "document"):
        super().__init__()
        self.device = torch.device(device)
        self.unified_dim = unified_dim
        self.text_prompt_name = text_prompt_name
        
        # Load text + audio by default if not specified
        if modalities is None:
            modalities = ["text", "audio"]
        self.modalities = modalities
        
        # 1. Text Teacher: EmbeddingGemma-300M (768d)
        if "text" in self.modalities:
            logger.info("Loading Text Teacher: google/embeddinggemma-300m...")
            token = os.environ.get("HF_TOKEN")
            from sentence_transformers import SentenceTransformer
            self.text_teacher = SentenceTransformer("google/embeddinggemma-300m", device=device, token=token)
            self.text_proj = None
        else:
            self.text_teacher = None
        
        # 2. Multimodal Teacher: Gemma 3n E4B (Audio 1536d)
        if "audio" in self.modalities:
            logger.info("Loading Multimodal Teacher: google/gemma-3n-E4B...")
            model_id = "google/gemma-3n-E4B"
            token = os.environ.get("HF_TOKEN")
            from transformers import AutoProcessor, AutoModel
            self.mm_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
            self.mm_teacher = AutoModel.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device,
                token=token
            )
            # Projection layers
            self.audio_proj = nn.Linear(1536, unified_dim).to(device)
        else:
            self.mm_teacher = None

    @torch.no_grad()
    def get_text_embedding(self, text, prompt_name: str | None = None):
        """Generates unified 512d text embedding."""
        prompt_name = prompt_name or self.text_prompt_name
        encode_kwargs = {
            "convert_to_tensor": True,
            "normalize_embeddings": False,
            "truncate_dim": self.unified_dim,
            "prompt_name": prompt_name,
        }
        text_batch_size = getattr(self, "text_batch_size", None)
        if text_batch_size:
            encode_kwargs["batch_size"] = int(text_batch_size)
        emb = self.text_teacher.encode(text, **encode_kwargs)
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        if emb.size(1) >= self.unified_dim:
            emb = emb[:, :self.unified_dim]
        else:
            pad = emb.new_zeros((emb.size(0), self.unified_dim - emb.size(1)))
            emb = torch.cat([emb, pad], dim=1)
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def get_audio_embedding(self, audio_data, sample_rate=16000):
        """Generates unified 512d audio embedding."""
        if isinstance(audio_data, np.ndarray) and len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
            
        if isinstance(audio_data, list):
             prompt = ["<audio_soft_token>"] * len(audio_data)
        else:
             prompt = "<audio_soft_token>"
             
        inputs = self.mm_processor(text=prompt, audio=audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(self.device)
        outputs = self.mm_teacher(**inputs, output_hidden_states=True)
        
        if hasattr(self.mm_teacher, 'audio_tower'):
            # Gemma3n expects audio_mel_mask True for invalid/padded positions.
            audio_mel_mask = inputs.get('input_features_mask')
            if audio_mel_mask is None:
                audio_mel_mask = torch.zeros(
                    inputs['input_features'].shape[:2],
                    dtype=torch.bool,
                    device=inputs['input_features'].device,
                )
            else:
                audio_mel_mask = ~audio_mel_mask.bool()
            # Pass all audio-related inputs with correct names
            audio_out = self.mm_teacher.audio_tower(
                audio_mel=inputs['input_features'], 
                audio_mel_mask=audio_mel_mask
            )
            features = audio_out[0] if isinstance(audio_out, tuple) else audio_out.last_hidden_state
        elif hasattr(self.mm_teacher, 'model') and hasattr(self.mm_teacher.model, 'audio_tower'):
             audio_mel_mask = inputs.get('input_features_mask')
             if audio_mel_mask is None:
                 audio_mel_mask = torch.zeros(
                     inputs['input_features'].shape[:2],
                     dtype=torch.bool,
                     device=inputs['input_features'].device,
                 )
             else:
                 audio_mel_mask = ~audio_mel_mask.bool()
             audio_out = self.mm_teacher.model.audio_tower(
                 audio_mel=inputs['input_features'], 
                 audio_mel_mask=audio_mel_mask
             )
             features = audio_out[0] if isinstance(audio_out, tuple) else audio_out.last_hidden_state
        else:
            features = getattr(outputs, "audio_hidden_states", [outputs.last_hidden_state])[-1]
            
        logger.info(f"Audio features raw shape: {features.shape}")
        pooled = features.mean(dim=1)
        logger.info(f"Audio pooled shape: {pooled.shape}")
        return self.audio_proj(pooled.to(torch.float32))

    def get_embedding(self, data, modality, prompt_name: str | None = None):
        """General interface for generating teacher embeddings."""
        if modality == 'text':
            return self.get_text_embedding(data, prompt_name=prompt_name)
        elif modality == 'audio':
            return self.get_audio_embedding(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")


class Gemma3nHiddenStateTeacher(nn.Module):
    """
    Gemma-3n hidden-state teacher for text + audio.
    Returns pooled hidden states from a specified layer.
    """
    def __init__(
        self,
        device="cuda",
        model_id: str = "google/gemma-3n-E4B",
        layer: int = -1,
        layer_indices: tuple[int, ...] | None = None,
        text_prompt_name: str | None = "document",
        audio_prompt: str = "<audio_soft_token>",
        audio_sample_rate: int = 16000,
        dtype: torch.dtype = torch.bfloat16,
        audio_source: str | None = None,
        slice_dim: int = 512,
        text_mode: str | None = None,
        text_debug: bool | None = None,
        stream_idx_text: int | None = None,
        stream_idx_audio: int | None = None,
        max_bytes: int | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_id = model_id
        self.layer = int(layer)
        if layer_indices is None:
            layer_indices = (self.layer,)
        self.layer_indices = tuple(int(idx) for idx in layer_indices)
        self.text_prompt_name = text_prompt_name
        self.audio_prompt = audio_prompt
        self.audio_sample_rate = int(audio_sample_rate)
        env_audio_source = os.environ.get("GEMMA3N_AUDIO_SOURCE")
        audio_source = audio_source or env_audio_source or "llm_hidden"
        audio_source = audio_source.lower().strip()
        if audio_source not in ("llm_hidden", "audio_tower"):
            raise ValueError(f"Unsupported audio_source: {audio_source}")
        self.audio_source = audio_source
        self.slice_dim = int(slice_dim) if slice_dim else None
        env_text_mode = os.environ.get("GEMMA3N_TEXT_MODE")
        text_mode = text_mode or env_text_mode or "templated"
        text_mode = text_mode.lower().strip()
        if text_mode not in ("raw", "templated"):
            raise ValueError(f"Unsupported text_mode: {text_mode}")
        self.text_mode = text_mode
        if text_debug is None:
            env_text_debug = os.environ.get("GEMMA3N_TEXT_DEBUG", "").strip().lower()
            text_debug = env_text_debug in ("1", "true", "yes", "on")
        self.text_debug = bool(text_debug)
        self._text_debug_logged = False
        if stream_idx_text is None:
            env_stream_text = os.environ.get("GEMMA3N_STREAM_IDX_TEXT")
            if env_stream_text:
                try:
                    stream_idx_text = int(env_stream_text)
                except ValueError:
                    raise ValueError(f"Invalid GEMMA3N_STREAM_IDX_TEXT: {env_stream_text}") from None
        if stream_idx_audio is None:
            env_stream_audio = os.environ.get("GEMMA3N_STREAM_IDX_AUDIO")
            if env_stream_audio:
                try:
                    stream_idx_audio = int(env_stream_audio)
                except ValueError:
                    raise ValueError(f"Invalid GEMMA3N_STREAM_IDX_AUDIO: {env_stream_audio}") from None
        self.stream_idx_text = stream_idx_text
        self.stream_idx_audio = stream_idx_audio
        self.max_bytes = int(max_bytes) if max_bytes else None
        self.prompt_templates = {
            "query": "query: {text}",
            "document": "document: {text}",
        }
        token = os.environ.get("HF_TOKEN")
        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Gemma3n teacher requested but Gemma3n dependencies are missing."
            ) from exc
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.audio_prompt = self._resolve_audio_prompt(audio_prompt)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            token=token,
        )
        self.model.eval()
        self.base_dim = self._infer_hidden_dim()
        self.layer_count = len(self.layer_indices)
        raw_dim = self.base_dim * self.layer_count
        if self.slice_dim is not None:
            self.output_dim = min(self.slice_dim, raw_dim)
        else:
            self.output_dim = raw_dim
        self.audio_projector = self._find_audio_projector()
        self._audio_debug_logged = False
        self._audio_feature_checked = False
        env_pool_debug = os.environ.get("TEACHER_DEBUG_POOL", "").strip().lower()
        self._pool_debug = env_pool_debug in ("1", "true", "yes", "on")
        env_pool_debug_n = os.environ.get("TEACHER_DEBUG_POOL_N", "1").strip()
        try:
            self._pool_debug_remaining = int(env_pool_debug_n)
        except ValueError:
            self._pool_debug_remaining = 1
        if not self._pool_debug:
            self._pool_debug_remaining = 0
        self.cache_key = self._build_cache_key()

    def _build_cache_key(self) -> str:
        layer_str = ",".join(str(idx) for idx in self.layer_indices)
        max_bytes = self.max_bytes if self.max_bytes is not None else "na"
        stream_text = "na" if self.stream_idx_text is None else str(self.stream_idx_text)
        stream_audio = "na" if self.stream_idx_audio is None else str(self.stream_idx_audio)
        return (
            f"{self.model_id}|layers={layer_str}|audio={self.audio_source}|slice={self.output_dim}"
            f"|text={self.text_mode}|stream_text={stream_text}|stream_audio={stream_audio}"
            f"|max_bytes={max_bytes}|audio_prompt={self.audio_prompt}"
        )

    def _resolve_audio_prompt(self, prompt: str) -> str:
        if not prompt or self.tokenizer is None:
            return prompt
        try:
            if prompt in self.tokenizer.get_vocab() or prompt in self.tokenizer.all_special_tokens:
                return prompt
        except Exception:
            return prompt
        for tok in self.tokenizer.all_special_tokens or []:
            if "audio" in tok.lower():
                logger.warning("Audio prompt %r not found; using %r instead.", prompt, tok)
                return tok
        logger.warning("Audio prompt %r not found in tokenizer; proceeding unchanged.", prompt)
        return prompt

    def _find_audio_projector(self) -> nn.Module | None:
        candidates = (
            "audio_projector",
            "audio_proj",
            "audio_projection",
            "mm_projector",
            "multi_modal_projector",
        )
        roots = [self.model, getattr(self.model, "model", None)]
        for root in roots:
            if root is None:
                continue
            for name in candidates:
                proj = getattr(root, name, None)
                if proj is None:
                    continue
                out_features = getattr(proj, "out_features", None)
                if out_features is not None and int(out_features) != self.base_dim:
                    continue
                return proj
        return None

    def _match_hidden_dim(self, hidden: torch.Tensor) -> torch.Tensor:
        dim = hidden.size(-1)
        if dim == self.base_dim:
            return hidden
        if dim > self.base_dim:
            return hidden[..., : self.base_dim]
        pad = self.base_dim - dim
        return F.pad(hidden, (0, pad))

    def _infer_hidden_dim(self) -> int:
        cfg = self.model.config
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return int(cfg.text_config.hidden_size)
        return 2048

    def _select_hidden_layers(self, hidden_states):
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        layers = []
        for idx in self.layer_indices:
            layer_idx = idx
            if layer_idx < 0:
                layer_idx = len(hidden_states) + layer_idx
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                raise IndexError(f"Hidden state index {idx} out of range.")
            layers.append(hidden_states[layer_idx])
        return layers

    def _select_hidden_layer(self, hidden_states, layer: int) -> torch.Tensor:
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        layer_idx = layer
        if layer_idx < 0:
            layer_idx = len(hidden_states) + layer_idx
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise IndexError(f"Hidden state index {layer} out of range.")
        return hidden_states[layer_idx]

    def _select_stream(
        self,
        hidden: torch.Tensor,
        stream_idx: int | None,
        label: str,
    ) -> torch.Tensor:
        if hidden.dim() == 4:
            if stream_idx is None:
                raise AssertionError(
                    f"{label} hidden has shape {tuple(hidden.shape)} but stream_idx is None."
                )
            if stream_idx < 0 or stream_idx >= hidden.size(0):
                raise AssertionError(
                    f"{label} stream_idx {stream_idx} out of range for hidden shape {tuple(hidden.shape)}"
                )
            if self._pool_debug_remaining > 0:
                logger.info(
                    "Pool debug[%s] select stream=%d hidden=%s",
                    label,
                    stream_idx,
                    tuple(hidden.shape),
                )
            return hidden[stream_idx]
        if hidden.dim() > 4:
            raise AssertionError(
                f"{label} hidden has unsupported shape {tuple(hidden.shape)}; "
                "expected 2D/3D or 4D with stream selection."
            )
        return hidden

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        debug_label: str = "",
    ) -> torch.Tensor:
        debug = self._pool_debug_remaining > 0
        if debug:
            self._pool_debug_remaining -= 1
            logger.info(
                "Pool debug%s hidden=%s attn_mask=%s input_ids=%s input_features=%s",
                f"[{debug_label}]" if debug_label else "",
                tuple(hidden.shape),
                None if attention_mask is None else tuple(attention_mask.shape),
                None if input_ids is None else tuple(input_ids.shape),
                None if input_features is None else tuple(input_features.shape),
            )
        if hidden.dim() == 2:
            return hidden
        if hidden.dim() > 3:
            raise AssertionError(
                f"Pooling expects 2D/3D hidden after stream selection; got {tuple(hidden.shape)}"
            )
        if attention_mask is None:
            if hidden.dim() == 3:
                pooled = hidden.mean(dim=1)
            else:
                pooled = hidden.view(hidden.size(0), -1, hidden.size(-1)).mean(dim=1)
            if pooled.dim() != 2:
                raise AssertionError(f"Pooling without mask produced shape {tuple(pooled.shape)}")
            if debug:
                logger.info(
                    "Pool debug%s batch_dim=0 seq_dim=1 pooled=%s",
                    f"[{debug_label}]" if debug_label else "",
                    tuple(pooled.shape),
                )
            return pooled
        mask = attention_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() != 2:
            raise AssertionError(f"attention_mask must be 2D, got {tuple(mask.shape)}")
        batch_size, seq_len = mask.shape
        dims = list(range(hidden.dim() - 1))
        batch_candidates = [dim for dim in dims if hidden.size(dim) == batch_size]
        seq_candidates = [dim for dim in dims if hidden.size(dim) == seq_len]
        if not batch_candidates or not seq_candidates:
            raise AssertionError(
                "Could not match attention_mask dims to hidden shape: "
                f"hidden={tuple(hidden.shape)} mask={tuple(mask.shape)}"
            )
        candidate_pairs = [(b, s) for b in batch_candidates for s in seq_candidates if b != s]
        if not candidate_pairs:
            raise AssertionError(
                "No valid (batch_dim, seq_dim) pairs for hidden "
                f"{tuple(hidden.shape)} mask {tuple(mask.shape)}"
            )
        adjacent_pairs = [pair for pair in candidate_pairs if pair[1] == pair[0] + 1]
        if adjacent_pairs:
            candidate_pairs = adjacent_pairs
        last_two = (len(dims) - 2, len(dims) - 1)
        if last_two in candidate_pairs:
            batch_dim, seq_dim = last_two
        else:
            batch_dim, seq_dim = max(candidate_pairs, key=lambda pair: (pair[0], pair[1]))
        other_dims = [dim for dim in dims if dim not in (batch_dim, seq_dim)]
        permute_order = [batch_dim, seq_dim] + other_dims + [hidden.dim() - 1]
        hidden = hidden.permute(*permute_order)
        if hidden.dim() != 3:
            raise AssertionError(f"Pooling produced unexpected hidden shape {tuple(hidden.shape)}")
        if hidden.size(0) != batch_size or hidden.size(1) != seq_len:
            raise AssertionError(
                "Pooling permute mismatch: hidden "
                f"{tuple(hidden.shape)} mask {tuple(mask.shape)}"
            )
        mask = mask.to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        pooled = pooled / denom
        if pooled.dim() != 2 or pooled.size(0) != batch_size or pooled.size(1) != hidden.size(-1):
            raise AssertionError(
                "Pooled shape invariant failed: pooled "
                f"{tuple(pooled.shape)} hidden {tuple(hidden.shape)}"
            )
        if debug:
            logger.info(
                "Pool debug%s batch_dim=%s seq_dim=%s extra_dims=%s pooled=%s",
                f"[{debug_label}]" if debug_label else "",
                batch_dim,
                seq_dim,
                other_dims,
                tuple(pooled.shape),
            )
        return pooled

    def _apply_prompt(self, text: str, prompt_name: str | None) -> str:
        if not prompt_name:
            return text
        template = self.prompt_templates.get(prompt_name)
        if template:
            return template.format(text=text)
        return f"{prompt_name}: {text}"

    def _prompt_prefix(self, prompt_name: str | None) -> str:
        if not prompt_name:
            return ""
        template = self.prompt_templates.get(prompt_name)
        if template:
            return template.format(text="")
        return f"{prompt_name}: "

    def _prompt_length(self, prompt_text: str) -> int:
        if not prompt_text or self.tokenizer is None:
            return 0
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded.get("input_ids", []))

    def _mask_prompt_tokens(
        self,
        attention_mask: torch.Tensor | None,
        prompt_lengths: list[int],
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        mask = attention_mask.clone()
        if mask.dim() != 2:
            return mask
        for i, plen in enumerate(prompt_lengths):
            if plen <= 0:
                continue
            if plen >= mask.size(1):
                continue
            mask[i, :plen] = 0
        return mask

    def _postprocess(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.slice_dim is not None and embeddings.size(1) > self.slice_dim:
            embeddings = embeddings[:, : self.slice_dim].contiguous()
        return F.normalize(embeddings.float(), p=2, dim=-1)

    @torch.no_grad()
    def get_text_embedding(
        self,
        text,
        prompt_name: str | None = None,
        layer_override: int | None = None,
    ):
        if isinstance(text, str):
            text = [text]
        raw_texts = [t if isinstance(t, str) else str(t) for t in text]
        prompt_name = prompt_name or self.text_prompt_name
        prefix_len = 0
        if self.text_mode == "templated" and prompt_name:
            texts = [self._apply_prompt(t, prompt_name) for t in raw_texts]
            prefix = self._prompt_prefix(prompt_name)
            prefix_len = self._prompt_length(prefix) if prefix else 0
        else:
            texts = raw_texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs.get("input_ids")
        hashes = None
        unique_ratio = None
        if input_ids is not None:
            hashes = []
            for row in input_ids:
                digest = hashlib.sha1(row.cpu().numpy().tobytes()).hexdigest()
                hashes.append(digest)
            unique_ratio = len(set(hashes)) / max(1, len(hashes))
            if unique_ratio < 0.5:
                logger.warning(
                    "Teacher text input_ids uniqueness ratio low: %.2f; "
                    "templating/masking may be collapsing captions.",
                    unique_ratio,
                )
        if self.text_debug:
            if len(set(texts)) == 1:
                raise RuntimeError("Teacher text inputs are identical within batch.")
            if unique_ratio is not None and unique_ratio <= 0.8:
                raise RuntimeError(
                    f"Teacher text input_ids uniqueness ratio too low: {unique_ratio:.2f}"
                )
            if not self._text_debug_logged:
                logger.info("Teacher text debug raw=%s", raw_texts[:3])
                logger.info("Teacher text debug final=%s", texts[:3])
                if hashes is not None:
                    logger.info("Teacher text debug input_ids sha1=%s", hashes[:3])
                if unique_ratio is not None:
                    logger.info("Teacher text debug uniqueness_ratio=%.2f", unique_ratio)
                self._text_debug_logged = True
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model outputs missing hidden_states.")
        if layer_override is not None:
            layers = [self._select_hidden_layer(hidden_states, layer_override)]
        else:
            layers = self._select_hidden_layers(hidden_states)
        layers = [
            self._select_stream(layer, self.stream_idx_text, "text")
            for layer in layers
        ]
        attn_mask = inputs.get("attention_mask")
        if self.text_mode == "templated" and prefix_len:
            content_mask = self._mask_prompt_tokens(attn_mask, [prefix_len] * len(texts))
        else:
            content_mask = attn_mask
        pooled_layers = [
            self._pool_hidden(
                layer,
                content_mask,
                input_ids=inputs.get("input_ids"),
                debug_label="text",
            )
            for layer in layers
        ]
        pooled = torch.cat(pooled_layers, dim=-1)
        return self._postprocess(pooled).to(torch.float32)

    @torch.no_grad()
    def get_audio_embedding(
        self,
        audio,
        sample_rate: int | None = None,
        layer_override: int | None = None,
    ):
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            audio = [audio]
        sample_rate = int(sample_rate or self.audio_sample_rate)
        if isinstance(audio, list):
            prompt = [self.audio_prompt] * len(audio)
        else:
            prompt = self.audio_prompt
        inputs = self.processor(
            text=prompt,
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        if not self._audio_debug_logged:
            keys = list(inputs.keys())
            logger.info("Gemma3n audio processor keys: %s", keys)
            feats = inputs.get("input_features")
            if feats is not None:
                feat_std = feats.float().std().item()
                logger.info(
                    "Gemma3n audio input_features shape=%s std=%.6f",
                    tuple(feats.shape),
                    feat_std,
                )
            self._audio_debug_logged = True

        if self.audio_source == "audio_tower":
            audio_features = None
            audio_mask = None
            audio_mel_mask = None
            audio_inputs = inputs.get("input_features")
            if audio_inputs is not None:
                input_features_mask = inputs.get("input_features_mask")
                if input_features_mask is not None:
                    audio_mel_mask = ~input_features_mask.bool()
                    audio_mask = (~audio_mel_mask).to(torch.long)
                if hasattr(self.model, "audio_tower"):
                    audio_out = self.model.audio_tower(audio_mel=audio_inputs, audio_mel_mask=audio_mel_mask)
                    audio_features = audio_out[0] if isinstance(audio_out, tuple) else audio_out.last_hidden_state
                elif hasattr(self.model, "model") and hasattr(self.model.model, "audio_tower"):
                    audio_out = self.model.model.audio_tower(audio_mel=audio_inputs, audio_mel_mask=audio_mel_mask)
                    audio_features = audio_out[0] if isinstance(audio_out, tuple) else audio_out.last_hidden_state

            if audio_features is not None:
                feat_std = audio_features.float().std().item()
                if not self._audio_feature_checked:
                    if feat_std <= 1e-6:
                        logger.warning(
                            "Audio tower features appear collapsed (std=%.6f).", feat_std
                        )
                    self._audio_feature_checked = True
                if feat_std > 1e-6:
                    if self.audio_projector is not None:
                        audio_features = self.audio_projector(audio_features)
                    audio_features = self._match_hidden_dim(audio_features)
                    audio_features = self._select_stream(
                        audio_features,
                        self.stream_idx_audio,
                        "audio_tower",
                    )
                    pooled = self._pool_hidden(
                        audio_features,
                        audio_mask,
                        input_ids=inputs.get("input_ids"),
                        input_features=inputs.get("input_features"),
                        debug_label="audio_tower",
                    )
                    if self.layer_count > 1:
                        pooled = torch.cat([pooled] * self.layer_count, dim=-1)
                    return self._postprocess(pooled).to(torch.float32)

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model outputs missing hidden_states.")
        if layer_override is not None:
            layers = [self._select_hidden_layer(hidden_states, layer_override)]
        else:
            layers = self._select_hidden_layers(hidden_states)
        layers = [
            self._select_stream(layer, self.stream_idx_audio, "audio_llm")
            for layer in layers
        ]
        attn_mask = inputs.get("attention_mask")
        pooled_layers = [
            self._pool_hidden(
                layer,
                attn_mask,
                input_ids=inputs.get("input_ids"),
                input_features=inputs.get("input_features"),
                debug_label="audio_llm",
            )
            for layer in layers
        ]
        pooled = torch.cat(pooled_layers, dim=-1)
        return self._postprocess(pooled).to(torch.float32)

    def get_embedding(self, data, modality, prompt_name: str | None = None):
        if modality == "text":
            return self.get_text_embedding(data, prompt_name=prompt_name)
        if modality == "audio":
            return self.get_audio_embedding(data, sample_rate=self.audio_sample_rate)
        raise ValueError(f"Unsupported modality for Gemma3nHiddenStateTeacher: {modality}")


class M2DClapTeacher(nn.Module):
    """
    M2D-CLAP teacher for text + audio embeddings using the portable runtime.
    """
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        repo_path: str | None = None,
        text_prompt_name: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.repo_path = repo_path or os.environ.get("M2D_REPO")
        self.text_prompt_name = text_prompt_name
        self.model = None
        self.cfg = None
        self.output_dim = None
        self.cache_key = self._build_cache_key()
        self._load_model()
        self._infer_output_dim()

    def _build_cache_key(self) -> str:
        basename = os.path.basename(self.checkpoint)
        return f"m2d_clap:{basename}"

    def _load_model(self) -> None:
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"M2D checkpoint not found: {self.checkpoint}")
        if not self.repo_path:
            raise RuntimeError("M2D repo path missing. Set M2D_REPO or pass repo_path.")
        sys.path.insert(0, self.repo_path)
        cfg = None
        model = None
        try:
            from m2d.runtime_audio import Config, RuntimeM2D

            cfg = Config()
            cfg.weight_file = self.checkpoint
            model = RuntimeM2D(cfg=cfg, weight_file=self.checkpoint, encoder_only=True)
        except Exception:
            import importlib.util

            portable_path = os.path.join(self.repo_path, "examples", "portable_m2d.py")
            if not os.path.exists(portable_path):
                raise RuntimeError("portable_m2d.py not found in M2D repo.")
            spec = importlib.util.spec_from_file_location("portable_m2d", portable_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load portable_m2d module spec.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cfg = module.Config()
            model = module.PortableM2D(self.checkpoint, cfg=cfg)

        self.cfg = cfg
        self.model = model.to(self.device)
        self.model.eval()

    def _infer_output_dim(self) -> None:
        if self.output_dim is not None:
            return
        try:
            with torch.no_grad():
                emb = self.get_text_embedding(["test"])
            self.output_dim = int(emb.size(1))
        except Exception:
            self.output_dim = 768

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    @torch.no_grad()
    def get_text_embedding(self, text, prompt_name: str | None = None):
        texts = text if isinstance(text, list) else [text]
        with self._autocast():
            emb = self.model.encode_clap_text(texts, truncate=True)
        return F.normalize(emb.float(), p=2, dim=-1)

    @torch.no_grad()
    def get_audio_embedding(self, audio_data, sample_rate: int = 16000):
        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim == 1:
                audio_list = [audio_data]
            else:
                audio_list = [audio_data[i] for i in range(audio_data.shape[0])]
        else:
            audio_list = list(audio_data)
        max_len = max(audio.shape[0] for audio in audio_list)
        padded = []
        for audio in audio_list:
            if audio.shape[0] < max_len:
                audio = np.pad(audio, (0, max_len - audio.shape[0]), mode="constant")
            padded.append(audio.astype(np.float32))
        batch = torch.tensor(np.stack(padded), dtype=torch.float32, device=self.device)
        with self._autocast():
            emb = self.model.encode_clap_audio(batch)
        return F.normalize(emb.float(), p=2, dim=-1)

    def get_embedding(self, data, modality, prompt_name: str | None = None):
        if modality == "text":
            return self.get_text_embedding(data, prompt_name=prompt_name)
        if modality == "audio":
            return self.get_audio_embedding(data)
        raise ValueError(f"Unsupported modality for M2D-CLAP: {modality}")
