import io
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class MonoidDatasetProcessorMixin:
    """
    Mixin for processing raw examples into model inputs.
    Used by both Map-style and Iterable-style datasets.
    """
    def __init__(
        self,
        modality='text',
        max_bytes=1024,
        teacher_dim=512,
        teacher=None,
        text_prompt_mix=None,
        text_prompt_mix_mode="random",
        audio_sample_rate=16000,
        audio_random_crop=True,
    ):
        self.modality = modality
        self.max_bytes = max_bytes
        self.teacher_dim = teacher_dim
        self.teacher = teacher
        self.text_prompt_mix = [p for p in (text_prompt_mix or []) if p]
        self.text_prompt_mix_mode = text_prompt_mix_mode
        self._prompt_mix_idx = 0
        self.audio_sample_rate = int(audio_sample_rate) if audio_sample_rate else 16000
        self.audio_random_crop = bool(audio_random_crop)
        self.audio_max_samples = None
        self.logger = logging.getLogger(__name__)

    def _choose_prompt_name(self):
        if not self.text_prompt_mix:
            return None
        if self.text_prompt_mix_mode == "alternate":
            prompt = self.text_prompt_mix[self._prompt_mix_idx % len(self.text_prompt_mix)]
            self._prompt_mix_idx += 1
            return prompt
        return random.choice(self.text_prompt_mix)

    def _resolve_modality(self, example):
        if self.modality != "multimodal":
            return self.modality
        explicit = example.get("modality") or example.get("_modality")
        if explicit:
            return explicit
        audio_value = example.get("audio")
        if audio_value is not None:
            return "audio"
        if example.get("caption") is not None and example.get("text") is None:
            return "audio"
        return "text"

    def _pad_bytes(self, raw_bytes):
        byte_indices = list(raw_bytes)
        if len(byte_indices) > self.max_bytes:
            byte_indices = byte_indices[:self.max_bytes]
        else:
            byte_indices.extend([0] * (self.max_bytes - len(byte_indices)))
        length = min(len(raw_bytes), self.max_bytes)
        return torch.tensor(byte_indices, dtype=torch.long), length

    def _extract_text(self, example):
        for key in (
            "text",
            "sentence",
            "query",
            "document",
            "anchor",
            "positive",
            "sentence1",
            "sentence2",
            "question",
            "answer",
        ):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("set", "sentences", "answers"):
            value = example.get(key)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item
        return ""

    def _extract_pair_id(self, example):
        for key in ("pair_id", "audiocap_id", "audio_id", "sample_id", "uid", "id", "_id", "clip_id"):
            value = example.get(key)
            if value is not None:
                return str(value)
        youtube_id = example.get("youtube_id")
        start_time = example.get("start_time")
        if youtube_id is not None and start_time is not None:
            return f"{youtube_id}:{start_time}"
        return None

    def process_item(self, example, idx=None):
        item_modality = self._resolve_modality(example)
        # 1. Get Raw Bytes
        teacher_input = None
        if item_modality == 'text':
            raw_bytes = self._process_text(example)
        elif item_modality == 'audio':
            raw_bytes, teacher_input = self._process_audio(example)
        else:
            raw_bytes = b''
            
        # 2. Pad/Truncate
        byte_indices = list(raw_bytes)
        
        if len(byte_indices) > self.max_bytes:
            byte_indices = byte_indices[:self.max_bytes]
        else:
            padding = [0] * (self.max_bytes - len(byte_indices))
            byte_indices = byte_indices + padding
        length = min(len(raw_bytes), self.max_bytes)
            
        x_bytes = torch.tensor(byte_indices, dtype=torch.long)
        
        # 3. Get Teacher Input (Raw Data)
        # We return the raw input so the main process can batch-infer on GPU
        if teacher_input is None:
            teacher_input = self.get_teacher_input(example, item_modality=item_modality)
        prompt_name = None
        if item_modality == 'text':
            prompt_name = example.get('prompt_name')
            if prompt_name is None:
                role = example.get('text_role')
                if role in ("query", "document"):
                    prompt_name = role
                elif self.text_prompt_mix:
                    prompt_name = self._choose_prompt_name()
        pair_id = self._extract_pair_id(example)
        
        return {
            'bytes': x_bytes,
            'teacher_input': teacher_input,
            'teacher_prompt': prompt_name,
            'teacher_cache_input': raw_bytes,
            'modality': item_modality,
            'length': length,
            'pair_id': pair_id,
        }

    def _process_text(self, example):
        text = self._extract_text(example)
        bytes_data = text.encode('utf-8')
        return bytes_data

    def _process_audio(self, example):
        audio_dict = example.get('audio')
        if audio_dict is None:
            audio = np.zeros(max(1, self.max_bytes // 2), dtype=np.float32)
        else:
            fallback_path = example.get("file") or example.get("path")
            audio, sample_rate = self._extract_audio_array(audio_dict, fallback_path=fallback_path)
            if audio is None:
                audio = np.zeros(max(1, self.max_bytes // 2), dtype=np.float32)
            else:
                audio = self._ensure_mono(audio)
                audio = self._resample_audio(audio, sample_rate)

        orig_len = int(audio.shape[0])
        if self.audio_max_samples is not None:
            max_samples = int(self.audio_max_samples)
            if max_samples <= 0:
                max_samples = orig_len
            else:
                max_samples = max(1, max_samples)
        else:
            max_samples = max(1, self.max_bytes // 2)
        crop_start = 0
        crop_end = min(orig_len, max_samples)
        if orig_len > max_samples:
            if self.audio_random_crop:
                crop_start = random.randint(0, orig_len - max_samples)
            crop_end = crop_start + max_samples
            audio = audio[crop_start:crop_end]
        elif orig_len < max_samples:
            pad = max_samples - audio.shape[0]
            audio = np.pad(audio, (0, pad), mode="constant")

        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)
        self._last_audio_crop = {
            "start": int(crop_start),
            "end": int(crop_end),
            "orig_len": int(orig_len),
            "max_samples": int(max_samples),
            "random": bool(self.audio_random_crop),
        }
        return pcm16.tobytes(), audio.astype(np.float32)

    def _extract_audio_array(self, audio_dict, fallback_path=None):
        if hasattr(audio_dict, "get_all_samples"):
            return self._samples_to_numpy(audio_dict.get_all_samples())
        if hasattr(audio_dict, "data") and hasattr(audio_dict, "sample_rate"):
            return self._samples_to_numpy(audio_dict)
        if isinstance(audio_dict, dict):
            audio_array = audio_dict.get("array")
            sample_rate = audio_dict.get("sampling_rate") or self.audio_sample_rate
            if audio_array is None:
                path = audio_dict.get("path") or fallback_path
                if path and not os.path.isabs(path) and fallback_path:
                    path = fallback_path
                if path and os.path.exists(path):
                    return self._load_audio_path(path)
                if audio_dict.get("bytes"):
                    return self._load_audio_bytes(audio_dict.get("bytes"))
            return audio_array, sample_rate
        if isinstance(audio_dict, np.ndarray):
            return audio_dict, self.audio_sample_rate
        if isinstance(audio_dict, list):
            return np.asarray(audio_dict, dtype=np.float32), self.audio_sample_rate
        return None, None

    def _samples_to_numpy(self, samples):
        data = getattr(samples, "data", None)
        if data is None:
            return None, None
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=0)
        sample_rate = getattr(samples, "sample_rate", self.audio_sample_rate)
        return data, sample_rate

    def _ensure_mono(self, audio):
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            return audio.mean(axis=0)
        return audio.reshape(-1)

    def _resample_audio(self, audio, sample_rate):
        if not sample_rate or sample_rate == self.audio_sample_rate:
            return audio
        try:
            import torchaudio
        except Exception as exc:
            self.logger.warning("torchaudio unavailable; keeping sample_rate=%s (%s)", sample_rate, exc)
            return audio
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, self.audio_sample_rate)
        return audio_tensor.squeeze(0).cpu().numpy()

    def _load_audio_path(self, path):
        try:
            import torchaudio
        except Exception as exc:
            self.logger.warning("torchaudio unavailable; cannot load audio path %s (%s)", path, exc)
            return None, None
        try:
            audio_tensor, sample_rate = torchaudio.load(path)
        except Exception as exc:
            self.logger.warning("Failed to load audio path %s: %s", path, exc)
            return None, None
        audio = audio_tensor.mean(dim=0).cpu().numpy()
        return audio, sample_rate

    def _load_audio_bytes(self, audio_bytes):
        try:
            import torchaudio
        except Exception as exc:
            self.logger.warning("torchaudio unavailable; cannot load audio bytes (%s)", exc)
            return None, None
        try:
            with io.BytesIO(audio_bytes) as bio:
                audio_tensor, sample_rate = torchaudio.load(bio)
            audio = audio_tensor.mean(dim=0).cpu().numpy()
            return audio, sample_rate
        except Exception as exc:
            self.logger.warning("Failed to load audio bytes via torchaudio: %s", exc)
        try:
            import soundfile as sf
        except Exception as exc:
            self.logger.warning("soundfile unavailable; cannot load audio bytes (%s)", exc)
            return None, None
        try:
            data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            self.logger.warning("Failed to load audio bytes via soundfile: %s", exc)
            return None, None
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=0)
        return data, sample_rate

    def get_teacher_input(self, example, item_modality=None):
        if item_modality is None:
            item_modality = self._resolve_modality(example)
        if item_modality == 'text':
            return self._extract_text(example)
        elif item_modality == 'audio':
            audio_dict = example.get('audio', {})
            return audio_dict.get('array')
        return None


def _get_caption(example):
    for key in ("caption", "text", "transcript", "sentence"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


class MonoidDatasetProcessor(MonoidDatasetProcessorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MonoidDataset(Dataset, MonoidDatasetProcessorMixin):
    """
    Map-style dataset (supports index access).
    """
    def __init__(
        self,
        hf_dataset,
        modality='text',
        max_bytes=1024,
        teacher_dim=512,
        teacher=None,
        text_prompt_mix=None,
        text_prompt_mix_mode="random",
        audio_sample_rate=16000,
        audio_random_crop=True,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality,
            max_bytes,
            teacher_dim,
            teacher,
            text_prompt_mix=text_prompt_mix,
            text_prompt_mix_mode=text_prompt_mix_mode,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        self.dataset = hf_dataset
        # No caching needed in worker if we return raw inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.process_item(self.dataset[idx], idx)


class MonoidPrecomputedTeacherDataset(Dataset, MonoidDatasetProcessorMixin):
    """
    Map-style dataset with precomputed teacher embeddings stored in a memmap.
    """
    def __init__(
        self,
        hf_dataset,
        embeddings_path: str,
        embed_dim: int = 512,
        dtype: str = "float16",
        modality: str = "text",
        max_bytes: int = 1024,
        teacher_dim: int = 512,
        teacher=None,
        text_prompt_mix=None,
        text_prompt_mix_mode="random",
        audio_sample_rate=16000,
        audio_random_crop=True,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality,
            max_bytes,
            teacher_dim,
            teacher,
            text_prompt_mix=text_prompt_mix,
            text_prompt_mix_mode=text_prompt_mix_mode,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        self.dataset = hf_dataset
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.embeddings = np.memmap(
            embeddings_path,
            mode="r",
            dtype=self.dtype,
            shape=(len(hf_dataset), self.embed_dim),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.process_item(self.dataset[idx], idx)
        emb = torch.from_numpy(self.embeddings[idx]).float()
        item["teacher_emb"] = emb
        item["teacher_input"] = None
        return item

class MonoidIterableDataset(IterableDataset, MonoidDatasetProcessorMixin):
    """
    Iterable-style dataset for streaming large datasets.
    Handles correct sharding for multi-process DataLoader.
    """
    def __init__(
        self,
        hf_dataset,
        modality='text',
        max_bytes=1024,
        teacher_dim=512,
        teacher=None,
        text_prompt_mix=None,
        text_prompt_mix_mode="random",
        audio_sample_rate=16000,
        audio_random_crop=True,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality,
            max_bytes,
            teacher_dim,
            teacher,
            text_prompt_mix=text_prompt_mix,
            text_prompt_mix_mode=text_prompt_mix_mode,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        self.dataset = hf_dataset

    def __iter__(self):
        worker_info = get_worker_info()
        
        # Simple sharding logic for HF datasets
        iterator = self.dataset
        if worker_info is not None:
            if hasattr(iterator, 'shard') and worker_info.num_workers > 1:
                try:
                    iterator = iterator.shard(
                        num_shards=worker_info.num_workers,
                        index=worker_info.id,
                    )
                except Exception as exc:
                    self.logger.warning("Iterator shard failed (%s); using unsharded iterator.", exc)
        
        # Shuffle Buffer Implementation
        buffer_size = 10000
        buffer = []
        
        for example in iterator:
            if len(buffer) < buffer_size:
                buffer.append(example)
                yield self.process_item(example, idx=None)
            else:
                # Reservoir sampling / Random swap
                idx = np.random.randint(0, len(buffer))
                yield self.process_item(buffer[idx], idx=None)
                buffer[idx] = example
                
        # Drain buffer
        np.random.shuffle(buffer)
        for example in buffer:
            yield self.process_item(example, idx=None)

class PairedAudioTextDataset(Dataset, MonoidDatasetProcessorMixin):
    """
    Map-style dataset yielding paired text/audio items from the same example.
    """
    def __init__(
        self,
        hf_dataset,
        max_bytes=1024,
        audio_sample_rate=16000,
        audio_random_crop=True,
        audio_max_seconds=10.0,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality="multimodal",
            max_bytes=max_bytes,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        if audio_max_seconds is not None:
            if audio_max_seconds > 0:
                self.audio_max_samples = int(round(audio_max_seconds * self.audio_sample_rate))
            else:
                self.audio_max_samples = 0
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        caption = _get_caption(example)
        if caption is None or example.get("audio") is None:
            raise ValueError("PairedAudioTextDataset requires caption and audio fields.")
        text_raw = self._process_text({"text": caption})
        text_bytes, text_length = self._pad_bytes(text_raw)
        audio_raw, _ = self._process_audio(example)
        audio_crop = getattr(self, "_last_audio_crop", None)
        audio_bytes, audio_length = self._pad_bytes(audio_raw)
        return {
            "text_bytes": text_bytes,
            "text_length": text_length,
            "text_caption": caption,
            "text_pair_id": self._extract_pair_id(example),
            "audio_bytes": audio_bytes,
            "audio_length": audio_length,
            "audio_crop": audio_crop,
            "audio_pair_id": self._extract_pair_id(example),
            "pair_id": self._extract_pair_id(example),
        }


class PairedAudioTextIterableDataset(IterableDataset, MonoidDatasetProcessorMixin):
    """
    Iterable dataset yielding paired text/audio items for streaming datasets.
    """
    def __init__(
        self,
        hf_dataset,
        max_bytes=1024,
        audio_sample_rate=16000,
        audio_random_crop=True,
        audio_max_seconds=10.0,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality="multimodal",
            max_bytes=max_bytes,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        if audio_max_seconds is not None:
            if audio_max_seconds > 0:
                self.audio_max_samples = int(round(audio_max_seconds * self.audio_sample_rate))
            else:
                self.audio_max_samples = 0
        self.dataset = hf_dataset

    def __iter__(self):
        worker_info = get_worker_info()
        iterator = self.dataset
        if worker_info is not None and hasattr(iterator, "shard"):
            iterator = iterator.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        buffer_size = 10000
        buffer = []

        def _emit(example):
            caption = _get_caption(example)
            if caption is None or example.get("audio") is None:
                return None
            text_raw = self._process_text({"text": caption})
            text_bytes, text_length = self._pad_bytes(text_raw)
            audio_raw, _ = self._process_audio(example)
            audio_crop = getattr(self, "_last_audio_crop", None)
            audio_bytes, audio_length = self._pad_bytes(audio_raw)
            return {
                "text_bytes": text_bytes,
                "text_length": text_length,
                "text_caption": caption,
                "text_pair_id": self._extract_pair_id(example),
                "audio_bytes": audio_bytes,
                "audio_length": audio_length,
                "audio_crop": audio_crop,
                "audio_pair_id": self._extract_pair_id(example),
                "pair_id": self._extract_pair_id(example),
            }

        for example in iterator:
            if len(buffer) < buffer_size:
                buffer.append(example)
                item = _emit(example)
                if item is not None:
                    yield item
            else:
                idx = np.random.randint(0, len(buffer))
                item = _emit(buffer[idx])
                buffer[idx] = example
                if item is not None:
                    yield item

        np.random.shuffle(buffer)
        for example in buffer:
            item = _emit(example)
            if item is not None:
                yield item

class RetrievalPairsDataset(Dataset, MonoidDatasetProcessorMixin):
    """
    Map-style dataset yielding paired query/document items.
    Each index returns a list of two processed items.
    """
    def __init__(
        self,
        pairs,
        modality='text',
        max_bytes=1024,
        teacher_dim=512,
        teacher=None,
        text_prompt_mix=None,
        text_prompt_mix_mode="random",
        audio_sample_rate=16000,
        audio_random_crop=True,
    ):
        MonoidDatasetProcessorMixin.__init__(
            self,
            modality,
            max_bytes,
            teacher_dim,
            teacher,
            text_prompt_mix=text_prompt_mix,
            text_prompt_mix_mode=text_prompt_mix_mode,
            audio_sample_rate=audio_sample_rate,
            audio_random_crop=audio_random_crop,
        )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query_text, doc_text = self.pairs[idx]
        query_item = self.process_item({"text": query_text, "text_role": "query"}, idx)
        doc_item = self.process_item({"text": doc_text, "text_role": "document"}, idx)
        return [query_item, doc_item]

def collate_fn(batch):
    if batch and isinstance(batch[0], list):
        batch = [item for sub in batch for item in sub]
    bytes_list = [item['bytes'] for item in batch]
    teacher_input_list = [item['teacher_input'] for item in batch]
    teacher_prompt_list = [item.get('teacher_prompt') for item in batch]
    teacher_cache_list = [item.get('teacher_cache_input') for item in batch]
    modalities = [item.get('modality') for item in batch]
    pair_ids = [item.get('pair_id') for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    teacher_emb_list = [item.get('teacher_emb') for item in batch]
    teacher_emb = None
    if all(emb is not None for emb in teacher_emb_list):
        teacher_emb = torch.stack(teacher_emb_list)
    
    # We return teacher_input as a list (raw data), not stacked tensor yet
    return {
        'bytes': torch.stack(bytes_list),
        'teacher_input': teacher_input_list,
        'teacher_prompt': teacher_prompt_list,
        'teacher_cache_input': teacher_cache_list,
        'modality': modalities,
        'pair_id': pair_ids,
        'teacher_emb': teacher_emb,
        'lengths': lengths,
    }


def collate_paired_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("Paired batch is empty.")
    text_bytes = torch.stack([item["text_bytes"] for item in batch])
    audio_bytes = torch.stack([item["audio_bytes"] for item in batch])
    text_lengths = torch.tensor([item["text_length"] for item in batch], dtype=torch.long)
    audio_lengths = torch.tensor([item["audio_length"] for item in batch], dtype=torch.long)
    text_captions = [item.get("text_caption") for item in batch]
    text_pair_ids = [item.get("text_pair_id") for item in batch]
    audio_pair_ids = [item.get("audio_pair_id") for item in batch]
    audio_crop = [item.get("audio_crop") for item in batch]
    pair_ids = [item.get("pair_id") for item in batch]
    return {
        "text_bytes": text_bytes,
        "text_lengths": text_lengths,
        "text_captions": text_captions,
        "text_pair_ids": text_pair_ids,
        "audio_bytes": audio_bytes,
        "audio_lengths": audio_lengths,
        "audio_crop": audio_crop,
        "audio_pair_ids": audio_pair_ids,
        "pair_id": pair_ids,
    }
