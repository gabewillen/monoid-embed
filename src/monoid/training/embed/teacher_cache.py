import hashlib
import json
import os
import queue
import sqlite3
import threading
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from filelock import FileLock, Timeout


def _hash_teacher_input(
    teacher_input,
    prompt: Optional[str],
    modality: str,
    extra: Optional[str] = None,
) -> str:
    h = hashlib.sha1()
    h.update(modality.encode("utf-8"))
    if prompt:
        h.update(prompt.encode("utf-8"))
    if extra:
        h.update(extra.encode("utf-8"))

    if teacher_input is None:
        h.update(b"<none>")
        return h.hexdigest()

    if isinstance(teacher_input, str):
        h.update(teacher_input.encode("utf-8"))
        return h.hexdigest()

    if isinstance(teacher_input, (bytes, bytearray)):
        h.update(bytes(teacher_input))
        return h.hexdigest()

    if torch.is_tensor(teacher_input):
        data = teacher_input.detach().cpu().numpy().tobytes()
        h.update(data)
        return h.hexdigest()

    if isinstance(teacher_input, np.ndarray):
        h.update(teacher_input.tobytes())
        return h.hexdigest()

    if hasattr(teacher_input, "tobytes"):
        h.update(teacher_input.tobytes())
        return h.hexdigest()

    h.update(repr(teacher_input).encode("utf-8"))
    return h.hexdigest()


def hash_batch(
    teacher_inputs: Iterable,
    prompts: Optional[Iterable[Optional[str]]],
    modalities,
    extra: Optional[str] = None,
) -> List[str]:
    hashes = []
    inputs_list = list(teacher_inputs)
    prompts_list = list(prompts) if prompts is not None else [None] * len(inputs_list)
    if isinstance(modalities, str):
        modalities_list = [modalities] * len(inputs_list)
    else:
        modalities_list = list(modalities)
    for idx in range(len(inputs_list)):
        prompt = prompts_list[idx] if idx < len(prompts_list) else None
        modality = modalities_list[idx] if idx < len(modalities_list) else modalities_list[-1]
        hashes.append(_hash_teacher_input(inputs_list[idx], prompt, modality, extra=extra))
    return hashes


class TeacherEmbeddingCache:
    def __init__(
        self,
        cache_dir: str,
        embed_dim: int = 512,
        dtype: str = "float16",
        read_lock_timeout: float = 0.1,
        write_lock_timeout: float = 0.5,
    ):
        self.cache_dir = cache_dir
        self.embed_dim = embed_dim
        self.dtype = np.dtype(dtype)
        os.makedirs(cache_dir, exist_ok=True)

        self.data_path = os.path.join(cache_dir, "embeddings.f16")
        self.db_path = os.path.join(cache_dir, "index.sqlite")
        self.meta_path = os.path.join(cache_dir, "meta.json")
        self._read_lock = FileLock(os.path.join(cache_dir, "cache.read.lock"))
        self._write_lock = FileLock(os.path.join(cache_dir, "cache.write.lock"))
        self._read_lock_timeout = read_lock_timeout
        self._write_lock_timeout = write_lock_timeout

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, idx INTEGER)"
        )
        self._conn.commit()

        self._count = self._load_count()
        self._ensure_file_size()
        self._memmap = None
        self._refresh_memmap()
        self._write_meta()
        self._hit = 0
        self._miss = 0

    def _load_count(self) -> int:
        cur = self._conn.execute("SELECT MAX(idx) FROM embeddings")
        row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0]) + 1
        return 0

    def _ensure_file_size(self) -> None:
        bytes_needed = self._count * self.embed_dim * self.dtype.itemsize
        if not os.path.exists(self.data_path):
            with open(self.data_path, "wb") as f:
                if bytes_needed:
                    f.seek(bytes_needed - 1)
                    f.write(b"\0")
        else:
            current = os.path.getsize(self.data_path)
            if current < bytes_needed:
                with open(self.data_path, "r+b") as f:
                    f.seek(bytes_needed - 1)
                    f.write(b"\0")

    def _refresh_memmap(self) -> None:
        if self._count == 0:
            self._memmap = None
            return
        self._memmap = np.memmap(
            self.data_path,
            mode="r+",
            dtype=self.dtype,
            shape=(self._count, self.embed_dim),
        )

    def _write_meta(self) -> None:
        meta = {
            "embed_dim": self.embed_dim,
            "dtype": str(self.dtype),
            "count": self._count,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def get_many(self, hashes: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        try:
            self._read_lock.acquire(timeout=self._read_lock_timeout)
        except Timeout:
            self._miss += len(hashes)
            return [None] * len(hashes), list(range(len(hashes)))
        results: List[Optional[np.ndarray]] = [None] * len(hashes)
        missing = []
        try:
            for idx, h in enumerate(hashes):
                row = self._conn.execute(
                    "SELECT idx FROM embeddings WHERE hash = ?",
                    (h,),
                ).fetchone()
                if row is None:
                    missing.append(idx)
                    continue
                if self._memmap is None:
                    self._refresh_memmap()
                results[idx] = np.array(self._memmap[row[0]], copy=True)
        finally:
            self._read_lock.release()
        hits = len(hashes) - len(missing)
        self._hit += hits
        self._miss += len(missing)
        return results, missing

    def put_many(self, hashes: List[str], embeddings: np.ndarray) -> None:
        try:
            self._write_lock.acquire(timeout=self._write_lock_timeout)
        except Timeout:
            return
        try:
            with self._lock:
                for h, emb in zip(hashes, embeddings):
                    row = self._conn.execute(
                        "SELECT idx FROM embeddings WHERE hash = ?",
                        (h,),
                    ).fetchone()
                    if row is not None:
                        continue
                    idx = self._count
                    self._append_embedding(emb)
                    self._conn.execute(
                        "INSERT INTO embeddings(hash, idx) VALUES (?, ?)",
                        (h, idx),
                    )
                    self._count += 1
                self._conn.commit()
                self._refresh_memmap()
                self._write_meta()
        finally:
            self._write_lock.release()

    def _append_embedding(self, emb: np.ndarray) -> None:
        emb = emb.astype(self.dtype, copy=False)
        with open(self.data_path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            f.write(emb.tobytes(order="C"))

    def take_stats(self) -> Tuple[int, int]:
        hits, misses = self._hit, self._miss
        self._hit = 0
        self._miss = 0
        return hits, misses


class TeacherPrefetcher:
    def __init__(
        self,
        loader: Iterable,
        teacher,
        cache: TeacherEmbeddingCache,
        device: torch.device,
        modality: str,
        prefetch_batches: int = 2,
        cache_key_extra: Optional[str] = None,
        use_thread: bool = True,
    ):
        self.loader = iter(loader)
        self.teacher = teacher
        self.cache = cache
        self.device = device
        self.modality = modality
        self.cache_key_extra = cache_key_extra
        self.use_thread = bool(use_thread)
        self.queue = None
        if self.use_thread:
            self.queue = queue.Queue(maxsize=max(1, prefetch_batches))
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        self._last_hits = 0
        self._last_misses = 0

    def _embed_teacher_batch(self, inputs, prompts, modalities):
        if isinstance(modalities, str):
            modalities_list = [modalities] * len(inputs)
        else:
            modalities_list = list(modalities)
        prompts_list = list(prompts) if prompts is not None else [None] * len(inputs)

        grouped = {}
        for idx, modality in enumerate(modalities_list):
            prompt = prompts_list[idx] if idx < len(prompts_list) else None
            prompt_key = prompt or self.teacher.text_prompt_name
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

    def _process_batch(self, batch):
        teacher_input = batch["teacher_input"]
        prompts = batch.get("teacher_prompt")
        cache_inputs = batch.get("teacher_cache_input", teacher_input)
        modalities = self.modality
        if self.modality == "multimodal":
            modalities = batch.get("modality", self.modality)
        hashes = hash_batch(cache_inputs, prompts, modalities, extra=self.cache_key_extra)
        cached, missing = self.cache.get_many(hashes)

        if missing:
            if self.teacher is None:
                raise RuntimeError("Teacher cache miss and no teacher available.")
            teacher_input = batch["teacher_input"]
            teacher_prompts = prompts
            missing_inputs = [teacher_input[i] for i in missing]
            missing_prompts = [teacher_prompts[i] for i in missing] if teacher_prompts else None
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

        teacher_emb = torch.from_numpy(np.stack(cached)).float()
        batch["teacher_emb"] = teacher_emb
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
        hits, misses = self.cache.take_stats()
        self._last_hits = hits
        self._last_misses = misses
        return hits, misses
