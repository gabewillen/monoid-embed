#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig


def _stub_mteb_registry() -> None:
    # Avoid importing heavy model registries (transformers/torchvision) when we only
    # evaluate a custom in-repo model.
    import types

    stub = types.ModuleType("mteb.models.model_implementations")
    stub.MODEL_REGISTRY = {}
    sys.modules["mteb.models.model_implementations"] = stub


def _stub_sentence_transformers() -> None:
    import types

    class _StubModel:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("sentence_transformers is not available in this environment.")

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = _StubModel
    st.CrossEncoder = _StubModel
    sys.modules["sentence_transformers"] = st

    ce = types.ModuleType("sentence_transformers.cross_encoder")
    ce.CrossEncoder = _StubModel
    sys.modules["sentence_transformers.cross_encoder"] = ce


def _text_to_bytes(text: str | None, max_bytes: int) -> tuple[list[int], int]:
    raw = (text or "").encode("utf-8")
    if max_bytes and max_bytes > 0:
        length = min(len(raw), max_bytes)
        byte_indices = list(raw[:max_bytes])
    else:
        length = len(raw)
        byte_indices = list(raw)
    return byte_indices, length


def _unwrap_checkpoint_state(state: object) -> tuple[object, bool]:
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        return state, False
    has_orig_mod = any(key.startswith("_orig_mod.") or "._orig_mod." in key for key in state)
    if not has_orig_mod:
        return state, False
    remapped: dict = {}
    for key, value in state.items():
        new_key = key.replace("._orig_mod.", ".")
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        remapped[new_key] = value
    return remapped, True


def _start_heartbeat(seconds: int, tasks_count: int) -> None:
    if seconds <= 0:
        return
    start_time = time.time()

    def _worker() -> None:
        while True:
            time.sleep(seconds)
            elapsed = time.time() - start_time
            print(f"HEARTBEAT: {elapsed:.1f}s elapsed, tasks={tasks_count}", flush=True)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


class _TextBytesDataset(Dataset):
    def __init__(self, texts: list[str], max_bytes: int) -> None:
        self._texts = texts
        self._max_bytes = max_bytes

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        byte_indices, length = _text_to_bytes(self._texts[idx], self._max_bytes)
        return byte_indices, length


class MonoidMTEBModel:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        max_bytes: int,
        pool: str,
        normalize: bool,
        quantized: bool,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        matryoshka_dim: int | None,
        model_name: str | None,
        model_revision: str | None,
    ) -> None:
        self.device = device
        self.max_bytes = max_bytes
        self.pool = pool
        self.normalize = normalize
        self.quantized = quantized
        self.batch_size = batch_size
        self.num_workers = max(0, int(num_workers))
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.matryoshka_dim = matryoshka_dim
        self.model_name = model_name
        self.model_revision = model_revision

        config = MonoidEmbedConfig()
        config.use_quantized = quantized
        config.pool_strategy = pool
        config.normalize_output = normalize
        if matryoshka_dim:
            dims = tuple(getattr(config, "matryoshka_dims", ()) or ())
            if matryoshka_dim not in dims:
                raise ValueError(f"matryoshka_dim {matryoshka_dim} not in {dims}")

        self.model = MonoidEmbed(config).to(device)
        state = torch.load(checkpoint, map_location=device)
        state, has_orig_mod = _unwrap_checkpoint_state(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys: {unexpected}")
        if has_orig_mod:
            print("Info: stripped _orig_mod prefixes from checkpoint state.")
        self.model.eval()

    def _embed_texts(self, texts: Iterable[str], batch_size: int | None = None) -> np.ndarray:
        if batch_size is None:
            batch_size = self.batch_size
        texts = list(texts)
        embeddings = []
        use_workers = self.num_workers > 0
        if use_workers:
            dataset = _TextBytesDataset(texts, self.max_bytes)

            def _collate(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
                if not batch:
                    return torch.empty((0, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
                batch_bytes, batch_lengths = zip(*batch)
                if self.max_bytes and self.max_bytes > 0:
                    max_len = self.max_bytes
                else:
                    max_len = max(len(row) for row in batch_bytes)
                padded = [
                    row + [0] * (max_len - len(row)) if len(row) < max_len else row[:max_len]
                    for row in batch_bytes
                ]
                return (
                    torch.tensor(padded, dtype=torch.long),
                    torch.tensor(batch_lengths, dtype=torch.long),
                )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.device.type == "cuda",
                persistent_workers=True,
                collate_fn=_collate,
            )
            for bytes_tensor, lengths_tensor in loader:
                if self.device.type == "cuda":
                    bytes_tensor = bytes_tensor.to(self.device, non_blocking=True)
                    lengths_tensor = lengths_tensor.to(self.device, non_blocking=True)
                else:
                    bytes_tensor = bytes_tensor.to(self.device)
                    lengths_tensor = lengths_tensor.to(self.device)
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = self.model(bytes_tensor, lengths=lengths_tensor)
                    else:
                        out = self.model(bytes_tensor, lengths=lengths_tensor)
                emb = out["embeddings"].float()
                if self.matryoshka_dim:
                    emb = emb[:, : self.matryoshka_dim]
                    if self.normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu().numpy())
        else:
            for idx in range(0, len(texts), batch_size):
                batch = texts[idx:idx + batch_size]
                batch_bytes = []
                batch_lengths = []
                for text in batch:
                    byte_indices, length = _text_to_bytes(text, self.max_bytes)
                    batch_bytes.append(byte_indices)
                    batch_lengths.append(length)
                if self.max_bytes and self.max_bytes > 0:
                    max_len = self.max_bytes
                else:
                    max_len = max((len(row) for row in batch_bytes), default=0)
                padded = [
                    row + [0] * (max_len - len(row)) if len(row) < max_len else row[:max_len]
                    for row in batch_bytes
                ]
                bytes_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
                lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = self.model(bytes_tensor, lengths=lengths_tensor)
                    else:
                        out = self.model(bytes_tensor, lengths=lengths_tensor)
                emb = out["embeddings"].float()
                if self.matryoshka_dim:
                    emb = emb[:, : self.matryoshka_dim]
                    if self.normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        batch_size = kwargs.get("batch_size", self.batch_size)

        def _as_list(value) -> list[str]:
            if value is None:
                return [""]
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            return [value]

        def _texts_from_mapping(mapping: dict) -> list[str]:
            if "text" in mapping:
                return _as_list(mapping["text"])
            if "query" in mapping:
                return _as_list(mapping["query"])
            if "body" in mapping or "title" in mapping:
                bodies = _as_list(mapping.get("body"))
                titles = _as_list(mapping.get("title"))
                if len(titles) == 1 and len(bodies) > 1:
                    titles = titles * len(bodies)
                if len(bodies) == 1 and len(titles) > 1:
                    bodies = bodies * len(titles)
                return [(t + "\n" + b).strip() for t, b in zip(titles, bodies)]
            raise ValueError(f"Unsupported mapping format for encoding: {mapping.keys()}")

        # Handle simple list of strings (standard MTEB behavior)
        if isinstance(inputs, list):
            if not inputs:
                return np.zeros((0, self.matryoshka_dim or 0), dtype=np.float32)
            if isinstance(inputs[0], str):
                return self._embed_texts(inputs, batch_size=batch_size)
            if isinstance(inputs[0], dict):
                texts = []
                for row in inputs:
                    texts.extend(_texts_from_mapping(row))
                return self._embed_texts(texts, batch_size=batch_size)

        if isinstance(inputs, dict):
            return self._embed_texts(_texts_from_mapping(inputs), batch_size=batch_size)

        if hasattr(inputs, "column_names"):
            columns = getattr(inputs, "column_names", [])
            if "text" in columns:
                return self._embed_texts(list(inputs["text"]), batch_size=batch_size)
            if "query" in columns:
                return self._embed_texts(list(inputs["query"]), batch_size=batch_size)
            if "title" in columns or "body" in columns:
                titles = list(inputs["title"]) if "title" in columns else [""] * len(inputs["body"])
                bodies = list(inputs["body"]) if "body" in columns else [""] * len(titles)
                texts = [(t + "\n" + b).strip() for t, b in zip(titles, bodies)]
                return self._embed_texts(texts, batch_size=batch_size)

        all_embeddings = []
        for batch in inputs:
            if isinstance(batch, str):
                all_embeddings.append(self._embed_texts([batch], batch_size=batch_size))
            elif isinstance(batch, dict):
                all_embeddings.append(self._embed_texts(_texts_from_mapping(batch), batch_size=batch_size))
            else:
                raise ValueError(f"Unsupported batch format for encoding: {type(batch)}")
        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1, embeddings2):
        e1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        e2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        return e1 @ e2.t()

    def similarity_pairwise(self, embeddings1, embeddings2):
        e1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        e2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        return (e1 * e2).sum(dim=-1)

    @property
    def mteb_model_meta(self):
        if not self.model_name:
            return None
        from mteb.models.model_meta import ModelMeta

        return ModelMeta(
            loader=None,
            name=self.model_name,
            revision=self.model_revision,
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            reference=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            modalities=["text"],
            model_type=["dense"],
            citation=None,
            contacts=None,
        )


class MonoidMTEBSearch:
    def __init__(
        self,
        encoder: MonoidMTEBModel,
        max_queries: int | None,
        max_docs: int | None,
        smoke_output_dir: str | None,
    ) -> None:
        self.encoder = encoder
        self.max_queries = max_queries
        self.max_docs = max_docs
        self.smoke_output_dir = smoke_output_dir
        self._doc_texts: list[str] = []
        self._doc_ids: list[str] = []
        self._corpus = None

    @property
    def mteb_model_meta(self):
        return self.encoder.mteb_model_meta

    def index(
        self,
        corpus,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        encode_kwargs=None,
    ) -> None:
        self._corpus = corpus

    def encode(self, *args, **kwargs):
        return self.encoder.encode(*args, **kwargs)

    def search(
        self,
        queries,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        top_k: int,
        encode_kwargs=None,
        top_ranked=None,
    ):
        if self._corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        query_count = len(queries)
        if self.max_queries is not None:
            query_count = min(query_count, self.max_queries)
            queries = queries.select(range(query_count))
        query_ids = [str(row["id"]) for row in queries]
        query_texts = [row["text"] for row in queries]
        print(f"SMOKE queries: {len(query_texts)}")

        relevant_by_query = {}
        relevant_doc_ids = set()
        if task_metadata is not None and task_metadata.dataset:
            from datasets import load_dataset

            dataset_path = task_metadata.dataset.get("path")
            dataset_name = task_metadata.dataset.get("name")
            qrels = None
            if dataset_path:
                try:
                    if dataset_name:
                        qrels = load_dataset(dataset_path, dataset_name, split=hf_split)
                    else:
                        qrels = load_dataset(dataset_path, split=hf_split)
                except Exception:
                    qrels = None
            if qrels is not None:
                for row in qrels:
                    if row.get("score", 0) <= 0:
                        continue
                    qid = str(row.get("query-id"))
                    if qid not in query_ids:
                        continue
                    cid = str(row.get("corpus-id"))
                    relevant_by_query.setdefault(qid, []).append(cid)
                    relevant_doc_ids.add(cid)

        corpus = self._corpus
        if self.max_docs is not None:
            rng = np.random.default_rng(1234)
            selected_docs = {}
            negatives = []
            seen = 0
            target_negatives = max(self.max_docs - len(relevant_doc_ids), 0)
            for row in corpus:
                doc_id = str(row["id"])
                title = row.get("title") or ""
                text = row.get("text") or ""
                doc_text = (title + "\n" + text).strip()
                if doc_id in relevant_doc_ids:
                    selected_docs[doc_id] = doc_text
                    continue
                if target_negatives == 0:
                    continue
                seen += 1
                if len(negatives) < target_negatives:
                    negatives.append((doc_id, doc_text))
                else:
                    j = rng.integers(0, seen)
                    if j < target_negatives:
                        negatives[int(j)] = (doc_id, doc_text)
            self._doc_ids = list(selected_docs.keys()) + [doc_id for doc_id, _ in negatives]
            self._doc_texts = list(selected_docs.values()) + [text for _, text in negatives]
            print(f"SMOKE docs: {len(self._doc_texts)} (relevant kept: {len(selected_docs)})")
        else:
            self._doc_ids = [str(row["id"]) for row in corpus]
            self._doc_texts = []
            for row in corpus:
                title = row.get("title") or ""
                text = row.get("text") or ""
                self._doc_texts.append((title + "\n" + text).strip())

        self._write_smoke_set(
            task_metadata=task_metadata,
            hf_split=hf_split,
            query_ids=query_ids,
            query_texts=query_texts,
            doc_ids=self._doc_ids,
            doc_texts=self._doc_texts,
        )

        doc_emb = self.encoder._embed_texts(self._doc_texts)
        query_emb = self.encoder._embed_texts(query_texts)
        scores = torch.as_tensor(query_emb) @ torch.as_tensor(doc_emb).t()
        k = min(top_k, scores.size(1))
        top_vals, top_idx = scores.topk(k, dim=1)
        results = {}
        for q_i, qid in enumerate(query_ids):
            results[qid] = {
                self._doc_ids[idx]: float(score)
                for idx, score in zip(top_idx[q_i].tolist(), top_vals[q_i].tolist())
            }

        for qid in query_ids[:5]:
            rel_ids = relevant_by_query.get(qid, [])
            rel_in_corpus = [cid for cid in rel_ids if cid in set(self._doc_ids)]
            if not rel_in_corpus and rel_ids:
                print(f"SMOKE warning: no relevant docs in capped corpus for {qid}")
            top10 = sorted(results[qid].items(), key=lambda item: item[1], reverse=True)[:10]
            qidx = query_ids.index(qid)
            print("SMOKE query:", query_texts[qidx])
            print("SMOKE relevant ids:", rel_ids)
            print("SMOKE top10:", top10)
        return results

    def _write_smoke_set(
        self,
        *,
        task_metadata,
        hf_split: str | None,
        query_ids: list[str],
        query_texts: list[str],
        doc_ids: list[str],
        doc_texts: list[str],
    ) -> None:
        if not self.smoke_output_dir:
            return
        task_name = getattr(task_metadata, "name", None) or "task"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("_") or "task"
        os.makedirs(self.smoke_output_dir, exist_ok=True)
        path = os.path.join(self.smoke_output_dir, f"smoke_set_{safe_name}.jsonl")
        meta = {"type": "meta", "task": task_name, "split": hf_split}
        if task_metadata is not None and task_metadata.dataset:
            meta["dataset"] = task_metadata.dataset.get("path")
            meta["dataset_name"] = task_metadata.dataset.get("name")
        meta["max_queries"] = self.max_queries
        meta["max_docs"] = self.max_docs
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta) + "\n")
            for qid, text in zip(query_ids, query_texts):
                handle.write(json.dumps({"type": "query", "id": qid, "text": text}) + "\n")
            for doc_id, text in zip(doc_ids, doc_texts):
                handle.write(json.dumps({"type": "doc", "id": doc_id, "text": text}) + "\n")
        print(f"SMOKE set saved: {path}")

    @property
    def mteb_model_meta(self):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with MonoidEmbed checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_bytes", type=int, default=0)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--matryoshka_dim", type=int, default=0)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--task_types", type=str, nargs="*", default=["Retrieval"])
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="mteb_results")
    parser.add_argument("--full_retrieval", action="store_true", help="Run full retrieval suite (otherwise use lite English list).")
    parser.add_argument("--full_english", action="store_true", help="Run full English MTEB suite.")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--heartbeat_seconds", type=int, default=0)
    args = parser.parse_args()

    _stub_mteb_registry()
    _stub_sentence_transformers()
    try:
        from mteb import evaluate, get_tasks
        from mteb.abstasks import AbsTaskRetrieval
        from mteb.types import PromptType
    except Exception as exc:
        raise SystemExit(
            f"Failed to import mteb: {exc}. Install with: uv pip install --python .venv/bin/python -r requirements-mteb.txt"
        ) from exc

    env_device = os.environ.get("TORCH_DEVICE")
    device_name = args.device
    if env_device:
        device_name = env_device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Normalize: {args.normalize}")
    print(f"Max bytes: {args.max_bytes if args.max_bytes > 0 else 'none'}")
    if args.matryoshka_dim:
        print(f"Matryoshka dim: {args.matryoshka_dim}")

    encoder = MonoidMTEBModel(
        checkpoint=args.checkpoint,
        device=device,
        max_bytes=args.max_bytes,
        pool=args.pool,
        normalize=args.normalize,
        quantized=args.quantized,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        matryoshka_dim=args.matryoshka_dim or None,
        model_name=args.model_name,
        model_revision=args.model_revision,
    )

    config = encoder.model.config
    print(f"Embedding dim: {getattr(config, 'embedding_dim', None)}")
    print(f"Matryoshka dims: {getattr(config, 'matryoshka_dims', None)}")

    language = args.language
    if language in {"en", "eng"}:
        language = "eng"

    tasks = None
    task_kwargs = {
        "languages": [language] if language else None,
        "eval_splits": [args.split],
        "exclusive_language_filter": bool(language),
    }
    lite_retrieval_tasks = [
        "SciFact",
        "NFCorpus",
        "TRECCOVID",
        "SCIDOCS",
        "Touche2020Retrieval.v3",
    ]

    if args.full_english:
        if not language:
            language = "eng"
            task_kwargs["languages"] = [language]
            task_kwargs["exclusive_language_filter"] = True
        tasks = get_tasks(**task_kwargs)
    elif args.tasks:
        tasks = get_tasks(tasks=args.tasks, **task_kwargs)
    else:
        if (
            not args.full_retrieval
            and args.task_types == ["Retrieval"]
            and language == "eng"
        ):
            tasks = get_tasks(tasks=lite_retrieval_tasks, **task_kwargs)
            print(f"Using lite retrieval task list: {lite_retrieval_tasks}")
        else:
            tasks = get_tasks(task_types=args.task_types, **task_kwargs)

    tasks = list(tasks)
    if language == "eng":
        tasks = [task for task in tasks if set(task.metadata.languages) == {"eng"}]

    model_modalities = set(encoder.mteb_model_meta.modalities or []) if encoder.mteb_model_meta else set()
    if model_modalities:
        filtered = []
        skipped = 0
        for task in tasks:
            if isinstance(task, AbsTaskRetrieval):
                query_mods = set(task.metadata.get_modalities(PromptType.query))
                doc_mods = set(task.metadata.get_modalities(PromptType.document))
                if query_mods.issubset(model_modalities) and doc_mods.issubset(model_modalities):
                    filtered.append(task)
                else:
                    skipped += 1
            else:
                task_mods = set(task.metadata.modalities or [])
                if not task_mods or task_mods.issubset(model_modalities):
                    filtered.append(task)
                else:
                    skipped += 1
        if filtered:
            tasks = filtered
        if skipped:
            print(f"Filtered out {skipped} tasks due to modality mismatch.", flush=True)

    _start_heartbeat(args.heartbeat_seconds, len(tasks))

    smoke_mode = args.max_queries is not None or args.max_docs is not None
    if smoke_mode:
        print("SMOKE_MODE=1")
        print(f"SMOKE max_queries: {args.max_queries}")
        print(f"SMOKE max_docs: {args.max_docs}")
        model = MonoidMTEBSearch(encoder, args.max_queries, args.max_docs, args.output_dir)
    else:
        model = encoder

    model_name = encoder.model_name or "no_model_name_available"
    model_revision = encoder.model_revision or "no_revision_available"
    model_dir = model_name.replace("/", "__").replace(" ", "_")
    output_root = Path(args.output_dir) / model_dir / model_revision
    output_root.mkdir(parents=True, exist_ok=True)
    if encoder.mteb_model_meta is not None:
        with (output_root / "model_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(encoder.mteb_model_meta.to_dict(), handle, default=str)

    overwrite_results = smoke_mode
    if not overwrite_results:
        tasks = [task for task in tasks if not (output_root / f"{task.metadata.name}.json").exists()]
    if not tasks:
        print("No tasks to run (all results exist).")
        return

    encode_kwargs = {"batch_size": args.batch_size}
    result = evaluate(
        model,
        tasks,
        encode_kwargs=encode_kwargs,
        cache=None,
        show_progress_bar=False,
        raise_error=True,
    )

    for task_result in result.task_results:
        task_path = output_root / f"{task_result.task_name}.json"
        task_result.to_disk(task_path)


if __name__ == "__main__":
    main()
