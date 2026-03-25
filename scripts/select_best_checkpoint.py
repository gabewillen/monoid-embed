#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig


def _text_to_bytes(text: str | None, max_bytes: int) -> Tuple[List[int], int]:
    raw = (text or "").encode("utf-8")
    length = min(len(raw), max_bytes)
    byte_indices = list(raw[:max_bytes])
    if len(byte_indices) < max_bytes:
        byte_indices.extend([0] * (max_bytes - len(byte_indices)))
    return byte_indices, length


def _format_doc_text(title: str | None, text: str | None) -> str:
    title = title or ""
    text = text or ""
    if title:
        return (title + " " + text).strip()
    return text.strip()


def _format_query_text(row: dict, mteb_parity: bool) -> str:
    query = row.get("text") or ""
    if mteb_parity and "instruction" in row and row.get("instruction") is not None:
        return f"{query} {row.get('instruction')}"
    return query


def _log_parity_debug(
    logger,
    query_samples: list[tuple[str, str]],
    doc_samples: list[tuple[str, str]],
    joiner: str,
    max_bytes: int,
    trunc_query: int,
    trunc_doc: int,
    normalize_output: bool,
    whitespace_collapse: bool,
) -> None:
    logger.info("MTEB parity debug: joiner=%r", joiner)
    logger.info("MTEB parity debug: max_bytes=%s", max_bytes if max_bytes else "none")
    logger.info("MTEB parity debug: truncated_queries=%d truncated_docs=%d", trunc_query, trunc_doc)
    logger.info("MTEB parity debug: normalize_output=%s whitespace_collapse=%s", normalize_output, whitespace_collapse)
    for idx, (qid, text) in enumerate(query_samples):
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logger.info("MTEB parity debug: query[%d] id=%s sha256=%s text=%r", idx, qid, sha, text)
    for idx, (doc_id, text) in enumerate(doc_samples):
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logger.info("MTEB parity debug: doc[%d] id=%s sha256=%s text=%r", idx, doc_id, sha, text)


def _build_retrieval_eval(args, logger, normalize_output: bool):
    try:
        qrels = load_dataset(args.dataset, split=args.split)
    except Exception as exc:
        logger.error("Failed to load qrels: %s", exc)
        return None

    qrels_by_query = {}
    for row in qrels:
        if row.get("score", 0) <= 0:
            continue
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        qrels_by_query.setdefault(qid, set()).add(cid)

    if not qrels_by_query:
        logger.error("No positive qrels found.")
        return None

    rng = random.Random(args.seed)
    query_texts = {}
    query_ids: list[str] = []
    try:
        if args.mteb_parity:
            queries_ds = load_dataset(args.dataset, "queries", split="queries")
            for row in queries_ds:
                qid = str(row["_id"])
                if qid not in qrels_by_query:
                    continue
                query_texts[qid] = _format_query_text(row, mteb_parity=True)
                query_ids.append(qid)
        else:
            query_ids = list(qrels_by_query.keys())
            rng.shuffle(query_ids)
            if args.positives_per_query is None or args.positives_per_query <= 0:
                max_queries_by_docs = args.queries
            else:
                max_queries_by_docs = max(1, args.docs // max(1, args.positives_per_query))
            query_ids = query_ids[: min(args.queries, max_queries_by_docs, len(query_ids))]
            queries_ds = load_dataset(args.dataset, "queries", split="queries", streaming=True)
            query_set = set(query_ids)
            for row in queries_ds:
                qid = str(row["_id"])
                if qid in query_set:
                    query_texts[qid] = _format_query_text(row, mteb_parity=False)
                    if len(query_texts) == len(query_set):
                        break
    except Exception as exc:
        logger.error("Failed to load queries: %s", exc)
        return None

    query_ids = [qid for qid in query_ids if qid in query_texts]
    if not query_ids:
        logger.error("No query texts found.")
        return None

    pos_doc_ids = set()
    query_pos_docs = {}
    for qid in query_ids:
        pos_ids = list(qrels_by_query.get(qid, set()))
        if not pos_ids:
            continue
        if not args.mteb_parity:
            rng.shuffle(pos_ids)
            if args.positives_per_query is not None and args.positives_per_query > 0:
                pos_ids = pos_ids[: max(1, args.positives_per_query)]
        query_pos_docs[qid] = pos_ids
        pos_doc_ids.update(pos_ids)

    if not query_pos_docs:
        logger.error("No queries with positives after sampling.")
        return None

    doc_texts = {}
    needed_pos = set(pos_doc_ids)
    use_full = args.full_corpus or args.dataset == "mteb/scifact" or args.mteb_parity
    try:
        if use_full:
            corpus_ds = load_dataset(args.dataset, "corpus", split="corpus")
            corpus_rows = list(corpus_ds)
            corpus_ids = [str(row["_id"]) for row in corpus_rows]
            if args.mteb_parity:
                chosen = set(corpus_ids)
            else:
                doc_budget = args.docs if args.docs and args.docs < len(corpus_ids) else len(corpus_ids)
                chosen = set(needed_pos)
                remaining = [doc_id for doc_id in corpus_ids if doc_id not in chosen]
                rng.shuffle(remaining)
                for doc_id in remaining:
                    if len(chosen) >= doc_budget:
                        break
                    chosen.add(doc_id)
            for row in corpus_rows:
                doc_id = str(row["_id"])
                if doc_id not in chosen:
                    continue
                doc_texts[doc_id] = _format_doc_text(row.get("title"), row.get("text"))
        else:
            corpus_ds = load_dataset(args.dataset, "corpus", split="corpus", streaming=True)
            scanned = 0
            for row in corpus_ds:
                doc_id = str(row["_id"])
                include = doc_id in needed_pos or len(doc_texts) < args.docs
                if include and doc_id not in doc_texts:
                    doc_texts[doc_id] = _format_doc_text(row.get("title"), row.get("text"))
                    needed_pos.discard(doc_id)
                scanned += 1
                if len(doc_texts) >= args.docs and not needed_pos:
                    break
                if args.max_corpus and scanned >= args.max_corpus:
                    break
    except Exception as exc:
        logger.error("Failed to load corpus: %s", exc)
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
        logger.error("No queries with positives in subset.")
        return None

    doc_bytes = []
    doc_lengths = []
    trunc_doc = 0
    for text in doc_texts.values():
        if args.max_bytes and args.max_bytes > 0 and len(text.encode("utf-8")) > args.max_bytes:
            trunc_doc += 1
        byte_indices, length = _text_to_bytes(text, args.max_bytes)
        doc_bytes.append(byte_indices)
        doc_lengths.append(length)

    query_bytes = []
    query_lengths = []
    trunc_query = 0
    for text in query_text_list:
        if args.max_bytes and args.max_bytes > 0 and len(text.encode("utf-8")) > args.max_bytes:
            trunc_query += 1
        byte_indices, length = _text_to_bytes(text, args.max_bytes)
        query_bytes.append(byte_indices)
        query_lengths.append(length)

    logger.info(
        "Retrieval eval ready: %d queries, %d docs, k=%s",
        len(query_text_list),
        len(doc_bytes),
        args.k,
    )
    if args.mteb_parity_debug:
        query_samples = [(qid, query_texts[qid]) for qid in query_ids[:3]]
        doc_samples = list(doc_texts.items())[:3]
        _log_parity_debug(
            logger,
            query_samples=query_samples,
            doc_samples=doc_samples,
            joiner=" ",
            max_bytes=args.max_bytes,
            trunc_query=trunc_query,
            trunc_doc=trunc_doc,
            normalize_output=normalize_output,
            whitespace_collapse=False,
        )

    return {
        "query_bytes": query_bytes,
        "query_lengths": query_lengths,
        "doc_bytes": doc_bytes,
        "doc_lengths": doc_lengths,
        "relevances": relevances,
        "k_values": args.k,
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
    model_was_training = model.training
    model.eval()
    try:
        doc_emb = _embed_batches(
            model, device, retrieval["doc_bytes"], retrieval["doc_lengths"], batch_size
        ).float()
        query_emb = _embed_batches(
            model, device, retrieval["query_bytes"], retrieval["query_lengths"], batch_size
        ).float()
        sim = query_emb @ doc_emb.t()
        metrics = {}
        for k in retrieval["k_values"]:
            k_eff = min(k, sim.size(1))
            topk = sim.topk(k_eff, dim=1).indices
            recalls = []
            for idx, rel_set in enumerate(retrieval["relevances"]):
                if not rel_set:
                    recalls.append(0.0)
                    continue
                hits = sum(1 for doc_idx in topk[idx].tolist() if doc_idx in rel_set)
                recalls.append(hits / float(len(rel_set)))
            metrics[f"recall@{k}"] = sum(recalls) / max(1, len(recalls))
        return metrics
    finally:
        if model_was_training:
            model.train()


def _unwrap_checkpoint_state(state):
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        return state, False
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state)
    if has_orig_mod:
        state = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    return state, has_orig_mod


def _list_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    candidates = []
    for name in os.listdir(run_dir):
        if not name.endswith(".pt"):
            continue
        stem = name[:-3]
        if not stem.isdigit():
            continue
        candidates.append((int(stem), os.path.join(run_dir, name)))
    candidates.sort()
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints and select the best by SciFact recall."
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name under checkpoints/.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--preset", type=str, default=None, help="Monoid preset (sets MONOID_PRESET).")
    parser.add_argument("--dataset", type=str, default="mteb/scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--positives_per_query", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_corpus", type=int, default=200000)
    parser.add_argument("--full_corpus", action="store_true")
    parser.add_argument("--max_bytes", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 10])
    parser.add_argument("--tie_tol", type=float, default=0.005)
    parser.add_argument("--output", type=str, default="-", help="Write JSON summary to this path.")
    parser.add_argument("--mteb_parity", action="store_true", help="Match MTEB SciFact formatting and full dataset usage.")
    parser.add_argument("--mteb_parity_debug", action="store_true", help="Log MTEB parity debug samples and hashes.")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    args = parser.parse_args()

    if args.preset:
        os.environ["MONOID_PRESET"] = args.preset.strip().lower()

    run_dir = args.run_dir
    if not run_dir:
        if not args.run_name:
            raise SystemExit("Provide --run-dir or --run-name.")
        run_dir = os.path.join(args.checkpoints_dir, args.run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Checkpoint directory not found: {run_dir}")

    k_values = sorted(set(args.k + [1, 10]))
    args.k = k_values

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("checkpoint-selector")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = MonoidEmbedConfig()
    if args.normalize is not None:
        config.normalize_output = args.normalize
    model = MonoidEmbed(config).to(device)

    retrieval = _build_retrieval_eval(args, logger, config.normalize_output)
    if retrieval is None:
        raise SystemExit(1)

    checkpoints = _list_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No numeric checkpoints found in {run_dir}.")

    results = []
    best = None
    for step, path in checkpoints:
        state = torch.load(path, map_location=device)
        state, compiled_state = _unwrap_checkpoint_state(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Checkpoint %s load warnings: missing=%s unexpected=%s", path, missing, unexpected)
        metrics = _run_retrieval_eval(model, device, retrieval, args.batch_size)
        r1 = float(metrics.get("recall@1", 0.0))
        r10 = float(metrics.get("recall@10", 0.0))
        results.append(
            {
                "step": step,
                "path": path,
                "recall@1": r1,
                "recall@10": r10,
                "compiled_state": compiled_state,
                "metrics": metrics,
            }
        )

        if best is None:
            best = results[-1]
            continue
        if r1 > best["recall@1"] + args.tie_tol:
            best = results[-1]
            continue
        if abs(r1 - best["recall@1"]) <= args.tie_tol:
            if r10 > best["recall@10"]:
                best = results[-1]
                continue
            if abs(r10 - best["recall@10"]) < 1e-12 and step < best["step"]:
                best = results[-1]

    summary = {
        "run_dir": run_dir,
        "dataset": args.dataset,
        "split": args.split,
        "k_values": k_values,
        "best": best,
        "results": results,
    }

    if args.output == "-":
        print(json.dumps(summary, indent=2))
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
