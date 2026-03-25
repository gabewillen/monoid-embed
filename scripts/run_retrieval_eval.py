#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monoid.embed import MonoidEmbed, MonoidEmbedConfig


def _text_to_bytes(text, max_bytes):
    raw = (text or "").encode("utf-8")
    length = min(len(raw), max_bytes)
    byte_indices = list(raw[:max_bytes])
    if len(byte_indices) < max_bytes:
        byte_indices.extend([0] * (max_bytes - len(byte_indices)))
    return byte_indices, length


def _build_retrieval_eval(args, logger):
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
    query_ids = list(qrels_by_query.keys())
    rng.shuffle(query_ids)
    if args.positives_per_query is None or args.positives_per_query <= 0:
        max_queries_by_docs = args.queries
    else:
        max_queries_by_docs = max(1, args.docs // max(1, args.positives_per_query))
    query_ids = query_ids[: min(args.queries, max_queries_by_docs, len(query_ids))]

    query_texts = {}
    try:
        queries_ds = load_dataset(args.dataset, "queries", split="queries", streaming=True)
        query_set = set(query_ids)
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in query_set:
                query_texts[qid] = row["text"]
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
    use_full = args.full_corpus or args.dataset == "mteb/scifact"
    try:
        if use_full:
            corpus_ds = load_dataset(args.dataset, "corpus", split="corpus")
            corpus_rows = list(corpus_ds)
            corpus_ids = [str(row["_id"]) for row in corpus_rows]
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
                title = row.get("title") or ""
                text = row.get("text") or ""
                doc_texts[doc_id] = (title + "\n" + text).strip()
        else:
            corpus_ds = load_dataset(args.dataset, "corpus", split="corpus", streaming=True)
            scanned = 0
            for row in corpus_ds:
                doc_id = str(row["_id"])
                include = doc_id in needed_pos or len(doc_texts) < args.docs
                if include and doc_id not in doc_texts:
                    title = row.get("title") or ""
                    text = row.get("text") or ""
                    doc_texts[doc_id] = (title + "\n" + text).strip()
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
    for text in doc_texts.values():
        byte_indices, length = _text_to_bytes(text, args.max_bytes)
        doc_bytes.append(byte_indices)
        doc_lengths.append(length)

    query_bytes = []
    query_lengths = []
    for text in query_text_list:
        byte_indices, length = _text_to_bytes(text, args.max_bytes)
        query_bytes.append(byte_indices)
        query_lengths.append(length)

    logger.info(
        "Retrieval eval ready: %d queries, %d docs, k=%s",
        len(query_text_list),
        len(doc_bytes),
        args.k,
    )

    return {
        "query_bytes": query_bytes,
        "query_lengths": query_lengths,
        "doc_bytes": doc_bytes,
        "doc_lengths": doc_lengths,
        "relevances": relevances,
        "k_values": args.k,
    }


def _load_smoke_set(path, logger):
    meta = {}
    queries = []
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record_type = payload.get("type")
                if record_type == "meta":
                    meta = payload
                elif record_type == "query":
                    queries.append((str(payload.get("id")), payload.get("text") or ""))
                elif record_type == "doc":
                    docs.append((str(payload.get("id")), payload.get("text") or ""))
    except Exception as exc:
        logger.error("Failed to load smoke set: %s", exc)
        return None

    if not queries or not docs:
        logger.error("Smoke set missing queries or docs.")
        return None

    return {"meta": meta, "queries": queries, "docs": docs}


def _build_retrieval_eval_from_smoke(smoke, args, logger):
    meta = smoke.get("meta", {})
    dataset_name = meta.get("dataset_name")
    if meta.get("dataset") and args.dataset != meta.get("dataset"):
        logger.warning("Overriding dataset with smoke metadata: %s", meta.get("dataset"))
        args.dataset = meta.get("dataset")
    if meta.get("split") and args.split != meta.get("split"):
        logger.warning("Overriding split with smoke metadata: %s", meta.get("split"))
        args.split = meta.get("split")

    try:
        if dataset_name:
            qrels = load_dataset(args.dataset, dataset_name, split=args.split)
        else:
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

    query_ids = [qid for qid, _ in smoke["queries"]]
    query_texts = {qid: text for qid, text in smoke["queries"]}
    doc_ids = [doc_id for doc_id, _ in smoke["docs"]]
    doc_texts = {doc_id: text for doc_id, text in smoke["docs"]}

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    query_text_list = []
    relevances = []
    dropped = 0
    for qid in query_ids:
        rel_ids = qrels_by_query.get(qid, set())
        rel_in_corpus = {doc_id_to_idx[cid] for cid in rel_ids if cid in doc_id_to_idx}
        if not rel_in_corpus:
            dropped += 1
            continue
        query_text_list.append(query_texts[qid])
        relevances.append(rel_in_corpus)

    if not query_text_list:
        logger.error("No queries with positives in smoke set.")
        return None

    logger.info("SMOKE_SET=1")
    logger.info("Smoke queries: %d (dropped: %d)", len(query_text_list), dropped)
    logger.info("Smoke docs: %d", len(doc_ids))

    doc_bytes = []
    doc_lengths = []
    for doc_id in doc_ids:
        byte_indices, length = _text_to_bytes(doc_texts[doc_id], args.max_bytes)
        doc_bytes.append(byte_indices)
        doc_lengths.append(length)

    query_bytes = []
    query_lengths = []
    for text in query_text_list:
        byte_indices, length = _text_to_bytes(text, args.max_bytes)
        query_bytes.append(byte_indices)
        query_lengths.append(length)

    logger.info(
        "Retrieval eval ready: %d queries, %d docs, k=%s",
        len(query_text_list),
        len(doc_bytes),
        args.k,
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


def _run_retrieval_eval(model, device, retrieval, batch_size, hit_rate: bool):
    model_was_training = model.training
    model.eval()
    try:
        doc_emb = _embed_batches(model, device, retrieval["doc_bytes"], retrieval["doc_lengths"], batch_size).float()
        query_emb = _embed_batches(model, device, retrieval["query_bytes"], retrieval["query_lengths"], batch_size).float()
        sim = query_emb @ doc_emb.t()
        metrics = {}
        for k in retrieval["k_values"]:
            k_eff = min(k, sim.size(1))
            topk = sim.topk(k_eff, dim=1).indices
            if hit_rate:
                hits = 0
                for idx, rel_set in enumerate(retrieval["relevances"]):
                    if any(doc_idx in rel_set for doc_idx in topk[idx].tolist()):
                        hits += 1
                metrics[k] = hits / len(retrieval["relevances"])
            else:
                recalls = []
                for idx, rel_set in enumerate(retrieval["relevances"]):
                    if not rel_set:
                        continue
                    retrieved = sum(1 for doc_idx in topk[idx].tolist() if doc_idx in rel_set)
                    recalls.append(retrieved / len(rel_set))
                metrics[k] = float(sum(recalls)) / max(1, len(recalls))
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


def main():
    parser = argparse.ArgumentParser(description="Run lightweight retrieval eval for MonoidEmbed.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mteb/scifact")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_corpus", type=int, default=200000)
    parser.add_argument("--full_corpus", action="store_true", help="Use full non-streaming corpus when available")
    parser.add_argument("--max_bytes", type=int, default=1024)
    parser.add_argument(
        "--positives_per_query",
        type=int,
        default=None,
        help="Limit positives per query (default: all).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--pool", type=str, default="mean")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--hit_rate", action="store_true", help="Report hit-rate (legacy behavior).")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for eval.")
    parser.add_argument("--smoke_set_jsonl", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("retrieval-monoid")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.smoke_set_jsonl:
        smoke = _load_smoke_set(args.smoke_set_jsonl, logger)
        if smoke is None:
            raise SystemExit(1)
        retrieval = _build_retrieval_eval_from_smoke(smoke, args, logger)
    else:
        retrieval = _build_retrieval_eval(args, logger)
    if retrieval is None:
        raise SystemExit(1)

    config = MonoidEmbedConfig()
    config.use_quantized = args.quantized
    config.pool_strategy = args.pool
    config.normalize_output = args.normalize

    model = MonoidEmbed(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state, compiled_state = _unwrap_checkpoint_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    if args.compile or compiled_state:
        if device.type != "cuda":
            logger.warning("Compile requested but device is %s; skipping torch.compile.", device)
        else:
            logger.info("Compiling model for retrieval eval...")
            try:
                import torch._inductor.config as inductor_config

                inductor_config.triton.cudagraphs = False
                logger.info("torch.compile: disabled cudagraphs for stability.")
            except Exception as exc:
                logger.warning("torch.compile: failed to disable cudagraphs (%s).", exc)
            model = torch.compile(model, mode="max-autotune")

    metrics = _run_retrieval_eval(model, device, retrieval, args.batch_size, args.hit_rate)
    print("Retrieval metrics:")
    for k in sorted(metrics.keys()):
        print(f"recall@{k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
