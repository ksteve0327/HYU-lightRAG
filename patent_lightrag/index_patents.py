from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from functools import partial
from pathlib import Path

from patent_lightrag.common import (
    LIGHTRAG_ROOT,
    PATHS,
    ensure_dirs,
    lightrag_addon_params,
    load_dotenv_file,
    prompt_runtime_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index structured patent documents with LightRAG.")
    parser.add_argument("--docs", default=str(PATHS.docs_jsonl))
    parser.add_argument("--working-dir", default=str(Path("data/lightrag_runs/patent_rag_storage")))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--limit", type=int, default=0, help="Optional document limit for dry runs.")
    parser.add_argument("--stats", default=str(PATHS.index_stats))
    return parser.parse_args()


def load_docs(path: Path, limit: int = 0) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def missing_runtime_dependencies() -> list[str]:
    import importlib.util

    missing = []
    for module in ["openai", "numpy"]:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def storage_doc_status(working_dir: Path) -> dict[str, dict[str, object]]:
    status_path = working_dir / "kv_store_doc_status.json"
    if not status_path.exists():
        return {}
    with status_path.open(encoding="utf-8") as f:
        return json.load(f)


def storage_status_counts(
    working_dir: Path,
    primary_doc_ids: set[str] | None = None,
) -> dict[str, int]:
    status_data = storage_doc_status(working_dir)
    records = []
    duplicate_records = 0
    for doc_id, record in status_data.items():
        metadata = record.get("metadata") if isinstance(record, dict) else {}
        if isinstance(metadata, dict) and metadata.get("is_duplicate"):
            duplicate_records += 1
            continue
        if primary_doc_ids is not None and doc_id not in primary_doc_ids:
            continue
        records.append(record)
    counts = dict(Counter(record.get("status", "unknown") for record in records))
    if duplicate_records:
        counts["duplicate_status_records"] = duplicate_records
    return counts


def graphml_counts(working_dir: Path) -> dict[str, int]:
    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        return {"graph_nodes": 0, "graph_edges": 0}
    nodes = 0
    edges = 0
    for _, elem in ET.iterparse(graph_path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "node":
            nodes += 1
        elif tag == "edge":
            edges += 1
        elem.clear()
    return {"graph_nodes": nodes, "graph_edges": edges}


async def main_async() -> None:
    args = parse_args()
    ensure_dirs()
    load_dotenv_file(Path(args.env))
    missing = missing_runtime_dependencies()
    if missing:
        raise RuntimeError(
            "Missing Python packages for LightRAG indexing: "
            + ", ".join(missing)
            + ". Run this with LightRAG's configured virtual environment."
        )

    sys.path.insert(0, str(LIGHTRAG_ROOT))
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    docs = load_docs(Path(args.docs), args.limit)
    if not docs:
        raise RuntimeError(f"No documents found in {args.docs}")

    llm_model = os.getenv("LLM_MODEL", "gpt-5.5")
    llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11435/v1")
    llm_key = os.getenv("LLM_BINDING_API_KEY", "codex-proxy")
    llm_timeout = int(os.getenv("LLM_TIMEOUT", "300"))
    reasoning_effort = os.getenv("OPENAI_LLM_REASONING_EFFORT", "xhigh")

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=llm_host,
            api_key=llm_key,
            timeout=llm_timeout,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "https://openrouter.ai/api/v1")
    embedding_key = os.getenv("EMBEDDING_BINDING_API_KEY", "")
    if not embedding_key or "PASTE_OPENROUTER" in embedding_key:
        raise RuntimeError("Set EMBEDDING_BINDING_API_KEY in LightRAG-main/.env before indexing.")

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    embedding_token_limit = int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192"))
    embedding_send_dim = os.getenv("EMBEDDING_SEND_DIM", "false").lower() in {"1", "true", "yes", "on"}

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=embedding_token_limit,
        send_dimensions=embedding_send_dim,
        model_name=embedding_model,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            base_url=embedding_host,
            api_key=embedding_key,
        ),
    )

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = [str(doc["id"]) for doc in docs]
    primary_doc_ids = set(doc_ids)
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
        entity_extract_max_gleaning=int(os.getenv("MAX_GLEANING", "1")),
        addon_params=lightrag_addon_params(),
    )
    started = time.time()
    await rag.initialize_storages()
    try:
        existing_status = storage_doc_status(working_dir)
        if primary_doc_ids and primary_doc_ids.issubset(existing_status):
            print(
                json.dumps(
                    {
                        "resume": True,
                        "message": "All primary document IDs already exist; processing queued/failed statuses without re-enqueueing.",
                        "primary_documents": len(primary_doc_ids),
                    },
                    ensure_ascii=False,
                )
            )
            await rag.apipeline_process_enqueue_documents()
        else:
            await rag.ainsert(
                [str(doc["text"]) for doc in docs],
                ids=doc_ids,
                file_paths=[str(doc["file_path"]) for doc in docs],
            )
        status_counts = storage_status_counts(working_dir, primary_doc_ids)
        graph_counts = graphml_counts(working_dir)
        stats = {
            "document_count": len(docs),
            "processed_documents": status_counts.get("processed", 0),
            "failed_documents": status_counts.get("failed", 0),
            "completed": status_counts.get("processed", 0) == len(docs),
            "status_counts": status_counts,
            "working_dir": str(working_dir.resolve()),
            "elapsed_seconds": round(time.time() - started, 2),
            "llm_model": llm_model,
            "llm_host": llm_host,
            "reasoning_effort": reasoning_effort,
            "embedding_model": embedding_model,
            "embedding_host": embedding_host,
            "embedding_dim": embedding_dim,
            **prompt_runtime_config(),
            **graph_counts,
            "limit": args.limit,
        }
        write_json(Path(args.stats), stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main_async())
