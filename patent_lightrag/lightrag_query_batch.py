from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from patent_lightrag.common import (
    LIGHTRAG_ROOT,
    ROOT,
    lightrag_addon_params,
    load_dotenv_file,
    prompt_runtime_config,
)
from patent_lightrag.index_patents import missing_runtime_dependencies


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed query set against a LightRAG storage.")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--query-file", default=str(DEFAULT_EXPERIMENT_DIR / "queries" / "eval_queries_15.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_EXPERIMENT_DIR / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--modes", nargs="+", default=["naive", "local", "global", "hybrid"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-top-k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    if not queries:
        raise RuntimeError(f"No queries found in {path}")
    return queries


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = str(row.get("query_id", ""))
            mode = str(row.get("mode", ""))
            if query_id and mode and row.get("status") == "success":
                done.add((query_id, mode))
    return done


async def main_async() -> None:
    args = parse_args()
    load_dotenv_file(Path(args.env))
    missing = missing_runtime_dependencies()
    if missing:
        raise RuntimeError("Missing Python packages for LightRAG querying: " + ", ".join(missing))

    sys.path.insert(0, str(LIGHTRAG_ROOT))
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    queries = load_queries(Path(args.query_file))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(output_path) if args.resume else set()

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
    embedding_func = EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
        send_dimensions=os.getenv("EMBEDDING_SEND_DIM", "false").lower() in {"1", "true", "yes", "on"},
        model_name=embedding_model,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", ""),
        ),
    )

    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        addon_params=lightrag_addon_params(),
    )
    await rag.initialize_storages()

    runtime = {
        "system": "LightRAG",
        "working_dir": args.working_dir,
        "llm_model": llm_model,
        "llm_host": llm_host,
        "reasoning_effort": reasoning_effort,
        "embedding_model": embedding_model,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        **prompt_runtime_config(),
    }

    total = len(queries) * len(args.modes)
    completed = 0
    try:
        with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:
            for query in queries:
                query_id = str(query.get("query_id") or query.get("id") or "")
                question = str(query.get("question") or query.get("query") or "")
                if not query_id or not question:
                    raise RuntimeError(f"Invalid query row: {query}")
                for mode in args.modes:
                    completed += 1
                    if (query_id, mode) in done:
                        print(json.dumps({"skip": query_id, "mode": mode, "completed": completed, "total": total}, ensure_ascii=False), flush=True)
                        continue
                    started = time.time()
                    param = QueryParam(
                        mode=mode,
                        top_k=args.top_k,
                        chunk_top_k=args.chunk_top_k,
                        enable_rerank=False,
                    )
                    try:
                        data_result = await rag.aquery_data(question, param=param)
                        answer = await rag.aquery(question, param=param)
                        row = {
                            "status": "success",
                            "query_id": query_id,
                            "category": query.get("category"),
                            "type": query.get("type"),
                            "question": question,
                            "mode": mode,
                            "data": data_result,
                            "answer": answer,
                            "elapsed_seconds": round(time.time() - started, 3),
                            "runtime": runtime,
                        }
                    except Exception as exc:
                        row = {
                            "status": "failure",
                            "query_id": query_id,
                            "category": query.get("category"),
                            "type": query.get("type"),
                            "question": question,
                            "mode": mode,
                            "error": repr(exc),
                            "elapsed_seconds": round(time.time() - started, 3),
                            "runtime": runtime,
                        }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "mode": mode,
                                "status": row["status"],
                                "elapsed_seconds": row["elapsed_seconds"],
                                "completed": completed,
                                "total": total,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main_async())
