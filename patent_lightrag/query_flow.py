from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
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
from patent_lightrag.index_patents import missing_runtime_dependencies


DEFAULT_QUERY = "AI 반도체 특허에서 신경망 연산 가속기와 메모리 아키텍처는 어떤 기술 관계를 보이는가?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local/global/hybrid LightRAG queries and save flow data.")
    parser.add_argument("--working-dir", default=str(Path("data/lightrag_runs/patent_rag_storage")))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--output", default=str(PATHS.query_results))
    parser.add_argument("--modes", nargs="+", default=["local", "global", "hybrid"])
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    ensure_dirs()
    load_dotenv_file(Path(args.env))
    missing = missing_runtime_dependencies()
    if missing:
        raise RuntimeError("Missing Python packages for LightRAG querying: " + ", ".join(missing))

    sys.path.insert(0, str(LIGHTRAG_ROOT))
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

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
    results = {
        "query": args.query,
        "modes": {},
        "runtime": {
            "llm_model": llm_model,
            "llm_host": llm_host,
            "reasoning_effort": reasoning_effort,
            "embedding_model": embedding_model,
            **prompt_runtime_config(),
        },
    }
    try:
        for mode in args.modes:
            param = QueryParam(mode=mode, top_k=10, chunk_top_k=10, enable_rerank=False)
            data_result = await rag.aquery_data(args.query, param=param)
            answer = await rag.aquery(args.query, param=param)
            results["modes"][mode] = {
                "data": data_result,
                "answer": answer,
            }
        write_json(Path(args.output), results)
        print(json.dumps({"query": args.query, "modes": list(results["modes"])}, ensure_ascii=False, indent=2))
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main_async())
