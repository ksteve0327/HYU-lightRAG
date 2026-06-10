from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

from patent_lightrag.common import LIGHTRAG_ROOT, load_dotenv_file, redact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime readiness for the patent LightRAG workflow.")
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--check-openrouter", action="store_true", help="Only use after entering a real OpenRouter API key.")
    return parser.parse_args()


def check_module(name: str) -> dict[str, object]:
    return {"module": name, "available": importlib.util.find_spec(name) is not None}


def get_json(url: str, timeout: int = 5) -> dict[str, object]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ok": True, "data": json.loads(response.read().decode("utf-8"))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_openrouter_embedding(env: dict[str, str]) -> dict[str, object]:
    api_key = env.get("EMBEDDING_BINDING_API_KEY", "")
    if not api_key or "PASTE_OPENROUTER" in api_key:
        return {"ok": False, "error": "EMBEDDING_BINDING_API_KEY is still a placeholder."}

    host = env.get("EMBEDDING_BINDING_HOST", "https://openrouter.ai/api/v1").rstrip("/")
    model = env.get("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    payload = json.dumps({"model": model, "input": "preflight smoke test", "encoding_format": "float"}).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        embedding = body.get("data", [{}])[0].get("embedding", [])
        return {"ok": True, "model": model, "dimension": len(embedding)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")[:1000]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    args = parse_args()
    env = load_dotenv_file(Path(args.env))
    checks = {
        "env_path": str(Path(args.env).resolve()),
        "codex_proxy": get_json("http://localhost:11435/health", timeout=3),
        "python_modules": [check_module(name) for name in ["openai", "numpy", "tiktoken"]],
        "llm": {
            "host": env.get("LLM_BINDING_HOST"),
            "model": env.get("LLM_MODEL"),
            "reasoning": env.get("OPENAI_LLM_REASONING_EFFORT"),
        },
        "embedding": {
            "host": env.get("EMBEDDING_BINDING_HOST"),
            "model": env.get("EMBEDDING_MODEL"),
            "dim": env.get("EMBEDDING_DIM"),
            "api_key": redact(env.get("EMBEDDING_BINDING_API_KEY", "")),
        },
    }
    if args.check_openrouter:
        checks["openrouter_embedding_smoke"] = check_openrouter_embedding(env)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
