from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, load_dotenv_file, write_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_EXTERNAL_DIR = ROOT / "external" / "graphrag"
DEFAULT_TAG = "v3.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Microsoft GraphRAG smoke/update/query stages for Patent-100.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR))
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument(
        "--stage",
        choices=[
            "setup",
            "smoke",
            "update",
            "fresh-full",
            "query-smoke",
            "query-full",
            "query-full-repair",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--query-file", default=str(DEFAULT_EXPERIMENT_DIR / "queries" / "eval_queries_15.jsonl"))
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"cmd": cmd, "cwd": str(cwd) if cwd else None}, ensure_ascii=False), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text_inputs(root: Path, docs_path: Path, clear: bool = False) -> None:
    docs = load_jsonl(docs_path)
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for file in input_dir.glob("*.txt"):
            file.unlink()
    for doc in docs:
        doc_id = str(doc.get("id", "unknown")).replace("/", "_")
        (input_dir / f"{doc_id}.txt").write_text(str(doc.get("text", "")), encoding="utf-8")


def graph_env(env_path: Path, external_dir: Path | None = None) -> dict[str, str]:
    loaded = load_dotenv_file(env_path)
    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = loaded.get("LLM_BINDING_API_KEY", "codex-proxy") or "codex-proxy"
    env["GRAPHRAG_EMBEDDING_API_KEY"] = loaded.get("EMBEDDING_BINDING_API_KEY", "")
    env["OPENAI_API_KEY"] = env["GRAPHRAG_API_KEY"]
    if external_dir is not None:
        package_path = external_dir / "packages" / "graphrag"
        if package_path.exists():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(package_path) if not existing else f"{package_path}{os.pathsep}{existing}"
    return env


def ensure_repo(external_dir: Path, tag: str) -> dict[str, str]:
    external_dir.parent.mkdir(parents=True, exist_ok=True)
    if external_dir.exists() and not (external_dir / ".git").exists():
        shutil.rmtree(external_dir)
    if not external_dir.exists():
        result = run(["git", "clone", "--depth", "1", "--branch", tag, "https://github.com/microsoft/graphrag.git", str(external_dir)])
        if result.returncode:
            result = run(["git", "clone", "https://github.com/microsoft/graphrag.git", str(external_dir)])
        if result.returncode:
            raise RuntimeError(result.stdout)
    result = run(["git", "checkout", tag], cwd=external_dir)
    if result.returncode:
        raise RuntimeError(result.stdout)
    commit = run(["git", "rev-parse", "HEAD"], cwd=external_dir)
    return {"repo": str(external_dir), "tag": tag, "commit": commit.stdout.strip()}


def choose_graphrag_python() -> str:
    candidates = [
        os.getenv("GRAPHRAG_PYTHON", ""),
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
        "python3.12",
        "python3.11",
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        result = run([candidate, "--version"])
        if result.returncode == 0:
            version = result.stdout.strip()
            if "Python 3.10" in version or "Python 3.11" in version or "Python 3.12" in version:
                return candidate
    raise RuntimeError("GraphRAG requires Python 3.10-3.12; set GRAPHRAG_PYTHON to a compatible interpreter.")


def ensure_venv(external_dir: Path) -> Path:
    venv = external_dir / ".venv"
    python = venv / "bin" / "python"
    if not python.exists():
        result = run([choose_graphrag_python(), "-m", "venv", str(venv)])
        if result.returncode:
            raise RuntimeError(result.stdout)
    result = run([str(python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=external_dir)
    if result.returncode:
        raise RuntimeError(result.stdout)
    package_dir = external_dir / "packages" / "graphrag"
    install_target = str(package_dir) if package_dir.exists() else "."
    result = run([str(python), "-m", "pip", "install", "-e", install_target], cwd=external_dir)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return python


def graphrag_bin(external_dir: Path) -> Path:
    return external_dir / ".venv" / "bin" / "graphrag"


def patch_settings(root: Path) -> None:
    settings = root / "settings.yaml"
    if not settings.exists():
        return
    text = settings.read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        text = re.sub(
            r"(completion_models:\n\s+default_completion_model:\n(?:\s+.*\n)*?\s+api_key:\s+\$\{GRAPHRAG_API_KEY\}[^\n]*\n)",
            r"\1    api_base: http://localhost:11435/v1\n",
            text,
            count=1,
        )
        text = re.sub(
            r"(embedding_models:\n\s+default_embedding_model:\n(?:\s+.*\n)*?\s+api_key:)\s+\$\{GRAPHRAG_API_KEY\}",
            r"\1 ${GRAPHRAG_EMBEDDING_API_KEY}",
            text,
            count=1,
        )
        text = re.sub(
            r"(embedding_models:\n\s+default_embedding_model:\n(?:\s+.*\n)*?\s+api_key:\s+\$\{GRAPHRAG_EMBEDDING_API_KEY\}[^\n]*\n)",
            r"\1    api_base: https://openrouter.ai/api/v1\n",
            text,
            count=1,
        )
        text = re.sub(
            r"entity_types:\s*\[[^\]]*\]",
            "entity_types: [TechComponent,Architecture,Operation,Method,Material,PerformanceMetric,ApplicationDomain]",
            text,
        )
        settings.write_text(text, encoding="utf-8")
        return
    data = yaml.safe_load(text) or {}

    completion_models = data.setdefault("completion_models", {})
    for _, model_cfg in completion_models.items():
        if not isinstance(model_cfg, dict):
            continue
        model_cfg["model_provider"] = "openai"
        model_cfg["model"] = "gpt-5.5"
        model_cfg["api_base"] = "http://localhost:11435/v1"
        model_cfg["api_key"] = "${GRAPHRAG_API_KEY}"
        model_cfg["auth_method"] = "api_key"

    embedding_models = data.setdefault("embedding_models", {})
    for _, model_cfg in embedding_models.items():
        if not isinstance(model_cfg, dict):
            continue
        model_cfg["model_provider"] = "openai"
        model_cfg["model"] = "openai/text-embedding-3-large"
        model_cfg["api_base"] = "https://openrouter.ai/api/v1"
        model_cfg["api_key"] = "${GRAPHRAG_EMBEDDING_API_KEY}"
        model_cfg["auth_method"] = "api_key"

    chunking = data.setdefault("chunking", {})
    if isinstance(chunking, dict):
        chunking["size"] = 1200
        chunking["overlap"] = 100

    extract_graph = data.setdefault("extract_graph", {})
    if isinstance(extract_graph, dict):
        extract_graph["entity_types"] = [
            "TechComponent",
            "Architecture",
            "Operation",
            "Method",
            "Material",
            "PerformanceMetric",
            "ApplicationDomain",
        ]
    settings.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def init_workspace(root: Path, external_dir: Path, env: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "settings.yaml").exists():
        result = run(
            [
                str(graphrag_bin(external_dir)),
                "init",
                "--root",
                str(root),
                "--model",
                "gpt-5.5",
                "--embedding",
                "openai/text-embedding-3-large",
                "--force",
            ],
            env=env,
        )
        if result.returncode:
            raise RuntimeError(result.stdout)
    patch_settings(root)


def index_workspace(root: Path, external_dir: Path, env: dict[str, str], update: bool = False) -> dict[str, Any]:
    init_workspace(root, external_dir, env)
    command = "update" if update else "index"
    method = "standard-update" if update else "standard"
    if not update:
        dry = run([str(graphrag_bin(external_dir)), command, "--root", str(root), "--method", method, "--dry-run"], env=env)
        if dry.returncode:
            raise RuntimeError(dry.stdout)
    started = time.time()
    result = run([str(graphrag_bin(external_dir)), command, "--root", str(root), "--method", method, "--verbose"], env=env)
    return {
        "root": str(root),
        "command": command,
        "method": method,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "output_excerpt": result.stdout[-10000:],
    }


def query_workspace(root: Path, external_dir: Path, env: dict[str, str], query_file: Path, output: Path) -> None:
    methods = ["basic", "local", "global"]
    queries = load_jsonl(query_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for query in queries:
            question = str(query.get("question", ""))
            for method in methods:
                started = time.time()
                result = run(
                    [
                        str(graphrag_bin(external_dir)),
                        "query",
                        "--root",
                        str(root),
                        "--method",
                        method,
                        question,
                    ],
                    env=env,
                )
                row = {
                    "status": "success" if result.returncode == 0 else "failure",
                    "query_id": query.get("query_id"),
                    "category": query.get("category"),
                    "type": query.get("type"),
                    "question": question,
                    "method": method,
                    "answer": result.stdout.strip(),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "returncode": result.returncode,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps({"query_id": row["query_id"], "method": method, "status": row["status"]}, ensure_ascii=False), flush=True)


def load_existing_query_rows(output: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not output.exists():
        return rows
    with output.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(json.dumps({"warning": "skip_parse_error", "line": line_no}, ensure_ascii=False), flush=True)
                continue
            query_id = str(row.get("query_id", ""))
            method = str(row.get("method", ""))
            if query_id and method:
                rows[(query_id, method)] = row
    return rows


def write_query_rows(output: Path, queries: list[dict[str, Any]], rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    methods = ["basic", "local", "global"]
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for query in queries:
            query_id = str(query.get("query_id", ""))
            for method in methods:
                row = rows.get((query_id, method))
                if row is not None:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)


def query_workspace_repair(root: Path, external_dir: Path, env: dict[str, str], query_file: Path, output: Path) -> None:
    methods = ["basic", "local", "global"]
    queries = load_jsonl(query_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_query_rows(output)

    if output.exists():
        backup = output.with_name(f"{output.stem}.backup_{time.strftime('%Y%m%d_%H%M%S')}{output.suffix}")
        shutil.copy2(output, backup)
        print(json.dumps({"backup": str(backup)}, ensure_ascii=False), flush=True)

    expected: list[tuple[dict[str, Any], str]] = [(query, method) for query in queries for method in methods]
    failures = [
        (query, method)
        for query, method in expected
        if existing.get((str(query.get("query_id", "")), method), {}).get("status") not in {None, "success"}
    ]
    missing = [
        (query, method)
        for query, method in expected
        if (str(query.get("query_id", "")), method) not in existing
    ]
    repair_targets = failures + [target for target in missing if target not in failures]

    print(
        json.dumps(
            {
                "existing": len(existing),
                "failures_to_retry": len(failures),
                "missing_to_run": len(missing),
                "repair_targets": [
                    {"query_id": query.get("query_id"), "method": method}
                    for query, method in repair_targets
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for query, method in repair_targets:
        question = str(query.get("question", ""))
        started = time.time()
        result = run(
            [
                str(graphrag_bin(external_dir)),
                "query",
                "--root",
                str(root),
                "--method",
                method,
                question,
            ],
            env=env,
        )
        row = {
            "status": "success" if result.returncode == 0 else "failure",
            "query_id": query.get("query_id"),
            "category": query.get("category"),
            "type": query.get("type"),
            "question": question,
            "method": method,
            "answer": result.stdout.strip(),
            "elapsed_seconds": round(time.time() - started, 3),
            "returncode": result.returncode,
        }
        existing[(str(row["query_id"]), method)] = row
        write_query_rows(output, queries, existing)
        print(
            json.dumps(
                {
                    "query_id": row["query_id"],
                    "method": method,
                    "status": row["status"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "remaining": len(repair_targets) - repair_targets.index((query, method)) - 1,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def copy_smoke_to_full(experiment_dir: Path) -> Path:
    smoke_root = experiment_dir / "graphrag_smoke_20"
    full_root = experiment_dir / "graphrag_full_100_update"
    if full_root.exists():
        shutil.rmtree(full_root)
    shutil.copytree(smoke_root, full_root)
    return full_root


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    external_dir = Path(args.external_dir)
    env = graph_env(Path(args.env), external_dir)
    manifest_path = experiment_dir / "graphrag_manifest.json"
    manifest = {
        "experiment_dir": str(experiment_dir),
        "external_dir": str(external_dir),
        "tag": args.tag,
        "stages": [],
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def save(stage: str, payload: dict[str, Any]) -> None:
        manifest.setdefault("stages", []).append({"stage": stage, **payload})
        write_json(manifest_path, manifest)
        print(json.dumps({"stage": stage, **payload}, ensure_ascii=False, indent=2), flush=True)

    if args.stage in {"setup", "all"}:
        repo = ensure_repo(external_dir, args.tag)
        ensure_venv(external_dir)
        save("setup", repo)

    if args.stage in {"smoke", "all"}:
        ensure_repo(external_dir, args.tag)
        ensure_venv(external_dir)
        smoke_root = experiment_dir / "graphrag_smoke_20"
        write_text_inputs(smoke_root, experiment_dir / "dataset" / "graphrag_smoke_20.jsonl", clear=True)
        save("smoke", index_workspace(smoke_root, external_dir, env, update=False))

    if args.stage in {"update", "all"}:
        ensure_repo(external_dir, args.tag)
        ensure_venv(external_dir)
        full_root = copy_smoke_to_full(experiment_dir)
        write_text_inputs(full_root, experiment_dir / "dataset" / "graphrag_remaining_80.jsonl", clear=False)
        save("update", index_workspace(full_root, external_dir, env, update=True))

    if args.stage == "fresh-full":
        ensure_repo(external_dir, args.tag)
        ensure_venv(external_dir)
        fresh_root = experiment_dir / "graphrag_full_100_fresh"
        write_text_inputs(fresh_root, experiment_dir / "dataset" / "patents_100.jsonl", clear=True)
        save("fresh-full", index_workspace(fresh_root, external_dir, env, update=False))

    if args.stage in {"query-smoke", "all"}:
        query_workspace(
            experiment_dir / "graphrag_smoke_20",
            external_dir,
            env,
            Path(args.query_file),
            experiment_dir / "graphrag_smoke_20" / "query_results_smoke_methods.jsonl",
        )

    if args.stage in {"query-full", "all"}:
        root = experiment_dir / "graphrag_full_100_fresh"
        if not root.exists():
            root = experiment_dir / "graphrag_full_100_update"
        query_workspace(
            root,
            external_dir,
            env,
            Path(args.query_file),
            root / "query_results_15_methods.jsonl",
        )

    if args.stage == "query-full-repair":
        root = experiment_dir / "graphrag_full_100_fresh"
        if not root.exists():
            root = experiment_dir / "graphrag_full_100_update"
        query_workspace_repair(
            root,
            external_dir,
            env,
            Path(args.query_file),
            root / "query_results_15_methods.jsonl",
        )


if __name__ == "__main__":
    main()
