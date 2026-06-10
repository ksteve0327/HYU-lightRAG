from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from patent_lightrag.common import ROOT, estimate_tokens, write_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
PATENT_RE = re.compile(r"\b(?:\d{4}-\d{6,}|\d{2}-\d{5,})\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for Patent-100 RAG query results.")
    parser.add_argument("--lightrag-results", default=str(DEFAULT_EXPERIMENT_DIR / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"))
    parser.add_argument("--graphrag-results", default=str(DEFAULT_EXPERIMENT_DIR / "graphrag_full_100_update" / "query_results_15_methods.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_EXPERIMENT_DIR / "evaluation" / "auto_metrics.json"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_patents_from_text(text: str) -> set[str]:
    return set(PATENT_RE.findall(text or ""))


def source_patents_from_data(data: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    body = data.get("data") if isinstance(data, dict) else {}
    if not isinstance(body, dict):
        return found
    for section in ["references", "chunks", "entities", "relationships"]:
        values = body.get(section, [])
        if not isinstance(values, list):
            continue
        for row in values:
            if not isinstance(row, dict):
                continue
            for key in ["file_path", "source_id", "content", "reference_id"]:
                value = row.get(key)
                if isinstance(value, str):
                    found.update(source_patents_from_text(value))
    return found


def light_or_graph_system(row: dict[str, Any]) -> str:
    if "method" in row:
        return f"graphrag_{row.get('method')}"
    return f"lightrag_{row.get('mode')}"


def record_metrics(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or row.get("response") or "")
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    body = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else {}
    entities = body.get("entities", []) if isinstance(body, dict) else []
    relationships = body.get("relationships", []) if isinstance(body, dict) else []
    chunks = body.get("chunks", []) if isinstance(body, dict) else []
    references = body.get("references", []) if isinstance(body, dict) else []
    source_patents = source_patents_from_data(data if isinstance(data, dict) else {})
    source_patents.update(source_patents_from_text(answer))
    return {
        "system": light_or_graph_system(row),
        "query_id": row.get("query_id"),
        "category": row.get("category"),
        "type": row.get("type"),
        "status": row.get("status", "success"),
        "answer_chars": len(answer),
        "answer_token_estimate": estimate_tokens(answer) if answer else 0,
        "empty_answer": not bool(answer.strip()),
        "retrieved_entity_count": len(entities) if isinstance(entities, list) else 0,
        "retrieved_relation_count": len(relationships) if isinstance(relationships, list) else 0,
        "retrieved_chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "reference_count": len(references) if isinstance(references, list) else 0,
        "unique_source_patent_count": len(source_patents),
        "source_patents": sorted(source_patents),
        "patent_citation_count": len(PATENT_RE.findall(answer)),
        "elapsed_seconds": row.get("elapsed_seconds"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_system_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        system = str(record.get("system", "unknown"))
        category = str(record.get("category", "unknown"))
        by_system[system].append(record)
        by_system_category[(system, category)].append(record)

    def avg(rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    system_rows = []
    for system, rows in sorted(by_system.items()):
        system_rows.append(
            {
                "system": system,
                "queries": len(rows),
                "success": sum(1 for row in rows if row.get("status") == "success"),
                "empty_answers": sum(1 for row in rows if row.get("empty_answer")),
                "avg_answer_chars": avg(rows, "answer_chars"),
                "avg_retrieved_entities": avg(rows, "retrieved_entity_count"),
                "avg_retrieved_relations": avg(rows, "retrieved_relation_count"),
                "avg_retrieved_chunks": avg(rows, "retrieved_chunk_count"),
                "avg_unique_source_patents": avg(rows, "unique_source_patent_count"),
                "avg_latency_seconds": avg(rows, "elapsed_seconds"),
            }
        )

    system_category_rows = []
    for (system, category), rows in sorted(by_system_category.items()):
        system_category_rows.append(
            {
                "system": system,
                "category": category,
                "queries": len(rows),
                "avg_answer_chars": avg(rows, "answer_chars"),
                "avg_retrieved_entities": avg(rows, "retrieved_entity_count"),
                "avg_retrieved_relations": avg(rows, "retrieved_relation_count"),
                "avg_retrieved_chunks": avg(rows, "retrieved_chunk_count"),
                "avg_unique_source_patents": avg(rows, "unique_source_patent_count"),
                "avg_latency_seconds": avg(rows, "elapsed_seconds"),
            }
        )

    return {
        "records": records,
        "system_summary": system_rows,
        "system_category_summary": system_category_rows,
        "counts": dict(Counter(record.get("system") for record in records)),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.lightrag_results)) + read_jsonl(Path(args.graphrag_results))
    records = [record_metrics(row) for row in rows]
    payload = summarize(records)
    write_json(Path(args.output), payload)
    print(json.dumps({"records": len(records), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
