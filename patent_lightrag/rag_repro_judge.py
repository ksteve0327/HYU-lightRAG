from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, load_dotenv_file, write_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_PAIRS = [
    ("lightrag_hybrid", "lightrag_naive"),
    ("lightrag_hybrid", "graphrag_global"),
    ("lightrag_hybrid", "graphrag_local"),
    ("graphrag_global", "lightrag_naive"),
]
RUBRICS = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Technical correctness",
    "Hallucination risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini pairwise judge for Patent-100 RAG outputs.")
    parser.add_argument("--lightrag-results", default=str(DEFAULT_EXPERIMENT_DIR / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"))
    parser.add_argument("--graphrag-results", default=str(DEFAULT_EXPERIMENT_DIR / "graphrag_full_100_update" / "query_results_15_methods.jsonl"))
    parser.add_argument("--query-file", default=str(DEFAULT_EXPERIMENT_DIR / "queries" / "eval_queries_15.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_EXPERIMENT_DIR / "evaluation" / "judge_results.jsonl"))
    parser.add_argument("--summary", default=str(DEFAULT_EXPERIMENT_DIR / "evaluation" / "judge_summary.json"))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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


def label_for(row: dict[str, Any]) -> str:
    if row.get("method"):
        return f"graphrag_{row.get('method')}"
    return f"lightrag_{row.get('mode')}"


def build_answer_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        query_id = str(row.get("query_id", ""))
        answer = str(row.get("answer") or row.get("response") or "")
        if not query_id or not answer.strip():
            continue
        lookup[label_for(row)][query_id] = row
    return lookup


def load_done(path: Path) -> set[tuple[str, str]]:
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
            if row.get("status") == "success":
                done.add((str(row.get("pair_id", "")), str(row.get("query_id", ""))))
    return done


def clip(text: str, limit: int = 9000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return f"""
You are evaluating two retrieval-augmented answers to the same AI semiconductor patent question.
Use only the question and the two answers. Do not assume hidden evidence.

Rubrics:
- Comprehensiveness: which answer covers the relevant aspects and details more completely?
- Diversity: which answer gives richer technical perspectives and source/evidence variety?
- Empowerment: which answer better helps a researcher understand the issue and make follow-up judgments?
- Technical correctness: which answer is more technically precise and less internally inconsistent?
- Hallucination risk: which answer has lower risk of unsupported claims? Choose the safer answer.

Return strict JSON with this shape:
{{
  "Comprehensiveness": {{"winner": "A|B|Tie", "explanation": "..."}},
  "Diversity": {{"winner": "A|B|Tie", "explanation": "..."}},
  "Empowerment": {{"winner": "A|B|Tie", "explanation": "..."}},
  "Technical correctness": {{"winner": "A|B|Tie", "explanation": "..."}},
  "Hallucination risk": {{"winner": "A|B|Tie", "explanation": "..."}},
  "Overall": {{"winner": "A|B|Tie", "explanation": "..."}}
}}

Question:
{question}

Answer A:
{clip(answer_a)}

Answer B:
{clip(answer_b)}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


async def call_judge(prompt: str) -> dict[str, Any]:
    from openai import AsyncOpenAI

    api_key = (
        os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError("Set EVAL_LLM_BINDING_API_KEY or EMBEDDING_BINDING_API_KEY for Gemini judge.")
    client = AsyncOpenAI(
        base_url=os.getenv("EVAL_LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )
    response = await client.chat.completions.create(
        model=os.getenv("EVAL_LLM_MODEL", "google/gemini-3.5-flash"),
        messages=[
            {"role": "system", "content": "You are a strict JSON-only evaluator for RAG answer quality."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or "{}"
    return parse_json_object(text)


def winner_to_label(winner: str, answer_a_label: str, answer_b_label: str) -> str:
    normalized = winner.strip().lower()
    if normalized.startswith("a"):
        return answer_a_label
    if normalized.startswith("b"):
        return answer_b_label
    return "Tie"


def summarize(path: Path, output: Path) -> None:
    rows = read_jsonl(path)
    wins: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        if row.get("status") != "success":
            continue
        pair_id = str(row.get("pair_id"))
        result = row.get("judge_result")
        if not isinstance(result, dict):
            continue
        for rubric in RUBRICS + ["Overall"]:
            block = result.get(rubric)
            if not isinstance(block, dict):
                continue
            label = winner_to_label(
                str(block.get("winner", "Tie")),
                str(row.get("answer_a_label")),
                str(row.get("answer_b_label")),
            )
            wins[rubric][label] += 1
            by_pair[pair_id][rubric][label] += 1

    summary = {
        "judge_model": os.getenv("EVAL_LLM_MODEL", "google/gemini-3.5-flash"),
        "total_judgments": len(rows),
        "rubric_wins": {rubric: dict(counter) for rubric, counter in wins.items()},
        "pair_wins": {
            pair: {rubric: dict(counter) for rubric, counter in rubrics.items()}
            for pair, rubrics in by_pair.items()
        },
    }
    write_json(output, summary)


async def main_async() -> None:
    args = parse_args()
    load_dotenv_file(Path(args.env))
    rows = read_jsonl(Path(args.lightrag_results)) + read_jsonl(Path(args.graphrag_results))
    queries = {str(row.get("query_id")): row for row in read_jsonl(Path(args.query_file))}
    lookup = build_answer_lookup(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output_path) if args.resume else set()

    jobs: list[tuple[str, str, str, str]] = []
    for left, right in DEFAULT_PAIRS:
        pair_id = f"{left}__vs__{right}"
        query_ids = sorted(set(lookup.get(left, {})) & set(lookup.get(right, {})))
        for query_id in query_ids:
            jobs.append((pair_id, query_id, left, right))
    if args.limit:
        jobs = jobs[: args.limit]

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:
        for index, (pair_id, query_id, left, right) in enumerate(jobs, start=1):
            if (pair_id, query_id) in done:
                print(json.dumps({"skip": pair_id, "query_id": query_id, "index": index, "total": len(jobs)}, ensure_ascii=False), flush=True)
                continue
            swap = index % 2 == 0
            answer_a_label, answer_b_label = (right, left) if swap else (left, right)
            answer_a = str(lookup[answer_a_label][query_id].get("answer") or lookup[answer_a_label][query_id].get("response") or "")
            answer_b = str(lookup[answer_b_label][query_id].get("answer") or lookup[answer_b_label][query_id].get("response") or "")
            question = str(queries.get(query_id, {}).get("question") or lookup[left][query_id].get("question") or "")
            started = time.time()
            try:
                result = await call_judge(judge_prompt(question, answer_a, answer_b))
                row = {
                    "status": "success",
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "question": question,
                    "answer_a_label": answer_a_label,
                    "answer_b_label": answer_b_label,
                    "judge_result": result,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            except Exception as exc:
                row = {
                    "status": "failure",
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "question": question,
                    "answer_a_label": answer_a_label,
                    "answer_b_label": answer_b_label,
                    "error": repr(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "query_id": query_id,
                        "status": row["status"],
                        "elapsed_seconds": row["elapsed_seconds"],
                        "index": index,
                        "total": len(jobs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summarize(output_path, Path(args.summary))


if __name__ == "__main__":
    asyncio.run(main_async())
