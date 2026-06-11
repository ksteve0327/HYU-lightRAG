from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, estimate_tokens, load_dotenv_file, write_json
from patent_lightrag.rag_repro_judge import winner_to_label


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "evaluation_length_control"
SYSTEMS = ["lightrag_naive", "lightrag_hybrid", "graphrag_global", "graphrag_local"]
PAIRS = [
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
    parser = argparse.ArgumentParser(description="Run length-controlled RAG answer evaluation.")
    parser.add_argument(
        "--stage",
        choices=["normalize", "judge-original", "judge-normalized", "summarize", "all"],
        required=True,
    )
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--target-min-chars", type=int, default=1100)
    parser.add_argument("--target-max-chars", type=int, default=1300)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def clean_answer_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\x1b\[[0-9;]*m", "", value)
    lines = [
        line for line in value.splitlines()
        if "LiteLLM:WARNING" not in line and "could not pre-load" not in line
    ]
    return "\n".join(lines).strip()


def label_for(row: dict[str, Any]) -> str:
    if row.get("method"):
        return f"graphrag_{row.get('method')}"
    return f"lightrag_{row.get('mode')}"


def load_source_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    light = read_jsonl(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl")
    graph = read_jsonl(experiment_dir / "graphrag_full_100_fresh" / "query_results_15_methods.jsonl")
    selected: list[dict[str, Any]] = []
    for row in light + graph:
        system = label_for(row)
        answer = clean_answer_text(row.get("answer", ""))
        if row.get("status") == "success" and system in SYSTEMS and answer:
            selected.append({**row, "system": system, "clean_answer": answer})
    return selected


def load_queries(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("query_id")): row
        for row in read_jsonl(experiment_dir / "queries" / "eval_queries_15.jsonl")
    }


def normalized_prompt(question: str, answer: str, min_chars: int, max_chars: int) -> str:
    return f"""
아래 RAG 답변을 길이 편향 평가용으로 정규화하라.

규칙:
- 한국어로 작성한다.
- {min_chars}-{max_chars}자 범위를 목표로 한다.
- 원 답변에 없는 새로운 사실, 특허 번호, 수치, 인과관계를 추가하지 않는다.
- 원 답변의 핵심 주장, 기술 근거, 출처/특허 언급은 가능한 유지한다.
- 장황한 서론, 반복 문장, 과도한 목록은 제거한다.
- 형식은 가능하면 "요약", "핵심 근거", "한계" 순서로 간결하게 쓴다.
- 답변 본문만 출력하고 설명 문구나 JSON은 출력하지 않는다.

질문:
{question}

원 답변:
{answer}
""".strip()


async def normalize_answer(prompt: str, env_path: Path) -> str:
    load_dotenv_file(env_path)
    sys.path.insert(0, str(LIGHTRAG_ROOT))
    from lightrag.llm.openai import openai_complete_if_cache

    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "gpt-5.5"),
        prompt,
        base_url=os.getenv("LLM_BINDING_HOST", "http://localhost:11435/v1"),
        api_key=os.getenv("LLM_BINDING_API_KEY", "codex-proxy"),
        timeout=int(os.getenv("LLM_TIMEOUT", "900")),
        reasoning_effort=os.getenv("OPENAI_LLM_REASONING_EFFORT", "xhigh"),
    )


def normalize_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("system")), str(row.get("query_id")))


async def run_normalize(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "normalized_answers_1200.jsonl"
    length_path = output_dir / "answer_length_before_after.json"
    source_rows = load_source_rows(experiment_dir)
    existing_rows = read_jsonl(output_path) if args.resume else []
    existing = {
        normalize_key(row): row
        for row in existing_rows
        if row.get("status") == "success" and row.get("answer")
    }
    output_rows = {normalize_key(row): row for row in existing_rows}

    jobs = [row for row in source_rows if normalize_key(row) not in existing]
    if args.limit:
        jobs = jobs[: args.limit]

    print(json.dumps({"stage": "normalize", "total_source": len(source_rows), "todo": len(jobs)}, ensure_ascii=False), flush=True)
    for index, row in enumerate(jobs, start=1):
        started = time.time()
        key = normalize_key(row)
        try:
            normalized = clean_answer_text(
                await normalize_answer(
                    normalized_prompt(
                        str(row.get("question", "")),
                        str(row.get("clean_answer", "")),
                        args.target_min_chars,
                        args.target_max_chars,
                    ),
                    Path(args.env),
                )
            )
            out = {
                "status": "success",
                "system": row.get("system"),
                "query_id": row.get("query_id"),
                "category": row.get("category"),
                "type": row.get("type"),
                "question": row.get("question"),
                "answer": normalized,
                "original_answer": row.get("clean_answer"),
                "original_chars": len(str(row.get("clean_answer", ""))),
                "normalized_chars": len(normalized),
                "original_token_estimate": estimate_tokens(str(row.get("clean_answer", ""))),
                "normalized_token_estimate": estimate_tokens(normalized),
                "elapsed_seconds": round(time.time() - started, 3),
                "target_min_chars": args.target_min_chars,
                "target_max_chars": args.target_max_chars,
            }
        except Exception as exc:
            out = {
                "status": "failure",
                "system": row.get("system"),
                "query_id": row.get("query_id"),
                "category": row.get("category"),
                "type": row.get("type"),
                "question": row.get("question"),
                "error": repr(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        output_rows[key] = out
        ordered = sorted(output_rows.values(), key=lambda item: (SYSTEMS.index(str(item.get("system"))) if str(item.get("system")) in SYSTEMS else 99, str(item.get("query_id"))))
        write_jsonl(output_path, ordered)
        print(
            json.dumps(
                {
                    "stage": "normalize",
                    "system": row.get("system"),
                    "query_id": row.get("query_id"),
                    "status": out["status"],
                    "normalized_chars": out.get("normalized_chars"),
                    "elapsed_seconds": out["elapsed_seconds"],
                    "completed": index,
                    "todo": len(jobs),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    final_rows = read_jsonl(output_path)
    summary = summarize_lengths(source_rows, final_rows, args.target_min_chars, args.target_max_chars)
    write_json(length_path, summary)
    print(json.dumps({"stage": "normalize-summary", **summary["counts"]}, ensure_ascii=False), flush=True)


def answer_lookup(rows: list[dict[str, Any]], normalized: bool = False) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "success":
            continue
        system = str(row.get("system") or label_for(row))
        answer = clean_answer_text(row.get("answer", ""))
        if system in SYSTEMS and row.get("query_id") and answer:
            payload = dict(row)
            payload["answer"] = answer
            payload["normalized"] = normalized
            lookup[system][str(row.get("query_id"))] = payload
    return lookup


def judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return f"""
You are evaluating two retrieval-augmented answers to the same AI semiconductor patent question.
Use only the question and the two answers. Do not assume hidden evidence.

Important anti-verbosity rule:
- Do not reward an answer merely because it is longer.
- If two answers contain the same useful information, prefer the more concise and precise answer.
- Comprehensiveness means necessary coverage for the question, not number of paragraphs.
- Diversity means valid distinct technical perspectives, not a longer list.
- Empowerment means helping a researcher make a better follow-up judgment, not more wording.

Rubrics:
- Comprehensiveness: which answer covers the needed aspects more completely without padding?
- Diversity: which answer gives richer valid technical perspectives and evidence variety?
- Empowerment: which answer better helps a researcher understand and act on the issue?
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
{answer_a}

Answer B:
{answer_b}
""".strip()


async def call_judge(prompt: str, env_path: Path) -> dict[str, Any]:
    from openai import AsyncOpenAI

    load_dotenv_file(env_path)
    api_key = (
        os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError("Set EVAL_LLM_BINDING_API_KEY, OPENROUTER_API_KEY, or EMBEDDING_BINDING_API_KEY.")
    client = AsyncOpenAI(
        base_url=os.getenv("EVAL_LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )
    response = await client.chat.completions.create(
        model=os.getenv("EVAL_LLM_MODEL", "google/gemini-3.5-flash"),
        messages=[
            {"role": "system", "content": "You are a strict JSON-only evaluator for RAG answer quality. Penalize verbosity that does not add useful evidence."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def done_judge_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("pair_id")), str(row.get("query_id")))
        for row in read_jsonl(path)
        if row.get("status") == "success"
    }


async def run_judge(args: argparse.Namespace, normalized: bool) -> None:
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / ("judge_normalized_verbosity_aware.jsonl" if normalized else "judge_original_verbosity_aware.jsonl")
    queries = load_queries(experiment_dir)
    rows = read_jsonl(output_dir / "normalized_answers_1200.jsonl") if normalized else load_source_rows(experiment_dir)
    lookup = answer_lookup(rows, normalized=normalized)
    done = done_judge_keys(output_path) if args.resume else set()
    jobs: list[tuple[str, str, str, str]] = []
    for left, right in PAIRS:
        pair_id = f"{left}__vs__{right}"
        query_ids = sorted(set(lookup.get(left, {})) & set(lookup.get(right, {})))
        for query_id in query_ids:
            if (pair_id, query_id) not in done:
                jobs.append((pair_id, query_id, left, right))
    if args.limit:
        jobs = jobs[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "judge-normalized" if normalized else "judge-original", "todo": len(jobs)}, ensure_ascii=False), flush=True)
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:
        for index, (pair_id, query_id, left, right) in enumerate(jobs, start=1):
            swap = index % 2 == 0
            answer_a_label, answer_b_label = (right, left) if swap else (left, right)
            answer_a = str(lookup[answer_a_label][query_id].get("answer") or "")
            answer_b = str(lookup[answer_b_label][query_id].get("answer") or "")
            question = str(queries.get(query_id, {}).get("question") or lookup[left][query_id].get("question") or "")
            started = time.time()
            try:
                result = await call_judge(judge_prompt(question, answer_a, answer_b), Path(args.env))
                row = {
                    "status": "success",
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "question": question,
                    "answer_a_label": answer_a_label,
                    "answer_b_label": answer_b_label,
                    "answer_a_chars": len(answer_a),
                    "answer_b_chars": len(answer_b),
                    "judge_result": result,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "length_control": "normalized" if normalized else "original",
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
                    "length_control": "normalized" if normalized else "original",
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        "stage": "judge-normalized" if normalized else "judge-original",
                        "pair_id": pair_id,
                        "query_id": query_id,
                        "status": row["status"],
                        "elapsed_seconds": row["elapsed_seconds"],
                        "completed": index,
                        "todo": len(jobs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def canonical_judge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest successful attempt per pair/query key."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        key = (str(row.get("pair_id")), str(row.get("query_id")))
        by_key[key] = row
    return list(by_key.values())


def summarize_judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_rows = canonical_judge_rows(rows)
    wins: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in canonical_rows:
        result = row.get("judge_result", {})
        pair_id = str(row.get("pair_id"))
        query_type = str(row.get("type") or "unknown")
        for rubric in RUBRICS + ["Overall"]:
            block = result.get(rubric, {}) if isinstance(result, dict) else {}
            label = winner_to_label(
                str(block.get("winner", "Tie")),
                str(row.get("answer_a_label")),
                str(row.get("answer_b_label")),
            )
            wins[rubric][label] += 1
            by_pair[pair_id][rubric][label] += 1
            by_type[query_type][rubric][label] += 1
    return {
        "attempt_rows": len(rows),
        "success_attempt_rows": sum(1 for row in rows if row.get("status") == "success"),
        "failure_attempt_rows": sum(1 for row in rows if row.get("status") != "success"),
        "deduped_success_rows": len(canonical_rows),
        "rubric_wins": {rubric: dict(counter) for rubric, counter in wins.items()},
        "pair_wins": {
            pair: {rubric: dict(counter) for rubric, counter in rubrics.items()}
            for pair, rubrics in by_pair.items()
        },
        "query_type_wins": {
            query_type: {rubric: dict(counter) for rubric, counter in rubrics.items()}
            for query_type, rubrics in by_type.items()
        },
    }


def summarize_lengths(source_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]], min_chars: int, max_chars: int) -> dict[str, Any]:
    before: dict[str, list[int]] = defaultdict(list)
    after: dict[str, list[int]] = defaultdict(list)
    for row in source_rows:
        before[str(row.get("system"))].append(len(str(row.get("clean_answer", ""))))
    for row in normalized_rows:
        if row.get("status") == "success":
            after[str(row.get("system"))].append(int(row.get("normalized_chars") or len(str(row.get("answer", "")))))

    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "in_target": 0}
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "in_target": sum(1 for value in values if min_chars <= value <= max_chars),
        }

    return {
        "target": {"min_chars": min_chars, "max_chars": max_chars},
        "counts": {
            "source_rows": len(source_rows),
            "normalized_rows": len(normalized_rows),
            "normalized_success": sum(1 for row in normalized_rows if row.get("status") == "success"),
            "normalized_failure": sum(1 for row in normalized_rows if row.get("status") != "success"),
        },
        "before": {system: stats(values) for system, values in sorted(before.items())},
        "after": {system: stats(values) for system, values in sorted(after.items())},
    }


def run_summarize(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    source_rows = load_source_rows(experiment_dir)
    normalized_rows = read_jsonl(output_dir / "normalized_answers_1200.jsonl")
    original_judge = read_jsonl(output_dir / "judge_original_verbosity_aware.jsonl")
    normalized_judge = read_jsonl(output_dir / "judge_normalized_verbosity_aware.jsonl")
    queries = load_queries(experiment_dir)
    for row in original_judge + normalized_judge:
        query = queries.get(str(row.get("query_id")), {})
        row["type"] = query.get("type", row.get("type"))
        row["category"] = query.get("category", row.get("category"))
    payload = {
        "lengths": summarize_lengths(source_rows, normalized_rows, args.target_min_chars, args.target_max_chars),
        "judge_original_verbosity_aware": summarize_judge(original_judge),
        "judge_normalized_verbosity_aware": summarize_judge(normalized_judge),
    }
    write_json(output_dir / "judge_length_control_summary.json", payload)
    write_json(output_dir / "query_type_breakdown_length_control.json", {
        "original": payload["judge_original_verbosity_aware"]["query_type_wins"],
        "normalized": payload["judge_normalized_verbosity_aware"]["query_type_wins"],
    })
    print(json.dumps({"stage": "summarize", "output": str(output_dir)}, ensure_ascii=False), flush=True)


async def main_async() -> None:
    args = parse_args()
    if args.stage in {"normalize", "all"}:
        await run_normalize(args)
    if args.stage in {"judge-original", "all"}:
        await run_judge(args, normalized=False)
    if args.stage in {"judge-normalized", "all"}:
        await run_judge(args, normalized=True)
    if args.stage in {"summarize", "all"}:
        run_summarize(args)


if __name__ == "__main__":
    asyncio.run(main_async())
