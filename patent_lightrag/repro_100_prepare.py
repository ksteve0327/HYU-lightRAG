from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from patent_lightrag.common import (
    MID_CATEGORY_FIELD,
    RAW_CSV,
    ROOT,
    build_doc_record,
    clean_value,
    estimate_tokens,
    read_csv_rows,
    summarize_rows,
    write_csv,
    write_json,
)


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_SEED = 20260609
TARGET_CATEGORIES = ["AA", "AB", "AC", "AD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Patent-100 reproducibility dataset and fixed evaluation queries."
    )
    parser.add_argument("--input", default=str(RAW_CSV))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--smoke-per-category", type=int, default=5)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(
    rows: list[dict[str, str]],
    seed: int,
    per_category: int,
    smoke_per_category: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    smoke: list[dict[str, str]] = []
    remaining: list[dict[str, str]] = []
    selected_by_category: dict[str, list[str]] = {}
    smoke_by_category: dict[str, list[str]] = {}
    remaining_by_category: dict[str, list[str]] = {}
    available_by_category: dict[str, int] = {}

    for category in TARGET_CATEGORIES:
        candidates = [
            row for row in rows if clean_value(row.get(MID_CATEGORY_FIELD)) == category
        ]
        candidates = sorted(
            candidates,
            key=lambda row: (
                clean_value(row.get("patent_id")),
                clean_value(row.get("출원번호")),
            ),
        )
        available_by_category[category] = len(candidates)
        if len(candidates) < per_category:
            raise RuntimeError(
                f"Category {category} has only {len(candidates)} rows; {per_category} required."
            )
        category_selected = rng.sample(candidates, per_category)
        category_selected = sorted(
            category_selected,
            key=lambda row: (
                clean_value(row.get("patent_id")),
                clean_value(row.get("출원번호")),
            ),
        )
        category_smoke = category_selected[:smoke_per_category]
        category_remaining = category_selected[smoke_per_category:]

        selected.extend(category_selected)
        smoke.extend(category_smoke)
        remaining.extend(category_remaining)
        selected_by_category[category] = [clean_value(row.get("patent_id")) for row in category_selected]
        smoke_by_category[category] = [clean_value(row.get("patent_id")) for row in category_smoke]
        remaining_by_category[category] = [clean_value(row.get("patent_id")) for row in category_remaining]

    selected = sorted(selected, key=lambda row: (clean_value(row.get(MID_CATEGORY_FIELD)), clean_value(row.get("patent_id"))))
    smoke = sorted(smoke, key=lambda row: (clean_value(row.get(MID_CATEGORY_FIELD)), clean_value(row.get("patent_id"))))
    remaining = sorted(remaining, key=lambda row: (clean_value(row.get(MID_CATEGORY_FIELD)), clean_value(row.get("patent_id"))))
    manifest = {
        "seed": seed,
        "per_category": per_category,
        "smoke_per_category": smoke_per_category,
        "categories": TARGET_CATEGORIES,
        "available_by_category": available_by_category,
        "selected_ids_by_category": selected_by_category,
        "smoke_ids_by_category": smoke_by_category,
        "remaining_ids_by_category": remaining_by_category,
        "total_selected": len(selected),
        "total_smoke": len(smoke),
        "total_remaining": len(remaining),
        "git_commit": git_commit(),
    }
    return selected, smoke, remaining, manifest


def build_docs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [build_doc_record(row) for row in rows]


def pick_fact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            len(clean_value(row.get("대표청구항"))),
            len(clean_value(row.get("요약"))),
        ),
        reverse=True,
    )
    picked: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    for row in ranked:
        category = clean_value(row.get(MID_CATEGORY_FIELD))
        if category in seen_categories:
            continue
        picked.append(row)
        seen_categories.add(category)
        if len(picked) == 2:
            break
    return picked or ranked[:2]


def build_queries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    facts = pick_fact_rows(rows)
    fact_1 = facts[0] if facts else rows[0]
    fact_2 = facts[1] if len(facts) > 1 else rows[-1]

    return [
        {
            "query_id": "AA-1",
            "category": "AA",
            "type": "category_specific",
            "question": "AI 코어 및 가속기 특허에서 신경망 연산 가속기 구조는 어떤 방식으로 연산 병렬성을 높이는가?",
            "expected_focus": "parallel compute architecture, NPU, matrix operation, dataflow",
        },
        {
            "query_id": "AA-2",
            "category": "AA",
            "type": "category_specific",
            "question": "NPU 또는 행렬 연산 장치 관련 특허에서 메모리 접근 병목을 줄이는 기술은 무엇인가?",
            "expected_focus": "memory access bottleneck, buffer, scheduler, data movement",
        },
        {
            "query_id": "AB-1",
            "category": "AB",
            "type": "category_specific",
            "question": "인-메모리 컴퓨팅 또는 PIM 구조가 AI 연산 성능을 높이는 핵심 원리는 무엇인가?",
            "expected_focus": "PIM, in-memory computing, analog/digital matrix operation",
        },
        {
            "query_id": "AB-2",
            "category": "AB",
            "type": "category_specific",
            "question": "SRAM 또는 DRAM 기반 메모리 아키텍처와 AI 연산 회로의 결합 방식은 어떻게 나타나는가?",
            "expected_focus": "SRAM, DRAM, memory architecture, compute circuit coupling",
        },
        {
            "query_id": "AC-1",
            "category": "AC",
            "type": "category_specific",
            "question": "AI 반도체 패키징 또는 인터커넥트 기술은 대역폭과 전력 효율을 어떻게 개선하는가?",
            "expected_focus": "packaging, interconnect, bandwidth, power efficiency",
        },
        {
            "query_id": "AC-2",
            "category": "AC",
            "type": "category_specific",
            "question": "칩렛, 적층, TSV, 인터포저 계열 기술이 AI 가속기 구조와 어떤 관계를 갖는가?",
            "expected_focus": "chiplet, stacking, TSV, interposer, accelerator architecture",
        },
        {
            "query_id": "AD-1",
            "category": "AD",
            "type": "category_specific",
            "question": "AI 반도체 제조 또는 공정 특허에서 신뢰성이나 수율 개선을 위한 핵심 방법은 무엇인가?",
            "expected_focus": "manufacturing process, reliability, yield, process control",
        },
        {
            "query_id": "AD-2",
            "category": "AD",
            "type": "category_specific",
            "question": "소자 구조나 공정 제어가 AI 연산 성능 또는 전력 효율에 미치는 영향은 무엇인가?",
            "expected_focus": "device structure, process control, performance, energy efficiency",
        },
        {
            "query_id": "X-1",
            "category": "cross",
            "type": "cross_category",
            "question": "연산 가속기, 메모리 아키텍처, 패키징 기술은 AI 반도체에서 어떤 기술적 의존 관계를 갖는가?",
            "expected_focus": "accelerator-memory-packaging dependency graph",
        },
        {
            "query_id": "X-2",
            "category": "cross",
            "type": "cross_category",
            "question": "PIM과 NPU 방식은 데이터 이동 비용을 줄이는 관점에서 어떻게 다르게 접근하는가?",
            "expected_focus": "PIM vs NPU, data movement, memory proximity",
        },
        {
            "query_id": "X-3",
            "category": "cross",
            "type": "cross_category",
            "question": "AI 반도체 특허에서 전력 효율, 대역폭, 지연시간 개선 기술은 어떤 관계망을 형성하는가?",
            "expected_focus": "power efficiency, bandwidth, latency, relation network",
        },
        {
            "query_id": "F-1",
            "category": clean_value(fact_1.get(MID_CATEGORY_FIELD)),
            "type": "fact_check",
            "target_patent_id": clean_value(fact_1.get("patent_id")),
            "target_application_no": clean_value(fact_1.get("출원번호")),
            "question": f"특허 {clean_value(fact_1.get('patent_id'))}의 주요 구성요소, 해결하려는 문제, 대표청구항의 기술 범위를 설명하라.",
            "expected_focus": "single patent structure, claim elements, problem-solution",
        },
        {
            "query_id": "F-2",
            "category": clean_value(fact_2.get(MID_CATEGORY_FIELD)),
            "type": "fact_check",
            "target_patent_id": clean_value(fact_2.get("patent_id")),
            "target_application_no": clean_value(fact_2.get("출원번호")),
            "question": f"특허 {clean_value(fact_2.get('patent_id'))}의 대표청구항에 포함된 회로, 메모리, 연산 구성요소를 source patent 기준으로 정리하라.",
            "expected_focus": "single patent claim components, source-grounded answer",
        },
        {
            "query_id": "C-1",
            "category": "cross",
            "type": "comparison",
            "question": "인-메모리 컴퓨팅 방식과 hardware operator fusion 방식의 차이를 비교하라.",
            "expected_focus": "comparison, in-memory computing, operator fusion",
        },
        {
            "query_id": "E-1",
            "category": "AC",
            "type": "exploratory",
            "question": "AI 반도체 패키징 기술 동향을 특허 근거 중심으로 탐색적으로 요약하라.",
            "expected_focus": "packaging trend, patent evidence, source coverage",
        },
    ]


def write_graphrag_input(root: Path, docs: list[dict[str, object]]) -> None:
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old_file in input_dir.glob("*.txt"):
        old_file.unlink()
    for doc in docs:
        doc_id = str(doc.get("id", "unknown")).replace("/", "_")
        text = str(doc.get("text", ""))
        (input_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    dataset_dir = experiment_dir / "dataset"
    query_dir = experiment_dir / "queries"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_csv_rows(Path(args.input))
    selected, smoke, remaining, manifest = select_rows(
        rows,
        seed=args.seed,
        per_category=args.per_category,
        smoke_per_category=args.smoke_per_category,
    )
    docs = build_docs(selected)
    smoke_docs = build_docs(smoke)
    remaining_docs = build_docs(remaining)
    queries = build_queries(selected)

    write_csv(dataset_dir / "patents_100.csv", fieldnames, selected)
    write_csv(dataset_dir / "graphrag_smoke_20.csv", fieldnames, smoke)
    write_csv(dataset_dir / "graphrag_remaining_80.csv", fieldnames, remaining)
    write_jsonl(dataset_dir / "patents_100.jsonl", docs)
    write_jsonl(dataset_dir / "graphrag_smoke_20.jsonl", smoke_docs)
    write_jsonl(dataset_dir / "graphrag_remaining_80.jsonl", remaining_docs)
    write_jsonl(query_dir / "eval_queries_15.jsonl", queries)

    token_counts = [estimate_tokens(str(doc.get("text", ""))) for doc in docs]
    manifest.update(
        {
            "input": str(Path(args.input)),
            "sample_summary": summarize_rows(selected),
            "smoke_summary": summarize_rows(smoke),
            "remaining_summary": summarize_rows(remaining),
            "doc_count": len(docs),
            "query_count": len(queries),
            "token_estimate": {
                "total": sum(token_counts),
                "average": round(sum(token_counts) / max(1, len(token_counts)), 2),
            },
            "category_counts": dict(Counter(str(doc.get("category", "")) for doc in docs)),
            "files": {
                "patents_100_jsonl": str(dataset_dir / "patents_100.jsonl"),
                "queries_jsonl": str(query_dir / "eval_queries_15.jsonl"),
            },
        }
    )
    write_json(dataset_dir / "patents_100_manifest.json", manifest)
    write_graphrag_input(experiment_dir / "graphrag_smoke_20", smoke_docs)
    write_graphrag_input(experiment_dir / "graphrag_full_100_update", smoke_docs)
    print(
        json.dumps(
            {
                "experiment_dir": str(experiment_dir),
                "docs": len(docs),
                "queries": len(queries),
                "seed": args.seed,
                "category_counts": manifest["category_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
