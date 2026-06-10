from __future__ import annotations

import csv
import html
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LIGHTRAG_ROOT = ROOT / "LightRAG-main"
RAW_CSV = ROOT / "patent_rawdata.csv"
DATA_DIR = ROOT / "data" / "patents"
RUN_DIR = ROOT / "data" / "lightrag_runs"
REPORT_DIR = ROOT / "reports"

DEFAULT_SEED = 42
DEFAULT_TARGET_PER_CATEGORY = 50
DEFAULT_DRY_RUN_LIMIT = 20

MID_CATEGORY_FIELD = "중분류"
MID_CATEGORY_NAME_FIELD = "중분류명"
SUB_CATEGORY_FIELD = "소분류"
SUB_CATEGORY_NAME_FIELD = "소분류명"

IDENTIFIER_FIELDS = ["patent_id", "출원번호", "공개번호", "등록번호"]
TEXT_FIELDS = [
    "발명의 명칭",
    "요약",
    "AI요약(목적)",
    "AI요약(솔루션)",
    "AI요약(목적+솔루션)",
    "대표청구항",
]
CLASSIFICATION_FIELDS = [
    "중분류",
    "중분류명",
    "소분류",
    "소분류명",
    "메인 IPC",
    "메인 IPC 설명",
    "전체 IPC",
    "전체 IPC 설명",
    "메인 CPC",
    "전체 CPC",
]
OWNER_FIELDS = [
    "출원인정리",
    "출원인정규화",
    "출원인유형",
    "현재권리자정리",
    "현재권리자정규화",
    "현재권리자유형",
]
COUNTRY_DATE_FIELDS = [
    "국가코드",
    "출원인 국적",
    "현재권리자 국적",
    "출원일",
    "출원연도",
    "공개일",
    "등록일",
    "법적상태",
]


@dataclass(frozen=True)
class ArtifactPaths:
    sample_csv: Path = DATA_DIR / "patent_sample_200.csv"
    sample_manifest: Path = DATA_DIR / "sampling_manifest.json"
    docs_jsonl: Path = DATA_DIR / "patent_docs.jsonl"
    docs_manifest: Path = DATA_DIR / "docs_manifest.json"
    index_stats: Path = RUN_DIR / "index_stats.json"
    query_results: Path = RUN_DIR / "query_results.json"
    html_report: Path = REPORT_DIR / "lightrag_flow_3_1_3_4.html"


PATHS = ArtifactPaths()


def ensure_dirs() -> None:
    for path in [DATA_DIR, RUN_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text in {"", "-", "nan", "None"}:
        return ""
    return " ".join(text.split())


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(
    rows: list[dict[str, str]],
    category_field: str = MID_CATEGORY_FIELD,
    strata_field: str = SUB_CATEGORY_FIELD,
    target_per_category: int = DEFAULT_TARGET_PER_CATEGORY,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        category = clean_value(row.get(category_field))
        if category:
            by_category[category].append(row)

    sampled: list[dict[str, str]] = []
    manifest_categories: dict[str, object] = {}

    for category in sorted(by_category):
        category_rows = by_category[category]
        if len(category_rows) < target_per_category:
            raise ValueError(
                f"Category {category} has only {len(category_rows)} rows, "
                f"but {target_per_category} are required."
            )

        by_strata: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in category_rows:
            stratum = clean_value(row.get(strata_field)) or "UNKNOWN"
            by_strata[stratum].append(row)

        total = len(category_rows)
        allocation: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        for stratum, stratum_rows in sorted(by_strata.items()):
            ideal = target_per_category * len(stratum_rows) / total
            base = min(len(stratum_rows), int(ideal))
            allocation[stratum] = base
            remainders.append((ideal - base, stratum))

        remaining = target_per_category - sum(allocation.values())
        for _, stratum in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if allocation[stratum] < len(by_strata[stratum]):
                allocation[stratum] += 1
                remaining -= 1

        if remaining:
            for stratum in sorted(by_strata):
                while remaining and allocation[stratum] < len(by_strata[stratum]):
                    allocation[stratum] += 1
                    remaining -= 1
                if remaining <= 0:
                    break

        category_sample: list[dict[str, str]] = []
        stratum_manifest: dict[str, object] = {}
        for stratum in sorted(by_strata):
            candidates = sorted(
                by_strata[stratum],
                key=lambda row: (
                    clean_value(row.get("patent_id")),
                    clean_value(row.get("출원번호")),
                ),
            )
            take = allocation[stratum]
            selected = rng.sample(candidates, take) if take else []
            selected = sorted(selected, key=lambda row: clean_value(row.get("patent_id")))
            category_sample.extend(selected)
            stratum_manifest[stratum] = {
                "available": len(candidates),
                "sampled": take,
                "sub_category_name": clean_value(
                    selected[0].get(SUB_CATEGORY_NAME_FIELD)
                    if selected
                    else candidates[0].get(SUB_CATEGORY_NAME_FIELD)
                ),
            }

        category_sample = sorted(category_sample, key=lambda row: clean_value(row.get("patent_id")))
        sampled.extend(category_sample)
        manifest_categories[category] = {
            "category_name": clean_value(category_sample[0].get(MID_CATEGORY_NAME_FIELD)),
            "available": len(category_rows),
            "sampled": len(category_sample),
            "strata": stratum_manifest,
        }

    sampled = sorted(
        sampled,
        key=lambda row: (
            clean_value(row.get(MID_CATEGORY_FIELD)),
            clean_value(row.get(SUB_CATEGORY_FIELD)),
            clean_value(row.get("patent_id")),
        ),
    )
    manifest = {
        "seed": seed,
        "target_per_category": target_per_category,
        "total_sampled": len(sampled),
        "categories": manifest_categories,
    }
    return sampled, manifest


def structured_patent_text(row: dict[str, str]) -> str:
    sections: list[str] = []

    def add(label: str, field: str) -> None:
        value = clean_value(row.get(field))
        if value:
            sections.append(f"{label}: {value}")

    sections.append("문서 유형: AI 반도체 특허")
    for field in IDENTIFIER_FIELDS:
        add(field, field)
    for field in TEXT_FIELDS:
        add(field, field)
    for field in CLASSIFICATION_FIELDS:
        add(field, field)
    for field in OWNER_FIELDS:
        add(field, field)
    for field in COUNTRY_DATE_FIELDS:
        add(field, field)

    return "\n".join(sections)


def patent_metadata(row: dict[str, str]) -> dict[str, str]:
    metadata_fields = (
        IDENTIFIER_FIELDS
        + [MID_CATEGORY_FIELD, MID_CATEGORY_NAME_FIELD, SUB_CATEGORY_FIELD, SUB_CATEGORY_NAME_FIELD]
        + ["출원연도", "출원일", "국가코드", "출원인정규화", "현재권리자정규화"]
    )
    return {field: clean_value(row.get(field)) for field in metadata_fields}


def build_doc_record(row: dict[str, str]) -> dict[str, object]:
    patent_id = clean_value(row.get("patent_id"))
    application_no = clean_value(row.get("출원번호"))
    return {
        "id": patent_id,
        "file_path": f"patent://{application_no or patent_id}",
        "category": clean_value(row.get(MID_CATEGORY_FIELD)),
        "category_name": clean_value(row.get(MID_CATEGORY_NAME_FIELD)),
        "sub_category": clean_value(row.get(SUB_CATEGORY_FIELD)),
        "sub_category_name": clean_value(row.get(SUB_CATEGORY_NAME_FIELD)),
        "text": structured_patent_text(row),
        "metadata": patent_metadata(row),
    }


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    category_counts = Counter(clean_value(row.get(MID_CATEGORY_FIELD)) for row in rows)
    sub_counts = Counter(
        (
            clean_value(row.get(MID_CATEGORY_FIELD)),
            clean_value(row.get(SUB_CATEGORY_FIELD)),
        )
        for row in rows
    )
    years = Counter(clean_value(row.get("출원연도")) for row in rows if clean_value(row.get("출원연도")))
    return {
        "rows": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "sub_category_counts": {f"{cat}/{sub}": count for (cat, sub), count in sorted(sub_counts.items())},
        "year_counts": dict(sorted(years.items())),
    }


def load_dotenv_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
        os.environ.setdefault(key.strip(), value)
    return env


def lightrag_addon_params() -> dict[str, str]:
    params = {"language": os.getenv("SUMMARY_LANGUAGE", "Korean")}
    prompt_file = os.getenv("ENTITY_TYPE_PROMPT_FILE", "").strip()
    if prompt_file:
        params["entity_type_prompt_file"] = prompt_file
    return params


def prompt_runtime_config() -> dict[str, str]:
    return {
        "summary_language": os.getenv("SUMMARY_LANGUAGE", "Korean"),
        "prompt_dir": os.getenv("PROMPT_DIR", ""),
        "entity_type_prompt_file": os.getenv("ENTITY_TYPE_PROMPT_FILE", ""),
        "entity_extraction_use_json": os.getenv("ENTITY_EXTRACTION_USE_JSON", "false"),
    }


def redact(value: str) -> str:
    if not value:
        return ""
    if "PASTE_" in value or value == "codex-proxy":
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:5]}...{value[-4:]}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def read_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = []
    for line_no in range(start, min(end, len(lines)) + 1):
        selected.append(f"{line_no:>4}: {lines[line_no - 1]}")
    return "\n".join(selected)
