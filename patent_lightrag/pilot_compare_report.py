from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, html_escape, read_json


EXPERIMENT_DIR = ROOT / "experiments" / "patent_prompt_pilot_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a baseline-vs-pilot LightRAG report.")
    parser.add_argument("--baseline-metrics", default=str(EXPERIMENT_DIR / "baseline_metrics.json"))
    parser.add_argument("--pilot-metrics", default=str(EXPERIMENT_DIR / "pilot_metrics.json"))
    parser.add_argument("--baseline-stats", default=str(ROOT / "data" / "lightrag_runs" / "index_stats.json"))
    parser.add_argument("--pilot-stats", default=str(EXPERIMENT_DIR / "index_stats.json"))
    parser.add_argument("--baseline-query", default=str(ROOT / "data" / "lightrag_runs" / "query_results.json"))
    parser.add_argument("--pilot-query", default=str(EXPERIMENT_DIR / "query_results.json"))
    parser.add_argument("--pilot-manifest", default=str(EXPERIMENT_DIR / "pilot_docs_manifest.json"))
    parser.add_argument("--prompt", default=str(LIGHTRAG_ROOT / "prompts" / "entity_type" / "patent_ai_semiconductor.yml"))
    parser.add_argument("--baseline-working-dir", default=str(ROOT / "data" / "lightrag_runs" / "patent_rag_storage_full_200"))
    parser.add_argument("--pilot-working-dir", default=str(EXPERIMENT_DIR / "storage"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "lightrag_flow_3_1_3_4_pilot_compare.html"))
    return parser.parse_args()


def table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p class='empty'>No rows.</p>"
    headers = list(rows[0])
    head = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html_escape(row.get(header, ''))}</td>"
                for header in headers
            )
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def metric_rows(baseline: dict[str, object], pilot: dict[str, object]) -> list[dict[str, object]]:
    specs = [
        ("graph_nodes", "Graph nodes", ""),
        ("graph_edges", "Graph edges", ""),
        ("metadata_relation_ratio", "Metadata relation ratio", "<= 0.25"),
        ("technical_relation_ratio", "Technical relation ratio", ">= 0.40"),
        ("excluded_entity_ratio", "Excluded entity ratio", "<= 0.10"),
        ("entity_type_entropy", "Entity type entropy", "maintain/improve"),
        ("degree_ge_50_hub_count", "Degree >= 50 hub count", "decrease"),
    ]
    rows = []
    for key, label, target in specs:
        rows.append(
            {
                "metric": label,
                "baseline": baseline.get(key, ""),
                "pilot": pilot.get(key, ""),
                "target": target,
            }
        )
    return rows


def runtime_rows(baseline: dict[str, object], pilot: dict[str, object]) -> list[dict[str, object]]:
    keys = [
        "document_count",
        "processed_documents",
        "failed_documents",
        "completed",
        "elapsed_seconds",
        "llm_model",
        "reasoning_effort",
        "entity_type_prompt_file",
        "prompt_dir",
        "entity_extraction_use_json",
    ]
    return [
        {
            "field": key,
            "baseline": baseline.get(key, ""),
            "pilot": pilot.get(key, ""),
        }
        for key in keys
    ]


def clip(value: object, limit: int = 1400) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def relation_label(pair: object) -> str:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return f"{pair[0]} -> {pair[1]}"
    return str(pair)


def load_vdb_rows(path: Path) -> list[dict[str, object]]:
    data = read_json(path, {}) or {}
    if not isinstance(data, dict):
        return []
    rows = data.get("data", [])
    return rows if isinstance(rows, list) else []


def run_examples(working_dir: Path, preferred_doc: str = "") -> dict[str, object]:
    full_entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    entity_chunks = read_json(working_dir / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(working_dir / "kv_store_relation_chunks.json", {}) or {}
    entity_vdb = load_vdb_rows(working_dir / "vdb_entities.json")
    relation_vdb = load_vdb_rows(working_dir / "vdb_relationships.json")

    doc_id = preferred_doc if preferred_doc in full_entities else ""
    if not doc_id and isinstance(full_entities, dict) and full_entities:
        doc_id = sorted(full_entities)[0]
    entity_record = full_entities.get(doc_id, {}) if isinstance(full_entities, dict) else {}
    relation_record = full_relations.get(doc_id, {}) if isinstance(full_relations, dict) else {}

    entity_rows = [
        row
        for row in entity_vdb
        if not doc_id or row.get("file_path") == doc_id
    ][:5]
    relation_rows = [
        row
        for row in relation_vdb
        if not doc_id or row.get("file_path") == doc_id
    ][:5]

    duplicate_entities = [
        {"key": key, "count": value.get("count"), "chunk_ids": ", ".join(value.get("chunk_ids", [])[:5])}
        for key, value in entity_chunks.items()
        if isinstance(value, dict) and int(value.get("count", 0)) > 1
    ][:5] if isinstance(entity_chunks, dict) else []
    duplicate_relations = [
        {"key": key.replace("<SEP>", " -> "), "count": value.get("count"), "chunk_ids": ", ".join(value.get("chunk_ids", [])[:5])}
        for key, value in relation_chunks.items()
        if isinstance(value, dict) and int(value.get("count", 0)) > 1
    ][:5] if isinstance(relation_chunks, dict) else []

    return {
        "doc_id": doc_id,
        "r_entities": entity_record.get("entity_names", [])[:16]
        if isinstance(entity_record, dict)
        else [],
        "r_relations": [
            relation_label(pair)
            for pair in relation_record.get("relation_pairs", [])[:16]
        ]
        if isinstance(relation_record, dict)
        else [],
        "p_entity_profiles": [
            {
                "entity_name": row.get("entity_name"),
                "entity_type": row.get("entity_type"),
                "content": clip(row.get("content"), 500),
            }
            for row in entity_rows
        ],
        "p_relation_profiles": [
            {
                "src_id": row.get("src_id"),
                "tgt_id": row.get("tgt_id"),
                "content": clip(row.get("content"), 500),
            }
            for row in relation_rows
        ],
        "d_entity_examples": duplicate_entities,
        "d_relation_examples": duplicate_relations,
    }


def query_summary(query_results: dict[str, object]) -> list[dict[str, object]]:
    modes = query_results.get("modes", {}) if isinstance(query_results, dict) else {}
    rows: list[dict[str, object]] = []
    if not isinstance(modes, dict):
        return rows
    for mode, payload in modes.items():
        payload = payload if isinstance(payload, dict) else {}
        data = payload.get("data", {})
        body = data.get("data", {}) if isinstance(data, dict) else {}
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
        rows.append(
            {
                "mode": mode,
                "high_keywords": ", ".join(keywords.get("high_level", []))
                if isinstance(keywords.get("high_level", []), list)
                else "",
                "low_keywords": ", ".join(keywords.get("low_level", []))
                if isinstance(keywords.get("low_level", []), list)
                else "",
                "entities": len(body.get("entities", [])) if isinstance(body, dict) else 0,
                "relationships": len(body.get("relationships", [])) if isinstance(body, dict) else 0,
                "chunks": len(body.get("chunks", [])) if isinstance(body, dict) else 0,
                "answer_excerpt": clip(payload.get("answer", ""), 700),
            }
        )
    return rows


def render_json(title: str, value: object) -> str:
    return (
        f"<details><summary>{html_escape(title)}</summary>"
        f"<pre>{html_escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>"
    )


def main() -> None:
    args = parse_args()
    baseline_metrics = read_json(Path(args.baseline_metrics), {}) or {}
    pilot_metrics = read_json(Path(args.pilot_metrics), {}) or {}
    baseline_stats = read_json(Path(args.baseline_stats), {}) or {}
    pilot_stats = read_json(Path(args.pilot_stats), {}) or {}
    baseline_query = read_json(Path(args.baseline_query), {}) or {}
    pilot_query = read_json(Path(args.pilot_query), {}) or {}
    pilot_manifest = read_json(Path(args.pilot_manifest), {}) or {}
    prompt_text = Path(args.prompt).read_text(encoding="utf-8") if Path(args.prompt).exists() else ""
    baseline_examples = run_examples(Path(args.baseline_working_dir), preferred_doc="16-175229")
    pilot_examples = run_examples(Path(args.pilot_working_dir))

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LightRAG Patent Prompt Pilot Compare</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f7f8fa; }}
    header {{ padding: 28px 36px; background: #1c2733; color: #fff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    section {{ background: #fff; border: 1px solid #dbe2e8; border-radius: 8px; padding: 22px; margin: 18px 0; }}
    h1, h2, h3 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d8dee4; padding: 9px; vertical-align: top; text-align: left; }}
    th {{ background: #f0f3f6; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #101820; color: #e6edf3; padding: 14px; border-radius: 6px; overflow-x: auto; }}
    details {{ margin: 12px 0; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .note {{ border-left: 4px solid #607d3b; background: #fbfcf2; padding: 12px; border-radius: 6px; }}
    .empty {{ color: #6b7280; background: #fafafa; border: 1px dashed #ccd2d8; padding: 12px; border-radius: 6px; }}
    .cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
<header>
  <h1>LightRAG 3.1-3.4 Patent Prompt Pilot Compare</h1>
  <p>기존 200건 baseline과 특허 특화 프롬프트 20건 pilot을 같은 기준으로 비교합니다.</p>
</header>
<main>
  <section>
    <h2>Experiment Setup</h2>
    <p class="note">LLM 설정은 유지했습니다: <code>gpt-5.5</code> + <code>xhigh</code>. 실험 변수는 특허 특화 entity/relation extraction prompt입니다.</p>
    {table(runtime_rows(baseline_stats, pilot_stats))}
    {render_json("Pilot sampling manifest", pilot_manifest)}
  </section>

  <section>
    <h2>Patent-Specific Prompt</h2>
    <p>아래 YAML은 LightRAG text extraction mode에서 요구하는 <code>entity_types_guidance</code>와 <code>entity_extraction_examples</code>를 모두 포함합니다.</p>
    <pre>{html_escape(prompt_text[:18000])}</pre>
  </section>

  <section>
    <h2>Quantitative Metrics</h2>
    <p class="note">metadata relation은 특허번호, IPC/CPC, 국가/법적상태, 날짜, 분류, 출원인/권리자 관계를 포함하도록 휴리스틱으로 분류했습니다. technical relation은 회로/메모리/연산/아키텍처/제어/최적화 등 기술 관계 키워드와 entity type을 기준으로 분류했습니다.</p>
    {table(metric_rows(baseline_metrics, pilot_metrics))}
    <div class="cols">
      <div>
        <h3>Baseline top hubs</h3>
        {table(baseline_metrics.get("top_hubs", [])[:10] if isinstance(baseline_metrics.get("top_hubs"), list) else [])}
      </div>
      <div>
        <h3>Pilot top hubs</h3>
        {table(pilot_metrics.get("top_hubs", [])[:10] if isinstance(pilot_metrics.get("top_hubs"), list) else [])}
      </div>
    </div>
    {render_json("Baseline metrics raw", baseline_metrics)}
    {render_json("Pilot metrics raw", pilot_metrics)}
  </section>

  <section>
    <h2>3.1 R/P/D Examples</h2>
    <div class="cols">
      <div>
        <h3>Baseline R/P/D sample: {html_escape(baseline_examples.get("doc_id"))}</h3>
        {render_json("R extracted entities/relations", {"entities": baseline_examples.get("r_entities"), "relations": baseline_examples.get("r_relations")})}
        {render_json("P key-value profiles", {"entities": baseline_examples.get("p_entity_profiles"), "relations": baseline_examples.get("p_relation_profiles")})}
        {render_json("D dedupe examples", {"entities": baseline_examples.get("d_entity_examples"), "relations": baseline_examples.get("d_relation_examples")})}
      </div>
      <div>
        <h3>Pilot R/P/D sample: {html_escape(pilot_examples.get("doc_id"))}</h3>
        {render_json("R extracted entities/relations", {"entities": pilot_examples.get("r_entities"), "relations": pilot_examples.get("r_relations")})}
        {render_json("P key-value profiles", {"entities": pilot_examples.get("p_entity_profiles"), "relations": pilot_examples.get("p_relation_profiles")})}
        {render_json("D dedupe examples", {"entities": pilot_examples.get("d_entity_examples"), "relations": pilot_examples.get("d_relation_examples")})}
      </div>
    </div>
  </section>

  <section>
    <h2>3.2-3.3 Retrieval / Answer Compare</h2>
    <h3>Baseline query</h3>
    {table(query_summary(baseline_query))}
    <h3>Pilot query</h3>
    {table(query_summary(pilot_query))}
  </section>

  <section>
    <h2>200-Document Rerun Decision</h2>
    <p class="note">목표는 metadata relation ratio 25% 이하, technical relation ratio 40% 이상, excluded entity ratio 10% 이하입니다. 이 세 지표와 query answer 품질이 통과하면 200건을 같은 프롬프트와 <code>gpt-5.5 + xhigh</code>로 재실행합니다.</p>
  </section>
</main>
</body>
</html>
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
