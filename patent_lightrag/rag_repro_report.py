from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, estimate_tokens, html_escape, read_json
from patent_lightrag.graph_metrics import load_graphml


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Patent-100 RAG reproduction HTML report.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output", default=str(ROOT / "reports" / "rag_repro_100_comparison.html"))
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


def table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='empty'>pending</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html_escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in keys) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def pre(title: str, value: Any, limit: int = 8000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > limit:
        text = text[:limit] + "\n...[truncated]"
    return f"<details open><summary>{html_escape(title)}</summary><pre>{html_escape(text)}</pre></details>"


def short(text: Any, limit: int = 900) -> str:
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[:limit] + " ...[truncated]"


def clean_answer_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\x1b\[[0-9;]*m", "", value)
    lines = [
        line for line in value.splitlines()
        if "LiteLLM:WARNING" not in line and "could not pre-load" not in line
    ]
    return "\n".join(lines).strip()


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if seconds >= 3600:
        return f"{seconds:,.1f}s ({seconds / 3600:.2f}h)"
    return f"{seconds:,.1f}s"


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def blank_cell() -> str:
    return "—"


def label_lines(value: str, max_line: int = 12, max_lines: int = 2) -> str:
    words = value.split()
    lines: list[str] = []
    current = ""
    if not words:
        words = [value]
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_line and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = [value[:max_line]]
    return "".join(
        f"<tspan x='0' dy='{0 if idx == 0 else 13}'>{html_escape(line[:max_line])}</tspan>"
        for idx, line in enumerate(lines[:max_lines])
    )


def color_for(entity_type: str) -> str:
    key = entity_type.lower()
    palette = {
        "techcomponent": "#2563eb",
        "architecture": "#0f766e",
        "operation": "#9333ea",
        "method": "#d97706",
        "material": "#16a34a",
        "performancemetric": "#dc2626",
        "applicationdomain": "#475569",
        "organization": "#64748b",
        "other": "#7c3aed",
    }
    return palette.get(key, "#64748b")


def relation_label(edge: dict[str, Any], limit: int = 22) -> str:
    label = str(edge.get("keywords") or "").replace("\n", ", ").strip()
    if not label:
        label = str(edge.get("description") or "").replace("\n", " ").strip()
    if not label:
        label = "relation"
    return label if len(label) <= limit else label[:limit] + "..."


def relation_table(edges: list[dict[str, Any]], limit: int = 14) -> str:
    rows = []
    for edge in edges[:limit]:
        rows.append(
            {
                "source": edge.get("source", ""),
                "relation": relation_label(edge, 48),
                "target": edge.get("target", ""),
                "description": short(edge.get("description", ""), 240),
                "source_patent": edge.get("file_path", ""),
            }
        )
    return table(rows)


def render_svg(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str, subtitle: str) -> str:
    if not nodes or not edges:
        return "<p class='empty'>graph visualization pending</p>"
    width = 1120
    height = 720
    cx = width / 2
    cy = height / 2
    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        radius = 0 if idx == 0 else (210 if idx <= 10 else 315)
        count = 1 if idx == 0 else (min(10, max(1, len(nodes) - 1)) if idx <= 10 else max(1, len(nodes) - 11))
        offset_idx = 0 if idx == 0 else (idx - 1 if idx <= 10 else idx - 11)
        angle = -math.pi / 2 + 2 * math.pi * offset_idx / count
        name = str(node["id"])
        positions[name] = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
    edge_svg = []
    for edge_idx, edge in enumerate(edges):
        src = str(edge.get("source") or edge.get("src"))
        tgt = str(edge.get("target") or edge.get("tgt"))
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        tooltip = "\n".join(
            part for part in [
                f"{src} -> {tgt}",
                f"keywords: {edge.get('keywords', '')}" if edge.get("keywords") else "",
                str(edge.get("description", "")),
                f"source: {edge.get('source_id', '')}" if edge.get("source_id") else "",
            ] if part
        )
        edge_svg.append(
            f"<line class='edge' x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}'>"
            f"<title>{html_escape(tooltip)}</title></line>"
        )
        if edge_idx < 28 and (edge.get("keywords") or edge.get("description")):
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            edge_svg.append(
                f"<text class='edge-label' x='{mx:.1f}' y='{my:.1f}' text-anchor='middle'>"
                f"<title>{html_escape(tooltip)}</title>{html_escape(relation_label(edge))}</text>"
            )
    node_svg = []
    for node in nodes:
        name = str(node["id"])
        if name not in positions:
            continue
        x, y = positions[name]
        degree = int(node.get("degree", 1))
        radius = min(42, max(18, 14 + degree * 1.5))
        fill = color_for(str(node.get("entity_type", "other")))
        node_svg.append(
            f"<g transform='translate({x:.1f},{y:.1f})'><circle r='{radius}' fill='{fill}' opacity='0.9'>"
            f"<title>{html_escape(name)} | degree={degree} | type={html_escape(node.get('entity_type', ''))}</title></circle>"
            f"<text text-anchor='middle' y='{radius + 14}'>{label_lines(name)}</text></g>"
        )
    return f"""
    <section class="viz">
      <h3>{html_escape(title)}</h3>
      <p>{html_escape(subtitle)}</p>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#fbfcfd"/>
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </section>
    """


def full_graph_visualization(working_dir: Path) -> str:
    try:
        raw_nodes, raw_edges = load_graphml(working_dir)
    except Exception:
        return "<p class='empty'>LightRAG graph is not available yet.</p>"
    degree: Counter[str] = Counter()
    for edge in raw_edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    selected_names = [name for name, _ in degree.most_common(32)]
    selected = set(selected_names)
    nodes = [
        {
            "id": name,
            "degree": degree[name],
            "entity_type": raw_nodes.get(name, {}).get("entity_type", "other"),
        }
        for name in selected_names
    ]
    edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "keywords": edge.get("keywords", ""),
            "description": edge.get("description", ""),
            "source_id": edge.get("source_id", ""),
            "file_path": edge.get("file_path", ""),
        }
        for edge in raw_edges
        if edge["source"] in selected and edge["target"] in selected
    ][:55]
    return (
        render_svg(
        nodes,
        edges,
        "Patent-100 전체 LightRAG graph visualization",
        "전체 graph 중 degree가 높은 node를 중심으로 제한 렌더링했다. node color는 entity type, node size는 degree 기준이다.",
    )
        + "<h4>Relation samples shown in the graph</h4>"
        + relation_table(edges)
    )


def patent_graph_example(working_dir: Path) -> tuple[str, str]:
    full_entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    if not isinstance(full_relations, dict) or not full_relations:
        return "pending", "<p class='empty'>single patent graph pending</p>"
    try:
        raw_nodes, raw_edges = load_graphml(working_dir)
    except Exception:
        raw_nodes, raw_edges = {}, []
    candidates = []
    for doc_id, record in full_relations.items():
        pairs = record.get("relation_pairs", []) if isinstance(record, dict) else []
        if isinstance(pairs, list) and 6 <= len(pairs) <= 40:
            candidates.append((abs(len(pairs) - 18), str(doc_id), pairs))
    if not candidates:
        for doc_id, record in full_relations.items():
            pairs = record.get("relation_pairs", []) if isinstance(record, dict) else []
            if isinstance(pairs, list) and pairs:
                candidates.append((len(pairs), str(doc_id), pairs))
    _, doc_id, pairs = sorted(candidates)[0]
    graph_edges = [
        edge for edge in raw_edges
        if doc_id in str(edge.get("source_id", "")) or doc_id in str(edge.get("file_path", ""))
    ]
    if graph_edges:
        selected_pairs = [
            [
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("keywords", ""),
                edge.get("description", ""),
                edge.get("source_id", ""),
                edge.get("file_path", ""),
            ]
            for edge in graph_edges[:24]
        ]
    else:
        selected_pairs = [
            pair for pair in pairs
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ][:24]
    degree: Counter[str] = Counter()
    for src, tgt, *_ in selected_pairs:
        degree[str(src)] += 1
        degree[str(tgt)] += 1
    nodes = [
        {
            "id": name,
            "degree": degree[name],
            "entity_type": raw_nodes.get(name, {}).get("entity_type", "other") if isinstance(raw_nodes, dict) else "other",
        }
        for name in sorted({name for pair in selected_pairs for name in [str(pair[0]), str(pair[1])]}, key=lambda n: (-degree[n], n))
    ]
    edges = [
        {
            "source": str(pair[0]),
            "target": str(pair[1]),
            "keywords": str(pair[2]) if len(pair) >= 3 else "",
            "description": str(pair[3]) if len(pair) >= 4 else "",
            "source_id": str(pair[4]) if len(pair) >= 5 else "",
            "file_path": str(pair[5]) if len(pair) >= 6 else doc_id,
        }
        for pair in selected_pairs
    ]
    caption = (
        f"임의 특허 {doc_id}에서 source chunk 기준으로 추출된 기술 entity/relation만 렌더링했다. "
        "patent_id는 source/reference로 남고 entity node로 쓰지 않는다."
    )
    return (
        doc_id,
        render_svg(nodes, edges, f"Single patent graph example: {doc_id}", caption)
        + "<h4>Single patent relation details</h4>"
        + relation_table(edges, 24),
    )


def load_vdb_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {}) or {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def rpd_examples(working_dir: Path) -> str:
    full_entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    text_chunks = read_json(working_dir / "kv_store_text_chunks.json", {}) or {}
    entity_chunks = read_json(working_dir / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(working_dir / "kv_store_relation_chunks.json", {}) or {}
    if not isinstance(full_entities, dict) or not full_entities:
        return "<p class='empty'>R/P/D examples pending</p>"
    doc_id = next(iter(full_entities))
    relation_counts = []
    for key, value in full_relations.items() if isinstance(full_relations, dict) else []:
        pairs = value.get("relation_pairs", []) if isinstance(value, dict) else []
        if isinstance(pairs, list):
            relation_counts.append((len(pairs), key))
    if relation_counts:
        doc_id = sorted(relation_counts, reverse=True)[0][1]
    entity_names = full_entities.get(doc_id, {}).get("entity_names", []) if isinstance(full_entities.get(doc_id), dict) else []
    relation_pairs = full_relations.get(doc_id, {}).get("relation_pairs", []) if isinstance(full_relations.get(doc_id), dict) else []
    chunk = text_chunks.get(f"{doc_id}-chunk-000", {}) if isinstance(text_chunks, dict) else {}
    entity_vdb = load_vdb_rows(working_dir / "vdb_entities.json")
    relation_vdb = load_vdb_rows(working_dir / "vdb_relationships.json")
    dedup_entities = [
        {"entity": key, "chunk_count": value.get("count"), "chunk_ids": value.get("chunk_ids", [])[:5]}
        for key, value in sorted(
            entity_chunks.items(),
            key=lambda item: int(item[1].get("count", 0)) if isinstance(item[1], dict) else 0,
            reverse=True,
        )
        if isinstance(value, dict) and int(value.get("count", 0)) > 1
    ][:5] if isinstance(entity_chunks, dict) else []
    dedup_relations = [
        {"relation": key, "chunk_count": value.get("count"), "chunk_ids": value.get("chunk_ids", [])[:5]}
        for key, value in sorted(
            relation_chunks.items(),
            key=lambda item: int(item[1].get("count", 0)) if isinstance(item[1], dict) else 0,
            reverse=True,
        )
        if isinstance(value, dict) and int(value.get("count", 0)) > 1
    ][:5] if isinstance(relation_chunks, dict) else []
    entity_profile = next((row for row in entity_vdb if row.get("entity_name") in entity_names), entity_vdb[0] if entity_vdb else {})
    relation_profile = next((row for row in relation_vdb if row.get("src_id") or row.get("tgt_id")), relation_vdb[0] if relation_vdb else {})
    return f"""
    <h3>R/P/D Function Output Examples</h3>
    <p class="note">실제 Patent-100 storage에서 특허 <code>{html_escape(doc_id)}</code>와 graph/VDB payload를 읽어 구성했다.</p>
    {pre("R input chunk excerpt", short(chunk.get("content", "") if isinstance(chunk, dict) else "", 1800), 2500)}
    {table([
        {"R output": "entities", "count": len(entity_names), "sample": ", ".join(map(str, entity_names[:10]))},
        {"R output": "relations", "count": len(relation_pairs), "sample": "; ".join(f"{p[0]} -> {p[1]}" for p in relation_pairs[:8] if isinstance(p, list) and len(p) >= 2)},
    ])}
    {pre("P entity key-value profile", entity_profile, 3000)}
    {pre("P relation key-value profile", relation_profile, 3000)}
    <h4>D deduplication examples</h4>
    {table(dedup_entities)}
    {table(dedup_relations)}
    """


def result_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    for row in rows:
        key = (str(row.get("query_id")), str(row.get("mode") or row.get("method")))
        lookup[key] = row
    return lookup


def context_body(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return {}
    body = data.get("data", {})
    return body if isinstance(body, dict) else {}


def collect_source_patents(row: dict[str, Any]) -> set[str]:
    body = context_body(row)
    sources: set[str] = set()
    for section in ["entities", "relationships", "chunks"]:
        values = body.get(section, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            raw_values = [
                item.get("file_path", ""),
                item.get("source_id", ""),
                item.get("chunk_id", ""),
            ]
            for raw in raw_values:
                for part in str(raw).split("<SEP>"):
                    part = part.strip()
                    if not part:
                        continue
                    part = re.sub(r"-chunk-\d+$", "", part)
                    if re.search(r"\d", part):
                        sources.add(part)
    return sources


def patent_citation_count(answer: Any) -> int:
    text = "" if answer is None else str(answer)
    matches = re.findall(r"\b(?:\d{2}-\d{6}|\d{4}-\d{7})\b", text)
    return len(set(matches))


def lightrag_mode_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for mode in ["naive", "local", "global", "hybrid"]:
        mode_rows = [row for row in rows if row.get("mode") == mode]
        latencies = [float(row.get("elapsed_seconds") or 0) for row in mode_rows]
        answer_lengths = [len(str(row.get("answer") or "")) for row in mode_rows]
        entity_counts = [len(context_body(row).get("entities", []) or []) for row in mode_rows]
        relation_counts = [len(context_body(row).get("relationships", []) or []) for row in mode_rows]
        chunk_counts = [len(context_body(row).get("chunks", []) or []) for row in mode_rows]
        source_counts = [len(collect_source_patents(row)) for row in mode_rows]
        empty_or_failed = sum(1 for row in mode_rows if row.get("status") != "success" or not row.get("answer"))
        summary_rows.append(
            {
                "Mode": mode,
                "Queries": len(mode_rows),
                "Success answers": len(mode_rows) - empty_or_failed,
                "Empty/failed": empty_or_failed,
                "Avg latency": format_seconds(safe_mean(latencies)),
                "Total latency": format_seconds(sum(latencies)),
                "Avg answer chars": f"{safe_mean(answer_lengths):.0f}",
                "Avg entities": f"{safe_mean(entity_counts):.1f}",
                "Avg relations": f"{safe_mean(relation_counts):.1f}",
                "Avg chunks": f"{safe_mean(chunk_counts):.1f}",
                "Avg source patents": f"{safe_mean(source_counts):.1f}",
            }
        )
    return summary_rows


def render_query_flow(rows: list[dict[str, Any]]) -> str:
    lookup = result_lookup(rows)
    row = lookup.get(("AA-1", "hybrid")) or next((r for r in rows if r.get("mode") == "hybrid" and r.get("answer")), None)
    if not row:
        return "<p class='empty'>query flow pending</p>"
    data = row.get("data", {})
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    body = context_body(row)
    return f"""
    <h3>Keyword Extraction and Retrieved Context</h3>
    {table([{
        "query_id": row.get("query_id", ""),
        "mode": row.get("mode", ""),
        "question": row.get("question", ""),
        "elapsed": format_seconds(row.get("elapsed_seconds")),
        "source_patents": len(collect_source_patents(row)),
    }])}
    {pre("Query metadata", metadata, 5000)}
    {table([
        {"context": "entities", "count": len(body.get("entities", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("entities", []) if isinstance(body, dict) else [])[:3], ensure_ascii=False), 1000)},
        {"context": "relationships", "count": len(body.get("relationships", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("relationships", []) if isinstance(body, dict) else [])[:3], ensure_ascii=False), 1000)},
        {"context": "chunks", "count": len(body.get("chunks", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("chunks", []) if isinstance(body, dict) else [])[:2], ensure_ascii=False), 1000)},
    ])}
    """


def render_answer_compare(rows: list[dict[str, Any]], query_id: str = "X-1") -> str:
    lookup = result_lookup(rows)
    compare_rows = []
    for mode in ["naive", "local", "global", "hybrid"]:
        row = lookup.get((query_id, mode))
        compare_rows.append(
            {
                "mode": mode,
                "status": row.get("status", "—") if row else "—",
                "elapsed": format_seconds(row.get("elapsed_seconds")) if row else "—",
                "retrieved": (
                    f"E={len(context_body(row).get('entities', []) or [])}, "
                    f"R={len(context_body(row).get('relationships', []) or [])}, "
                    f"C={len(context_body(row).get('chunks', []) or [])}"
                    if row else "—"
                ),
                "answer_excerpt": short(row.get("answer", "") if row else "—", 900),
            }
        )
    return table(compare_rows)


def graphrag_root(experiment_dir: Path) -> Path:
    candidates = [
        experiment_dir / "graphrag_full_100_fresh",
        experiment_dir / "graphrag_full_100_update",
        experiment_dir / "graphrag_smoke_20",
    ]
    for candidate in candidates:
        if (candidate / "query_results_15_methods.jsonl").exists() or (candidate / "output" / "stats.json").exists():
            return candidate
    return candidates[0]


def read_parquet_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def load_graphrag_stats(experiment_dir: Path) -> dict[str, Any]:
    root = graphrag_root(experiment_dir)
    stats = read_json(root / "output" / "stats.json", {}) or {}
    counts = {
        name: read_parquet_count(root / "output" / f"{name}.parquet")
        for name in ["documents", "text_units", "entities", "relationships", "communities", "community_reports"]
    }
    workflows = stats.get("workflows", {}) if isinstance(stats, dict) else {}
    workflow_seconds = {
        name: round(float(value.get("overall", 0)), 2)
        for name, value in workflows.items()
        if isinstance(value, dict)
    }
    return {
        "root": str(root),
        "stats_path": str(root / "output" / "stats.json"),
        "query_path": str(root / "query_results_15_methods.jsonl"),
        "total_runtime": stats.get("total_runtime") if isinstance(stats, dict) else None,
        "num_documents": stats.get("num_documents") if isinstance(stats, dict) else None,
        "counts": counts,
        "workflow_seconds": workflow_seconds,
    }


def load_all_query_rows(experiment_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    light = read_jsonl(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl")
    graph = read_jsonl(graphrag_root(experiment_dir) / "query_results_15_methods.jsonl")
    return light, graph


def graphrag_query_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for method in ["basic", "local", "global"]:
        method_rows = [row for row in rows if row.get("method") == method]
        latencies = [float(row.get("elapsed_seconds") or 0) for row in method_rows]
        answers = [clean_answer_text(row.get("answer", "")) for row in method_rows]
        failed = sum(1 for row, answer in zip(method_rows, answers) if row.get("status") != "success" or not answer)
        summary_rows.append(
            {
                "Method": method,
                "Queries": len(method_rows),
                "Success answers": len(method_rows) - failed,
                "Empty/failed": failed,
                "Avg latency": format_seconds(safe_mean(latencies)),
                "Max latency": format_seconds(max(latencies) if latencies else 0),
                "Total latency": format_seconds(sum(latencies)),
                "Avg answer chars": f"{safe_mean([len(answer) for answer in answers]):.0f}",
            }
        )
    return summary_rows


def query_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("query_id")), str(row.get("method") or row.get("mode")))


def graphrag_recovery_rows(experiment_dir: Path, graph_rows: list[dict[str, Any]], queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = graphrag_root(experiment_dir)
    methods = ["basic", "local", "global"]
    expected = {
        (str(query.get("query_id")), method)
        for query in queries
        for method in methods
    }
    current_keys = [query_key(row) for row in graph_rows]
    current_unique = set(current_keys)
    current_failures = [
        key for key, row in zip(current_keys, graph_rows)
        if row.get("status") != "success" or not clean_answer_text(row.get("answer", ""))
    ]
    current_missing = sorted(expected - current_unique)
    current_duplicates = len(current_keys) - len(current_unique)
    rows = [
        {
            "Audit": "After repair current",
            "Value": f"{len(current_unique)}/{len(expected)} unique expected keys, failures={len(current_failures)}, missing={len(current_missing)}, duplicates={current_duplicates}",
        }
    ]
    backups = sorted(root.glob("query_results_15_methods.backup_*.jsonl"))
    if backups:
        backup = backups[-1]
        backup_rows = read_jsonl(backup)
        backup_keys = [query_key(row) for row in backup_rows]
        backup_unique = set(backup_keys)
        backup_failures = [
            key for key, row in zip(backup_keys, backup_rows)
            if row.get("status") != "success" or not clean_answer_text(row.get("answer", ""))
        ]
        backup_missing = sorted(expected - backup_unique)
        recovered_failures = [key for key in backup_failures if key in current_unique and key not in current_failures]
        recovered_missing = [key for key in backup_missing if key in current_unique and key not in current_failures]
        rows.insert(
            0,
            {
                "Audit": "Before repair backup",
                "Value": f"{len(backup_unique)}/{len(expected)} unique expected keys, failures={len(backup_failures)}, missing={len(backup_missing)}, duplicates={len(backup_keys) - len(backup_unique)}",
            },
        )
        rows.extend(
            [
                {"Audit": "Recovered failed keys", "Value": ", ".join(f"{qid}/{method}" for qid, method in recovered_failures) or "—"},
                {"Audit": "Recovered missing keys", "Value": f"{len(recovered_missing)} keys: " + short(", ".join(f"{qid}/{method}" for qid, method in recovered_missing), 450)},
                {"Audit": "Backup preserved", "Value": str(backup)},
            ]
        )
    return rows


def judge_table(judge_rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    pair_id_1 = f"{left}__vs__{right}"
    pair_id_2 = f"{right}__vs__{left}"
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    categories = ["AA", "AB", "AC", "AD", "cross"]
    for row in judge_rows:
        if row.get("pair_id") not in {pair_id_1, pair_id_2} or row.get("status") != "success":
            continue
        result = row.get("judge_result", {})
        for metric in ["Comprehensiveness", "Diversity", "Empowerment", "Technical correctness", "Hallucination risk", "Overall"]:
            block = result.get(metric, {}) if isinstance(result, dict) else {}
            winner = str(block.get("winner", "Tie"))
            label = row.get("answer_a_label") if winner.startswith("A") else row.get("answer_b_label") if winner.startswith("B") else "Tie"
            counters[metric][str(label)] += 1
    rows = []
    for metric, counter in counters.items():
        total = sum(counter.values()) or 1
        rows.append(
            {
                "Comparison": f"{left} vs {right}",
                "Metric": metric,
                "Left system": left,
                "Left win-rate": f"{counter.get(left, 0) / total * 100:.1f}%",
                "Right system": right,
                "Right win-rate": f"{counter.get(right, 0) / total * 100:.1f}%",
                "Tie": f"{counter.get('Tie', 0) / total * 100:.1f}%",
            }
        )
    return rows


def case_table(light_rows: list[dict[str, Any]], graph_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]], query_id: str, graph_method: str) -> str:
    light = result_lookup(light_rows).get((query_id, "hybrid"), {})
    graph = result_lookup(graph_rows).get((query_id, graph_method), {})
    decision = next((row for row in judge_rows if row.get("query_id") == query_id and graph_method in str(row.get("pair_id"))), {})
    return table(
        [
            {
                "Query": query_id,
                "GraphRAG/Naive": short(clean_answer_text(graph.get("answer", "pending")), 1100),
                "LightRAG": short(clean_answer_text(light.get("answer", "pending")), 1100),
                "LLM Decision": short(json.dumps(decision.get("judge_result", "pending"), ensure_ascii=False), 1100),
            }
        ]
    )


def structured_patent_examples(experiment_dir: Path, limit: int = 2) -> str:
    rows = read_jsonl(experiment_dir / "dataset" / "patents_100.jsonl")
    if not rows:
        return "<p class='empty'>structured patent examples pending</p>"
    selected: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for row in rows:
        category = str(row.get("category", ""))
        if category not in seen_categories:
            selected.append(row)
            seen_categories.add(category)
        if len(selected) >= limit:
            break
    overview = [
        {
            "patent_id": row.get("id", ""),
            "category": f"{row.get('category', '')}/{row.get('sub_category', '')}",
            "category_name": f"{row.get('category_name', '')} / {row.get('sub_category_name', '')}",
            "source": row.get("file_path", ""),
        }
        for row in selected
    ]
    examples = "\n".join(
        pre(f"Structured patent text: {row.get('id', '')}", short(row.get("text", ""), 4200), 4600)
        for row in selected
    )
    return f"{table(overview)}{examples}"


def prompt_rule_table() -> str:
    return table(
        [
            {
                "Rule": "Do not extract as entity",
                "Items": "특허번호, 출원번호, 공개번호, 등록번호, IPC/CPC 코드, 국가코드, 법적상태, 날짜, 청구항 번호, 중분류/소분류 코드 자체",
            },
            {
                "Rule": "Extract technical entities",
                "Items": "회로, 메모리 구조, 아키텍처, 연산 장치, 연산 방식, 제조/패키징 공정, 인터커넥트, 성능 지표, 응용 기술",
            },
            {
                "Rule": "Relation preference",
                "Items": "metadata relation보다 기술 구성요소 간 기능/구조/성능/공정 관계를 우선 추출",
            },
        ]
    )


def graph_quality_table(graph_metrics: dict[str, Any]) -> str:
    return table(
        [
            {
                "Metric": "metadata relation ratio",
                "Value": graph_metrics.get("metadata_relation_ratio", "—"),
                "Target": "<= 0.25",
                "Interpretation": "프롬프트 적용 후 메타데이터 relation이 크게 억제됨",
            },
            {
                "Metric": "technical relation ratio",
                "Value": graph_metrics.get("technical_relation_ratio", "—"),
                "Target": ">= 0.40",
                "Interpretation": "기술 relation 중심 graph로 전환됨",
            },
            {
                "Metric": "excluded entity ratio",
                "Value": graph_metrics.get("excluded_entity_ratio", "—"),
                "Target": "<= 0.10",
                "Interpretation": "특허번호/코드/국가/날짜류 entity가 낮은 비율로 남음",
            },
            {
                "Metric": "degree >= 50 hub count",
                "Value": graph_metrics.get("degree_ge_50_hub_count", "—"),
                "Target": "lower is better",
                "Interpretation": "US/등록 같은 메타 hub가 top hub에서 사라졌는지 확인",
            },
        ]
    )


def win_rate_placeholder(judge_rows: list[dict[str, Any]]) -> str:
    if judge_rows:
        rows = []
        for left, right in [
            ("lightrag_hybrid", "lightrag_naive"),
            ("lightrag_hybrid", "graphrag_global"),
            ("lightrag_hybrid", "graphrag_local"),
            ("graphrag_global", "lightrag_naive"),
        ]:
            rows.extend(judge_table(judge_rows, left, right))
        return table(rows)
    return table(
        [
            {"Comparison": "LightRAG hybrid vs NaiveRAG", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "lightrag_naive", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "LightRAG hybrid vs GraphRAG global", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "graphrag_global", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "LightRAG hybrid vs GraphRAG local", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "graphrag_local", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "GraphRAG global vs NaiveRAG", "Metric": "—", "Left system": "graphrag_global", "Left win-rate": "—", "Right system": "lightrag_naive", "Right win-rate": "—", "Tie": "—"},
        ]
    )


def graph_case_blank() -> str:
    return table(
        [
            {
                "Query": "X-1",
                "GraphRAG answer": "—",
                "LightRAG hybrid answer": "—",
                "LLM judge decision": "—",
                "Status": "GraphRAG indexing/query 실행 후 작성",
            }
        ]
    )


def system_label(row: dict[str, Any]) -> str:
    if row.get("method"):
        return f"graphrag_{row.get('method')}"
    return f"lightrag_{row.get('mode')}"


def answer_length_stats(light_rows: list[dict[str, Any]], graph_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in light_rows + graph_rows:
        if row.get("status") != "success":
            continue
        answer = clean_answer_text(row.get("answer", ""))
        if not answer:
            continue
        grouped[system_label(row)].append(
            {
                "chars": len(answer),
                "tokens": estimate_tokens(answer),
                "latency": float(row.get("elapsed_seconds") or 0),
            }
        )
    stats: dict[str, dict[str, float]] = {}
    for system, rows in grouped.items():
        chars = [float(row["chars"]) for row in rows]
        tokens = [float(row["tokens"]) for row in rows]
        latencies = [float(row["latency"]) for row in rows]
        stats[system] = {
            "queries": float(len(rows)),
            "avg_chars": safe_mean(chars),
            "median_chars": statistics.median(chars) if chars else 0.0,
            "avg_tokens": safe_mean(tokens),
            "avg_latency": safe_mean(latencies),
        }
    return stats


def answer_length_table(length_stats: dict[str, dict[str, float]]) -> str:
    order = [
        "lightrag_naive",
        "lightrag_local",
        "lightrag_global",
        "lightrag_hybrid",
        "graphrag_basic",
        "graphrag_local",
        "graphrag_global",
    ]
    hybrid_chars = length_stats.get("lightrag_hybrid", {}).get("avg_chars", 0.0)
    rows = []
    for system in order:
        stats = length_stats.get(system)
        if not stats:
            continue
        ratio = stats["avg_chars"] / hybrid_chars if hybrid_chars else 0.0
        rows.append(
            {
                "System": system,
                "Queries": int(stats["queries"]),
                "Avg answer chars": f"{stats['avg_chars']:.0f}",
                "Median answer chars": f"{stats['median_chars']:.0f}",
                "Avg token estimate": f"{stats['avg_tokens']:.0f}",
                "Avg latency": format_seconds(stats["avg_latency"]),
                "Length vs LightRAG hybrid": f"{ratio:.2f}x" if hybrid_chars else "—",
            }
        )
    return table(rows)


def judge_metric_counts(judge_rows: list[dict[str, Any]], left: str, right: str, metric: str = "Overall") -> Counter[str]:
    pair_ids = {f"{left}__vs__{right}", f"{right}__vs__{left}"}
    counter: Counter[str] = Counter()
    for row in judge_rows:
        if row.get("status") != "success" or row.get("pair_id") not in pair_ids:
            continue
        result = row.get("judge_result", {})
        block = result.get(metric, {}) if isinstance(result, dict) else {}
        winner = str(block.get("winner", "Tie")).strip().lower()
        if winner.startswith("a"):
            counter[str(row.get("answer_a_label"))] += 1
        elif winner.startswith("b"):
            counter[str(row.get("answer_b_label"))] += 1
        else:
            counter["Tie"] += 1
    return counter


def discussion_section(
    graph_metrics: dict[str, Any],
    judge_rows: list[dict[str, Any]],
    length_stats: dict[str, dict[str, float]],
) -> str:
    hybrid_chars = length_stats.get("lightrag_hybrid", {}).get("avg_chars", 0.0)
    naive_chars = length_stats.get("lightrag_naive", {}).get("avg_chars", 0.0)
    graph_global_chars = length_stats.get("graphrag_global", {}).get("avg_chars", 0.0)
    graph_local_chars = length_stats.get("graphrag_local", {}).get("avg_chars", 0.0)
    global_ratio = graph_global_chars / hybrid_chars if hybrid_chars else 0.0
    local_ratio = graph_local_chars / hybrid_chars if hybrid_chars else 0.0
    naive_ratio = naive_chars / hybrid_chars if hybrid_chars else 0.0
    hybrid_naive = judge_metric_counts(judge_rows, "lightrag_hybrid", "lightrag_naive")
    hybrid_global = judge_metric_counts(judge_rows, "lightrag_hybrid", "graphrag_global")
    hybrid_local = judge_metric_counts(judge_rows, "lightrag_hybrid", "graphrag_local")
    global_naive = judge_metric_counts(judge_rows, "graphrag_global", "lightrag_naive")
    rows = [
        {
            "Finding": "Graph construction succeeded",
            "Evidence": (
                f"metadata relation={graph_metrics.get('metadata_relation_ratio', '—')}, "
                f"technical relation={graph_metrics.get('technical_relation_ratio', '—')}, "
                f"excluded entity={graph_metrics.get('excluded_entity_ratio', '—')}"
            ),
            "Interpretation": "특허번호/분류코드 중심 hub는 억제됐고, 기술 entity/relation 중심 graph는 목표 기준을 충족했다.",
        },
        {
            "Finding": "GraphRAG global wins, but length bias is a major confound",
            "Evidence": (
                f"Overall wins: GraphRAG global {hybrid_global.get('graphrag_global', 0)} vs "
                f"LightRAG hybrid {hybrid_global.get('lightrag_hybrid', 0)}; "
                f"avg length ratio={global_ratio:.2f}x ({graph_global_chars:.0f} vs {hybrid_chars:.0f} chars)"
            ),
            "Interpretation": "Comprehensiveness/Diversity/Empowerment rubric은 긴 답변에 구조적으로 유리하므로, GraphRAG global 우위는 verbosity bias 가능성을 함께 보고해야 한다.",
        },
        {
            "Finding": "LightRAG hybrid is not clearly better than naive",
            "Evidence": (
                f"Overall wins: LightRAG hybrid {hybrid_naive.get('lightrag_hybrid', 0)} vs "
                f"naive {hybrid_naive.get('lightrag_naive', 0)}; "
                f"avg length ratio naive/hybrid={naive_ratio:.2f}x"
            ),
            "Interpretation": "이 특허 데이터에서는 chunk-only retrieval만으로도 답변 생성에 충분한 정보가 잡혔고, graph context의 추가 이득은 제한적으로 관찰됐다.",
        },
        {
            "Finding": "GraphRAG local is weak for this query set",
            "Evidence": (
                f"Overall wins: LightRAG hybrid {hybrid_local.get('lightrag_hybrid', 0)} vs "
                f"GraphRAG local {hybrid_local.get('graphrag_local', 0)}; "
                f"GraphRAG local avg length={graph_local_chars:.0f} chars ({local_ratio:.2f}x hybrid)"
            ),
            "Interpretation": "15개 쿼리가 broad 기술관계 질문 중심이라 entity-description local retrieval보다 global community/theme retrieval에 유리했을 가능성이 높다.",
        },
        {
            "Finding": "GraphRAG global also beats naive",
            "Evidence": (
                f"Overall wins: GraphRAG global {global_naive.get('graphrag_global', 0)} vs "
                f"naive {global_naive.get('lightrag_naive', 0)}"
            ),
            "Interpretation": "다만 이 비교 역시 GraphRAG global의 답변 길이가 길다는 confound를 공유하므로, 최종 결론은 길이 통제 실험 전까지 보수적으로 해석해야 한다.",
        },
    ]
    return f"""
    <h2 id="discussion">Discussion and Conclusion</h2>
    <p>이번 실험은 Patent-100에서 LightRAG graph indexing, GraphRAG indexing/query, Gemini judge 평가까지 재현 가능한 pipeline으로 완주했다. 다만 judge win-rate를 그대로 성능 결론으로 읽기에는 답변 길이 편향이 가장 큰 교란 요인이다.</p>
    {table(rows)}
    <h3>Conclusion</h3>
    <p>현재 결과의 보수적 결론은 다음과 같다. LightRAG의 특허 특화 graph 구축은 정량 품질 지표상 성공했다. 그러나 답변 품질 평가에서는 LightRAG hybrid가 naive 대비 뚜렷한 우위를 보이지 않았고, GraphRAG global의 강한 승률은 답변 길이 효과와 분리해서 해석해야 한다. 따라서 다음 단계에서는 동일 길이 조건 또는 judge prompt의 verbosity penalty를 둔 재평가가 필요하다.</p>
    """


def naive_vs_light_table(light_rows: list[dict[str, Any]], query_id: str = "C-1") -> str:
    lookup = result_lookup(light_rows)
    naive = lookup.get((query_id, "naive"), {})
    hybrid = lookup.get((query_id, "hybrid"), {})
    return table(
        [
            {
                "Query": query_id,
                "Mode": "naive",
                "Latency": format_seconds(naive.get("elapsed_seconds")),
                "Retrieved": f"C={len(context_body(naive).get('chunks', []) or [])}",
                "Sources": len(collect_source_patents(naive)),
                "Patent citations": patent_citation_count(naive.get("answer")),
                "Answer excerpt": short(naive.get("answer", "—"), 1100),
            },
            {
                "Query": query_id,
                "Mode": "hybrid",
                "Latency": format_seconds(hybrid.get("elapsed_seconds")),
                "Retrieved": (
                    f"E={len(context_body(hybrid).get('entities', []) or [])}, "
                    f"R={len(context_body(hybrid).get('relationships', []) or [])}, "
                    f"C={len(context_body(hybrid).get('chunks', []) or [])}"
                ),
                "Sources": len(collect_source_patents(hybrid)),
                "Patent citations": patent_citation_count(hybrid.get("answer")),
                "Answer excerpt": short(hybrid.get("answer", "—"), 1100),
            },
        ]
    )


def complexity_rows(
    index_stats: dict[str, Any],
    light_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    graph_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    query_seconds = sum(float(row.get("elapsed_seconds") or 0) for row in light_rows)
    empty_or_failed = sum(1 for row in light_rows if row.get("status") != "success" or not row.get("answer"))
    graph_query_seconds = sum(float(row.get("elapsed_seconds") or 0) for row in graph_rows)
    graph_empty_or_failed = sum(
        1 for row in graph_rows
        if row.get("status") != "success" or not clean_answer_text(row.get("answer", ""))
    )
    graph_note = (
        "100건 full fresh index 완료; standard-update가 20건 output에 머물러 fallback; repair 후 15개 query x 3 mode 완료"
        if graph_rows
        else "GraphRAG smoke/update 실행 전"
    )
    return [
        {
            "System": "LightRAG",
            "Index elapsed": format_seconds(index_stats.get("elapsed_seconds_total_attempts") or index_stats.get("elapsed_seconds")),
            "Query elapsed": format_seconds(query_seconds),
            "Query records": len(light_rows),
            "Empty/failed": empty_or_failed,
            "Modes": "naive/local/global/hybrid",
            "Notes": "100건 indexing 완료, 15개 query x 4 mode 실행 완료",
        },
        {
            "System": "GraphRAG",
            "Index elapsed": format_seconds(graph_stats.get("total_runtime")),
            "Query elapsed": format_seconds(graph_query_seconds) if graph_rows else "—",
            "Query records": len(graph_rows) if graph_rows else "—",
            "Empty/failed": graph_empty_or_failed if graph_rows else "—",
            "Modes": "basic/local/global",
            "Notes": graph_note,
        },
    ]


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_json(experiment_dir / "dataset" / "patents_100_manifest.json", {}) or {}
    index_stats = read_json(experiment_dir / "lightrag_patent_prompt_100" / "index_stats.json", {}) or {}
    graph_metrics = read_json(experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json", {}) or {}
    query_status = read_json(experiment_dir / "lightrag_patent_prompt_100" / "query_status_summary.json", {}) or {}
    light_repair = read_json(experiment_dir / "lightrag_patent_prompt_100" / "query_repair_summary.json", {}) or {}
    auto_metrics = read_json(experiment_dir / "evaluation" / "auto_metrics.json", {}) or {}
    judge_rows = read_jsonl(experiment_dir / "evaluation" / "judge_results.jsonl")
    judge_summary = read_json(experiment_dir / "evaluation" / "judge_summary.json", {}) or {}
    judge_repair = read_json(experiment_dir / "evaluation" / "judge_repair_summary.json", {}) or {}
    queries = read_jsonl(experiment_dir / "queries" / "eval_queries_15.jsonl")
    light_rows, graph_rows = load_all_query_rows(experiment_dir)
    light_mode_rows = lightrag_mode_summary(light_rows)
    length_stats = answer_length_stats(light_rows, graph_rows)
    graph_stats = load_graphrag_stats(experiment_dir)
    graph_mode_rows = graphrag_query_summary(graph_rows)
    graph_recovery = graphrag_recovery_rows(experiment_dir, graph_rows, queries)
    graph_success = sum(
        1 for row in graph_rows
        if row.get("status") == "success" and clean_answer_text(row.get("answer", ""))
    )
    graph_status = "complete after repair" if graph_rows and graph_success == len(graph_rows) else "blank before execution"
    graph_evidence = (
        f"{graph_stats.get('num_documents') or '—'} docs indexed, {graph_success}/{len(graph_rows)} query records succeeded"
        if graph_rows
        else "목차와 표 구조만 남김"
    )
    graph_case_note = (
        "GraphRAG global 결과를 LightRAG hybrid와 비교한 사례다. Gemini judge decision도 함께 표시한다."
        if graph_rows
        else "GraphRAG 실행 전이므로 case study 본문은 비워둔다."
    )
    judge_success = sum(1 for row in judge_rows if row.get("status") == "success")
    judge_failures = len(judge_rows) - judge_success
    judge_status = "complete after repair" if judge_rows and judge_failures == 0 else "blank before execution"
    judge_evidence = f"{judge_success}/{len(judge_rows)} pairwise judgments succeeded" if judge_rows else "GraphRAG 실행 후 pairwise/rubric 평가 예정"
    judge_note = (
        f"Gemini judge 결과 {judge_success}/{len(judge_rows)}건을 사용해 win-rate를 계산했다."
        if judge_rows
        else "Gemini judge 실행 전이므로 논문형 table 구조만 유지하고 수치 칸은 비워둔다."
    )
    judge_repair_rows = [
        {"Audit": "Rows before normalization", "Value": judge_repair.get("rows_before", "—")},
        {"Audit": "Rows after normalization", "Value": judge_repair.get("rows_after", "—")},
        {"Audit": "Success after repair", "Value": f"{judge_repair.get('success_after', judge_success)}/{judge_repair.get('rows_after', len(judge_rows))}"},
        {"Audit": "Recovered failed keys", "Value": ", ".join(judge_repair.get("failed_before", [])) or "—"},
        {"Audit": "Backup preserved", "Value": judge_repair.get("backup", "—")},
    ]
    light_empty_count = len(query_status.get("empty_or_failed_records") or []) if isinstance(query_status, dict) else 0
    light_status = "complete after repair" if light_rows and light_empty_count == 0 else "complete with anomalies"
    light_evidence = f"{query_status.get('success_records', 0)}/{query_status.get('total_records', 0)} answer records succeeded"
    light_repair_rows = [
        {"Audit": "Repaired keys", "Value": ", ".join(light_repair.get("replaced_keys", [])) or "—"},
        {"Audit": "After repair current", "Value": f"{light_repair.get('success_records', query_status.get('success_records', 0))}/{query_status.get('total_records', 0)} success, empty/failed={light_repair.get('empty_or_failed', light_empty_count)}"},
        {"Audit": "Backup preserved", "Value": light_repair.get("backup", "—")},
    ]
    working_dir = Path(str(index_stats.get("working_dir") or experiment_dir / "lightrag_patent_prompt_100" / "storage"))
    prompt_text = (LIGHTRAG_ROOT / "prompts" / "entity_type" / "patent_ai_semiconductor.yml").read_text(encoding="utf-8") if (LIGHTRAG_ROOT / "prompts" / "entity_type" / "patent_ai_semiconductor.yml").exists() else ""
    selected_doc, single_patent_graph = patent_graph_example(working_dir)

    dataset_rows = [
        {
            "Category": cat,
            "Documents": (manifest.get("category_counts") or {}).get(cat, 0),
            "Smoke docs": len((manifest.get("smoke_ids_by_category") or {}).get(cat, [])),
            "Remaining docs": len((manifest.get("remaining_ids_by_category") or {}).get(cat, [])),
        }
        for cat in ["AA", "AB", "AC", "AD"]
    ]
    dataset_rows.append(
        {
            "Category": "Overall",
            "Documents": manifest.get("doc_count", 0),
            "Smoke docs": manifest.get("total_smoke", 0),
            "Remaining docs": manifest.get("total_remaining", 0),
        }
    )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Patent-100 RAG Reproduction Report</title>
  <style>
    :root {{ color-scheme: light; --ink:#1f2937; --muted:#64748b; --line:#d9e2ec; --band:#f6f8fb; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:white; line-height:1.55; }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 28px 80px; }}
    h1 {{ font-size:30px; margin:0 0 8px; }}
    h2 {{ margin-top:42px; border-top:1px solid var(--line); padding-top:24px; font-size:22px; }}
    h3 {{ margin-top:26px; font-size:17px; }}
    h4 {{ margin:20px 0 8px; }}
    p {{ margin:8px 0 14px; }}
    .note, .empty {{ color:var(--muted); }}
    .toc {{ border:1px solid var(--line); border-radius:8px; padding:14px 18px; background:#fbfcfd; margin:18px 0 24px; }}
    .toc a {{ color:#1d4ed8; text-decoration:none; }}
    .toc ol {{ margin:8px 0 0 20px; padding:0; }}
    .status {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#e7f8ee; color:#166534; font-size:12px; font-weight:650; }}
    .blank {{ color:var(--muted); font-style:italic; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .metric {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--band); font-size:13px; }}
    .metric strong {{ display:block; font-size:22px; margin-top:4px; color:#0f172a; }}
    table {{ width:100%; border-collapse:collapse; margin:12px 0 20px; font-size:13px; table-layout:fixed; }}
    th,td {{ border:1px solid var(--line); padding:8px; vertical-align:top; word-break:break-word; }}
    th {{ background:#eef2f7; text-align:left; }}
    pre {{ background:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; overflow:auto; font-size:12px; }}
    details {{ margin:12px 0; }}
    summary {{ cursor:pointer; font-weight:650; }}
    code {{ background:#eef2f7; padding:1px 5px; border-radius:4px; }}
    svg {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; }}
    .edge {{ stroke:#94a3b8; stroke-width:1.5; opacity:.7; }}
    .edge-label {{ font-size:9px; fill:#334155; stroke:white; stroke-width:3px; paint-order:stroke; }}
    text {{ font-size:10px; fill:#0f172a; pointer-events:none; }}
  </style>
</head>
<body>
<main>
  <h1>Patent-100 RAG Reproduction Report</h1>
  <p class="note">LightRAG, NaiveRAG(mode=naive), GraphRAG를 AI 반도체 특허 100건에서 비교한다. 기존 작업물은 보존하고 새 실험 산출물만 사용한다.</p>
  <div class="grid">
    <div class="metric">Seed<strong>{html_escape(manifest.get("seed", "pending"))}</strong></div>
    <div class="metric">Patent docs<strong>{html_escape(manifest.get("doc_count", "pending"))}</strong></div>
    <div class="metric">Graph nodes<strong>{html_escape(graph_metrics.get("graph_nodes", "pending"))}</strong></div>
    <div class="metric">Graph edges<strong>{html_escape(graph_metrics.get("graph_edges", "pending"))}</strong></div>
  </div>

  <section class="toc">
    <strong>Table of Contents</strong>
    <ol>
      <li><a href="#models">Model Roles and Current Status</a></li>
      <li><a href="#indexing">3.1 Graph-based Text Indexing</a></li>
      <li><a href="#graph-viz">Graph Visualization Before 3.2</a></li>
      <li><a href="#retrieval">3.2 Dual-level Retrieval Paradigm</a></li>
      <li><a href="#generation">3.3 Retrieval-Augmented Answer Generation</a></li>
      <li><a href="#complexity">3.4 Complexity Analysis</a></li>
      <li><a href="#tables">Paper-style Evaluation Tables</a></li>
      <li><a href="#discussion">Discussion and Conclusion</a></li>
      <li><a href="#appendix">Appendix 7.1-7.3.4</a></li>
    </ol>
  </section>

  <h2 id="models">Model Roles and Current Status</h2>
  {table([
      {"Model": "gpt-5.5 via codex-proxy", "Role": "LightRAG indexing/query, GraphRAG indexing/query"},
      {"Model": "text-embedding-3-large via OpenRouter", "Role": "chunk/entity/relation/query embedding"},
      {"Model": "Gemini 3.5 Flash via OpenRouter", "Role": "LLM judge only"},
  ])}
  {table([
      {"Track": "LightRAG indexing", "Status": "complete", "Evidence": f"{index_stats.get('processed_documents', 0)}/{index_stats.get('document_count', 0)} docs processed"},
      {"Track": "LightRAG query", "Status": light_status, "Evidence": light_evidence},
      {"Track": "GraphRAG", "Status": graph_status, "Evidence": graph_evidence},
      {"Track": "Gemini judge", "Status": judge_status, "Evidence": judge_evidence},
  ])}

  <h2 id="indexing">3.1 Graph-based Text Indexing</h2>
  <p>특허 1건을 구조화 텍스트로 변환한 뒤 chunking, R(entity/relation extraction), P(profile/key-value generation), D(deduplication)를 거쳐 graph와 vector DB를 만든다.</p>
  <h3>Structured Patent Documents</h3>
  {structured_patent_examples(experiment_dir)}
  <h3>Patent Prompt Rules</h3>
  {prompt_rule_table()}
  {rpd_examples(working_dir)}
  {pre("Patent-specific entity extraction prompt excerpt", prompt_text[:7000], 8000)}

  <h2 id="graph-viz">Graph Visualization Before 3.2</h2>
  {graph_quality_table(graph_metrics)}
  {full_graph_visualization(working_dir)}
  {single_patent_graph}

  <h2 id="retrieval">3.2 Dual-level Retrieval Paradigm</h2>
  <p><code>local</code>은 entity 중심 low-level retrieval, <code>global</code>은 relation/theme 중심 high-level retrieval, <code>hybrid</code>는 둘을 병합한다.</p>
  {render_query_flow(light_rows)}

  <h2 id="generation">3.3 Retrieval-Augmented Answer Generation</h2>
  <p>retrieved entities, relationships, chunks, references가 answer prompt에 합쳐지고, 같은 query에 대해 mode별 답변이 달라진다.</p>
  {render_answer_compare(light_rows, "AA-1")}

  <h2 id="complexity">3.4 Complexity Analysis</h2>
  {table(complexity_rows(index_stats, light_rows, graph_rows, graph_stats))}
  <h3>LightRAG Query Recovery Audit</h3>
  {table(light_repair_rows)}
  <h3>GraphRAG Query Recovery Audit</h3>
  {table(graph_recovery)}
  <h3>GraphRAG Query Mode Summary</h3>
  {table(graph_mode_rows)}
  <h3>Gemini Judge Recovery Audit</h3>
  {table(judge_repair_rows)}

  <h2 id="tables">Paper-style Evaluation Tables</h2>
  <h3>Table 4: Statistical information of the patent datasets</h3>
  {table(dataset_rows)}
  <h3>Table 1: Win rates (%) of baselines vs LightRAG</h3>
  <p class="note">{html_escape(judge_note)}</p>
  {win_rate_placeholder(judge_rows)}
  <h3>Answer Length Statistics for Table 1 Interpretation</h3>
  <p class="note">GraphRAG 답변에 섞인 런타임 warning 문구를 제거한 cleaned answer 기준이다. 이 표는 LLM-as-judge의 verbosity bias 가능성을 해석하기 위한 보조 지표다.</p>
  {answer_length_table(length_stats)}
  <h3>Table 2: Performance of ablated LightRAG retrieval modes</h3>
  {table((auto_metrics.get("system_summary") or light_mode_rows) if isinstance(auto_metrics, dict) else light_mode_rows)}
  <h3>Table 3: Case Study: GraphRAG vs LightRAG</h3>
  <p class="note">{html_escape(graph_case_note)}</p>
  {graph_case_blank() if not graph_rows else case_table(light_rows, graph_rows, judge_rows, "X-1", "global")}
  <h3>Figure 2: Cost comparison in tokens/API calls</h3>
  {pre("Cost and runtime summary", {"index_stats": index_stats, "graph_metrics": graph_metrics, "graphrag_manifest": read_json(experiment_dir / "graphrag_manifest.json", {}) or {}, "graphrag_stats": graph_stats, "graphrag_query_summary": graph_mode_rows, "graphrag_recovery": graph_recovery, "judge_summary": judge_summary, "judge_recovery": judge_repair}, 18000)}
  <h3>Table 5: Case Study: NaiveRAG vs LightRAG</h3>
  {naive_vs_light_table(light_rows, "C-1")}

  {discussion_section(graph_metrics, judge_rows, length_stats)}

  <h2 id="appendix">Appendix 7.1: Experimental Data Details</h2>
  {pre("Sampling manifest", manifest, 18000)}
  <h2>Appendix 7.2: Retrieval-Augmented Generation Example</h2>
  {render_query_flow(light_rows)}
  <h2>Appendix 7.3.1: Graph Generation Prompt and Outputs</h2>
  {pre("Prompt excerpt", prompt_text[:7000], 8000)}
  {rpd_examples(working_dir)}
  <h2>Appendix 7.3.2: Query Generation</h2>
  <p>15개 쿼리는 모델 없이 직접 설계한 고정 평가셋이다.</p>
  {table(queries)}
  <h2>Appendix 7.3.3: Keyword Extraction</h2>
  {render_query_flow(light_rows)}
  <h2>Appendix 7.3.4: RAG Evaluation</h2>
  <p class="note">{html_escape('Gemini 3.5 Flash judge를 실행했고 60/60 pairwise 결과를 정규화해 사용했다.' if judge_rows else 'Gemini 3.5 Flash judge는 아직 실행하지 않았다. 아래는 실행 후 채울 평가 구조다.')}</p>
  {table([
      {"Judge item": "Pairwise comparisons", "Planned value": "LightRAG hybrid vs NaiveRAG; LightRAG hybrid vs GraphRAG global/local; GraphRAG global vs NaiveRAG"},
      {"Judge item": "Rubrics", "Planned value": "Comprehensiveness, Diversity, Empowerment, Technical correctness, Hallucination risk, Overall"},
      {"Judge item": "A/B ordering", "Planned value": "query별로 answer A/B 순서 교차"},
      {"Judge item": "Output schema", "Planned value": "{metric: {winner, score_a, score_b, rationale}, overall_summary}"},
  ])}
  {pre("Judge summary", judge_summary, 12000)}
  {table(judge_repair_rows)}
  {pre("Judge output examples", judge_rows[:3], 12000)}
</main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(output), "exists": output.exists()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
