from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from patent_lightrag.common import (
    LIGHTRAG_ROOT,
    PATHS,
    ROOT,
    ensure_dirs,
    estimate_tokens,
    html_escape,
    load_dotenv_file,
    read_json,
    read_lines,
    redact,
)


PAPER_FLOW_MAP = {
    "3.1 Graph-based Text Indexing": {
        "paper_step": "Chunk -> R(entities, relations) -> P(key-value profiles) -> D(deduped graph)",
        "this_run": "특허 1건을 구조화 텍스트로 만들고 LightRAG `ainsert`가 chunking, entity/relation extraction, graph/vector upsert를 수행합니다.",
        "artifacts": "patent_docs.jsonl, graph_chunk_entity_relation.graphml, vdb_entities.json, vdb_relationships.json, vdb_chunks.json",
    },
    "3.1 Incremental Update": {
        "paper_step": "새 문서 D'를 같은 indexing 함수로 처리한 뒤 기존 V/E와 union",
        "this_run": "새 working dir에는 200건 초기 인덱스를 만들고, 같은 스크립트를 기존 storage에 다시 실행하면 incremental merge 경로로 들어갑니다.",
        "artifacts": "kv_store_doc_status.json, kv_store_full_docs.json, graph/vector stores",
    },
    "3.2 Dual-level Retrieval": {
        "paper_step": "query -> high/low keywords -> relation/entity vector search -> one-hop graph expansion",
        "this_run": "`local`, `global`, `hybrid` 모드를 각각 실행해 low-level entity 검색, high-level relation 검색, 병합 검색 결과를 비교합니다.",
        "artifacts": "query_results.json의 metadata.keywords, entities, relationships, chunks",
    },
    "3.3 Answer Generation": {
        "paper_step": "retrieved entity/relation/chunk values + query -> general-purpose LLM answer",
        "this_run": "`aquery_data`로 retrieval context를 저장하고 `aquery`로 최종 답변을 생성해 보고서에 같이 넣습니다.",
        "artifacts": "query_results.json의 Answer 영역",
    },
    "3.4 Complexity": {
        "paper_step": "indexing LLM calls ~= total tokens / chunk size, retrieval = keyword LLM call + vector search + graph lookup",
        "this_run": "문서 토큰 추정치, chunk size, 실제 graph/vector record 수를 실행 지표로 제시합니다.",
        "artifacts": "docs_manifest.json, index_stats.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML report for LightRAG paper sections 3.1-3.4.")
    parser.add_argument("--docs-manifest", default=str(PATHS.docs_manifest))
    parser.add_argument("--sample-manifest", default=str(PATHS.sample_manifest))
    parser.add_argument("--index-stats", default=str(PATHS.index_stats))
    parser.add_argument("--query-results", default=str(PATHS.query_results))
    parser.add_argument("--env", default=str(LIGHTRAG_ROOT / ".env"))
    parser.add_argument("--output", default=str(PATHS.html_report))
    return parser.parse_args()


def table_from_mapping(mapping: dict[str, object]) -> str:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        rows.append(f"<tr><th>{html_escape(key)}</th><td><pre>{html_escape(value)}</pre></td></tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def table_from_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p class='empty'>No rows.</p>"
    headers = list(rows[0])
    head = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(header, ''))}</td>" for header in headers) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def code_block(title: str, path: Path, start: int, end: int) -> str:
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    code = read_lines(path, start, end)
    return (
        f"<details open><summary>{html_escape(title)} - {html_escape(rel)}:{start}</summary>"
        f"<pre><code>{html_escape(code)}</code></pre></details>"
    )


def render_query_results(query_results: dict[str, object] | None) -> str:
    if not query_results:
        return "<p class='empty'>아직 query 결과가 없습니다. OpenRouter API 키 입력 후 query_flow.py를 실행하면 이 영역이 채워집니다.</p>"

    chunks = [f"<p><strong>질의:</strong> {html_escape(query_results.get('query'))}</p>"]
    modes = query_results.get("modes", {})
    if not isinstance(modes, dict):
        return "<p class='empty'>query_results 형식이 예상과 다릅니다.</p>"

    for mode, payload in modes.items():
        payload = payload if isinstance(payload, dict) else {}
        data = payload.get("data", {})
        answer = payload.get("answer", "")
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        data_body = data.get("data", {}) if isinstance(data, dict) else {}
        keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
        chunks.append(f"<section class='mode'><h4>{html_escape(mode)}</h4>")
        chunks.append(table_from_mapping({
            "high_level_keywords": keywords.get("high_level", []),
            "low_level_keywords": keywords.get("low_level", []),
            "entities": len(data_body.get("entities", [])) if isinstance(data_body, dict) else 0,
            "relationships": len(data_body.get("relationships", [])) if isinstance(data_body, dict) else 0,
            "chunks": len(data_body.get("chunks", [])) if isinstance(data_body, dict) else 0,
        }))
        chunks.append(f"<details><summary>Retrieved raw data</summary><pre>{html_escape(json.dumps(data, ensure_ascii=False, indent=2)[:20000])}</pre></details>")
        chunks.append(f"<details open><summary>Answer</summary><pre>{html_escape(answer)}</pre></details>")
        chunks.append("</section>")
    return "\n".join(chunks)


def short_label(value: object, max_chars: int = 22) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def svg_label_lines(value: object, max_line_chars: int = 13, max_lines: int = 2) -> list[str]:
    text = short_label(value, max_line_chars * max_lines)
    lines = []
    while text and len(lines) < max_lines:
        lines.append(text[:max_line_chars])
        text = text[max_line_chars:]
    if text and lines:
        lines[-1] = short_label(lines[-1], max_line_chars)
    return lines or [""]


def entity_type_color(entity_type: object) -> str:
    normalized = str(entity_type or "other").lower()
    if normalized == "artifact":
        return "#2f6f8f"
    if normalized == "concept":
        return "#7a6332"
    if normalized == "data":
        return "#687a3a"
    if normalized == "content":
        return "#73568a"
    if normalized == "organization":
        return "#2f7a67"
    if normalized == "method":
        return "#8a4f3c"
    return "#59636e"


def render_svg_node(name: str, entity_type: str, x: float, y: float, degree: int) -> str:
    radius = 12 + min(10, degree)
    lines = svg_label_lines(name)
    label_y = y + radius + 15
    tspans = []
    for idx, line in enumerate(lines):
        tspans.append(
            f"<tspan x='{x:.1f}' y='{label_y + idx * 13:.1f}'>{html_escape(line)}</tspan>"
        )
    return (
        f"<g class='graph-node'>"
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{entity_type_color(entity_type)}'>"
        f"<title>{html_escape(name)} | type={html_escape(entity_type)} | degree={degree}</title>"
        f"</circle>"
        f"<text text-anchor='middle'>{''.join(tspans)}</text>"
        f"</g>"
    )


def render_retrieval_graph(query_results: dict[str, object] | None) -> str:
    if not query_results:
        return "<p class='empty'>Query results are not available, so no retrieval graph can be rendered.</p>"

    modes = query_results.get("modes", {})
    if not isinstance(modes, dict):
        return "<p class='empty'>Query result format is invalid.</p>"
    payload = modes.get("hybrid") or modes.get("global") or modes.get("local")
    if not isinstance(payload, dict):
        return "<p class='empty'>No retrieval mode payload is available.</p>"
    data = payload.get("data", {})
    body = data.get("data", {}) if isinstance(data, dict) else {}
    if not isinstance(body, dict):
        return "<p class='empty'>No retrieval graph data is available.</p>"

    entities = body.get("entities", [])
    relationships = body.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list) or not relationships:
        return "<p class='empty'>No relationships are available for graph visualization.</p>"

    entity_types: dict[str, str] = {}
    entity_order: dict[str, int] = {}
    for idx, row in enumerate(entities):
        if not isinstance(row, dict):
            continue
        name = str(row.get("entity_name", ""))
        if not name:
            continue
        entity_types[name] = str(row.get("entity_type", "other"))
        entity_order.setdefault(name, idx)

    degree: Counter[str] = Counter()
    relation_rows: list[dict[str, object]] = []
    for row in relationships:
        if not isinstance(row, dict):
            continue
        src = str(row.get("src_id", ""))
        tgt = str(row.get("tgt_id", ""))
        if not src or not tgt:
            continue
        degree[src] += 1
        degree[tgt] += 1
        entity_types.setdefault(src, "other")
        entity_types.setdefault(tgt, "other")
        relation_rows.append(row)

    priority_names = {
        "AI 반도체 특허",
        "신경망 가속기",
        "신경망 연산 가속기",
        "AI 코어 및 가속기",
        "메모리",
        "메모리부",
        "3D 패키징 및 집적 기술",
    }

    scored_nodes = sorted(
        degree,
        key=lambda name: (
            name not in priority_names,
            -degree[name],
            entity_order.get(name, 999),
            name,
        ),
    )
    selected_nodes = set(scored_nodes[:28])
    selected_edges = [
        row
        for row in relation_rows
        if str(row.get("src_id", "")) in selected_nodes
        and str(row.get("tgt_id", "")) in selected_nodes
    ][:34]

    if len(selected_edges) < 8:
        selected_nodes = set()
        selected_edges = []
        for row in relation_rows:
            src = str(row.get("src_id", ""))
            tgt = str(row.get("tgt_id", ""))
            if len(selected_nodes | {src, tgt}) > 28:
                continue
            selected_nodes.update([src, tgt])
            selected_edges.append(row)
            if len(selected_edges) >= 34:
                break

    node_list = sorted(
        selected_nodes,
        key=lambda name: (
            name != "AI 반도체 특허",
            -degree[name],
            entity_order.get(name, 999),
            name,
        ),
    )
    if not node_list:
        return "<p class='empty'>No graph nodes were selected for visualization.</p>"

    width = 1080
    height = 720
    cx = width / 2
    cy = height / 2 - 20
    positions: dict[str, tuple[float, float]] = {}
    center = node_list[0]
    positions[center] = (cx, cy)
    ring_nodes = node_list[1:]
    inner = ring_nodes[:8]
    outer = ring_nodes[8:]

    import math

    for ring, radius, offset in [(inner, 210, -math.pi / 2), (outer, 315, -math.pi / 2 + 0.16)]:
        count = max(1, len(ring))
        for idx, name in enumerate(ring):
            angle = offset + (2 * math.pi * idx / count)
            positions[name] = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)

    edge_svg = []
    for row in selected_edges:
        src = str(row.get("src_id", ""))
        tgt = str(row.get("tgt_id", ""))
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        keywords = str(row.get("keywords", ""))
        edge_svg.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' class='graph-edge'>"
            f"<title>{html_escape(src)} -> {html_escape(tgt)} | {html_escape(keywords)}</title>"
            f"</line>"
        )

    node_svg = [
        render_svg_node(name, entity_types.get(name, "other"), *positions[name], degree[name])
        for name in node_list
    ]
    edge_rows = [
        {
            "src": row.get("src_id"),
            "tgt": row.get("tgt_id"),
            "keywords": row.get("keywords"),
            "file_path": row.get("file_path"),
        }
        for row in selected_edges[:14]
    ]
    entity_count = len(entities)
    relation_count = len(relationships)
    return f"""
    <section>
      <h2>Retrieval Subgraph Visualization</h2>
      <p>아래 그림은 전체 graph가 아니라 3.2 질의의 <code>hybrid</code> retrieval context에서 선택된 subgraph입니다. 전체 인덱스는 5,494 nodes / 9,015 edges이고, 여기서는 검색 결과의 핵심 entity/relation만 제한해서 표시합니다.</p>
      <div class="graph-summary">
        <span>retrieved entities: {entity_count}</span>
        <span>retrieved relationships: {relation_count}</span>
        <span>visualized nodes: {len(node_list)}</span>
        <span>visualized edges: {len(selected_edges)}</span>
      </div>
      <div class="graph-wrap">
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="Hybrid retrieval subgraph">
          <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#fbfcfd"/>
          {''.join(edge_svg)}
          {''.join(node_svg)}
        </svg>
      </div>
      <details open><summary>Visualized relationship sample</summary>
      {table_from_rows(edge_rows)}
      </details>
    </section>
    """


def infer_single_patent_node_type(name: str) -> str:
    if name in {"Intel", "Samsung Electronics", "TSMC", "Rebellions"}:
        return "organization"
    if name in {"US", "KR", "JP", "TW", "등록", "공개", "심사중"}:
        return "data"
    if name.startswith(("G06", "G11", "H01", "2020-", "112", "16-")):
        return "data"
    if "processor" in name.lower() or "프로세서" in name or "메모리" in name or "회로" in name:
        return "artifact"
    if "방법" in name or "연산" in name or "처리" in name:
        return "method"
    return "concept"


def load_single_patent_graph_example(
    index_stats: dict[str, object] | None,
    doc_id: str = "16-175229",
) -> dict[str, object]:
    if not index_stats:
        return {}
    working_dir = Path(str(index_stats.get("working_dir", "")))
    relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    if not isinstance(relations, dict) or not relations:
        return {}
    if doc_id not in relations:
        doc_id = next(iter(relations))

    relation_record = relations.get(doc_id, {})
    entity_record = entities.get(doc_id, {}) if isinstance(entities, dict) else {}
    relation_pairs = relation_record.get("relation_pairs", []) if isinstance(relation_record, dict) else []
    entity_names = entity_record.get("entity_names", []) if isinstance(entity_record, dict) else []
    if not isinstance(relation_pairs, list):
        relation_pairs = []
    if not isinstance(entity_names, list):
        entity_names = []

    preferred_edges = [
        ("Efficient analog in-memory matrix multiplication processor", "Intel"),
        ("Intel", "US"),
        ("Efficient analog in-memory matrix multiplication processor", "인-메모리 아날로그 병렬 처리"),
        ("인-메모리 아날로그 병렬 처리", "행렬 곱셈"),
        ("신경망", "인-메모리 아날로그 병렬 처리"),
        ("인-메모리 아날로그 병렬 처리", "인공 지능 프로세서"),
        ("메모리 회로", "아날로그 인-메모리 행렬 곱셈 시스템"),
        ("메모리 회로", "제1 메모리 배열 영역"),
        ("메모리 회로", "제2 메모리 배열 영역"),
        ("제1 메모리 배열 영역", "제1 행렬"),
        ("제2 메모리 배열 영역", "제2 행렬"),
        ("제1 행렬", "제2 행렬"),
        ("BL 기능 판독 회로", "제1 메모리 배열 영역"),
        ("BL 기능 판독 회로", "제2 메모리 배열 영역"),
        ("BL 아날로그 처리 회로", "아날로그 내적"),
        ("교차 비트 라인 기능 판독 연산", "아날로그 내적"),
        ("아날로그 내적", "행렬 곱셈"),
        ("Efficient analog in-memory matrix multiplication processor", "G06F-017/16"),
        ("Efficient analog in-memory matrix multiplication processor", "G06N-003/063"),
        ("Efficient analog in-memory matrix multiplication processor", "시스템 연동 및 플랫폼 통합"),
        ("Efficient analog in-memory matrix multiplication processor", "플랫폼 관리 및 최적화 기술"),
        ("Efficient analog in-memory matrix multiplication processor", "등록"),
    ]
    actual_pairs = {
        (str(pair[0]), str(pair[1]))
        for pair in relation_pairs
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    }
    selected_edges = [edge for edge in preferred_edges if edge in actual_pairs]
    if len(selected_edges) < 8:
        selected_edges = list(actual_pairs)[:22]

    selected_nodes = sorted({node for edge in selected_edges for node in edge})
    return {
        "doc_id": doc_id,
        "entity_count": len(entity_names),
        "relation_count": len(relation_pairs),
        "nodes": selected_nodes,
        "edges": selected_edges,
    }


def render_single_patent_graph(example: dict[str, object]) -> str:
    if not example:
        return "<p class='empty'>Single-patent graph example is not available.</p>"

    nodes = example.get("nodes", [])
    edges = example.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list) or not edges:
        return "<p class='empty'>Single-patent graph example has no edges.</p>"

    width = 1080
    height = 650
    positions: dict[str, tuple[float, float]] = {
        "Efficient analog in-memory matrix multiplication processor": (540, 310),
        "Intel": (540, 88),
        "US": (690, 82),
        "등록": (835, 110),
        "G06F-017/16": (875, 245),
        "G06N-003/063": (875, 365),
        "시스템 연동 및 플랫폼 통합": (790, 520),
        "플랫폼 관리 및 최적화 기술": (610, 560),
        "인-메모리 아날로그 병렬 처리": (310, 180),
        "행렬 곱셈": (210, 315),
        "신경망": (310, 470),
        "인공 지능 프로세서": (500, 500),
        "메모리 회로": (305, 72),
        "아날로그 인-메모리 행렬 곱셈 시스템": (160, 125),
        "제1 메모리 배열 영역": (150, 245),
        "제2 메모리 배열 영역": (150, 395),
        "제1 행렬": (310, 270),
        "제2 행렬": (310, 395),
        "BL 기능 판독 회로": (60, 315),
        "BL 아날로그 처리 회로": (420, 115),
        "교차 비트 라인 기능 판독 연산": (430, 615),
        "아날로그 내적": (415, 315),
    }

    fallback_nodes = [name for name in nodes if name not in positions]
    import math

    for idx, name in enumerate(fallback_nodes):
        angle = -math.pi / 2 + 2 * math.pi * idx / max(1, len(fallback_nodes))
        positions[str(name)] = (540 + math.cos(angle) * 275, 315 + math.sin(angle) * 245)

    degree: Counter[str] = Counter()
    edge_svg = []
    normalized_edges: list[tuple[str, str]] = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        src = str(edge[0])
        tgt = str(edge[1])
        if src not in positions or tgt not in positions:
            continue
        degree[src] += 1
        degree[tgt] += 1
        normalized_edges.append((src, tgt))
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        edge_svg.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' class='graph-edge'>"
            f"<title>{html_escape(src)} -> {html_escape(tgt)}</title>"
            f"</line>"
        )

    node_svg = [
        render_svg_node(str(name), infer_single_patent_node_type(str(name)), *positions[str(name)], degree[str(name)])
        for name in nodes
        if str(name) in positions
    ]
    edge_rows = [{"src": src, "tgt": tgt} for src, tgt in normalized_edges]
    return f"""
    <section>
      <h2>Single Patent Graph Example</h2>
      <p>아래는 임의 예시로 고른 특허 <code>{html_escape(example.get("doc_id"))}</code> 1건에서 나온 실제 entity/relation 일부입니다. 하나의 특허 안에서도 출원인, 국가, 법적상태, IPC/CPC 코드, 기술 구성요소, 기능 관계가 서로 섞여 연결되므로 tree hierarchy라기보다 일반 knowledge graph에 가깝습니다.</p>
      <div class="graph-summary">
        <span>patent entities: {html_escape(example.get("entity_count"))}</span>
        <span>patent relationships: {html_escape(example.get("relation_count"))}</span>
        <span>visualized nodes: {len(nodes)}</span>
        <span>visualized edges: {len(normalized_edges)}</span>
      </div>
      <div class="graph-wrap">
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="Single patent non-hierarchical graph">
          <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#fbfcfd"/>
          {''.join(edge_svg)}
          {''.join(node_svg)}
        </svg>
      </div>
      <p class="note">이 예시에서 <code>메모리 회로 -> 제1/제2 메모리 배열 영역 -> 제1/제2 행렬</code>, <code>아날로그 내적 -> 행렬 곱셈</code>, <code>Intel -> US</code>, <code>특허 -> IPC/CPC/법적상태</code>가 동시에 존재합니다. 이는 부모-자식 계층 하나로 정렬된 구조가 아니라, 여러 관계 타입이 같은 graph 안에 공존한다는 뜻입니다.</p>
      <details open><summary>Single-patent relationship sample</summary>
      {table_from_rows(edge_rows)}
      </details>
    </section>
    """


def normalized_index_stats(index_stats: dict[str, object] | None) -> dict[str, object] | None:
    if not index_stats:
        return index_stats
    stats = dict(index_stats)
    working_dir = Path(str(stats.get("working_dir", "")))
    status_path = working_dir / "kv_store_doc_status.json"
    if not status_path.exists():
        return stats

    status_data = read_json(status_path, {}) or {}
    primary_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    for record in status_data.values():
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata")
        status = str(record.get("status", "unknown"))
        if isinstance(metadata, dict) and metadata.get("is_duplicate"):
            duplicate_counts[status] += 1
        else:
            primary_counts[status] += 1

    if primary_counts:
        stats["processed_documents"] = primary_counts.get("processed", 0)
        stats["failed_documents"] = primary_counts.get("failed", 0)
        stats["completed"] = primary_counts.get("processed", 0) == stats.get("document_count")
        stats["status_counts"] = dict(primary_counts)
        if duplicate_counts:
            stats["duplicate_status_records"] = dict(duplicate_counts)
    return stats


def clip_text(value: object, limit: int = 1600) -> str:
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


def pick_vdb_row(
    rows: list[dict[str, object]],
    file_path: str,
    preferred_keys: list[str],
    key_field: str,
) -> dict[str, object]:
    for preferred in preferred_keys:
        for row in rows:
            if row.get("file_path") == file_path and row.get(key_field) == preferred:
                return row
    for row in rows:
        if row.get("file_path") == file_path:
            return row
    return {}


def compact_entity_profile(row: dict[str, object]) -> dict[str, object]:
    if not row:
        return {}
    return {
        "kv_key": row.get("entity_name"),
        "kv_store": "vdb_entities.json / graph node",
        "source_id": row.get("source_id"),
        "file_path": row.get("file_path"),
        "value_profile": clip_text(row.get("content"), 1400),
    }


def compact_relation_profile(row: dict[str, object]) -> dict[str, object]:
    if not row:
        return {}
    return {
        "kv_key": f"{row.get('src_id')}<SEP>{row.get('tgt_id')}",
        "kv_store": "vdb_relationships.json / graph edge",
        "source_id": row.get("source_id"),
        "file_path": row.get("file_path"),
        "value_profile": clip_text(row.get("content"), 1400),
    }


def chunk_dedupe_rows(
    chunk_store: dict[str, object],
    preferred_keys: list[str],
    relation: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in preferred_keys:
        record = chunk_store.get(key)
        if not isinstance(record, dict):
            continue
        chunk_ids = record.get("chunk_ids", [])
        if not isinstance(chunk_ids, list):
            chunk_ids = []
        rows.append(
            {
                "deduped_graph_key": key.replace("<SEP>", " -> ") if relation else key,
                "input_occurrences": record.get("count", len(chunk_ids)),
                "sample_source_chunks": "\n".join(str(chunk_id) for chunk_id in chunk_ids[:8]),
                "dedupe_result": "single graph edge + single relation profile"
                if relation
                else "single graph node + single entity profile",
            }
        )
    return rows


def load_incremental_update_examples(index_stats: dict[str, object] | None) -> dict[str, object]:
    if not index_stats:
        return {}
    working_dir = Path(str(index_stats.get("working_dir", "")))
    if not working_dir.exists():
        return {}

    full_entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    text_chunks = read_json(working_dir / "kv_store_text_chunks.json", {}) or {}
    entity_chunks = read_json(working_dir / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(working_dir / "kv_store_relation_chunks.json", {}) or {}

    preferred_docs = ["16-175229", "2022-0027364", "17-353393", "18-154340"]
    doc_id = next((candidate for candidate in preferred_docs if candidate in full_entities), "")
    if not doc_id and isinstance(full_entities, dict) and full_entities:
        doc_id = next(iter(full_entities))
    if not doc_id:
        return {}

    entity_record = full_entities.get(doc_id, {}) if isinstance(full_entities, dict) else {}
    relation_record = full_relations.get(doc_id, {}) if isinstance(full_relations, dict) else {}
    chunk_id = f"{doc_id}-chunk-000"
    chunk_record = text_chunks.get(chunk_id, {}) if isinstance(text_chunks, dict) else {}
    entity_names = entity_record.get("entity_names", []) if isinstance(entity_record, dict) else []
    relation_pairs = relation_record.get("relation_pairs", []) if isinstance(relation_record, dict) else []

    entity_vdb = load_vdb_rows(working_dir / "vdb_entities.json")
    relation_vdb = load_vdb_rows(working_dir / "vdb_relationships.json")
    entity_profile = compact_entity_profile(
        pick_vdb_row(
            entity_vdb,
            doc_id,
            [
                "Efficient analog in-memory matrix multiplication processor",
                "인-메모리 아날로그 병렬 처리",
                "메모리 회로",
            ],
            "entity_name",
        )
    )
    relation_profile = compact_relation_profile(
        pick_vdb_row(
            relation_vdb,
            doc_id,
            ["Efficient analog in-memory matrix multiplication processor"],
            "src_id",
        )
    )

    return {
        "doc_id": doc_id,
        "r_function": {
            "input_chunk_id": chunk_id,
            "input_chunk_excerpt": clip_text(chunk_record.get("content", ""), 1800)
            if isinstance(chunk_record, dict)
            else "",
            "extracted_entity_count": len(entity_names) if isinstance(entity_names, list) else 0,
            "extracted_entities_sample": entity_names[:14] if isinstance(entity_names, list) else [],
            "extracted_relation_count": len(relation_pairs) if isinstance(relation_pairs, list) else 0,
            "extracted_relations_sample": [
                relation_label(pair) for pair in relation_pairs[:14]
            ]
            if isinstance(relation_pairs, list)
            else [],
        },
        "p_function": {
            "entity_profile_example": entity_profile,
            "relation_profile_example": relation_profile,
        },
        "d_function": {
            "entity_dedup_examples": chunk_dedupe_rows(
                entity_chunks if isinstance(entity_chunks, dict) else {},
                [
                    "AI 코어 및 가속기",
                    "신경망 연산 가속기",
                    "G06N-003/063",
                    "Samsung Electronics",
                ],
            ),
            "relation_dedup_examples": chunk_dedupe_rows(
                relation_chunks if isinstance(relation_chunks, dict) else {},
                [
                    "AA<SEP>AI 코어 및 가속기",
                    "AAD<SEP>신경망 연산 가속기",
                    "AB<SEP>제조 및 패키징",
                    "KR<SEP>Samsung Electronics",
                ],
                relation=True,
            ),
        },
    }


def load_extraction_cache_example(index_stats: dict[str, object] | None) -> dict[str, object]:
    if not index_stats:
        return {}
    working_dir = Path(str(index_stats.get("working_dir", "")))
    cache = read_json(working_dir / "kv_store_llm_response_cache.json", {}) or {}
    if not isinstance(cache, dict):
        return {}

    preferred_chunk_ids = [
        "16-175229-chunk-000",
        "18-497672-chunk-000",
        "2022-0027364-chunk-000",
    ]
    extract_entries = []
    for cache_key, payload in cache.items():
        if not isinstance(payload, dict) or payload.get("cache_type") != "extract":
            continue
        extract_entries.append((str(cache_key), payload))

    for chunk_id in preferred_chunk_ids:
        for cache_key, payload in extract_entries:
            if payload.get("chunk_id") == chunk_id:
                return {
                    "cache_key": cache_key,
                    "chunk_id": chunk_id,
                    "output_excerpt": clip_text(payload.get("return", ""), 5000),
                }

    if extract_entries:
        cache_key, payload = extract_entries[0]
        return {
            "cache_key": cache_key,
            "chunk_id": payload.get("chunk_id"),
            "output_excerpt": clip_text(payload.get("return", ""), 5000),
        }
    return {}


def render_incremental_update_examples(examples: dict[str, object]) -> str:
    if not examples:
        return "<p class='empty'>Incremental update 예시 데이터를 찾지 못했습니다.</p>"

    r_function = examples.get("r_function", {})
    p_function = examples.get("p_function", {})
    d_function = examples.get("d_function", {})
    return f"""
    <h3>R/P/D Function Output Examples</h3>
    <p class="note">아래 예시는 최종 storage에서 실제 특허 <code>{html_escape(examples.get("doc_id"))}</code>와 공통 그래프 키를 읽어 만든 것입니다. vector payload는 제외하고, 사람이 검토할 수 있는 key/value/source만 표시했습니다.</p>
    <details open>
      <summary>R function: entity/relation extraction result</summary>
      <p>R은 chunk 텍스트에서 entity set과 relation set을 뽑습니다. 아래는 <code>{html_escape((r_function or {}).get("input_chunk_id"))}</code>에서 나온 실제 결과 일부입니다.</p>
      <h4>Input chunk excerpt</h4>
      <pre>{html_escape((r_function or {}).get("input_chunk_excerpt", ""))}</pre>
      <h4>R output sample</h4>
      {table_from_mapping({
          "extracted_entity_count": (r_function or {}).get("extracted_entity_count"),
          "extracted_entities_sample": (r_function or {}).get("extracted_entities_sample"),
          "extracted_relation_count": (r_function or {}).get("extracted_relation_count"),
          "extracted_relations_sample": (r_function or {}).get("extracted_relations_sample"),
      })}
    </details>
    <details open>
      <summary>P function: key-value profile generation</summary>
      <p>P는 R이 만든 entity/relation을 graph/VDB에 저장 가능한 key-value profile로 바꿉니다. value profile은 LLM이 생성한 설명, source chunk, file path를 포함합니다.</p>
      <h4>Entity profile</h4>
      {table_from_mapping((p_function or {}).get("entity_profile_example", {}) or {})}
      <h4>Relation profile</h4>
      {table_from_mapping((p_function or {}).get("relation_profile_example", {}) or {})}
    </details>
    <details open>
      <summary>D function: deduplication and merge result</summary>
      <p>D는 같은 entity key 또는 relation key가 여러 chunk/document에서 반복될 때 하나의 graph node/edge로 병합하고, source chunk 목록과 count를 유지합니다.</p>
      <h4>Entity dedup examples</h4>
      {table_from_rows((d_function or {}).get("entity_dedup_examples", []) or [])}
      <h4>Relation dedup examples</h4>
      {table_from_rows((d_function or {}).get("relation_dedup_examples", []) or [])}
    </details>
    """


def render_extraction_criteria(cache_example: dict[str, object]) -> str:
    cache_block = "<p class='empty'>이번 실행의 extract cache 예시를 찾지 못했습니다.</p>"
    if cache_example:
        cache_block = table_from_mapping(
            {
                "cache_key": cache_example.get("cache_key"),
                "chunk_id": cache_example.get("chunk_id"),
                "cached_llm_extract_output_excerpt": cache_example.get("output_excerpt"),
            }
        )
    return f"""
    <h3>Entity / Relationship Selection Criteria</h3>
    <p class="note">이번 실행의 entity와 relationship은 LightRAG의 extraction prompt가 정한 기준을 따릅니다. 즉, 입력 chunk 안에서 명확하고 의미 있는 entity를 먼저 고르고, 그 entity들 사이에 직접 명시된 의미 있는 관계만 relation으로 출력하게 되어 있습니다.</p>
    <h4>Prompt 기준 요약</h4>
    {table_from_mapping({
        "entity_selection": "입력 텍스트에 명확히 정의되고 의미 있는 대상. entity_name, entity_type, entity_description을 생성하며 description은 입력 텍스트에 근거해야 함.",
        "entity_type_basis": "Person, Organization, Location, Concept, Method, Content, Data, Artifact 등 기본 type guidance 중 하나. 맞지 않으면 Other.",
        "relationship_selection": "이미 추출된 entity들 사이의 직접적이고 명확하며 의미 있는 관계. n-ary 관계는 binary relation으로 분해.",
        "relationship_fields": "source_entity, target_entity, relationship_keywords, relationship_description.",
        "dedup_rule": "관계는 명시적으로 방향성이 없으면 undirected로 취급하고, 중복 relationship은 피함.",
        "output_order": "entity row를 모두 먼저 출력하고 relationship row를 뒤에 출력.",
        "run_language": "이번 실행은 addon_params language=Korean이므로 entity 설명, 관계 키워드/설명은 한국어 중심으로 생성됨. 고유명사는 원어 유지 가능.",
    })}
    <h4>Actual extract cache from this run</h4>
    <p>LightRAG cache에는 완성된 prompt 전체보다는, 해당 prompt로 LLM이 반환한 extraction 결과가 저장되어 있습니다. 아래는 이번 실행에서 저장된 실제 extract output 일부입니다.</p>
    {cache_block}
    <h4>Prompt template code</h4>
    {code_block("Extraction system prompt", LIGHTRAG_ROOT / "lightrag" / "prompt.py", 34, 104)}
    {code_block("Few-shot extraction examples", LIGHTRAG_ROOT / "lightrag" / "prompt.py", 133, 234)}
    """


def main() -> None:
    args = parse_args()
    ensure_dirs()
    env = load_dotenv_file(Path(args.env))
    sample_manifest = read_json(Path(args.sample_manifest), {}) or {}
    docs_manifest = read_json(Path(args.docs_manifest), {}) or {}
    index_stats = normalized_index_stats(read_json(Path(args.index_stats), None))
    query_results = read_json(Path(args.query_results), None)
    incremental_examples = load_incremental_update_examples(index_stats)
    extraction_cache_example = load_extraction_cache_example(index_stats)
    single_patent_graph_example = load_single_patent_graph_example(index_stats)
    sample_doc = (docs_manifest or {}).get("sample_document") or {}
    sample_text = sample_doc.get("text", "")
    doc_count = int((docs_manifest or {}).get("document_count") or 0)
    token_total = ((docs_manifest or {}).get("token_estimate") or {}).get("total", 0)
    estimated_chunks = round(int(token_total or 0) / 1200, 2) if token_total else 0
    flow_rows = [
        {"section": section, **payload}
        for section, payload in PAPER_FLOW_MAP.items()
    ]

    redacted_env = {
        key: "[configured, redacted]" if key.endswith("_API_KEY") and value else redact(value)
        for key, value in env.items()
        if key in {
            "LLM_BINDING_HOST",
            "LLM_MODEL",
            "OPENAI_LLM_REASONING_EFFORT",
            "EMBEDDING_BINDING_HOST",
            "EMBEDDING_BINDING_API_KEY",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIM",
            "EMBEDDING_TIMEOUT",
            "EMBEDDING_FUNC_MAX_ASYNC",
            "EMBEDDING_BATCH_NUM",
            "CHUNK_SIZE",
            "MAX_GLEANING",
            "SUMMARY_LANGUAGE",
        }
    }

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LightRAG 3.1-3.4 Flow Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #182026; background: #f6f7f8; }}
    header {{ padding: 28px 36px; background: #16212b; color: white; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    section {{ background: white; border: 1px solid #dde2e7; border-radius: 8px; padding: 22px; margin: 18px 0; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .metric {{ background: #f0f4f7; padding: 14px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d8dee4; padding: 9px; vertical-align: top; text-align: left; }}
    th {{ width: 240px; background: #f3f5f7; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0f1720; color: #e6edf3; padding: 14px; border-radius: 6px; overflow-x: auto; }}
    details {{ margin: 12px 0; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .empty {{ color: #6b7280; background: #fafafa; border: 1px dashed #ccd2d8; padding: 12px; border-radius: 6px; }}
    .mode {{ border-left: 4px solid #3f6f8f; }}
    .note {{ border-left: 4px solid #7b8f3f; background: #fbfcf3; padding: 12px; border-radius: 6px; }}
    .graph-summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .graph-summary span {{ background: #eef3f6; border: 1px solid #d8e1e8; border-radius: 4px; padding: 6px 9px; font-size: 13px; }}
    .graph-wrap {{ border: 1px solid #d8dee4; border-radius: 8px; overflow: hidden; background: #fbfcfd; }}
    .graph-edge {{ stroke: #9ca8b4; stroke-width: 1.3; stroke-opacity: 0.62; }}
    .graph-node text {{ font-size: 11px; fill: #182026; paint-order: stroke; stroke: #fbfcfd; stroke-width: 3px; stroke-linejoin: round; }}
  </style>
</head>
<body>
<header>
  <h1>LightRAG 3.1-3.4 Flow Report</h1>
  <p>AI 반도체 특허 데이터로 LightRAG 논문 Architecture 흐름을 코드와 실행 산출물 기준으로 추적합니다.</p>
</header>
<main>
  <section>
    <h2>Overview</h2>
    <div class="grid">
      <div class="metric">Sampled patents<strong>{doc_count}</strong></div>
      <div class="metric">Estimated tokens<strong>{html_escape(token_total)}</strong></div>
      <div class="metric">Estimated chunks @1200<strong>{estimated_chunks}</strong></div>
      <div class="metric">Processed patents<strong>{html_escape((index_stats or {}).get("processed_documents", "not run"))}</strong></div>
      <div class="metric">Failed patents<strong>{html_escape((index_stats or {}).get("failed_documents", "not run"))}</strong></div>
      <div class="metric">Indexed graph nodes<strong>{html_escape((index_stats or {}).get("graph_nodes", "not run"))}</strong></div>
      <div class="metric">Indexed graph edges<strong>{html_escape((index_stats or {}).get("graph_edges", "not run"))}</strong></div>
    </div>
    <h3>Runtime Configuration</h3>
    {table_from_mapping(redacted_env)}
    <h3>Paper-to-Run Mapping</h3>
    <p class="note">아래 표는 LightRAG 논문 3.1-3.4의 추상 함수/단계를 이번 특허 데이터 실행 산출물에 대응시킨 것입니다.</p>
    {table_from_rows(flow_rows)}
  </section>

  <section>
    <h2>3.1 Graph-based Text Indexing</h2>
    <p>특허 원천 CSV를 구조화 문서로 변환한 뒤, LightRAG의 fixed-token chunking, entity/relation extraction, graph/vector upsert 흐름으로 들어갑니다.</p>
    {render_extraction_criteria(extraction_cache_example)}
    <h3>Sample Structured Patent Document</h3>
    <pre>{html_escape(sample_text[:12000])}</pre>
    <h3>Sampling Manifest</h3>
    <details><summary>중분류/소분류 샘플링 결과</summary><pre>{html_escape(json.dumps(sample_manifest, ensure_ascii=False, indent=2)[:30000])}</pre></details>
    <h3>Code Flow</h3>
    {code_block("Patent structured text builder", ROOT / "patent_lightrag" / "common.py", 212, 260)}
    {code_block("LightRAG insert entrypoint", LIGHTRAG_ROOT / "lightrag" / "lightrag.py", 1275, 1335)}
    {code_block("Entity/relation extraction", LIGHTRAG_ROOT / "lightrag" / "operate.py", 3246, 3345)}
  </section>

  <section>
    <h2>3.1 Incremental Update</h2>
    <p>새 특허 문서는 동일한 indexing pipeline을 통과한 뒤 기존 storage에 merge됩니다. 전체 코퍼스를 재색인하는 방식이 아니라 새 chunk/entity/relation 결과를 기존 graph/vector store에 추가합니다.</p>
    {render_incremental_update_examples(incremental_examples)}
    <h3>Index Stats</h3>
    {table_from_mapping(index_stats or {"status": "not run", "reason": "OpenRouter API key required before indexing"})}
    <h3>R/P/D Code Hooks</h3>
    {code_block("R: parse LLM extraction output", LIGHTRAG_ROOT / "lightrag" / "operate.py", 1215, 1360)}
    {code_block("P: profile/summary generation", LIGHTRAG_ROOT / "lightrag" / "operate.py", 212, 420)}
    {code_block("D: rebuild entity from merged chunks", LIGHTRAG_ROOT / "lightrag" / "operate.py", 1417, 1615)}
    {code_block("D: rebuild relation from merged chunks", LIGHTRAG_ROOT / "lightrag" / "operate.py", 1660, 1915)}
    {code_block("Patent indexing wrapper", ROOT / "patent_lightrag" / "index_patents.py", 96, 159)}
  </section>

  {render_retrieval_graph(query_results)}
  {render_single_patent_graph(single_patent_graph_example)}

  <section>
    <h2>3.2 Dual-level Retrieval Paradigm</h2>
    <p><code>local</code>은 low-level keywords로 entity vector DB를 검색하고, <code>global</code>은 high-level keywords로 relationship vector DB를 검색합니다. <code>hybrid</code>는 두 결과를 병합해 논문상의 dual-level retrieval에 대응합니다.</p>
    {render_query_results(query_results)}
    <h3>Code Flow</h3>
    {code_block("KG query and keyword extraction", LIGHTRAG_ROOT / "lightrag" / "operate.py", 3673, 3765)}
    {code_block("Local entity retrieval", LIGHTRAG_ROOT / "lightrag" / "operate.py", 5022, 5075)}
    {code_block("Global relation retrieval", LIGHTRAG_ROOT / "lightrag" / "operate.py", 5297, 5355)}
  </section>

  <section>
    <h2>3.3 Retrieval-Augmented Answer Generation</h2>
    <p>검색된 entities, relationships, chunks가 context string으로 조립되고, <code>rag_response</code> system prompt에 들어가 최종 답변이 생성됩니다.</p>
    {code_block("Build query context", LIGHTRAG_ROOT / "lightrag" / "operate.py", 4899, 5005)}
    {code_block("Answer-generation call", LIGHTRAG_ROOT / "lightrag" / "operate.py", 3767, 3865)}
  </section>

  <section>
    <h2>3.4 Complexity Analysis</h2>
    <p>논문 기준으로 indexing 비용은 chunk 수에 비례하는 extraction LLM call이 핵심이고, retrieval 비용은 query keyword extraction 1회와 entity/relation vector search 및 graph lookup으로 구성됩니다.</p>
    {table_from_mapping({
        "documents": doc_count,
        "estimated_total_tokens": token_total,
        "estimated_chunks_at_1200": estimated_chunks,
        "expected_initial_extraction_llm_calls": f"about chunks * (1 initial + MAX_GLEANING 1)",
        "query_keyword_llm_calls": "1 per query mode unless cached",
        "local_vector_search": "entities_vdb.query(low_level_keywords)",
        "global_vector_search": "relationships_vdb.query(high_level_keywords)",
        "graph_expansion": "one-hop neighboring relations/entities from retrieved graph elements",
    })}
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
