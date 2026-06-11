from __future__ import annotations

import argparse
import html as html_lib
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
    parser.add_argument("--print-output", default=str(ROOT / "reports" / "rag_repro_100_comparison_print.html"))
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


def table(rows: list[dict[str, Any]], class_name: str = "") -> str:
    if not rows:
        return "<p class='empty'>pending</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html_escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in keys) + "</tr>")
    class_attr = f" class='{html_escape(class_name)}'" if class_name else ""
    return f"<div class='table-wrap'><table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def table_explanation(text: str) -> str:
    return f"<p class='table-explain'>{html_escape(text)}</p>"


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def print_css() -> str:
    return """
    body.print-version { background:#fff; }
    body.print-version main { max-width: none; padding: 24px 28px 60px; }
    body.print-version h1 { font-size: 28px; }
    body.print-version .print-toolbar { display:flex; gap:10px; align-items:center; margin:16px 0 22px; padding:12px 14px; border:1px solid var(--line); border-radius:8px; background:#f8fafc; color:#475569; font-size:13px; }
    body.print-version .print-toolbar button { border:1px solid #94a3b8; background:white; color:#0f172a; border-radius:6px; padding:7px 11px; font-weight:650; cursor:pointer; }
    body.print-version .table-wrap { overflow: visible; }
    body.print-version table { min-width: 0; table-layout: fixed; }
    body.print-version th, body.print-version td { word-break: normal; overflow-wrap: anywhere; }
    body.print-version .print-omitted { margin: 7px 0 12px; padding: 7px 10px; border: 1px dashed #cbd5e1; border-radius: 6px; background: #f8fafc; color: #64748b; font-size: 12px; }
    body.print-version .print-source-block { margin: 9px 0 15px; border: 1px solid #cbd5e1; border-radius: 7px; background: #fff; break-inside: avoid; page-break-inside: avoid; }
    body.print-version .print-source-title { margin: 0; padding: 7px 9px; background: #f1f5f9; border-bottom: 1px solid #cbd5e1; color: #0f172a; font-weight: 700; font-size: 12px; }
    body.print-version .print-source-pre { margin: 0; padding: 9px; white-space: pre-wrap; color: #111827; background: #fff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 9px; line-height: 1.34; }
    body.print-version .print-json-columns .print-source-pre { column-count: 3; column-gap: 18px; column-rule: 1px solid #e2e8f0; white-space: pre-wrap; }
    @media print {
      @page { size: A4 landscape; margin: 11mm; }
      :root { --ink:#111827; --muted:#475569; --line:#cbd5e1; --band:#f8fafc; }
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; font-size: 10.5px; line-height: 1.35; }
      main { max-width: none !important; padding: 0 !important; }
      h1 { font-size: 20px !important; margin-bottom: 4px !important; }
      h2 { break-before: page; page-break-before: always; margin-top: 0 !important; padding-top: 8px !important; font-size: 16px !important; }
      h2#models { break-before: auto; page-break-before: auto; }
      h3.print-page-break { break-before: page; page-break-before: always; padding-top: 8px !important; }
      h3 { font-size: 12.5px !important; margin-top: 14px !important; }
      h4 { font-size: 11px !important; margin-top: 10px !important; }
      p { margin: 5px 0 8px !important; }
      .print-toolbar { display:none !important; }
      .toc { break-inside: avoid; page-break-inside: avoid; padding: 8px 10px !important; margin: 10px 0 14px !important; }
      .grid { grid-template-columns: repeat(4, 1fr) !important; gap: 6px !important; margin: 8px 0 12px !important; }
      .metric { padding: 7px !important; font-size: 9px !important; }
      .metric strong { font-size: 15px !important; }
      .table-wrap { overflow: visible !important; margin: 7px 0 13px !important; border-color: #cbd5e1 !important; break-inside: auto; page-break-inside: auto; }
      .table-explain { margin: -6px 0 13px !important; padding: 6px 8px !important; font-size: 9.5px !important; border-left-width: 2px !important; }
      table { width:100% !important; min-width:0 !important; table-layout: fixed !important; font-size: 8.5px !important; }
      th,td { padding: 4px 5px !important; line-height: 1.28 !important; }
      table.answer-compare-table th:nth-child(1), table.answer-compare-table td:nth-child(1) { width:8% !important; }
      table.answer-compare-table th:nth-child(2), table.answer-compare-table td:nth-child(2) { width:9% !important; }
      table.answer-compare-table th:nth-child(3), table.answer-compare-table td:nth-child(3) { width:8% !important; }
      table.answer-compare-table th:nth-child(4), table.answer-compare-table td:nth-child(4) { width:12% !important; }
      table.answer-compare-table th:nth-child(5), table.answer-compare-table td:nth-child(5) { width:63% !important; }
      table.case-study-table th:nth-child(1), table.case-study-table td:nth-child(1) { width:7% !important; }
      table.case-study-table th:nth-child(2), table.case-study-table td:nth-child(2) { width:31% !important; }
      table.case-study-table th:nth-child(3), table.case-study-table td:nth-child(3) { width:31% !important; }
      table.case-study-table th:nth-child(4), table.case-study-table td:nth-child(4) { width:31% !important; }
      table.naive-light-table th:nth-child(1), table.naive-light-table td:nth-child(1) { width:6% !important; }
      table.naive-light-table th:nth-child(2), table.naive-light-table td:nth-child(2) { width:7% !important; }
      table.naive-light-table th:nth-child(3), table.naive-light-table td:nth-child(3) { width:8% !important; }
      table.naive-light-table th:nth-child(4), table.naive-light-table td:nth-child(4) { width:10% !important; }
      table.naive-light-table th:nth-child(5), table.naive-light-table td:nth-child(5) { width:7% !important; }
      table.naive-light-table th:nth-child(6), table.naive-light-table td:nth-child(6) { width:10% !important; }
      table.naive-light-table th:nth-child(7), table.naive-light-table td:nth-child(7) { width:52% !important; }
      table.cost-summary-table th:nth-child(1), table.cost-summary-table td:nth-child(1) { width:14% !important; }
      table.cost-summary-table th:nth-child(2), table.cost-summary-table td:nth-child(2) { width:13% !important; }
      table.cost-summary-table th:nth-child(3), table.cost-summary-table td:nth-child(3) { width:13% !important; }
      table.cost-summary-table th:nth-child(4), table.cost-summary-table td:nth-child(4) { width:30% !important; }
      table.cost-summary-table th:nth-child(5), table.cost-summary-table td:nth-child(5) { width:30% !important; }
      table.appendix-summary-table th:nth-child(1), table.appendix-summary-table td:nth-child(1) { width:18% !important; }
      table.appendix-summary-table th:nth-child(2), table.appendix-summary-table td:nth-child(2) { width:44% !important; }
      table.appendix-summary-table th:nth-child(3), table.appendix-summary-table td:nth-child(3) { width:38% !important; }
      tr { break-inside: avoid; page-break-inside: avoid; }
      .print-omitted { margin: 5px 0 9px !important; padding: 5px 7px !important; font-size: 8.5px !important; break-inside: avoid; page-break-inside: avoid; }
      .print-source-block { margin: 6px 0 11px !important; break-inside: avoid; page-break-inside: avoid; }
      .print-source-title { padding: 5px 7px !important; font-size: 8.8px !important; }
      .print-source-pre { padding: 6px 7px !important; font-size: 7.5px !important; line-height: 1.25 !important; }
      .print-json-columns .print-source-pre { column-count: 3 !important; column-gap: 10mm !important; column-rule: 0.2mm solid #e2e8f0 !important; font-size: 6.8px !important; line-height: 1.18 !important; }
      svg { max-height: 155mm; break-inside: avoid; page-break-inside: avoid; }
      .graph-legend { font-size: 8.5px !important; gap: 4px 8px !important; padding: 6px 8px !important; margin: 6px 0 12px !important; }
      .legend-dot { width:8px !important; height:8px !important; }
      a { color: inherit !important; text-decoration: none !important; }
    }
    """


def remove_print_heavy_blocks(html: str) -> str:
    keep_limits = {
        "Structured patent text: 15-091413": 2200,
        "R input chunk excerpt": 1500,
        "P entity key-value profile": 1000,
        "P relation key-value profile": 1000,
        "Patent-specific entity extraction prompt excerpt": 3200,
        "Query metadata": 1100,
        "Figure 2 raw JSON excerpt": 2200,
        "Appendix 7.1 manifest JSON excerpt": 2200,
        "Judge summary excerpt": 1800,
        "Judge output example excerpt": 1800,
    }
    keep_once = set(keep_limits)
    kept: set[str] = set()

    def replace_details(match: re.Match[str]) -> str:
        summary = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if summary not in keep_once or summary in kept:
            return ""
        kept.add(summary)
        body = match.group(2)
        pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", body, flags=re.DOTALL)
        if not pre_match:
            return ""
        text = html_lib.unescape(pre_match.group(1)).strip()
        limit = keep_limits[summary]
        if len(text) > limit:
            text = text[:limit].rstrip() + "\n...[print excerpt truncated]"
        summary_lower = summary.lower()
        extra_class = (
            " print-json-columns"
            if "json" in summary_lower
            or "metadata" in summary_lower
            or "profile" in summary_lower
            or "judge" in summary_lower
            else ""
        )
        return (
            f"<section class=\"print-source-block{extra_class}\">"
            f"<p class=\"print-source-title\">{html_escape(summary)}</p>"
            f"<pre class=\"print-source-pre\">{html_escape(text)}</pre>"
            "</section>"
        )

    html = re.sub(
        r"<details[^>]*>\s*<summary>(.*?)</summary>(.*?)</details>",
        replace_details,
        html,
        flags=re.DOTALL,
    )
    return html


def print_code_excerpt_block() -> str:
    code = """# .env prompt wiring
PROMPT_DIR=/Users/dabeenkim/Documents/GitHub/HYU-lightRAG/LightRAG-main/prompts
ENTITY_TYPE_PROMPT_FILE=patent_ai_semiconductor.yml
ENTITY_EXTRACTION_USE_JSON=false

# patent_lightrag/index_patents.py
rag = LightRAG(
    working_dir=str(working_dir),
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
    entity_extract_max_gleaning=int(os.getenv("MAX_GLEANING", "1")),
    addon_params=lightrag_addon_params(),
)

# LightRAG-main/lightrag/addon_params.py
"entity_type_prompt_file": get_env_value("ENTITY_TYPE_PROMPT_FILE", "", str)

# report regeneration
PYTHONPATH=. python3 patent_lightrag/rag_repro_report.py \\
  --output reports/rag_repro_100_comparison.html \\
  --print-output reports/rag_repro_100_comparison_print.html"""
    return (
        "<section class=\"print-source-block\">"
        "<p class=\"print-source-title\">Code/config excerpt: prompt wiring and report generation</p>"
        f"<pre class=\"print-source-pre\">{html_escape(code)}</pre>"
        "</section>"
    )


def build_print_html(html: str) -> str:
    html = remove_print_heavy_blocks(html)
    html = html.replace(
        "<title>Patent-100 RAG Reproduction Report</title>",
        "<title>Patent-100 RAG Reproduction Report - Print</title>",
        1,
    )
    html = html.replace("<body>", "<body class=\"print-version\">", 1)
    html = html.replace(
        "<h1>Patent-100 RAG Reproduction Report</h1>",
        (
            "<h1>Patent-100 RAG Reproduction Report - Print Version</h1>\n"
            "  <div class=\"print-toolbar\">"
            "<button type=\"button\" onclick=\"window.print()\">Print</button>"
            "<span>A4 landscape 기준으로 표 폭과 page break를 조정한 인쇄용 HTML이다. "
            "필요한 원문 excerpt만 남겼고, 브라우저 인쇄 설정에서 배경 그래픽을 켜면 그래프 색상이 유지된다.</span>"
            "</div>\n"
            "  <p class=\"print-omitted\">인쇄용에는 구조화 특허 1건, R/P/D 예시, patent prompt, query metadata, judge JSON 예시만 excerpt로 남겼다. "
            "중복 prompt, cost dump, manifest 전체, 긴 raw JSON은 화면용 HTML과 실험 산출물 파일에 보존되어 있다.</p>"
            f"{print_code_excerpt_block()}"
        ),
        1,
    )
    html = html.replace("  </style>", f"{print_css()}\n  </style>", 1)
    return html


def compact_for_report(value: Any, key: str = "") -> Any:
    key_lower = key.lower()
    if key_lower == "matrix" and isinstance(value, str):
        return f"[matrix omitted: {len(value):,} chars]"
    if key_lower == "vector" and isinstance(value, str):
        return f"[vector: compressed embedding, {len(value):,} chars]"
    if key_lower in {"selected_ids_by_category", "smoke_ids_by_category", "remaining_ids_by_category", "available_by_category"} and isinstance(value, dict):
        return {
            category: {
                "count": len(ids) if isinstance(ids, list) else len(ids) if hasattr(ids, "__len__") else "—",
                "sample": ids[:5] if isinstance(ids, list) else "omitted",
            }
            for category, ids in value.items()
        }
    if isinstance(value, dict):
        return {k: compact_for_report(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        if "embedding" in key_lower or "vector" in key_lower:
            return f"[vector: {len(value)}-dim]"
        return [compact_for_report(item, key) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) > 200 and set(stripped) <= {"."}:
            return f"[progress dots omitted: {len(value):,} chars]"
        if len(value) > 5000 and re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", value):
            return f"[encoded payload omitted: {len(value):,} chars]"
    return value


def compact_text_dump(text: str) -> str:
    text = re.sub(r"(?m)^\.{20,}$", "[progress dots omitted]", text)
    text = re.sub(r"\.{80,}", "[progress dots omitted]", text)
    return text


def pre(title: str, value: Any, limit: int = 8000, open: bool = False) -> str:
    compacted = compact_for_report(value)
    text = compacted if isinstance(compacted, str) else json.dumps(compacted, ensure_ascii=False, indent=2)
    text = compact_text_dump(text)
    if len(text) > limit:
        text = text[:limit] + "\n...[truncated]"
    open_attr = " open" if open else ""
    return f"<details{open_attr}><summary>{html_escape(title)}</summary><pre>{html_escape(text)}</pre></details>"


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


def graph_legend(nodes: list[dict[str, Any]]) -> str:
    labels = {
        "techcomponent": "기술 구성요소",
        "architecture": "아키텍처",
        "operation": "연산/동작",
        "method": "방법/공정",
        "material": "재료",
        "performancemetric": "성능 지표",
        "applicationdomain": "응용 분야",
        "organization": "기관/기업",
        "other": "기타",
    }
    present = sorted({str(node.get("entity_type", "other")).lower() for node in nodes})
    if not present:
        present = ["other"]
    items = []
    for key in present:
        label = labels.get(key, key)
        items.append(
            f"<span class='legend-item'><span class='legend-dot' style='background:{color_for(key)}'></span>{html_escape(label)}</span>"
        )
    return f"<div class='graph-legend'><strong>노드 색상</strong>{''.join(items)}<span class='legend-note'>노드 크기 = 그래프 degree, 굵은 파란 엣지 = Patent_count &gt; 1, 엣지 라벨 = 관계 키워드</span></div>"


def relation_label(edge: dict[str, Any], limit: int = 22) -> str:
    label = str(edge.get("keywords") or "").replace("\n", ", ").strip()
    if not label:
        label = str(edge.get("description") or "").replace("\n", " ").strip()
    if not label:
        label = "relation"
    return label if len(label) <= limit else label[:limit] + "..."


def edge_source_patents(edge: dict[str, Any]) -> list[str]:
    file_path = str(edge.get("file_path") or "")
    return sorted({part for part in file_path.split("<SEP>") if part})


def relation_table(edges: list[dict[str, Any]], limit: int = 14) -> str:
    rows = []
    for edge in edges[:limit]:
        source_patents = edge_source_patents(edge)
        rows.append(
            {
                "Source_entity": edge.get("source", ""),
                "Relation": relation_label(edge, 48),
                "Target_entity": edge.get("target", ""),
                "Description": short(edge.get("description", ""), 240),
                "Source_patents": edge.get("file_path", ""),
                "Patent_count": len(source_patents),
            }
        )
    return table(rows, class_name="relation-table")


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
        patent_count = len(edge_source_patents(edge))
        edge_class = "edge multi-edge" if patent_count > 1 else "edge"
        tooltip = "\n".join(
            part for part in [
                f"Source_entity: {src}",
                f"Target_entity: {tgt}",
                f"Relation keywords: {edge.get('keywords', '')}" if edge.get("keywords") else "",
                f"Patent_count: {patent_count}",
                str(edge.get("description", "")),
                f"source_chunk: {edge.get('source_id', '')}" if edge.get("source_id") else "",
            ] if part
        )
        edge_svg.append(
            f"<line class='{edge_class}' x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}'>"
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
        radius = min(32, max(13, 10 + degree * 0.9))
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
      {graph_legend(nodes)}
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
    multi_edges = [edge for edge in raw_edges if len(edge_source_patents(edge)) >= 2]
    multi_degree: Counter[str] = Counter()
    for edge in multi_edges:
        multi_degree[edge["source"]] += 1
        multi_degree[edge["target"]] += 1

    preferred_seeds = [
        "인공지능 가속기",
        "제1 데이터 버퍼",
        "컨볼루션 연산",
        "입력 데이터 슬라이스",
        "제2 데이터 버퍼",
        "제1 입력 시프트 레지스터",
    ]
    seed = next((name for name in preferred_seeds if multi_degree.get(name, 0) >= 2), "")
    if not seed and multi_degree:
        seed = sorted(multi_degree, key=lambda name: (-multi_degree[name], -degree[name], name))[0]
    if not seed:
        seed = "입력 데이터" if "입력 데이터" in degree else degree.most_common(1)[0][0]

    seed_multi_edges = [
        edge for edge in multi_edges
        if edge["source"] == seed or edge["target"] == seed
    ]
    cluster_patents = {patent for edge in seed_multi_edges for patent in edge_source_patents(edge)}
    if not cluster_patents and seed_multi_edges:
        cluster_patents = set(edge_source_patents(seed_multi_edges[0]))

    cluster_multi_edges = [
        edge for edge in multi_edges
        if cluster_patents.intersection(edge_source_patents(edge))
    ]
    selected_names = {seed}
    selected_edges: list[dict[str, Any]] = []

    def try_add_edge(edge: dict[str, Any]) -> None:
        if len(selected_edges) >= 40:
            return
        candidate_names = {str(edge["source"]), str(edge["target"])}
        if len(selected_names.union(candidate_names)) > 30:
            return
        if edge not in selected_edges:
            selected_edges.append(edge)
            selected_names.update(candidate_names)

    for edge in sorted(
        cluster_multi_edges,
        key=lambda item: (
            0 if item["source"] == seed or item["target"] == seed else 1,
            -len(edge_source_patents(item)),
            str(item.get("source", "")),
            str(item.get("target", "")),
        ),
    ):
        try_add_edge(edge)

    context_candidates = [
        edge for edge in raw_edges
        if cluster_patents.intersection(edge_source_patents(edge))
        and (edge["source"] in selected_names or edge["target"] in selected_names)
    ]
    for edge in sorted(
        context_candidates,
        key=lambda item: (
            0 if len(edge_source_patents(item)) >= 2 else 1,
            0 if item["source"] == seed or item["target"] == seed else 1,
            -max(degree[item["source"]], degree[item["target"]]),
            str(item.get("source", "")),
            str(item.get("target", "")),
        ),
    ):
        try_add_edge(edge)

    if not selected_edges:
        incident_edges = [
            edge for edge in raw_edges
            if edge["source"] == seed or edge["target"] == seed
        ]
        for edge in sorted(
            incident_edges,
            key=lambda item: (
                -degree[item["target"] if item["source"] == seed else item["source"]],
                str(item.get("file_path", "")),
            ),
        )[:28]:
            try_add_edge(edge)

    selected_names_ordered = [seed] + [
        name for name in sorted(
            selected_names - {seed},
            key=lambda item: (-multi_degree[item], -degree[item], item),
        )
    ]
    selected = set(selected_names_ordered)
    context_edges = [
        edge for edge in selected_edges
        if edge["source"] in selected and edge["target"] in selected
    ]
    context_edges = sorted(
        context_edges,
        key=lambda edge: (
            0 if len(edge_source_patents(edge)) >= 2 else 1,
            0 if edge["source"] == seed or edge["target"] == seed else 1,
            str(edge.get("source", "")),
            str(edge.get("target", "")),
        ),
    )[:40]
    nodes = [
        {
            "id": name,
            "degree": degree[name],
            "entity_type": raw_nodes.get(name, {}).get("entity_type", "other"),
        }
        for name in selected_names_ordered
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
        for edge in context_edges
    ]
    return (
        render_svg(
        nodes,
        edges,
        "Patent-100 LightRAG multi-patent relation cluster",
        (
            f"전체 graph에서 multi-patent relation이 가장 잘 보이는 entity '{seed}' 중심 cluster를 렌더링했다. "
            f"선택된 cluster patent={', '.join(sorted(cluster_patents)) or 'n/a'}. "
            "node color는 entity type, node size는 전체 graph degree, edge label은 LightRAG relation keywords 기준이다."
        ),
    )
        + "<h4>Relation samples shown in the graph</h4>"
        + "<p class='note'>표는 multi-patent relation을 먼저 정렬한다. <code>Patent_count &gt; 1</code>이면 동일한 entity relation이 둘 이상의 source patent에서 병합된 edge다.</p>"
        + relation_table(edges, 20)
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
    preferred_doc_ids = ["16-599306", "17-459782", "2023-0050666", "16-844988"]
    doc_id = ""
    pairs: list[Any] = []
    for preferred in preferred_doc_ids:
        record = full_relations.get(preferred)
        if isinstance(record, dict) and isinstance(record.get("relation_pairs"), list):
            doc_id = preferred
            pairs = record.get("relation_pairs", [])
            break
    if not doc_id:
        candidates = []
        for candidate_doc_id, record in full_relations.items():
            candidate_pairs = record.get("relation_pairs", []) if isinstance(record, dict) else []
            if isinstance(candidate_pairs, list) and 6 <= len(candidate_pairs) <= 40:
                candidates.append((abs(len(candidate_pairs) - 18), str(candidate_doc_id), candidate_pairs))
        if candidates:
            _, doc_id, pairs = sorted(candidates)[0]
    if not doc_id:
        candidates = []
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
        f"특허 {doc_id}에서 source chunk 기준으로 추출된 기술 entity/relation만 렌더링했다. "
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
    ], class_name="rpd-output-table")}
    {pre("P entity key-value profile", entity_profile, 3000)}
    {pre("P relation key-value profile", relation_profile, 3000)}
    <h4>D deduplication examples</h4>
    {table(dedup_entities, class_name="dedup-table")}
    {table(dedup_relations, class_name="dedup-table")}
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
    }], class_name="retrieval-summary-table")}
    {pre("Query metadata", metadata, 5000)}
    {table([
        {"context": "entities", "count": len(body.get("entities", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("entities", []) if isinstance(body, dict) else [])[:3], ensure_ascii=False), 1000)},
        {"context": "relationships", "count": len(body.get("relationships", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("relationships", []) if isinstance(body, dict) else [])[:3], ensure_ascii=False), 1000)},
        {"context": "chunks", "count": len(body.get("chunks", [])) if isinstance(body, dict) else 0, "sample": short(json.dumps((body.get("chunks", []) if isinstance(body, dict) else [])[:2], ensure_ascii=False), 1000)},
    ], class_name="context-sample-table")}
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
    return table(compare_rows, class_name="answer-compare-table")


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
        ],
        class_name="case-study-table",
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
    return f"{table(overview, class_name='structured-overview-table')}{examples}"


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
        ],
        class_name="prompt-rule-table",
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
        ],
        class_name="graph-quality-table",
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
        return table(rows, class_name="judge-win-table")
    return table(
        [
            {"Comparison": "LightRAG hybrid vs NaiveRAG", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "lightrag_naive", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "LightRAG hybrid vs GraphRAG global", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "graphrag_global", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "LightRAG hybrid vs GraphRAG local", "Metric": "—", "Left system": "lightrag_hybrid", "Left win-rate": "—", "Right system": "graphrag_local", "Right win-rate": "—", "Tie": "—"},
            {"Comparison": "GraphRAG global vs NaiveRAG", "Metric": "—", "Left system": "graphrag_global", "Left win-rate": "—", "Right system": "lightrag_naive", "Right win-rate": "—", "Tie": "—"},
        ],
        class_name="judge-win-table",
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
    return table(rows, class_name="length-stats-table")


def length_control_length_table(summary: dict[str, Any]) -> str:
    lengths = summary.get("lengths", {}) if isinstance(summary, dict) else {}
    before = lengths.get("before", {}) if isinstance(lengths, dict) else {}
    after = lengths.get("after", {}) if isinstance(lengths, dict) else {}
    target = lengths.get("target", {}) if isinstance(lengths, dict) else {}
    min_chars = target.get("min_chars", "—")
    max_chars = target.get("max_chars", "—")
    systems = ["lightrag_naive", "lightrag_hybrid", "graphrag_local", "graphrag_global"]
    rows = []
    for system in systems:
        b = before.get(system, {})
        a = after.get(system, {})
        if not b and not a:
            continue
        before_avg = float(b.get("avg", 0) or 0)
        after_avg = float(a.get("avg", 0) or 0)
        rows.append(
            {
                "System": system,
                "Original avg chars": f"{before_avg:.0f}" if before_avg else "—",
                "Normalized avg chars": f"{after_avg:.0f}" if after_avg else "—",
                "Change": f"{after_avg - before_avg:+.0f}" if before_avg and after_avg else "—",
                f"In target {min_chars}-{max_chars} before": f"{b.get('in_target', 0)}/{b.get('count', 0)}",
                f"In target {min_chars}-{max_chars} after": f"{a.get('in_target', 0)}/{a.get('count', 0)}",
            }
        )
    return table(rows, class_name="length-control-length-table")


def length_control_judge_table(summary: dict[str, Any]) -> str:
    if not summary:
        return "<p class='empty'>length-control judge pending</p>"
    stages = [
        ("Original answer + verbosity-aware judge", "judge_original_verbosity_aware"),
        ("Length-normalized answer + verbosity-aware judge", "judge_normalized_verbosity_aware"),
    ]
    pair_order = [
        "lightrag_hybrid__vs__lightrag_naive",
        "lightrag_hybrid__vs__graphrag_global",
        "lightrag_hybrid__vs__graphrag_local",
        "graphrag_global__vs__lightrag_naive",
    ]
    rows = []
    for stage_label, key in stages:
        stage = summary.get(key, {}) if isinstance(summary, dict) else {}
        pair_wins = stage.get("pair_wins", {}) if isinstance(stage, dict) else {}
        for pair_id in pair_order:
            left, right = pair_id.split("__vs__")
            overall = pair_wins.get(pair_id, {}).get("Overall", {})
            left_wins = int(overall.get(left, 0) or 0)
            right_wins = int(overall.get(right, 0) or 0)
            tie = int(overall.get("Tie", 0) or 0)
            winner = left if left_wins > right_wins else right if right_wins > left_wins else "Tie"
            rows.append(
                {
                    "Condition": stage_label,
                    "Left": left,
                    "Right": right,
                    "Left wins": left_wins,
                    "Right wins": right_wins,
                    "Tie": tie,
                    "Overall winner": winner,
                }
            )
    return table(rows, class_name="length-control-judge-table")


def query_type_pattern_table(summary: dict[str, Any]) -> str:
    normalized = (summary.get("judge_normalized_verbosity_aware") or {}) if isinstance(summary, dict) else {}
    query_type_wins = normalized.get("query_type_wins") or {}
    if not isinstance(query_type_wins, dict) or not query_type_wins:
        return "<p class='empty'>query type breakdown pending</p>"
    order = ["category_specific", "cross_category", "fact_check", "comparison", "exploratory"]
    rows = []
    for query_type in order:
        overall = (query_type_wins.get(query_type) or {}).get("Overall", {})
        if not isinstance(overall, dict):
            continue
        winner = max(overall.items(), key=lambda item: item[1])[0] if overall else "—"
        rows.append(
            {
                "Query type": query_type,
                "LightRAG hybrid": overall.get("lightrag_hybrid", 0),
                "Naive": overall.get("lightrag_naive", 0),
                "GraphRAG global": overall.get("graphrag_global", 0),
                "GraphRAG local": overall.get("graphrag_local", 0),
                "Tie": overall.get("Tie", 0),
                "Pattern": winner,
            }
        )
    return table(rows, class_name="query-type-table")


def keyword_extraction_summary(rows: list[dict[str, Any]], query_ids: list[str] | None = None) -> str:
    lookup = result_lookup(rows)
    selected_ids = query_ids or ["AA-1", "X-2", "F-1"]
    output_rows = []
    for query_id in selected_ids:
        row = lookup.get((query_id, "hybrid"))
        if not row:
            continue
        data = row.get("data", {})
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
        high_level = keywords.get("high_level", []) if isinstance(keywords, dict) else []
        low_level = keywords.get("low_level", []) if isinstance(keywords, dict) else []
        processing = metadata.get("processing_info", {}) if isinstance(metadata, dict) else {}
        output_rows.append(
            {
                "query_id": query_id,
                "question": row.get("question", ""),
                "high_level_keywords": ", ".join(map(str, high_level[:8])) if isinstance(high_level, list) and high_level else "—",
                "low_level_keywords": ", ".join(map(str, low_level[:10])) if isinstance(low_level, list) and low_level else "—",
                "retrieval_counts": (
                    f"E={processing.get('entities_after_truncation', '—')}, "
                    f"R={processing.get('relations_after_truncation', '—')}, "
                    f"C={processing.get('final_chunks_count', '—')}"
                ) if isinstance(processing, dict) else "—",
                "mode": row.get("mode", ""),
            }
        )
    return table(output_rows, class_name="keyword-summary-table")


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": manifest.get("seed"),
        "per_category": manifest.get("per_category"),
        "smoke_per_category": manifest.get("smoke_per_category"),
        "total_selected": manifest.get("total_selected"),
        "total_smoke": manifest.get("total_smoke"),
        "total_remaining": manifest.get("total_remaining"),
        "doc_count": manifest.get("doc_count"),
        "query_count": manifest.get("query_count"),
        "token_estimate": manifest.get("token_estimate"),
        "category_counts": manifest.get("category_counts"),
        "selected_ids_by_category": compact_for_report(manifest.get("selected_ids_by_category", {}), "selected_ids_by_category"),
        "files": manifest.get("files"),
    }


def graphrag_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = []
    for stage in manifest.get("stages", []) if isinstance(manifest.get("stages"), list) else []:
        if not isinstance(stage, dict):
            continue
        stages.append(
            {
                "stage": stage.get("stage"),
                "method": stage.get("method"),
                "command": stage.get("command"),
                "returncode": stage.get("returncode"),
                "elapsed_seconds": stage.get("elapsed_seconds"),
                "root": stage.get("root"),
                "repo": stage.get("repo"),
                "tag": stage.get("tag"),
                "commit": stage.get("commit"),
                "output_excerpt": "[progress output omitted]" if stage.get("output_excerpt") else "—",
            }
        )
    return {
        "experiment_dir": manifest.get("experiment_dir"),
        "external_dir": manifest.get("external_dir"),
        "tag": manifest.get("tag"),
        "stage_count": len(stages),
        "stages": stages,
    }


def cost_runtime_summary_table(
    index_stats: dict[str, Any],
    light_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    graph_stats: dict[str, Any],
    judge_rows: list[dict[str, Any]],
    length_control_summary: dict[str, Any],
) -> str:
    light_query_seconds = sum(float(row.get("elapsed_seconds") or 0) for row in light_rows)
    graph_query_seconds = sum(float(row.get("elapsed_seconds") or 0) for row in graph_rows)
    length_counts = ((length_control_summary.get("lengths") or {}).get("counts") or {}) if isinstance(length_control_summary, dict) else {}
    original_judge = (length_control_summary.get("judge_original_verbosity_aware") or {}) if isinstance(length_control_summary, dict) else {}
    normalized_judge = (length_control_summary.get("judge_normalized_verbosity_aware") or {}) if isinstance(length_control_summary, dict) else {}
    return table(
        [
            {
                "Stage": "LightRAG indexing",
                "Runtime": format_seconds(index_stats.get("elapsed_seconds")),
                "Total attempts": format_seconds(index_stats.get("elapsed_seconds_total_attempts")),
                "Records / call proxy": f"{index_stats.get('processed_documents', 0)}/{index_stats.get('document_count', 0)} docs, attempts={index_stats.get('attempt_count', '—')}",
                "Model path": f"{index_stats.get('llm_model', '—')} + {index_stats.get('embedding_model', '—')}",
            },
            {
                "Stage": "LightRAG query",
                "Runtime": format_seconds(light_query_seconds),
                "Total attempts": "—",
                "Records / call proxy": f"{len(light_rows)} records = 15 queries x 4 modes",
                "Model path": "gpt-5.5 answering, LightRAG naive/local/global/hybrid",
            },
            {
                "Stage": "GraphRAG indexing",
                "Runtime": format_seconds(graph_stats.get("total_runtime")),
                "Total attempts": "fallback full fresh",
                "Records / call proxy": f"{graph_stats.get('num_documents', '—')} docs, workflows={len(graph_stats.get('workflow_seconds', {}) or {})}",
                "Model path": "gpt-5.5 extraction/community reports",
            },
            {
                "Stage": "GraphRAG query",
                "Runtime": format_seconds(graph_query_seconds),
                "Total attempts": "repair after partial results",
                "Records / call proxy": f"{len(graph_rows)} records = 15 queries x 3 methods",
                "Model path": "GraphRAG basic/local/global",
            },
            {
                "Stage": "Gemini judge",
                "Runtime": "not timed in one batch",
                "Total attempts": "original + length-control",
                "Records / call proxy": (
                    f"base={sum(1 for row in judge_rows if row.get('status') == 'success')}/60, "
                    f"verbosity attempts={original_judge.get('attempt_rows', '—')}+{normalized_judge.get('attempt_rows', '—')}, "
                    f"normalization={length_counts.get('normalized_success', '—')}/{length_counts.get('normalized_rows', '—')}"
                ),
                "Model path": "Gemini 3.5 Flash judge, gpt-5.5 normalization",
            },
        ],
        class_name="cost-summary-table",
    )


def cost_raw_excerpt(
    index_stats: dict[str, Any],
    graph_metrics: dict[str, Any],
    graph_stats: dict[str, Any],
    graph_mode_rows: list[dict[str, Any]],
    judge_summary: dict[str, Any],
    length_control_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lightrag_index": {
            "document_count": index_stats.get("document_count"),
            "processed_documents": index_stats.get("processed_documents"),
            "elapsed_seconds_successful_attempt": index_stats.get("elapsed_seconds"),
            "elapsed_seconds_total_attempts": index_stats.get("elapsed_seconds_total_attempts"),
            "attempt_count": index_stats.get("attempt_count"),
            "llm_model": index_stats.get("llm_model"),
            "reasoning_effort": index_stats.get("reasoning_effort"),
            "embedding_model": index_stats.get("embedding_model"),
        },
        "lightrag_graph": {
            "graph_nodes": graph_metrics.get("graph_nodes"),
            "graph_edges": graph_metrics.get("graph_edges"),
            "metadata_relation_ratio": graph_metrics.get("metadata_relation_ratio"),
            "technical_relation_ratio": graph_metrics.get("technical_relation_ratio"),
            "excluded_entity_ratio": graph_metrics.get("excluded_entity_ratio"),
        },
        "graphrag_index": {
            "num_documents": graph_stats.get("num_documents"),
            "total_runtime": graph_stats.get("total_runtime"),
            "workflow_seconds": graph_stats.get("workflow_seconds"),
        },
        "graphrag_query_summary": graph_mode_rows,
        "judge": {
            "judge_model": judge_summary.get("judge_model"),
            "total_judgments": judge_summary.get("total_judgments"),
            "pair_wins": judge_summary.get("pair_wins"),
        },
        "length_control_counts": (length_control_summary.get("lengths") or {}).get("counts", {}) if isinstance(length_control_summary, dict) else {},
    }


def mode_ablation_display_rows(auto_metrics: dict[str, Any], light_mode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = (auto_metrics.get("system_summary") or light_mode_rows) if isinstance(auto_metrics, dict) else light_mode_rows
    display_rows = []
    for row in raw_rows:
        display_rows.append(
            {
                "System": row.get("system") or row.get("Mode") or row.get("Method") or "—",
                "Q": row.get("queries") or row.get("Queries") or "—",
                "OK": row.get("success") or row.get("Success answers") or "—",
                "Empty": row.get("empty_answers") if "empty_answers" in row else row.get("Empty/failed", "—"),
                "Avg chars": row.get("avg_answer_chars") or row.get("Avg answer chars") or "—",
                "Ent.": row.get("avg_retrieved_entities") if "avg_retrieved_entities" in row else row.get("Avg entities", "—"),
                "Rel.": row.get("avg_retrieved_relations") if "avg_retrieved_relations" in row else row.get("Avg relations", "—"),
                "Chunks": row.get("avg_retrieved_chunks") if "avg_retrieved_chunks" in row else row.get("Avg chunks", "—"),
                "Sources": row.get("avg_unique_source_patents") if "avg_unique_source_patents" in row else row.get("Avg source patents", "—"),
                "Latency(s)": row.get("avg_latency_seconds") if "avg_latency_seconds" in row else row.get("Avg latency", "—"),
            }
        )
    return display_rows


def appendix_data_summary_table(manifest: dict[str, Any]) -> str:
    token_estimate = manifest.get("token_estimate") or {}
    files = manifest.get("files") or {}
    return table(
        [
            {"Item": "Seed", "Value": manifest.get("seed", "—"), "Note": "random sampling reproducibility"},
            {"Item": "Sampling plan", "Value": f"{', '.join(manifest.get('categories', []) or [])} x {manifest.get('per_category', '—')}", "Note": f"total={manifest.get('total_selected', '—')} patents"},
            {"Item": "GraphRAG smoke split", "Value": f"{manifest.get('smoke_per_category', '—')} per category", "Note": f"smoke={manifest.get('total_smoke', '—')}, remaining={manifest.get('total_remaining', '—')}"},
            {"Item": "Query set", "Value": manifest.get("query_count", "—"), "Note": "fixed manual queries, no model-generated QA"},
            {"Item": "Token estimate", "Value": f"total={token_estimate.get('total', '—')}, avg={token_estimate.get('average', '—')}", "Note": "structured patent text estimate"},
            {"Item": "Dataset file", "Value": files.get("patents_100_jsonl", "—"), "Note": "print shows summary only"},
            {"Item": "Query file", "Value": files.get("queries_jsonl", "—"), "Note": "15-query JSONL"},
        ],
        class_name="appendix-summary-table",
    )


def appendix_category_sampling_table(manifest: dict[str, Any]) -> str:
    selected = manifest.get("selected_ids_by_category") or {}
    smoke = manifest.get("smoke_ids_by_category") or {}
    remaining = manifest.get("remaining_ids_by_category") or {}
    category_counts = manifest.get("category_counts") or {}
    rows = []
    for category in manifest.get("categories", []) or ["AA", "AB", "AC", "AD"]:
        selected_ids = selected.get(category, []) if isinstance(selected, dict) else []
        rows.append(
            {
                "Category": category,
                "Selected": category_counts.get(category, len(selected_ids)),
                "Smoke": len(smoke.get(category, [])) if isinstance(smoke, dict) else "—",
                "Remaining": len(remaining.get(category, [])) if isinstance(remaining, dict) else "—",
                "Patent ID sample": ", ".join(map(str, selected_ids[:5])) if isinstance(selected_ids, list) else "—",
            }
        )
    return table(rows, class_name="category-sampling-table")


def appendix_manifest_excerpt(manifest: dict[str, Any]) -> dict[str, Any]:
    selected = manifest.get("selected_ids_by_category") or {}
    return {
        "seed": manifest.get("seed"),
        "per_category": manifest.get("per_category"),
        "categories": manifest.get("categories"),
        "total_selected": manifest.get("total_selected"),
        "total_smoke": manifest.get("total_smoke"),
        "total_remaining": manifest.get("total_remaining"),
        "doc_count": manifest.get("doc_count"),
        "query_count": manifest.get("query_count"),
        "token_estimate": manifest.get("token_estimate"),
        "category_counts": manifest.get("category_counts"),
        "selected_id_sample_by_category": {
            category: ids[:8] if isinstance(ids, list) else ids
            for category, ids in selected.items()
        } if isinstance(selected, dict) else {},
        "files": manifest.get("files"),
    }


def length_control_audit_table(summary: dict[str, Any]) -> str:
    if not summary:
        return "<p class='empty'>length-control audit pending</p>"
    lengths = summary.get("lengths", {})
    counts = lengths.get("counts", {}) if isinstance(lengths, dict) else {}
    rows = [
        {
            "Audit": "Normalization",
            "Value": f"{counts.get('normalized_success', 0)}/{counts.get('normalized_rows', 0)} success",
            "Note": "same answer evidence, target 1100-1300 chars",
        }
    ]
    for label, key in [
        ("Original verbosity-aware judge", "judge_original_verbosity_aware"),
        ("Normalized verbosity-aware judge", "judge_normalized_verbosity_aware"),
    ]:
        stage = summary.get(key, {}) if isinstance(summary, dict) else {}
        rows.append(
            {
                "Audit": label,
                "Value": f"{stage.get('deduped_success_rows', 0)}/60 unique successes",
                "Note": (
                    f"attempts={stage.get('attempt_rows', 0)}, "
                    f"failed attempts recovered={stage.get('failure_attempt_rows', 0)}"
                ),
            }
        )
    return table(rows, class_name="audit-note-table")


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
    length_control_summary: dict[str, Any] | None = None,
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
    lc = length_control_summary or {}
    lc_pairs = ((lc.get("judge_normalized_verbosity_aware") or {}).get("pair_wins") or {}) if isinstance(lc, dict) else {}
    lc_original_pairs = ((lc.get("judge_original_verbosity_aware") or {}).get("pair_wins") or {}) if isinstance(lc, dict) else {}
    lc_hybrid_naive = (lc_pairs.get("lightrag_hybrid__vs__lightrag_naive") or {}).get("Overall", {})
    lc_hybrid_global = (lc_pairs.get("lightrag_hybrid__vs__graphrag_global") or {}).get("Overall", {})
    lc_hybrid_local = (lc_pairs.get("lightrag_hybrid__vs__graphrag_local") or {}).get("Overall", {})
    lc_global_naive = (lc_pairs.get("graphrag_global__vs__lightrag_naive") or {}).get("Overall", {})
    lc_original_hybrid_global = (lc_original_pairs.get("lightrag_hybrid__vs__graphrag_global") or {}).get("Overall", {})
    lc_lengths = ((lc.get("lengths") or {}).get("after") or {}) if isinstance(lc, dict) else {}
    lc_hybrid_chars = float((lc_lengths.get("lightrag_hybrid") or {}).get("avg", 0) or 0)
    lc_global_chars = float((lc_lengths.get("graphrag_global") or {}).get("avg", 0) or 0)
    lc_global_in_target = (lc_lengths.get("graphrag_global") or {}).get("in_target", 0)
    lc_global_count = (lc_lengths.get("graphrag_global") or {}).get("count", 0)
    lc_global_residual_ratio = lc_global_chars / lc_hybrid_chars if lc_hybrid_chars else 0.0
    lc_query_types = ((lc.get("judge_normalized_verbosity_aware") or {}).get("query_type_wins") or {}) if isinstance(lc, dict) else {}
    qt_category = (lc_query_types.get("category_specific") or {}).get("Overall", {})
    qt_cross = (lc_query_types.get("cross_category") or {}).get("Overall", {})
    qt_fact = (lc_query_types.get("fact_check") or {}).get("Overall", {})
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
            "Interpretation": "다만 이 비교 역시 GraphRAG global의 답변 길이가 길다는 confound를 공유한다. 길이 정규화 후에는 GraphRAG global 8 vs naive 7로 좁아졌으므로, 원본 답변 기준 우위는 보수적으로 해석한다.",
        },
    ]
    if lc_pairs:
        rows.extend(
            [
                {
                    "Finding": "Length normalization is incomplete",
                    "Evidence": (
                        f"After normalization: GraphRAG global avg={lc_global_chars:.0f} chars, "
                        f"LightRAG hybrid avg={lc_hybrid_chars:.0f} chars, residual ratio={lc_global_residual_ratio:.2f}x; "
                        f"GraphRAG global in target={lc_global_in_target}/{lc_global_count}"
                    ),
                    "Interpretation": "1100-1300자 목표로 압축했지만 근거 보존 제약 때문에 GraphRAG global이 여전히 더 길다. 따라서 normalized 결과도 완전한 동일 길이 비교가 아니라 verbosity-aware 보정 실험으로 해석해야 한다.",
                },
                {
                    "Finding": "Verbosity-aware re-judge reduces but does not remove the GraphRAG global effect",
                    "Evidence": (
                        f"After length normalization: GraphRAG global {lc_hybrid_global.get('graphrag_global', 0)} "
                        f"vs LightRAG hybrid {lc_hybrid_global.get('lightrag_hybrid', 0)}; "
                        f"GraphRAG global {lc_global_naive.get('graphrag_global', 0)} vs naive {lc_global_naive.get('lightrag_naive', 0)}"
                    ),
                    "Interpretation": "길이 보정 후에도 GraphRAG global은 hybrid 대비 우세하지만, naive 대비 우위는 8:7로 좁아져 답변 길이와 global community summary 효과가 함께 작용한 것으로 해석해야 한다.",
                },
                {
                    "Finding": "GraphRAG global becomes stronger after compression",
                    "Evidence": (
                        f"Original verbosity-aware judge: GraphRAG global {lc_original_hybrid_global.get('graphrag_global', 0)} "
                        f"vs hybrid {lc_original_hybrid_global.get('lightrag_hybrid', 0)}; "
                        f"normalized judge: GraphRAG global {lc_hybrid_global.get('graphrag_global', 0)} "
                        f"vs hybrid {lc_hybrid_global.get('lightrag_hybrid', 0)}"
                    ),
                    "Interpretation": "가능한 해석은 두 가지다. 첫째, GraphRAG global의 긴 community-summary 답변은 핵심 내용 밀도가 높아 압축 후에도 정보량이 유지됐을 수 있다. 둘째, 정규화 LLM이 GraphRAG 답변의 구조화된 핵심을 더 잘 보존하고 LightRAG 답변에서는 근거 일부를 압축 과정에서 잃었을 수 있다.",
                },
                {
                    "Finding": "LightRAG graph context still does not beat chunk-only naive",
                    "Evidence": (
                        f"After length normalization: naive {lc_hybrid_naive.get('lightrag_naive', 0)} "
                        f"vs LightRAG hybrid {lc_hybrid_naive.get('lightrag_hybrid', 0)}"
                    ),
                    "Interpretation": "특허 문서가 이미 구조화되어 있어 chunk retrieval만으로도 충분한 근거를 잡는 경우가 많고, hybrid graph context의 추가 정보가 judge 기준에서 일관된 이득으로 이어지지 않았다.",
                },
                {
                    "Finding": "GraphRAG local remains unsuitable for this broad query set",
                    "Evidence": (
                        f"After length normalization: LightRAG hybrid {lc_hybrid_local.get('lightrag_hybrid', 0)} "
                        f"vs GraphRAG local {lc_hybrid_local.get('graphrag_local', 0)}"
                    ),
                    "Interpretation": "질문셋이 기술 관계·동향 중심이라 local entity-description retrieval보다 LightRAG hybrid 또는 GraphRAG global 방식이 더 적합했다.",
                },
                {
                    "Finding": "Query type changes the conclusion",
                    "Evidence": (
                        f"category_specific Overall: GraphRAG global={qt_category.get('graphrag_global', 0)}; "
                        f"cross_category Overall: LightRAG hybrid={qt_cross.get('lightrag_hybrid', 0)} vs GraphRAG global={qt_cross.get('graphrag_global', 0)}; "
                        f"fact_check Overall: LightRAG hybrid={qt_fact.get('lightrag_hybrid', 0)} vs naive={qt_fact.get('lightrag_naive', 0)}"
                    ),
                    "Interpretation": "category-specific broad 질문은 GraphRAG global community summary가 강하고, cross-category 질문은 LightRAG hybrid의 entity/relation graph context가 더 유효하게 작동했다. fact-check에서도 hybrid가 naive보다 앞서지만 표본이 2개라 보조 근거로만 본다.",
                },
            ]
        )
        conclusion = (
            "length normalization과 verbosity-aware judge까지 반영한 보수적 결론은 다음과 같다. "
            "LightRAG의 특허 특화 graph 구축은 품질 지표상 성공했지만, 답변 품질에서는 LightRAG hybrid가 naive를 명확히 이기지 못했다. "
            "GraphRAG global은 category-specific broad query에서 강하지만, 길이 보정 후에도 평균 길이 차이가 남아 있어 완전한 동일 길이 비교는 아니다. "
            "반면 cross-category query에서는 LightRAG hybrid가 GraphRAG global을 앞서므로, graph context의 가치는 질문 유형별로 분리해서 판단해야 한다."
        )
    else:
        conclusion = (
            "현재 결과의 보수적 결론은 다음과 같다. LightRAG의 특허 특화 graph 구축은 정량 품질 지표상 성공했다. "
            "그러나 답변 품질 평가에서는 LightRAG hybrid가 naive 대비 뚜렷한 우위를 보이지 않았고, GraphRAG global의 강한 승률은 답변 길이 효과와 분리해서 해석해야 한다. "
            "따라서 다음 단계에서는 동일 길이 조건 또는 judge prompt의 verbosity penalty를 둔 재평가가 필요하다."
        )
    return f"""
    <h2 id="discussion">Discussion and Conclusion</h2>
    <p>이번 실험은 Patent-100에서 LightRAG graph indexing, GraphRAG indexing/query, Gemini judge 평가까지 재현 가능한 pipeline으로 완주했다. 다만 judge win-rate를 그대로 성능 결론으로 읽기에는 답변 길이 편향이 가장 큰 교란 요인이다.</p>
    {table(rows, class_name="discussion-table")}
    <h3>Conclusion</h3>
    <p>{html_escape(conclusion)}</p>
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
        ],
        class_name="naive-light-table",
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
    length_control_dir = experiment_dir / "evaluation_length_control"
    length_control_summary = read_json(length_control_dir / "judge_length_control_summary.json", {}) or {}
    length_control_original_rows = read_jsonl(length_control_dir / "judge_original_verbosity_aware.jsonl")
    length_control_normalized_rows = read_jsonl(length_control_dir / "judge_normalized_verbosity_aware.jsonl")
    queries = read_jsonl(experiment_dir / "queries" / "eval_queries_15.jsonl")
    light_rows, graph_rows = load_all_query_rows(experiment_dir)
    light_mode_rows = lightrag_mode_summary(light_rows)
    length_stats = answer_length_stats(light_rows, graph_rows)
    graph_stats = load_graphrag_stats(experiment_dir)
    graph_mode_rows = graphrag_query_summary(graph_rows)
    graph_recovery = graphrag_recovery_rows(experiment_dir, graph_rows, queries)
    graphrag_manifest = read_json(experiment_dir / "graphrag_manifest.json", {}) or {}
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
    main {{ max-width:1280px; margin:0 auto; padding:32px 32px 80px; }}
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
    .table-wrap {{ margin:16px 0 28px; overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:white; }}
    .table-explain {{ margin:-18px 0 26px; padding:9px 12px; border-left:3px solid #cbd5e1; background:#f8fafc; color:#475569; font-size:13px; }}
    table {{ width:100%; min-width:760px; border-collapse:separate; border-spacing:0; margin:0; font-size:13px; table-layout:auto; }}
    th,td {{ border:0; border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:10px 12px; vertical-align:top; line-height:1.55; word-break:keep-all; overflow-wrap:anywhere; }}
    th {{ background:#eef2f7; text-align:left; color:#0f172a; font-size:12px; font-weight:700; }}
    td {{ background:white; }}
    table.relation-table {{ min-width:1120px; table-layout:fixed; }}
    table.relation-table th:nth-child(1), table.relation-table td:nth-child(1) {{ width:16%; }}
    table.relation-table th:nth-child(2), table.relation-table td:nth-child(2) {{ width:15%; }}
    table.relation-table th:nth-child(3), table.relation-table td:nth-child(3) {{ width:16%; }}
    table.relation-table th:nth-child(4), table.relation-table td:nth-child(4) {{ width:31%; }}
    table.relation-table th:nth-child(5), table.relation-table td:nth-child(5) {{ width:14%; }}
    table.relation-table th:nth-child(6), table.relation-table td:nth-child(6) {{ width:8%; }}
    table.relation-table th, table.relation-table td {{ overflow-wrap:anywhere; }}
    table.answer-compare-table {{ table-layout:fixed; }}
    table.answer-compare-table th:nth-child(1), table.answer-compare-table td:nth-child(1) {{ width:8%; }}
    table.answer-compare-table th:nth-child(2), table.answer-compare-table td:nth-child(2) {{ width:9%; }}
    table.answer-compare-table th:nth-child(3), table.answer-compare-table td:nth-child(3) {{ width:8%; }}
    table.answer-compare-table th:nth-child(4), table.answer-compare-table td:nth-child(4) {{ width:12%; }}
    table.answer-compare-table th:nth-child(5), table.answer-compare-table td:nth-child(5) {{ width:63%; }}
    table.case-study-table {{ table-layout:fixed; }}
    table.case-study-table th:nth-child(1), table.case-study-table td:nth-child(1) {{ width:7%; }}
    table.case-study-table th:nth-child(2), table.case-study-table td:nth-child(2) {{ width:31%; }}
    table.case-study-table th:nth-child(3), table.case-study-table td:nth-child(3) {{ width:31%; }}
    table.case-study-table th:nth-child(4), table.case-study-table td:nth-child(4) {{ width:31%; }}
    table.naive-light-table {{ table-layout:fixed; }}
    table.naive-light-table th:nth-child(1), table.naive-light-table td:nth-child(1) {{ width:6%; }}
    table.naive-light-table th:nth-child(2), table.naive-light-table td:nth-child(2) {{ width:7%; }}
    table.naive-light-table th:nth-child(3), table.naive-light-table td:nth-child(3) {{ width:8%; }}
    table.naive-light-table th:nth-child(4), table.naive-light-table td:nth-child(4) {{ width:10%; }}
    table.naive-light-table th:nth-child(5), table.naive-light-table td:nth-child(5) {{ width:7%; }}
    table.naive-light-table th:nth-child(6), table.naive-light-table td:nth-child(6) {{ width:10%; }}
    table.naive-light-table th:nth-child(7), table.naive-light-table td:nth-child(7) {{ width:52%; }}
    table.cost-summary-table {{ table-layout:fixed; }}
    table.cost-summary-table th:nth-child(1), table.cost-summary-table td:nth-child(1) {{ width:14%; }}
    table.cost-summary-table th:nth-child(2), table.cost-summary-table td:nth-child(2) {{ width:13%; }}
    table.cost-summary-table th:nth-child(3), table.cost-summary-table td:nth-child(3) {{ width:13%; }}
    table.cost-summary-table th:nth-child(4), table.cost-summary-table td:nth-child(4) {{ width:30%; }}
    table.cost-summary-table th:nth-child(5), table.cost-summary-table td:nth-child(5) {{ width:30%; }}
    table.appendix-summary-table {{ table-layout:fixed; }}
    table.appendix-summary-table th:nth-child(1), table.appendix-summary-table td:nth-child(1) {{ width:18%; }}
    table.appendix-summary-table th:nth-child(2), table.appendix-summary-table td:nth-child(2) {{ width:44%; }}
    table.appendix-summary-table th:nth-child(3), table.appendix-summary-table td:nth-child(3) {{ width:38%; }}
    table.two-col-table,
    table.status-table,
    table.structured-overview-table,
    table.prompt-rule-table,
    table.rpd-output-table,
    table.dedup-table,
    table.graph-quality-table,
    table.retrieval-summary-table,
    table.context-sample-table,
    table.complexity-table,
    table.audit-table,
    table.audit-note-table,
    table.mode-summary-table,
    table.dataset-table,
    table.judge-win-table,
    table.length-stats-table,
    table.length-control-length-table,
    table.length-control-judge-table,
    table.query-type-table,
    table.mode-ablation-table,
    table.discussion-table,
    table.category-sampling-table,
    table.query-set-table,
    table.keyword-summary-table,
    table.judge-plan-table {{ table-layout:fixed; }}
    table.two-col-table th:nth-child(1), table.two-col-table td:nth-child(1) {{ width:32%; }}
    table.two-col-table th:nth-child(2), table.two-col-table td:nth-child(2) {{ width:68%; }}
    table.status-table th:nth-child(1), table.status-table td:nth-child(1) {{ width:24%; }}
    table.status-table th:nth-child(2), table.status-table td:nth-child(2) {{ width:18%; }}
    table.status-table th:nth-child(3), table.status-table td:nth-child(3) {{ width:58%; }}
    table.structured-overview-table th:nth-child(1), table.structured-overview-table td:nth-child(1) {{ width:14%; }}
    table.structured-overview-table th:nth-child(2), table.structured-overview-table td:nth-child(2) {{ width:16%; }}
    table.structured-overview-table th:nth-child(3), table.structured-overview-table td:nth-child(3) {{ width:38%; }}
    table.structured-overview-table th:nth-child(4), table.structured-overview-table td:nth-child(4) {{ width:32%; }}
    table.prompt-rule-table th:nth-child(1), table.prompt-rule-table td:nth-child(1) {{ width:24%; }}
    table.prompt-rule-table th:nth-child(2), table.prompt-rule-table td:nth-child(2) {{ width:76%; }}
    table.rpd-output-table th:nth-child(1), table.rpd-output-table td:nth-child(1) {{ width:18%; }}
    table.rpd-output-table th:nth-child(2), table.rpd-output-table td:nth-child(2) {{ width:10%; }}
    table.rpd-output-table th:nth-child(3), table.rpd-output-table td:nth-child(3) {{ width:72%; }}
    table.dedup-table th:nth-child(1), table.dedup-table td:nth-child(1) {{ width:24%; }}
    table.dedup-table th:nth-child(2), table.dedup-table td:nth-child(2) {{ width:12%; }}
    table.dedup-table th:nth-child(3), table.dedup-table td:nth-child(3) {{ width:64%; }}
    table.graph-quality-table th:nth-child(1), table.graph-quality-table td:nth-child(1) {{ width:24%; }}
    table.graph-quality-table th:nth-child(2), table.graph-quality-table td:nth-child(2) {{ width:12%; }}
    table.graph-quality-table th:nth-child(3), table.graph-quality-table td:nth-child(3) {{ width:18%; }}
    table.graph-quality-table th:nth-child(4), table.graph-quality-table td:nth-child(4) {{ width:46%; }}
    table.retrieval-summary-table th:nth-child(1), table.retrieval-summary-table td:nth-child(1) {{ width:8%; }}
    table.retrieval-summary-table th:nth-child(2), table.retrieval-summary-table td:nth-child(2) {{ width:7%; }}
    table.retrieval-summary-table th:nth-child(3), table.retrieval-summary-table td:nth-child(3) {{ width:58%; }}
    table.retrieval-summary-table th:nth-child(4), table.retrieval-summary-table td:nth-child(4) {{ width:10%; }}
    table.retrieval-summary-table th:nth-child(5), table.retrieval-summary-table td:nth-child(5) {{ width:17%; }}
    table.context-sample-table th:nth-child(1), table.context-sample-table td:nth-child(1) {{ width:15%; }}
    table.context-sample-table th:nth-child(2), table.context-sample-table td:nth-child(2) {{ width:8%; }}
    table.context-sample-table th:nth-child(3), table.context-sample-table td:nth-child(3) {{ width:77%; }}
    table.complexity-table th:nth-child(1), table.complexity-table td:nth-child(1) {{ width:12%; }}
    table.complexity-table th:nth-child(2), table.complexity-table td:nth-child(2) {{ width:12%; }}
    table.complexity-table th:nth-child(3), table.complexity-table td:nth-child(3) {{ width:12%; }}
    table.complexity-table th:nth-child(4), table.complexity-table td:nth-child(4) {{ width:10%; }}
    table.complexity-table th:nth-child(5), table.complexity-table td:nth-child(5) {{ width:9%; }}
    table.complexity-table th:nth-child(6), table.complexity-table td:nth-child(6) {{ width:17%; }}
    table.complexity-table th:nth-child(7), table.complexity-table td:nth-child(7) {{ width:28%; }}
    table.audit-table th:nth-child(1), table.audit-table td:nth-child(1) {{ width:24%; }}
    table.audit-table th:nth-child(2), table.audit-table td:nth-child(2) {{ width:76%; }}
    table.audit-note-table th:nth-child(1), table.audit-note-table td:nth-child(1) {{ width:22%; }}
    table.audit-note-table th:nth-child(2), table.audit-note-table td:nth-child(2) {{ width:34%; }}
    table.audit-note-table th:nth-child(3), table.audit-note-table td:nth-child(3) {{ width:44%; }}
    table.mode-summary-table th:nth-child(1), table.mode-summary-table td:nth-child(1) {{ width:10%; }}
    table.mode-summary-table th:nth-child(2), table.mode-summary-table td:nth-child(2) {{ width:8%; }}
    table.mode-summary-table th:nth-child(3), table.mode-summary-table td:nth-child(3) {{ width:14%; }}
    table.mode-summary-table th:nth-child(4), table.mode-summary-table td:nth-child(4) {{ width:12%; }}
    table.mode-summary-table th:nth-child(5), table.mode-summary-table td:nth-child(5) {{ width:12%; }}
    table.mode-summary-table th:nth-child(6), table.mode-summary-table td:nth-child(6) {{ width:12%; }}
    table.mode-summary-table th:nth-child(7), table.mode-summary-table td:nth-child(7) {{ width:14%; }}
    table.mode-summary-table th:nth-child(8), table.mode-summary-table td:nth-child(8) {{ width:18%; }}
    table.dataset-table th:nth-child(1), table.dataset-table td:nth-child(1) {{ width:22%; }}
    table.dataset-table th:nth-child(2), table.dataset-table td:nth-child(2),
    table.dataset-table th:nth-child(3), table.dataset-table td:nth-child(3),
    table.dataset-table th:nth-child(4), table.dataset-table td:nth-child(4) {{ width:26%; }}
    table.judge-win-table th:nth-child(1), table.judge-win-table td:nth-child(1) {{ width:26%; }}
    table.judge-win-table th:nth-child(2), table.judge-win-table td:nth-child(2) {{ width:17%; }}
    table.judge-win-table th:nth-child(3), table.judge-win-table td:nth-child(3) {{ width:14%; }}
    table.judge-win-table th:nth-child(4), table.judge-win-table td:nth-child(4) {{ width:11%; }}
    table.judge-win-table th:nth-child(5), table.judge-win-table td:nth-child(5) {{ width:14%; }}
    table.judge-win-table th:nth-child(6), table.judge-win-table td:nth-child(6) {{ width:11%; }}
    table.judge-win-table th:nth-child(7), table.judge-win-table td:nth-child(7) {{ width:7%; }}
    table.length-stats-table th:nth-child(1), table.length-stats-table td:nth-child(1) {{ width:18%; }}
    table.length-stats-table th:nth-child(2), table.length-stats-table td:nth-child(2) {{ width:8%; }}
    table.length-stats-table th:nth-child(3), table.length-stats-table td:nth-child(3),
    table.length-stats-table th:nth-child(4), table.length-stats-table td:nth-child(4),
    table.length-stats-table th:nth-child(5), table.length-stats-table td:nth-child(5),
    table.length-stats-table th:nth-child(6), table.length-stats-table td:nth-child(6) {{ width:12%; }}
    table.length-stats-table th:nth-child(7), table.length-stats-table td:nth-child(7) {{ width:26%; }}
    table.length-control-length-table th:nth-child(1), table.length-control-length-table td:nth-child(1) {{ width:15%; }}
    table.length-control-length-table th:nth-child(2), table.length-control-length-table td:nth-child(2),
    table.length-control-length-table th:nth-child(3), table.length-control-length-table td:nth-child(3) {{ width:16%; }}
    table.length-control-length-table th:nth-child(4), table.length-control-length-table td:nth-child(4) {{ width:9%; }}
    table.length-control-length-table th:nth-child(5), table.length-control-length-table td:nth-child(5),
    table.length-control-length-table th:nth-child(6), table.length-control-length-table td:nth-child(6) {{ width:22%; }}
    table.length-control-judge-table th:nth-child(1), table.length-control-judge-table td:nth-child(1) {{ width:30%; }}
    table.length-control-judge-table th:nth-child(2), table.length-control-judge-table td:nth-child(2),
    table.length-control-judge-table th:nth-child(3), table.length-control-judge-table td:nth-child(3) {{ width:15%; }}
    table.length-control-judge-table th:nth-child(4), table.length-control-judge-table td:nth-child(4),
    table.length-control-judge-table th:nth-child(5), table.length-control-judge-table td:nth-child(5) {{ width:8%; }}
    table.length-control-judge-table th:nth-child(6), table.length-control-judge-table td:nth-child(6) {{ width:6%; }}
    table.length-control-judge-table th:nth-child(7), table.length-control-judge-table td:nth-child(7) {{ width:18%; }}
    table.query-type-table th:nth-child(1), table.query-type-table td:nth-child(1) {{ width:20%; }}
    table.query-type-table th:nth-child(2), table.query-type-table td:nth-child(2) {{ width:15%; }}
    table.query-type-table th:nth-child(3), table.query-type-table td:nth-child(3) {{ width:8%; }}
    table.query-type-table th:nth-child(4), table.query-type-table td:nth-child(4) {{ width:18%; }}
    table.query-type-table th:nth-child(5), table.query-type-table td:nth-child(5) {{ width:16%; }}
    table.query-type-table th:nth-child(6), table.query-type-table td:nth-child(6) {{ width:6%; }}
    table.query-type-table th:nth-child(7), table.query-type-table td:nth-child(7) {{ width:17%; }}
    table.mode-ablation-table {{ font-size:12px; }}
    table.mode-ablation-table th:nth-child(1), table.mode-ablation-table td:nth-child(1) {{ width:18%; }}
    table.mode-ablation-table th:nth-child(2), table.mode-ablation-table td:nth-child(2),
    table.mode-ablation-table th:nth-child(3), table.mode-ablation-table td:nth-child(3) {{ width:5%; }}
    table.mode-ablation-table th:nth-child(4), table.mode-ablation-table td:nth-child(4) {{ width:6%; }}
    table.mode-ablation-table th:nth-child(5), table.mode-ablation-table td:nth-child(5) {{ width:12%; }}
    table.mode-ablation-table th:nth-child(6), table.mode-ablation-table td:nth-child(6),
    table.mode-ablation-table th:nth-child(7), table.mode-ablation-table td:nth-child(7),
    table.mode-ablation-table th:nth-child(8), table.mode-ablation-table td:nth-child(8) {{ width:8%; }}
    table.mode-ablation-table th:nth-child(9), table.mode-ablation-table td:nth-child(9) {{ width:13%; }}
    table.mode-ablation-table th:nth-child(10), table.mode-ablation-table td:nth-child(10) {{ width:17%; }}
    table.discussion-table th:nth-child(1), table.discussion-table td:nth-child(1) {{ width:24%; }}
    table.discussion-table th:nth-child(2), table.discussion-table td:nth-child(2) {{ width:34%; }}
    table.discussion-table th:nth-child(3), table.discussion-table td:nth-child(3) {{ width:42%; }}
    table.category-sampling-table th:nth-child(1), table.category-sampling-table td:nth-child(1) {{ width:10%; }}
    table.category-sampling-table th:nth-child(2), table.category-sampling-table td:nth-child(2),
    table.category-sampling-table th:nth-child(3), table.category-sampling-table td:nth-child(3),
    table.category-sampling-table th:nth-child(4), table.category-sampling-table td:nth-child(4) {{ width:10%; }}
    table.category-sampling-table th:nth-child(5), table.category-sampling-table td:nth-child(5) {{ width:60%; }}
    table.query-set-table th:nth-child(1), table.query-set-table td:nth-child(1) {{ width:8%; }}
    table.query-set-table th:nth-child(2), table.query-set-table td:nth-child(2) {{ width:8%; }}
    table.query-set-table th:nth-child(3), table.query-set-table td:nth-child(3) {{ width:16%; }}
    table.query-set-table th:nth-child(4), table.query-set-table td:nth-child(4) {{ width:38%; }}
    table.query-set-table th:nth-child(5), table.query-set-table td:nth-child(5) {{ width:30%; }}
    table.keyword-summary-table th:nth-child(1), table.keyword-summary-table td:nth-child(1) {{ width:8%; }}
    table.keyword-summary-table th:nth-child(2), table.keyword-summary-table td:nth-child(2) {{ width:36%; }}
    table.keyword-summary-table th:nth-child(3), table.keyword-summary-table td:nth-child(3) {{ width:22%; }}
    table.keyword-summary-table th:nth-child(4), table.keyword-summary-table td:nth-child(4) {{ width:20%; }}
    table.keyword-summary-table th:nth-child(5), table.keyword-summary-table td:nth-child(5) {{ width:10%; }}
    table.keyword-summary-table th:nth-child(6), table.keyword-summary-table td:nth-child(6) {{ width:4%; }}
    table.judge-plan-table th:nth-child(1), table.judge-plan-table td:nth-child(1) {{ width:25%; }}
    table.judge-plan-table th:nth-child(2), table.judge-plan-table td:nth-child(2) {{ width:75%; }}
    tbody tr:nth-child(even) td {{ background:#fafbfd; }}
    tr > *:last-child {{ border-right:0; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    pre {{ background:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; overflow:auto; font-size:12px; }}
    details {{ margin:12px 0; }}
    summary {{ cursor:pointer; font-weight:650; }}
    code {{ background:#eef2f7; padding:1px 5px; border-radius:4px; }}
    svg {{ width:100%; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; }}
    .graph-legend {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; margin:10px 0 22px; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; font-size:12px; color:#334155; }}
    .graph-legend strong {{ margin-right:4px; color:#0f172a; }}
    .legend-item {{ display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }}
    .legend-dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; box-shadow:0 0 0 1px rgba(15,23,42,.12); }}
    .legend-note {{ color:var(--muted); margin-left:auto; }}
    .edge {{ stroke:#94a3b8; stroke-width:1.5; opacity:.7; }}
    .multi-edge {{ stroke:#2563eb; stroke-width:2.6; opacity:.92; }}
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
	      <li><a href="#tables">Paper-style Evaluation Tables</a>
	        <ol>
	          <li><a href="#table-4">Table 4: Dataset Statistics</a></li>
	          <li><a href="#table-1">Table 1: Win rates</a></li>
	          <li><a href="#answer-length">Answer Length Statistics</a></li>
	          <li><a href="#length-control">Length Normalization + Verbosity-aware Judge</a></li>
	          <li><a href="#query-type">Query Type Breakdown</a></li>
	          <li><a href="#table-2">Table 2: LightRAG mode ablation</a></li>
	          <li><a href="#table-3">Table 3: GraphRAG vs LightRAG</a></li>
	          <li><a href="#figure-2">Figure 2: Cost/runtime summary</a></li>
	          <li><a href="#table-5">Table 5: NaiveRAG vs LightRAG</a></li>
	          <li><a href="#additional-case">Additional Case Study</a></li>
	        </ol>
	      </li>
	      <li><a href="#discussion">Discussion and Conclusion</a></li>
	      <li><a href="#appendix-7-1">Appendix</a>
	        <ol>
	          <li><a href="#appendix-7-1">7.1 Experimental Data Details</a></li>
	          <li><a href="#appendix-7-2">7.2 Retrieval-Augmented Generation Example</a></li>
	          <li><a href="#appendix-7-3-1">7.3.1 Graph Generation Prompt and Outputs</a></li>
	          <li><a href="#appendix-7-3-2">7.3.2 Query Generation</a></li>
	          <li><a href="#appendix-7-3-3">7.3.3 Keyword Extraction</a></li>
	          <li><a href="#appendix-7-3-4">7.3.4 RAG Evaluation</a></li>
	        </ol>
	      </li>
    </ol>
  </section>

  <h2 id="models">Model Roles and Current Status</h2>
  {table([
      {"Model": "gpt-5.5 via codex-proxy", "Role": "LightRAG indexing/query, GraphRAG indexing/query"},
      {"Model": "text-embedding-3-large via OpenRouter", "Role": "chunk/entity/relation/query embedding"},
      {"Model": "Gemini 3.5 Flash via OpenRouter", "Role": "LLM judge only"},
  ], class_name="two-col-table")}
  {table([
      {"Track": "LightRAG indexing", "Status": "complete", "Evidence": f"{index_stats.get('processed_documents', 0)}/{index_stats.get('document_count', 0)} docs processed"},
      {"Track": "LightRAG query", "Status": light_status, "Evidence": light_evidence},
      {"Track": "GraphRAG", "Status": graph_status, "Evidence": graph_evidence},
      {"Track": "Gemini judge", "Status": judge_status, "Evidence": judge_evidence},
  ], class_name="status-table")}

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
  {table_explanation("그래프 품질 표는 특허번호, 국가코드, 법적상태 같은 메타데이터 node/relation이 억제됐는지 확인하기 위한 품질 게이트다. 이번 Patent-100 graph는 technical relation 비율이 높고 excluded entity 비율이 낮아, 이후 retrieval 실험에 사용할 수 있는 기술 중심 graph로 판단했다.")}
  {full_graph_visualization(working_dir)}
  {single_patent_graph}

  <h2 id="retrieval">3.2 Dual-level Retrieval Paradigm</h2>
  <p><code>local</code>은 entity 중심 low-level retrieval, <code>global</code>은 relation/theme 중심 high-level retrieval, <code>hybrid</code>는 둘을 병합한다.</p>
  {render_query_flow(light_rows)}

  <h2 id="generation">3.3 Retrieval-Augmented Answer Generation</h2>
  <p>retrieved entities, relationships, chunks, references가 answer prompt에 합쳐지고, 같은 query에 대해 mode별 답변이 달라진다.</p>
  {render_answer_compare(light_rows, "AA-1")}
  {table_explanation("이 비교표는 같은 질문에서 retrieval mode가 답변 근거와 답변 형태를 어떻게 바꾸는지 보여준다. naive는 chunk 중심, local/global/hybrid는 entity와 relation context가 추가되며, 이 차이가 뒤의 judge 평가에서 실제 품질 차이로 이어지는지 확인한다.")}

  <h2 id="complexity">3.4 Complexity Analysis</h2>
  {table(complexity_rows(index_stats, light_rows, graph_rows, graph_stats), class_name="complexity-table")}
  {table_explanation("복잡도 표는 성공한 최종 실행 시간과 전체 재시도 비용을 구분해 읽어야 한다. LightRAG는 최종 성공 indexing 시간은 짧지만 이전 실패/재시도까지 포함하면 총 소요 시간이 커지고, GraphRAG는 community/report 생성 때문에 indexing 비용 구조가 더 무겁다.")}
  <h3>LightRAG Query Recovery Audit</h3>
  {table(light_repair_rows, class_name="audit-table")}
  {table_explanation("LightRAG query audit는 중간 실패나 누락이 최종 결과에 남아 있는지 확인하는 재현성 점검표다. 최종적으로 15개 query와 4개 mode 조합이 모두 채워졌는지를 이 표로 확인한다.")}
  <h3>GraphRAG Query Recovery Audit</h3>
  {table(graph_recovery, class_name="audit-table")}
  {table_explanation("GraphRAG query audit는 update/fresh 실행 과정에서 누락되거나 실패한 query-method 조합을 복구했는지 보여준다. GraphRAG는 실행 패턴이 LightRAG보다 무거워서 repair 로그를 별도로 남기는 것이 중요하다.")}
  <h3>GraphRAG Query Mode Summary</h3>
  {table(graph_mode_rows, class_name="mode-summary-table")}
  {table_explanation("GraphRAG mode summary는 basic, local, global의 지연시간과 답변 길이 차이를 비교한다. 이후 평가에서 global이 강하게 나온 원인을 볼 때, 이 표의 평균 답변 길이와 latency를 함께 봐야 한다.")}
  <h3>Gemini Judge Recovery Audit</h3>
  {table(judge_repair_rows, class_name="audit-table")}
  {table_explanation("Judge audit는 pairwise 평가 결과가 60개 모두 유효하게 수집됐는지 확인한다. judge 결과는 최종 승률의 근거이므로, 실패 attempt가 있더라도 deduped success가 모두 채워졌는지가 핵심이다.")}

  <h2 id="tables">Paper-style Evaluation Tables</h2>
	  <h3 id="table-4">Table 4: Statistical information of the patent datasets</h3>
  {table(dataset_rows, class_name="dataset-table")}
  {table_explanation("Table 4는 실험 데이터가 AA/AB/AC/AD 25건씩 균형 sampling됐는지, 그리고 graph/chunk/entity/relation 규모가 어느 정도인지 보여주는 기본 통계다. 이 표가 이후 모든 retrieval, judge, cost 비교의 모집단 정의 역할을 한다.")}
	  <h3 id="table-1" class="print-page-break">Table 1: Win rates (%) of baselines vs LightRAG</h3>
  <p class="note">{html_escape(judge_note)}</p>
  {win_rate_placeholder(judge_rows)}
  {table_explanation("Table 1은 Gemini judge가 pairwise로 고른 winner 비율이다. 원본 답변 기준으로는 GraphRAG global이 강하지만, 이 승률은 답변 길이 차이의 영향을 받을 수 있으므로 바로 성능 우위로 단정하지 않고 아래 길이 통계와 함께 해석한다.")}
	  <h3 id="answer-length" class="print-page-break">Answer Length Statistics for Table 1 Interpretation</h3>
  <p class="note">GraphRAG 답변에 섞인 런타임 warning 문구를 제거한 cleaned answer 기준이다. 이 표는 LLM-as-judge의 verbosity bias 가능성을 해석하기 위한 보조 지표다.</p>
  {answer_length_table(length_stats)}
  {table_explanation("답변 길이 표는 judge 승률의 confound를 확인하기 위한 보조 결과다. GraphRAG global 답변이 LightRAG hybrid보다 훨씬 길면 Comprehensiveness, Diversity, Empowerment rubric에서 구조적으로 유리할 수 있다.")}
	  <h3 id="length-control" class="print-page-break">Length Normalization + Verbosity-aware Judge</h3>
  <p class="note">답변 길이 편향을 줄이기 위해 같은 근거를 유지한 채 1100-1300자 목표로 답변을 재작성하고, Gemini judge prompt에 verbosity penalty를 명시했다. 일부 GraphRAG 답변은 근거 보존 때문에 목표 길이를 초과했으므로 완전한 동일 길이 실험은 아니며, 잔여 길이 차이를 함께 해석한다.</p>
  {length_control_audit_table(length_control_summary)}
  {table_explanation("이 audit 표는 길이 정규화와 verbosity-aware judge가 실제로 끝까지 수행됐는지 확인한다. 60개 pairwise 평가가 모두 성공해야 원본 judge 결과와 보정 judge 결과를 비교할 수 있다.")}
  {length_control_length_table(length_control_summary)}
  {table_explanation("길이 정규화 표는 원본 답변과 보정 답변의 평균 길이 변화를 보여준다. GraphRAG global은 압축 후에도 목표 길이에 완전히 들어오지 않았기 때문에, 보정 결과도 완전한 동일 길이 조건이 아니라 잔여 길이 차이가 남은 결과로 읽어야 한다.")}
  {length_control_judge_table(length_control_summary)}
  {table_explanation("verbosity-aware judge 표는 원본 답변과 길이 정규화 답변에서 winner가 어떻게 바뀌는지 보여준다. GraphRAG global은 hybrid 대비 여전히 강하지만 naive 대비 우위는 줄어들어, community summary 효과와 길이 효과가 함께 작용한 것으로 해석한다.")}
	  <h3 id="query-type" class="print-page-break">Normalized Verbosity-aware Judge by Query Type</h3>
  <p class="note">아래 표는 query type별 Overall winner count다. 하나의 query가 여러 pairwise 비교에 등장하므로 count는 query 수와 같지 않고, type별 경향을 보기 위한 breakdown이다.</p>
  {query_type_pattern_table(length_control_summary)}
  {table_explanation("Query type breakdown은 전체 평균만 보면 놓치는 패턴을 보여준다. category-specific broad query는 GraphRAG global이 강하고, cross-category query는 LightRAG hybrid가 더 강해 graph context의 효용이 질문 유형별로 달라진다.")}
	  <h3 id="table-2" class="print-page-break">Table 2: Performance of ablated LightRAG retrieval modes</h3>
  {table(mode_ablation_display_rows(auto_metrics, light_mode_rows), class_name="mode-ablation-table")}
  {table_explanation("Table 2는 LightRAG 내부 mode ablation이다. naive는 graph 없이 chunk embedding만 쓰고, local/global/hybrid는 entity/relation context를 추가하므로, graph retrieval이 latency와 source coverage를 얼마나 바꾸는지 확인하는 표다.")}
	  <h3 id="table-3" class="print-page-break">Table 3: Case Study: GraphRAG vs LightRAG</h3>
  <p class="note">{html_escape(graph_case_note)}</p>
  {graph_case_blank() if not graph_rows else case_table(light_rows, graph_rows, judge_rows, "X-1", "global")}
	  {table_explanation("Table 3은 aggregate win-rate만으로 보이지 않는 답변 양식 차이를 읽기 위한 사례다. GraphRAG global은 community summary를 통해 더 넓은 맥락을 제시하고, LightRAG hybrid는 검색된 entity/relation/chunk 근거에 더 직접적으로 묶이는 경향이 있다.")}
		  <h3 id="figure-2" class="print-page-break">Figure 2: Cost comparison in tokens/API calls</h3>
	  {cost_runtime_summary_table(index_stats, light_rows, graph_rows, graph_stats, judge_rows, length_control_summary)}
	  {pre("Figure 2 raw JSON excerpt", cost_raw_excerpt(index_stats, graph_metrics, graph_stats, graph_mode_rows, judge_summary, length_control_summary), 5000)}
	  {pre("Cost and runtime summary", {"index_stats": index_stats, "graph_metrics": graph_metrics, "graphrag_manifest_summary": graphrag_manifest_summary(graphrag_manifest), "raw_manifest_path": str(experiment_dir / "graphrag_manifest.json"), "graphrag_stats": graph_stats, "graphrag_query_summary": graph_mode_rows, "graphrag_recovery": graph_recovery, "judge_summary": judge_summary, "judge_recovery": judge_repair}, 12000)}
	  {table_explanation("Figure 2 항목은 raw cost/runtime 근거를 접어 둔 요약이다. LightRAG와 GraphRAG의 직접 비용 비교는 단순 query latency뿐 아니라 indexing 재시도, GraphRAG workflow, judge 호출까지 포함해서 해석해야 한다.")}
	  <h3 id="table-5" class="print-page-break">Table 5: Case Study: NaiveRAG vs LightRAG</h3>
  {naive_vs_light_table(light_rows, "C-1")}
  {table_explanation("Table 5는 LightRAG hybrid가 chunk-only naive 대비 어떤 추가 근거를 제공하는지 보는 사례다. 이번 실험에서는 hybrid가 항상 naive를 압도하지 않았으므로, 이 표는 graph context의 장점과 중복 가능성을 동시에 점검하는 용도다.")}
	  <h3 id="additional-case" class="print-page-break">Additional Case Study: LightRAG Hybrid vs GraphRAG Global</h3>
  <p class="note">category-specific broad query에서 GraphRAG global이 강하게 나온 사례다. 긴 community summary가 압축 후에도 핵심 근거를 유지하는지 확인하기 위한 읽기용 예시다.</p>
  {graph_case_blank() if not graph_rows else case_table(light_rows, graph_rows, judge_rows, "AA-1", "global")}
  {table_explanation("추가 case study는 GraphRAG global이 특히 강한 category-specific broad query를 보여준다. 이 사례는 GraphRAG의 장점을 설명하는 동시에, 긴 답변이 judge에 유리하게 작용했을 가능성을 같이 확인하기 위한 예시다.")}

  {discussion_section(graph_metrics, judge_rows, length_stats, length_control_summary)}

		  <h2 id="appendix-7-1">Appendix 7.1: Experimental Data Details</h2>
	  {appendix_data_summary_table(manifest)}
	  {appendix_category_sampling_table(manifest)}
	  {pre("Appendix 7.1 manifest JSON excerpt", appendix_manifest_excerpt(manifest), 5000)}
	  {pre("Sampling manifest summary", manifest_summary(manifest), 9000)}
	  <p class="note">전체 patent ID 목록은 raw JSON 파일 <code>{html_escape(str(experiment_dir / "dataset" / "patents_100_manifest.json"))}</code>에 보존했다.</p>
	  <h2 id="appendix-7-2">Appendix 7.2: Retrieval-Augmented Generation Example</h2>
  <p class="note">3.2 본문의 retrieval context 예시를 Appendix에도 남긴다. 7.3.3은 중복을 피하기 위해 keyword extraction 결과만 별도로 요약한다.</p>
  {render_query_flow(light_rows)}
	  <h2 id="appendix-7-3-1">Appendix 7.3.1: Graph Generation Prompt and Outputs</h2>
  {pre("Prompt excerpt", prompt_text[:7000], 8000)}
  {rpd_examples(working_dir)}
	  <h2 id="appendix-7-3-2">Appendix 7.3.2: Query Generation</h2>
  <p>15개 쿼리는 모델 없이 직접 설계한 고정 평가셋이다.</p>
  {table(queries, class_name="query-set-table")}
	  <h2 id="appendix-7-3-3">Appendix 7.3.3: Keyword Extraction</h2>
  {keyword_extraction_summary(light_rows)}
	  <h2 id="appendix-7-3-4">Appendix 7.3.4: RAG Evaluation</h2>
  <p class="note">{html_escape('Gemini 3.5 Flash judge를 실행했고 60/60 pairwise 결과를 정규화해 사용했다.' if judge_rows else 'Gemini 3.5 Flash judge는 아직 실행하지 않았다. 아래는 실행 후 채울 평가 구조다.')}</p>
  {table([
      {"Judge item": "Pairwise comparisons", "Planned value": "LightRAG hybrid vs NaiveRAG; LightRAG hybrid vs GraphRAG global/local; GraphRAG global vs NaiveRAG"},
      {"Judge item": "Rubrics", "Planned value": "Comprehensiveness, Diversity, Empowerment, Technical correctness, Hallucination risk, Overall"},
      {"Judge item": "A/B ordering", "Planned value": "query별로 answer A/B 순서 교차"},
      {"Judge item": "Output schema", "Planned value": "{metric: {winner, score_a, score_b, rationale}, overall_summary}"},
  ], class_name="judge-plan-table")}
  <p class="note">아래 JSON은 구조 확인용 excerpt다. 전체 raw 결과는 <code>{html_escape(str(experiment_dir / "evaluation"))}</code>와 <code>{html_escape(str(length_control_dir))}</code>에 보존했다.</p>
  {pre("Judge summary excerpt", judge_summary, 5000)}
  {table(judge_repair_rows, class_name="audit-table")}
  {pre("Judge output example excerpt", judge_rows[:1], 4500)}
  {pre("Length-control judge summary excerpt", length_control_summary, 6500)}
  {pre("Length-control original-answer judge example excerpt", length_control_original_rows[:1], 4500)}
  {pre("Length-control normalized-answer judge example excerpt", length_control_normalized_rows[:1], 4500)}
</main>
</body>
</html>
"""
    html = strip_trailing_whitespace(html)
    output.write_text(html, encoding="utf-8")
    print_output = Path(args.print_output)
    print_output.parent.mkdir(parents=True, exist_ok=True)
    print_output.write_text(strip_trailing_whitespace(build_print_html(html)), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "exists": output.exists(),
                "print_output": str(print_output),
                "print_exists": print_output.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
