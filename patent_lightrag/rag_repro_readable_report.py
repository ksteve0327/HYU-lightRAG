from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from patent_lightrag.common import LIGHTRAG_ROOT, ROOT, html_escape, read_json
from patent_lightrag.rag_repro_report import full_graph_visualization, patent_graph_example


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_OUTPUT = ROOT / "reports" / "rag_repro_100_readable.html"
DEFAULT_PRINT_OUTPUT = ROOT / "reports" / "rag_repro_100_readable_print.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a concise readable Patent-100 LightRAG report.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--print-output", default=str(DEFAULT_PRINT_OUTPUT))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def short(value: Any, limit: int = 420) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 12].rstrip() + " ...[truncated]"


def clean_answer(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\x1b\[[0-9;]*m", "", value)
    lines = [
        line for line in value.splitlines()
        if "LiteLLM:WARNING" not in line and "could not pre-load" not in line
    ]
    return "\n".join(lines).strip()


def strip_markdown_headings(text: str) -> str:
    return re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)


def render_inline_markdown(text: str) -> str:
    escaped = html_escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def answer_excerpt_html(text: Any, limit: int = 560) -> str:
    cleaned = strip_markdown_headings(clean_answer(text))
    excerpt = short(cleaned, limit)
    if excerpt.count("**") % 2 == 1:
        excerpt = excerpt.replace("**", "")
    return render_inline_markdown(excerpt)


def fmt_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{seconds / 3600:.2f}h" if seconds >= 3600 else f"{seconds:.1f}s"


def retrieved_count_text(entity_count: Any, relation_count: Any, chunk_count: Any) -> str:
    return f"Entity 개수={entity_count}, Relation 개수={relation_count}, Chunk 개수={chunk_count}"


def table(rows: list[dict[str, Any]], class_name: str = "table") -> str:
    if not rows:
        return "<p class='muted'>데이터 없음</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html_escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in keys) + "</tr>")
    return f"<div class='table-wrap'><table class='{class_name}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def code_block(value: Any, limit: int = 1600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"<pre class='code'>{html_escape(short(text, limit))}</pre>"


def code_block_preserve(value: Any, limit: int = 1600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    text = text if len(text) <= limit else text[: limit - 16].rstrip() + "\n...[truncated]"
    return f"<pre class='code'>{html_escape(text)}</pre>"


def preserve_excerpt(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 16].rstrip() + "\n...[truncated]"


def result_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("query_id")), str(row.get("mode") or row.get("method"))): row for row in rows}


def context_body(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return {}
    body = data.get("data", {})
    return body if isinstance(body, dict) else {}


def query_metadata(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data", {})
    if not isinstance(data, dict):
        return {}
    metadata = data.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def source_patents(row: dict[str, Any]) -> set[str]:
    patents: set[str] = set()
    for section in ["entities", "relationships", "chunks"]:
        for item in context_body(row).get(section, []) or []:
            if not isinstance(item, dict):
                continue
            for key in ["file_path", "source_id", "chunk_id"]:
                for part in str(item.get(key, "")).split("<SEP>"):
                    part = part.strip()
                    if not part:
                        continue
                    patents.add(re.sub(r"-chunk-\d+$", "", part))
    return patents


def load_vdb_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {}) or {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def chunk_stats(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tokens = [int(row.get("tokens", 0) or 0) for row in chunks.values() if isinstance(row, dict)]
    doc_counts = Counter(str(row.get("full_doc_id") or row.get("file_path") or "") for row in chunks.values() if isinstance(row, dict))
    return {
        "count": len(chunks),
        "avg": round(statistics.mean(tokens), 1) if tokens else 0,
        "max": max(tokens) if tokens else 0,
        "multi_chunk_docs": sum(1 for count in doc_counts.values() if count > 1),
    }


def multi_source_count(store: dict[str, Any]) -> int:
    return sum(1 for value in store.values() if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1)


def pick_relation_rich_doc(
    chunks: dict[str, Any],
    full_entities: dict[str, Any],
    full_relations: dict[str, Any],
) -> dict[str, Any]:
    relation_counts = []
    for doc_id, value in full_relations.items() if isinstance(full_relations, dict) else []:
        pairs = value.get("relation_pairs", []) if isinstance(value, dict) else []
        if isinstance(pairs, list):
            relation_counts.append((len(pairs), str(doc_id), pairs))
    if not relation_counts:
        return {"doc_id": "—", "chunk_id": "—", "tokens": "—", "entities": [], "relations": []}
    _, doc_id, relation_pairs = sorted(relation_counts, reverse=True)[0]
    entity_names = full_entities.get(doc_id, {}).get("entity_names", []) if isinstance(full_entities.get(doc_id), dict) else []
    chunk_id = next((key for key in chunks if str(key).startswith(f"{doc_id}-chunk-")), "")
    chunk = chunks.get(chunk_id, {}) if isinstance(chunks.get(chunk_id), dict) else {}
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id or "—",
        "tokens": chunk.get("tokens", "—"),
        "entities": list(entity_names) if isinstance(entity_names, list) else [],
        "relations": list(relation_pairs) if isinstance(relation_pairs, list) else [],
    }


def actual_data_flow_visual(
    manifest: dict[str, Any],
    index_stats: dict[str, Any],
    graph_metrics: dict[str, Any],
    working_dir: Path,
    chunks: dict[str, Any],
    full_entities: dict[str, Any],
    full_relations: dict[str, Any],
    entity_chunks: dict[str, Any],
    relation_chunks: dict[str, Any],
    light_rows: list[dict[str, Any]],
) -> str:
    stats = chunk_stats(chunks if isinstance(chunks, dict) else {})
    sample = pick_relation_rich_doc(chunks, full_entities, full_relations)
    hybrid = result_lookup(light_rows).get(("AA-1", "hybrid"), {})
    hybrid_body = context_body(hybrid)
    hybrid_retrieved = retrieved_count_text(
        len(hybrid_body.get("entities", []) or []),
        len(hybrid_body.get("relationships", []) or []),
        len(hybrid_body.get("chunks", []) or []),
    )
    vdb_counts = {
        "chunk": len(load_vdb_rows(working_dir / "vdb_chunks.json")),
        "entity": len(load_vdb_rows(working_dir / "vdb_entities.json")),
        "relation": len(load_vdb_rows(working_dir / "vdb_relationships.json")),
    }
    cards = [
        {
            "label": "01 input",
            "value": f"{manifest.get('total_selected', index_stats.get('document_count', '—'))} patents",
            "detail": f"AA/AB/AC/AD x {manifest.get('per_category', 25)}, seed={manifest.get('seed', '—')}",
        },
        {
            "label": "02 chunks",
            "value": f"{stats['count']} chunks",
            "detail": f"avg={stats['avg']} tokens, multi-chunk docs={stats['multi_chunk_docs']}",
        },
        {
            "label": "03 sample R",
            "value": str(sample["doc_id"]),
            "detail": f"{sample['chunk_id']} / {sample['tokens']} tokens / {len(sample['entities'])} entities / {len(sample['relations'])} relations",
        },
        {
            "label": "04 graph",
            "value": f"{graph_metrics.get('graph_nodes', '—')} nodes",
            "detail": f"{graph_metrics.get('graph_edges', '—')} edges, technical={float(graph_metrics.get('technical_relation_ratio', 0)) * 100:.1f}%",
        },
        {
            "label": "05 dedup",
            "value": f"{multi_source_count(entity_chunks)} entities",
            "detail": f"{multi_source_count(relation_chunks)} multi-source relations merged",
        },
        {
            "label": "06 vectors",
            "value": f"{sum(vdb_counts.values())} records",
            "detail": f"chunk={vdb_counts['chunk']}, entity={vdb_counts['entity']}, relation={vdb_counts['relation']}",
        },
        {
            "label": "07 retrieval",
            "value": "AA-1 hybrid",
            "detail": f"{hybrid_retrieved}, source patents={len(source_patents(hybrid)) if hybrid else '—'}",
        },
    ]
    card_html = "".join(
        f"""
        <article class="actual-node">
          <span>{html_escape(card['label'])}</span>
          <strong>{html_escape(card['value'])}</strong>
          <small>{html_escape(card['detail'])}</small>
        </article>
        """
        for card in cards
    )
    relation_examples = "; ".join(
        f"{pair[0]} -> {pair[1]}"
        for pair in sample["relations"][:3]
        if isinstance(pair, list) and len(pair) >= 2
    )
    entity_examples = ", ".join(map(str, sample["entities"][:6]))
    return f"""
    <div class="actual-run">
      <div class="actual-run-head">
        <span class="eyebrow">Actual Patent-100 run</span>
        <h3>실제 산출물 기준 데이터 이동</h3>
        <p>아래 값은 보고서 생성 시 LightRAG storage와 query result JSON에서 직접 읽은 값이다.</p>
      </div>
      <div class="actual-flow">{card_html}</div>
      <div class="actual-sample">
        <strong>Sample chunk path</strong>
        <p><code>{html_escape(sample['chunk_id'])}</code>에서 <b>{len(sample['entities'])}</b>개 entity와 <b>{len(sample['relations'])}</b>개 relation이 추출됐다.</p>
        <p><span>entities</span> {html_escape(entity_examples or '—')}</p>
        <p><span>relations</span> {html_escape(relation_examples or '—')}</p>
      </div>
    </div>
    """


def pct_bar(label: str, value: float, max_value: float, note: str = "") -> str:
    width = 0 if max_value <= 0 else min(100, max(2, value / max_value * 100))
    return f"""
    <div class="bar-row">
      <div class="bar-label"><strong>{html_escape(label)}</strong><span>{html_escape(note)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
      <div class="bar-value">{html_escape(value)}</div>
    </div>
    """


ENTITY_TYPE_LABELS_KO = {
    "techcomponent": "기술 구성요소",
    "architecture": "아키텍처",
    "operation": "연산/동작",
    "method": "방법/공정",
    "material": "재료/소자",
    "performancemetric": "성능 지표",
    "applicationdomain": "응용 분야",
    "organization": "기관/출원인",
    "other": "기타 기술 엔티티",
}


def entity_type_label_ko(label: str) -> str:
    normalized = str(label).replace("_", "").replace(" ", "").lower()
    return ENTITY_TYPE_LABELS_KO.get(normalized, str(label))


def entity_type_chart(type_counts: dict[str, int]) -> str:
    if not type_counts:
        return "<p class='muted'>entity type 통계 없음</p>"
    max_value = max(type_counts.values()) or 1
    total = sum(type_counts.values()) or 1
    rows = []
    for label, value in sorted(type_counts.items(), key=lambda item: item[1], reverse=True):
        rows.append(pct_bar(entity_type_label_ko(label), value, max_value, f"{value / total * 100:.1f}%"))
    return "<div class='bar-chart'>" + "".join(rows) + "</div>"


def quality_cards(graph_metrics: dict[str, Any]) -> str:
    cards = [
        ("Technical relations", f"{float(graph_metrics.get('technical_relation_ratio', 0)) * 100:.1f}%", "기술 구성요소 관계 중심"),
        ("Metadata relations", f"{float(graph_metrics.get('metadata_relation_ratio', 0)) * 100:.1f}%", "특허번호/국가/상태 hub 억제"),
        ("Excluded entities", f"{float(graph_metrics.get('excluded_entity_ratio', 0)) * 100:.2f}%", "제외 대상 entity 잔존 비율"),
        ("Degree >= 50 hubs", graph_metrics.get("degree_ge_50_hub_count", "—"), "과도한 metadata hub 없음"),
    ]
    return "".join(
        f"<article class='metric-card compact'><span>{html_escape(label)}</span><strong>{html_escape(value)}</strong><small>{html_escape(note)}</small></article>"
        for label, value, note in cards
    )


def model_cards(index_stats: dict[str, Any]) -> str:
    rows = [
        ("gpt-5.5", "codex-proxy", "indexing R/P/D, answering, GraphRAG query"),
        ("text-embedding-3-large", "OpenRouter", f"chunk/entity/relation/query vectors, dim={index_stats.get('embedding_dim', '—')}"),
        ("Gemini 3.5 Flash", "OpenRouter", "LLM judge only"),
    ]
    return "".join(
        f"<article class='mini-card'><span class='eyebrow'>{html_escape(channel)}</span><h3>{html_escape(model)}</h3><p>{html_escape(role)}</p></article>"
        for model, channel, role in rows
    )


def pipeline_html() -> str:
    steps = [
        ("01", "Structured Patent", "patent_id, 출원번호, 제목, 요약, 목적/솔루션, 대표청구항, 분류, 권리자, 국가, 연도"),
        ("02", "Chunking", "1200 token 단위로 특허 텍스트를 분할하고 source id를 유지"),
        ("03", "R Function", "LLM이 기술 entity와 relation tuple을 추출"),
        ("04", "P/D Functions", "profile 생성 후 동일 entity/relation을 source 기준으로 병합"),
        ("05", "Retrieval", "naive/local/global/hybrid가 query별 context를 구성"),
    ]
    return "".join(
        f"<article class='flow-step'><b>{num}</b><h3>{html_escape(title)}</h3><p>{html_escape(desc)}</p></article>"
        for num, title, desc in steps
    )


def prompt_assignment(source: str, key: str) -> str:
    pattern = rf'PROMPTS\["{re.escape(key)}"\]\s*=\s*"""(.*?)"""'
    match = re.search(pattern, source, re.S)
    return match.group(1).strip() if match else ""


def between_or_excerpt(text: str, start: str, end: str, limit: int) -> str:
    if start in text:
        tail = text.split(start, 1)[1]
        if end in tail:
            return preserve_excerpt(start + tail.split(end, 1)[0], limit)
    return preserve_excerpt(text, limit)


def patent_guidance_excerpt(prompt_text: str) -> str:
    if not prompt_text:
        return "patent_ai_semiconductor.yml 없음"
    guidance = prompt_text.split("entity_extraction_examples:", 1)[0].strip()
    return preserve_excerpt(guidance, 1050)


def patent_example_excerpt(prompt_text: str) -> str:
    if not prompt_text:
        return "patent_ai_semiconductor.yml 없음"
    marker = "---Output---"
    idx = prompt_text.find(marker)
    if idx < 0:
        return preserve_excerpt(prompt_text, 1200)
    return preserve_excerpt(prompt_text[idx: idx + 1500], 1500)


def rpd_prompt_excerpts(prompt_text: str) -> str:
    prompt_py = LIGHTRAG_ROOT / "lightrag" / "prompt.py"
    source = prompt_py.read_text(encoding="utf-8") if prompt_py.exists() else ""
    r_system = prompt_assignment(source, "entity_extraction_system_prompt")
    r_user = prompt_assignment(source, "entity_extraction_user_prompt")
    p_summary = prompt_assignment(source, "summarize_entity_descriptions")
    r_format = between_or_excerpt(r_system, "4. **Output Format:**", "6. **Output Order", 900)
    r_language = between_or_excerpt(r_system, "7. **Context & Language:**", "8. **Completion Signal", 500)
    r_user_task = between_or_excerpt(r_user, "---Task---", "---Input Text---", 650)
    r_prompt = f"""# R Function: entity / relation extraction prompt
System prompt 핵심 1: output schema
{r_format}

System prompt 핵심 2: language / context
{r_language}

User prompt 핵심:
{r_user_task}

# Patent-specific YAML guidance
{patent_guidance_excerpt(prompt_text)}

# Patent few-shot output example
{patent_example_excerpt(prompt_text)}"""
    p_prompt = f"""# P Function: entity/relation profile summary prompt
{preserve_excerpt(p_summary, 1550)}

# 실제 동작 조건
- description이 1개면 LLM 호출 없이 그대로 profile description으로 저장한다.
- 같은 entity/relation이 여러 chunk/source에서 반복되면 Description List를 JSONL로 묶어 위 prompt에 넣는다.
- LLM 요약 결과가 vdb_entities.json / vdb_relationships.json의 content/description이 되고 embedding 대상 profile이 된다."""
    r_prompt_ko = """# R Function 프롬프트 한국어 번역
역할:
- LLM은 Knowledge Graph Specialist로 동작한다.
- 입력 특허 텍스트에서 명확하고 의미 있는 entity와 relation을 추출한다.

Entity 추출 규칙:
- entity_name: entity 이름을 일관된 이름으로 정한다.
- entity_type: ---Entity Types---에 제공된 타입 중 하나로 분류한다. 맞는 타입이 없으면 Other를 사용한다.
- entity_description: 입력 텍스트에 있는 정보만 근거로 entity의 속성과 동작을 간결하지만 충분히 설명한다.

Relation 추출 규칙:
- 이미 추출한 entity 사이의 직접적이고 의미 있는 관계만 뽑는다.
- 하나의 문장이 여러 entity 관계를 포함하면 2개 entity 단위의 relation으로 분해한다.
- source_entity, target_entity, relationship_keywords, relationship_description을 생성한다.

출력 형식:
- entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description
- relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description
- entity row를 먼저 출력하고, 그 다음 relation row를 출력한다.
- 마지막에는 {completion_delimiter}를 출력한다.

특허 특화 규칙:
- 기술 knowledge graph에 필요한 entity/relation만 추출한다.
- 특허번호, 출원번호, 공개번호, 등록번호, IPC/CPC, 국가코드, 법적상태, 날짜, 청구항 번호, 중/소분류 코드는 entity로 추출하지 않는다.
- entity type은 TechComponent, Architecture, Operation, Method, Material, PerformanceMetric, ApplicationDomain, Organization, Other 중에서 고른다.
- relation은 추출된 기술 entity 사이에서만 만든다.
- 바람직한 relation keyword 예시는 구성요소, 포함, 연결, 저장, 판독, 제어, 생성, 수행, 최적화, 가속, 융합, 컴파일, 데이터 전송, 아키텍처 활용이다.

우리 실험에서의 의미:
- R Function은 특허 chunk를 읽고 기술 entity와 entity 간 relation tuple을 생성하는 LLM 단계다.
- 이 단계의 출력이 이후 graph node/edge의 후보가 된다."""
    p_prompt_ko = """# P Function 프롬프트 한국어 번역
역할:
- LLM은 Knowledge Graph Specialist로서 entity 또는 relation 설명 목록을 정리하고 종합한다.

작업:
- 주어진 entity 또는 relation에 대해 여러 description 조각을 하나의 포괄적이고 일관된 summary로 합친다.
- 입력 description list는 JSON 형식이며, 각 JSON 객체는 Description 필드를 가진다.
- 출력은 추가 설명 없이 plain text summary만 반환한다.

요약 규칙:
- 모든 description의 핵심 정보를 빠뜨리지 않고 통합한다.
- 객관적인 3인칭 관점으로 작성한다.
- summary 시작 부분에서 entity 또는 relation 이름을 명확히 언급한다.
- 서로 충돌하는 description이 있으면 같은 이름을 가진 별개 entity/relation인지 먼저 판단한다.
- 별개 entity/relation이면 각각 분리해 요약하고, 하나의 대상 내부 충돌이면 불확실성을 표시해 조정한다.
- summary는 지정된 token 길이 이내로 작성한다.
- 출력 언어는 Korean으로 설정된다.

실제 동작 조건:
- description이 1개뿐이면 LLM을 호출하지 않고 그대로 profile description으로 저장한다.
- 같은 entity/relation이 여러 chunk나 source patent에서 반복되면 description list를 만들고 이 P Function prompt로 LLM summary를 생성한다.
- 생성된 summary는 vdb_entities.json 또는 vdb_relationships.json의 profile content가 되고, embedding 대상이 된다.

우리 실험에서의 의미:
- P Function은 R Function이 뽑은 여러 중복 설명을 하나의 검색 가능한 profile로 정리하는 LLM 단계다.
- D Function은 이 profile과 source 정보를 이용해 같은 entity/relation을 병합한다."""
    return f"""
    <div class="prompt-excerpts">
      <article class="prompt-card">
        <span class="eyebrow">R Function prompt</span>
        <h3>Entity / Relation 추출 기준</h3>
        <p>LightRAG 기본 extraction prompt에 특허용 YAML guidance와 few-shot 예시를 주입해 gpt-5.5가 entity/relation tuple을 생성했다.</p>
        {code_block_preserve(r_prompt, 3900)}
      </article>
      <article class="prompt-card translation">
        <span class="eyebrow">R Function 한국어 번역</span>
        <h3>LLM이 entity / relation을 뽑는 규칙</h3>
        <p>위 R Function prompt를 보고서용으로 한국어로 옮긴 버전이다.</p>
        {code_block_preserve(r_prompt_ko, 2600)}
      </article>
    </div>
    <div class="prompt-excerpts prompt-translations">
      <article class="prompt-card">
        <span class="eyebrow">P Function prompt</span>
        <h3>Profile / Summary 생성 기준</h3>
        <p>동일 entity/relation의 description 조각이 여러 개일 때 LightRAG가 LLM으로 하나의 profile summary를 만든다.</p>
        {code_block_preserve(p_prompt, 2200)}
      </article>
      <article class="prompt-card translation">
        <span class="eyebrow">P Function prompt 한국어 번역</span>
        <h3>LLM이 profile summary를 만드는 규칙</h3>
        <p>위 P Function prompt를 보고서용으로 한국어로 옮긴 버전이다.</p>
        {code_block_preserve(p_prompt_ko, 2400)}
      </article>
    </div>
    """


def rpd_summary(
    working_dir: Path,
    chunks: dict[str, Any],
    full_entities: dict[str, Any],
    full_relations: dict[str, Any],
    entity_chunks: dict[str, Any],
    relation_chunks: dict[str, Any],
    llm_cache: dict[str, Any],
    vdb_entities: list[dict[str, Any]],
    vdb_relationships: list[dict[str, Any]],
) -> str:
    doc_id = ""
    relation_counts = []
    for key, value in full_relations.items() if isinstance(full_relations, dict) else []:
        pairs = value.get("relation_pairs", []) if isinstance(value, dict) else []
        relation_counts.append((len(pairs), str(key)))
    if relation_counts:
        doc_id = sorted(relation_counts, reverse=True)[0][1]
    entity_names = full_entities.get(doc_id, {}).get("entity_names", []) if isinstance(full_entities.get(doc_id), dict) else []
    relation_pairs = full_relations.get(doc_id, {}).get("relation_pairs", []) if isinstance(full_relations.get(doc_id), dict) else []
    chunk_key = next((key for key in chunks if str(key).startswith(f"{doc_id}-chunk-")), next(iter(chunks), ""))
    chunk = chunks.get(chunk_key, {}) if isinstance(chunks.get(chunk_key), dict) else {}
    dedup_entity = sorted(
        [
            (int(value.get("count", 0) or 0), key)
            for key, value in entity_chunks.items()
            if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1
        ],
        reverse=True,
    )[:3]
    dedup_relation = sorted(
        [
            (int(value.get("count", 0) or 0), key)
            for key, value in relation_chunks.items()
            if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1
        ],
        reverse=True,
    )[:3]
    rows = [
        {"Function": "Input chunk", "Actual output": f"{doc_id} / {chunk_key} / tokens={chunk.get('tokens', '—')}"},
        {"Function": "R: entities", "Actual output": f"{len(entity_names)}개: " + ", ".join(map(str, entity_names[:8]))},
        {"Function": "R: relations", "Actual output": f"{len(relation_pairs)}개: " + "; ".join(f"{pair[0]} -> {pair[1]}" for pair in relation_pairs[:5] if isinstance(pair, list) and len(pair) >= 2)},
        {"Function": "P: profiles", "Actual output": f"vdb_entities.json / vdb_relationships.json에 summary + source_id + vector 저장"},
        {"Function": "D: dedup", "Actual output": "Entity: " + "; ".join(f"{name}({count})" for count, name in dedup_entity) + " / Relation: " + "; ".join(f"{name.replace('<SEP>', ' -> ')}({count})" for count, name in dedup_relation)},
    ]
    excerpt = short(chunk.get("content", ""), 720)
    r_rows: list[str] = []
    extract_cache_items = sorted(
        [
            item
            for item in llm_cache.values()
            if isinstance(item, dict)
            and item.get("cache_type") == "extract"
            and item.get("chunk_id") == chunk_key
        ],
        key=lambda item: (int(item.get("create_time", 0) or 0), str(item.get("_id", ""))),
    )
    for item in extract_cache_items:
        for line in str(item.get("return", "")).splitlines():
            line = line.strip()
            if line.startswith("entity<|#|>") or line.startswith("relation<|#|>"):
                r_rows.append(line)
    r_entities = [line for line in r_rows if line.startswith("entity<|#|>")][:5]
    r_relations = [line for line in r_rows if line.startswith("relation<|#|>")][:5]
    r_output_sample = "\n".join(r_entities + r_relations) if r_entities or r_relations else "R Function cache sample not found."

    def profile_score(row: dict[str, Any], preferred_names: list[str]) -> tuple[int, int, int]:
        name = str(row.get("entity_name") or row.get("src_id") or "")
        tgt = str(row.get("tgt_id") or "")
        source_id = str(row.get("source_id", ""))
        file_path = str(row.get("file_path", ""))
        preferred = int(name in preferred_names or tgt in preferred_names)
        same_doc = int(file_path == doc_id)
        multi_source = int("<SEP>" in source_id)
        return (preferred, same_doc, multi_source)

    preferred_entities = ["제1 도전형 컬럼 영역", "게이트 패드부", "트렌치", "베이스 영역"]
    entity_profiles = [
        row for row in vdb_entities
        if isinstance(row, dict) and doc_id in str(row.get("file_path", ""))
    ]
    entity_profile = max(entity_profiles, key=lambda row: profile_score(row, preferred_entities), default={})

    preferred_relations = ["제1 도전형 컬럼 영역", "제2 도전형 컬럼 영역", "트렌치"]
    relation_profiles = [
        row for row in vdb_relationships
        if isinstance(row, dict) and doc_id in str(row.get("file_path", ""))
    ]
    relation_profile = max(relation_profiles, key=lambda row: profile_score(row, preferred_relations), default={})

    p_output_sample = "\n".join(
        [
            "# Entity profile output (vdb_entities.json)",
            f"entity_name: {entity_profile.get('entity_name', '—')}",
            f"content: {str(entity_profile.get('content', '—')).replace(chr(10), ' / ')}",
            f"source_id: {entity_profile.get('source_id', '—')}",
            f"file_path: {entity_profile.get('file_path', '—')}",
            "vector: [3072-dim embedding omitted]",
            "",
            "# Relationship profile output (vdb_relationships.json)",
            f"src_id: {relation_profile.get('src_id', '—')}",
            f"tgt_id: {relation_profile.get('tgt_id', '—')}",
            f"content: {str(relation_profile.get('content', '—')).replace(chr(10), ' / ')}",
            f"source_id: {relation_profile.get('source_id', '—')}",
            f"file_path: {relation_profile.get('file_path', '—')}",
            "vector: [3072-dim embedding omitted]",
        ]
    )
    return f"""
    <div class="split">
      <div>
        <h3>R/P/D 실제 출력</h3>
        {table(rows, "table compact-table")}
      </div>
      <div class="callout">
        <h3>Input chunk excerpt</h3>
        <p>{html_escape(excerpt)}</p>
      </div>
    </div>
    <div class="prompt-excerpts rpd-output-examples">
      <article class="prompt-card">
        <span class="eyebrow">R Function output example</span>
        <h3>LLM extraction tuple</h3>
        <p>{html_escape(chunk_key)}를 R Function에 넣었을 때 LLM cache에 저장된 실제 entity/relation tuple 일부다.</p>
        {code_block_preserve(r_output_sample, 2200)}
      </article>
      <article class="prompt-card">
        <span class="eyebrow">P Function output example</span>
        <h3>Profile stored for retrieval</h3>
        <p>R 결과가 entity/relation별로 병합된 뒤 vector DB에 저장된 profile 예시다. vector 원문은 길어서 생략했다.</p>
        {code_block_preserve(p_output_sample, 2200)}
      </article>
    </div>
    """


def relation_samples(vdb_relationships: list[dict[str, Any]], limit: int = 7) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for row in vdb_relationships:
        src = str(row.get("src_id", ""))
        tgt = str(row.get("tgt_id", ""))
        if not src or not tgt or (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        content = str(row.get("content", ""))
        keyword = content.split("\t", 1)[0].split("\n", 1)[0]
        rows.append(
            {
                "source_entity": src,
                "relation": short(keyword, 70),
                "target_entity": tgt,
                "source": row.get("file_path", "—"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def graph_snapshot_svg(graph_metrics: dict[str, Any], vdb_relationships: list[dict[str, Any]]) -> str:
    hubs = [item for item in graph_metrics.get("top_hubs", []) if isinstance(item, dict)][:8]
    hub_names = [str(item.get("entity", "")) for item in hubs]
    edges = []
    hub_set = set(hub_names)
    for row in vdb_relationships:
        src = str(row.get("src_id", ""))
        tgt = str(row.get("tgt_id", ""))
        if src in hub_set or tgt in hub_set:
            edges.append((src, tgt))
        if len(edges) >= 10:
            break
    if not edges:
        edges = [(row["source_entity"], row["target_entity"]) for row in relation_samples(vdb_relationships, 8)]
    node_names = list(dict.fromkeys(hub_names + [name for edge in edges for name in edge]))[:14]
    if not node_names:
        return "<p class='muted'>graph snapshot 없음</p>"
    width, height = 900, 430
    cx, cy = width / 2, height / 2
    positions: dict[str, tuple[float, float]] = {}
    for idx, name in enumerate(node_names):
        angle = -math.pi / 2 + 2 * math.pi * idx / max(1, len(node_names))
        radius = 155 if idx < 8 else 205
        positions[name] = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
    edge_svg = []
    for src, tgt in edges[:12]:
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        edge_svg.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' />")
    node_svg = []
    type_by_name = {str(item.get("entity")): str(item.get("entity_type", "other")) for item in hubs}
    for idx, name in enumerate(node_names):
        x, y = positions[name]
        degree = next((int(item.get("degree", 4) or 4) for item in hubs if item.get("entity") == name), 4)
        radius = 14 + min(22, degree * 0.55)
        node_type = type_by_name.get(name, "other")
        node_svg.append(
            f"<g><circle class='node node-{html_escape(node_type.lower())}' cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' />"
            f"<text x='{x:.1f}' y='{y + radius + 16:.1f}' text-anchor='middle'>{html_escape(short(name, 14))}</text></g>"
        )
    return f"""
    <svg class="graph-svg" viewBox="0 0 {width} {height}" role="img" aria-label="LightRAG graph snapshot">
      <defs><filter id="soft"><feDropShadow dx="0" dy="10" stdDeviation="12" flood-opacity=".12"/></filter></defs>
      <g class="edges">{''.join(edge_svg)}</g>
      <g class="nodes" filter="url(#soft)">{''.join(node_svg)}</g>
    </svg>
    """


def retrieval_mode_rows(light_rows: list[dict[str, Any]], query_id: str = "AA-1") -> list[dict[str, Any]]:
    lookup = result_lookup(light_rows)
    explain = {
        "naive": "chunk only",
        "local": "entity 주변 확장",
        "global": "relation/theme top-k",
        "hybrid": "local + global 병합",
    }
    rows = []
    for mode in ["naive", "local", "global", "hybrid"]:
        row = lookup.get((query_id, mode), {})
        body = context_body(row)
        metadata = query_metadata(row)
        keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
        rows.append(
            {
                "Mode": mode,
                "Meaning": explain[mode],
                "Retrieved": retrieved_count_text(
                    len(body.get("entities", []) or []),
                    len(body.get("relationships", []) or []),
                    len(body.get("chunks", []) or []),
                ),
                "Sources": len(source_patents(row)) if row else "—",
                "Keywords": short(", ".join((keywords.get("high_level") or []) + (keywords.get("low_level") or [])), 120) if isinstance(keywords, dict) else "—",
            }
        )
    return rows


def retrieved_counts(row: dict[str, Any]) -> str:
    body = context_body(row)
    return retrieved_count_text(
        len(body.get("entities", []) or []),
        len(body.get("relationships", []) or []),
        len(body.get("chunks", []) or []),
    )


def keyword_text(row: dict[str, Any], key: str) -> str:
    metadata = query_metadata(row)
    keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
    values = keywords.get(key, []) if isinstance(keywords, dict) else []
    return ", ".join(map(str, values)) or "—"


def qa_behavior_cards(light_rows: list[dict[str, Any]]) -> str:
    lookup = result_lookup(light_rows)
    specs = [
        ("AA-1", "Category-specific", "AI 가속기 병렬성"),
        ("X-2", "Cross-category", "PIM vs NPU"),
        ("F-1", "Fact-check", "특허 16-060058"),
    ]
    cards = []
    for query_id, query_type, title in specs:
        local = lookup.get((query_id, "local"), {})
        global_row = lookup.get((query_id, "global"), {})
        hybrid = lookup.get((query_id, "hybrid"), {})
        rows = [
            {
                "Mode": "local",
                "Keyword path": "low-level keyword -> entity 주변 확장",
                "Keywords": keyword_text(local, "low_level"),
                "Retrieved": retrieved_counts(local),
                "Sources": len(source_patents(local)) if local else "—",
            },
            {
                "Mode": "global",
                "Keyword path": "high-level keyword -> relation/theme top-k",
                "Keywords": keyword_text(global_row, "high_level"),
                "Retrieved": retrieved_counts(global_row),
                "Sources": len(source_patents(global_row)) if global_row else "—",
            },
            {
                "Mode": "hybrid",
                "Keyword path": "local + global context merge",
                "Keywords": short(keyword_text(hybrid, "high_level") + " / " + keyword_text(hybrid, "low_level"), 180),
                "Retrieved": retrieved_counts(hybrid),
                "Sources": len(source_patents(hybrid)) if hybrid else "—",
            },
        ]
        cards.append(
            f"""
            <article class="qa-test-card">
              <span class="eyebrow">{html_escape(query_type)} · {html_escape(query_id)}</span>
              <h3>{html_escape(title)}</h3>
              <p class="qa-question">{html_escape(local.get("question") or global_row.get("question") or hybrid.get("question") or "")}</p>
              {table(rows, "table compact-table qa-mode-table")}
              <div class="qa-hybrid-answer">
                <strong>Hybrid answer excerpt</strong>
                <p>{answer_excerpt_html(hybrid.get("answer", ""), 560)}</p>
              </div>
            </article>
            """
        )
    return "<div class='qa-test-grid'>" + "".join(cards) + "</div>"


def mode_metric_cards(auto_metrics: dict[str, Any]) -> str:
    summary = {
        str(row.get("system", "")).replace("lightrag_", ""): row
        for row in auto_metrics.get("system_summary", [])
        if str(row.get("system", "")).startswith("lightrag_")
    }
    def metric_1(value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "—"

    cards = []
    for mode in ["naive", "local", "global", "hybrid"]:
        row = summary.get(mode, {})
        cards.append(
            f"""
            <article class="mode-card">
              <h3>{html_escape(mode)}</h3>
              <strong>{html_escape(row.get('avg_latency_seconds', '—'))}s</strong>
              <p>평균 검색 근거: {html_escape(retrieved_count_text(metric_1(row.get('avg_retrieved_entities')), metric_1(row.get('avg_retrieved_relations')), metric_1(row.get('avg_retrieved_chunks'))))}</p>
            </article>
            """
        )
    return "".join(cards)


EVAL_RUBRICS = ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]
RUBRIC_LABELS_KO = {
    "Comprehensiveness": "포괄성",
    "Diversity": "다양성",
    "Empowerment": "활용성",
    "Overall": "종합",
}
CASE_STUDY_DECISION_KO = {
    "Comprehensiveness": "GraphRAG가 연산 실행, 데이터/명령 공급, 물리적 통합이라는 의존 구조를 더 체계적으로 설명해 질문의 범위를 더 넓게 다뤘다.",
    "Diversity": "GraphRAG가 systolic array, PIM/near-memory scheduling, active bridge chiplet, HBM, TSV, 열 관리 등 더 다양한 기술 관점을 포함했다.",
    "Empowerment": "GraphRAG가 AI 반도체 하드웨어 스택을 계층적으로 파악할 수 있는 구조를 제공해 후속 분석과 판단에 더 도움이 된다.",
    "Overall": "GraphRAG가 기술 의존 관계를 실행, 공급, 통합의 논리적 계층으로 정리해 더 포괄적이고 다양한 답변으로 평가됐다.",
}


def pair_wins_from_length_summary(length_summary: dict[str, Any]) -> dict[str, Any]:
    block = length_summary.get("judge_normalized_verbosity_aware", {}) if isinstance(length_summary, dict) else {}
    return block.get("pair_wins", {}) if isinstance(block, dict) else {}


def split_tie_pct(pair_wins: dict[str, Any], pair_id: str, rubric: str, left: str, right: str) -> tuple[str, str]:
    counts = pair_wins.get(pair_id, {}).get(rubric, {}) if isinstance(pair_wins, dict) else {}
    if not isinstance(counts, dict):
        return "—", "—"
    left_count = float(counts.get(left, 0) or 0)
    right_count = float(counts.get(right, 0) or 0)
    tie_count = float(counts.get("Tie", 0) or 0)
    total = left_count + right_count + tie_count
    if total <= 0:
        return "—", "—"
    left_pct = (left_count + tie_count * 0.5) / total * 100
    right_pct = (right_count + tie_count * 0.5) / total * 100
    return f"{left_pct:.1f}%", f"{right_pct:.1f}%"


def paper_win_rate_table(length_summary: dict[str, Any]) -> str:
    pair_wins = pair_wins_from_length_summary(length_summary)
    pairs = [
        ("lightrag_hybrid__vs__lightrag_naive", "lightrag_naive", "lightrag_hybrid", "NaiveRAG", "LightRAG"),
        ("lightrag_hybrid__vs__graphrag_global", "graphrag_global", "lightrag_hybrid", "GraphRAG", "LightRAG"),
    ]
    body = []
    for rubric in EVAL_RUBRICS:
        cells = [f"<td>{html_escape(RUBRIC_LABELS_KO.get(rubric, rubric))}</td>"]
        for pair_id, left, right, _, _ in pairs:
            left_pct, right_pct = split_tie_pct(pair_wins, pair_id, rubric, left, right)
            cells.append(f"<td class='num'>{html_escape(left_pct)}</td><td class='num emph'>{html_escape(right_pct)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    group_cells = "".join(f"<th colspan='2'>{html_escape(left_label)} vs {html_escape(right_label)}</th>" for _, _, _, left_label, right_label in pairs)
    label_cells = "".join(f"<th>{html_escape(left_label)}</th><th>{html_escape(right_label)}</th>" for _, _, _, left_label, right_label in pairs)
    return f"""
    <div class="table-wrap">
      <table class="paper-table paper-win-table">
        <caption>Table 1: Win rates (%) of baselines vs. LightRAG on Patent-100.</caption>
        <thead>
          <tr><th rowspan="2">Dimension</th>{group_cells}</tr>
          <tr>{label_cells}</tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    <p class="note">Length-normalized, verbosity-aware judge 기준이다. Tie는 양측에 0.5승씩 분배했다. 여기서 GraphRAG는 Microsoft GraphRAG의 global/community-report query mode를 의미한다.</p>
    """


def paper_ablation_table(auto_metrics: dict[str, Any]) -> str:
    summary = {
        str(row.get("system", "")): row
        for row in auto_metrics.get("system_summary", [])
        if isinstance(row, dict)
    }
    modes = [
        ("NaiveRAG", "lightrag_naive", "chunk only reference"),
        ("LightRAG-Local", "lightrag_local", "low-level keyword -> entity 주변 확장"),
        ("LightRAG-Global", "lightrag_global", "high-level keyword -> relation/theme top-k"),
        ("LightRAG-Hybrid", "lightrag_hybrid", "local + global context merge"),
    ]
    rows = []
    def metric_1(row: dict[str, Any], key: str) -> str:
        try:
            return f"{float(row.get(key, 0) or 0):.1f}"
        except (TypeError, ValueError):
            return "—"

    for label, key, role in modes:
        row = summary.get(key, {})
        retrieved = retrieved_count_text(
            f"{float(row.get('avg_retrieved_entities', 0) or 0):.1f}",
            f"{float(row.get('avg_retrieved_relations', 0) or 0):.1f}",
            f"{float(row.get('avg_retrieved_chunks', 0) or 0):.1f}",
        ) if row else "—"
        rows.append(
            f"""
            <tr>
              <td>{html_escape(label)}</td>
              <td>{html_escape(role)}</td>
              <td>{html_escape(retrieved)}</td>
              <td class="num">{html_escape(metric_1(row, "avg_unique_source_patents") if row else "—")}</td>
              <td class="num">{html_escape(metric_1(row, "avg_answer_chars") if row else "—")}</td>
              <td class="num">{html_escape(metric_1(row, "avg_latency_seconds") if row else "—")}</td>
            </tr>
            """
        )
    return f"""
    <div class="table-wrap">
      <table class="paper-table paper-ablation-table">
        <caption>Table 2: Performance of LightRAG retrieval modes, using NaiveRAG as reference.</caption>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Retrieval path</th>
            <th>Avg retrieved context</th>
            <th>Avg source patents</th>
            <th>Avg answer chars</th>
            <th>Avg latency(s)</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def winner_label(judge_row: dict[str, Any], rubric: str) -> str:
    result = judge_row.get("judge_result", {}).get(rubric, {}) if isinstance(judge_row, dict) else {}
    winner = result.get("winner") if isinstance(result, dict) else ""
    if winner == "Tie":
        return "Tie"
    if winner == "A":
        return str(judge_row.get("answer_a_label", "A"))
    if winner == "B":
        return str(judge_row.get("answer_b_label", "B"))
    return str(winner or "—")


def winner_display(label: str) -> str:
    return {
        "lightrag_hybrid": "LightRAG-Hybrid",
        "graphrag_global": "GraphRAG",
        "lightrag_naive": "NaiveRAG",
        "Tie": "Tie",
    }.get(label, label)


def paper_case_study_table(experiment_dir: Path, query_id: str = "X-1") -> str:
    answers = read_jsonl(experiment_dir / "evaluation_length_control" / "normalized_answers_1200.jsonl")
    answer_lookup = {(str(row.get("query_id")), str(row.get("system"))): row for row in answers}
    judge_rows = read_jsonl(experiment_dir / "evaluation_length_control" / "judge_normalized_verbosity_aware.jsonl")
    judge = next(
        (
            row for row in judge_rows
            if row.get("pair_id") == "lightrag_hybrid__vs__graphrag_global" and row.get("query_id") == query_id
        ),
        {},
    )
    if not judge:
        judge = next((row for row in judge_rows if row.get("pair_id") == "lightrag_hybrid__vs__graphrag_global"), {})
        query_id = str(judge.get("query_id", query_id))
    graph_answer = answer_lookup.get((query_id, "graphrag_global"), {})
    light_answer = answer_lookup.get((query_id, "lightrag_hybrid"), {})
    question = judge.get("question") or graph_answer.get("question") or light_answer.get("question") or "—"
    decision_lines = []
    for rubric in EVAL_RUBRICS:
        decision_lines.append(
            f"<p><strong>{html_escape(RUBRIC_LABELS_KO.get(rubric, rubric))}</strong>: "
            f"승자 = {html_escape(winner_display(winner_label(judge, rubric)))}. "
            f"{html_escape(CASE_STUDY_DECISION_KO.get(rubric, ''))}</p>"
        )
    return f"""
    <div class="table-wrap">
      <table class="paper-table paper-case-table">
        <caption>Table 3: Case study: comparison between LightRAG and GraphRAG.</caption>
        <tbody>
          <tr><th>Query</th><td>{html_escape(question)}</td></tr>
          <tr><th>GraphRAG</th><td>{answer_excerpt_html(graph_answer.get("answer", ""), 720)}</td></tr>
          <tr><th>LightRAG</th><td>{answer_excerpt_html(light_answer.get("answer", ""), 720)}</td></tr>
          <tr><th>LLM 판단</th><td>{''.join(decision_lines)}</td></tr>
        </tbody>
      </table>
    </div>
    """


def system_summary(auto_metrics: dict[str, Any], system: str) -> dict[str, Any]:
    rows = auto_metrics.get("system_summary", []) if isinstance(auto_metrics, dict) else []
    for row in rows:
        if isinstance(row, dict) and row.get("system") == system:
            return row
    return {}


def overall_win_text(length_summary: dict[str, Any], pair_id: str, labels: dict[str, str]) -> str:
    pair_wins = pair_wins_from_length_summary(length_summary)
    overall = pair_wins.get(pair_id, {}).get("Overall", {}) if isinstance(pair_wins, dict) else {}
    if not isinstance(overall, dict) or not overall:
        return "평가 결과 없음"
    return " vs ".join(f"{labels.get(key, key)} {value}" for key, value in overall.items())


def rq_discussion_cards(
    graph_metrics: dict[str, Any],
    auto_metrics: dict[str, Any],
    length_summary: dict[str, Any],
    index_stats: dict[str, Any],
) -> str:
    hybrid = system_summary(auto_metrics, "lightrag_hybrid")
    naive = system_summary(auto_metrics, "lightrag_naive")
    graph = system_summary(auto_metrics, "graphrag_global")
    hybrid_vs_naive = overall_win_text(
        length_summary,
        "lightrag_hybrid__vs__lightrag_naive",
        {"lightrag_naive": "NaiveRAG", "lightrag_hybrid": "LightRAG"},
    )
    graph_vs_light = overall_win_text(
        length_summary,
        "lightrag_hybrid__vs__graphrag_global",
        {"graphrag_global": "GraphRAG", "lightrag_hybrid": "LightRAG"},
    )
    technical_ratio = float(graph_metrics.get("technical_relation_ratio", 0) or 0) * 100
    metadata_ratio = float(graph_metrics.get("metadata_relation_ratio", 0) or 0) * 100
    excluded_ratio = float(graph_metrics.get("excluded_entity_ratio", 0) or 0) * 100
    final_index_seconds = float(index_stats.get("elapsed_seconds", 0) or 0)
    total_attempt_seconds = float(index_stats.get("elapsed_seconds_total_attempts", 0) or 0)
    attempt_count = index_stats.get("attempt_count", "—")
    cards = [
        {
            "rq": "RQ1",
            "question": "생성 성능: 기존 RAG 방법들과 비교했을 때 답변의 질이 얼마나 우수한가?",
            "answer": "Patent-100에서는 LightRAG가 NaiveRAG를 안정적으로 압도했다고 보기는 어렵다. GraphRAG와 비교하면 GraphRAG가 더 우세했다.",
            "points": [
                f"길이 보정 judge 기준 LightRAG vs NaiveRAG: {hybrid_vs_naive}",
                f"길이 보정 judge 기준 LightRAG vs GraphRAG: {graph_vs_light}",
                "따라서 LightRAG의 구현은 정상 동작했지만, 생성 품질 우위는 질문 유형과 비교 대상에 따라 달라진다.",
            ],
        },
        {
            "rq": "RQ2",
            "question": "구성 요소의 기여도: 이중 수준 검색(Dual-level Retrieval)과 그래프 기반 인덱싱이 품질 향상에 어떤 역할을 하는가?",
            "answer": "그래프 기반 인덱싱은 특허 메타데이터가 아니라 기술 entity와 relation 중심의 검색 근거를 만들었다. dual-level retrieval은 chunk-only 검색과 다른 근거 조합을 제공했다.",
            "points": [
                f"그래프 품질: 기술 relation {technical_ratio:.1f}%, metadata relation {metadata_ratio:.1f}%, excluded entity {excluded_ratio:.2f}%",
                "Hybrid 검색 평균: "
                + retrieved_count_text(
                    f"{float(hybrid.get('avg_retrieved_entities', 0) or 0):.1f}",
                    f"{float(hybrid.get('avg_retrieved_relations', 0) or 0):.1f}",
                    f"{float(hybrid.get('avg_retrieved_chunks', 0) or 0):.1f}",
                ),
                "Naive 검색 평균: "
                + retrieved_count_text(
                    f"{float(naive.get('avg_retrieved_entities', 0) or 0):.1f}",
                    f"{float(naive.get('avg_retrieved_relations', 0) or 0):.1f}",
                    f"{float(naive.get('avg_retrieved_chunks', 0) or 0):.1f}",
                ),
            ],
        },
        {
            "rq": "RQ3",
            "question": "사례 연구: 실제 시나리오에서 LightRAG가 보여주는 구체적인 장점은 무엇인가?",
            "answer": "LightRAG의 장점은 답변을 생성할 때 어떤 entity, relation, chunk가 근거로 쓰였는지 추적하기 쉽다는 점이다.",
            "points": [
                "특정 source patent와 기술 관계를 따라가야 하는 질문에서는 LightRAG의 retrieved entity/relation/chunk 구조가 해석에 유리하다.",
                "Table 3의 cross-category 사례에서는 GraphRAG가 community summary 기반으로 더 넓은 답변을 만들어 우세했다.",
                "즉, LightRAG는 넓은 요약보다 근거 추적성과 기술 관계 설명에 강점이 있다.",
            ],
        },
        {
            "rq": "RQ4",
            "question": "비용 및 적응성: 시스템 운영 비용(Token 사용량, API 호출)과 새로운 데이터 업데이트 속도는 어떠한가?",
            "answer": "LightRAG는 최종 성공 실행 기준으로는 빠르게 색인됐지만, 실제 실험에서는 재시도 비용이 컸다. query latency는 GraphRAG보다 낮았다.",
            "points": [
                f"LightRAG indexing 최종 성공 실행: {final_index_seconds:.1f}s",
                f"재시도 포함 총 시도 시간: {total_attempt_seconds / 3600:.2f}h, attempt={attempt_count}",
                f"평균 query latency: LightRAG hybrid {float(hybrid.get('avg_latency_seconds', 0) or 0):.1f}s, GraphRAG {float(graph.get('avg_latency_seconds', 0) or 0):.1f}s",
                "새 데이터 incremental update는 이번 readable 실험에서 별도 조건으로 직접 평가하지 않았으므로, 적응성 결론은 제한적이다.",
            ],
        },
    ]
    return "".join(
        f"""
        <article class="callout rq-card {'accent' if idx == 0 else ''}">
          <span class="eyebrow">{html_escape(card["rq"])}</span>
          <h3>{html_escape(card["question"])}</h3>
          <p class="rq-answer"><strong>답변</strong> {html_escape(card["answer"])}</p>
          <ul class="rq-points">
            {''.join(f"<li>{html_escape(point)}</li>" for point in card["points"])}
          </ul>
        </article>
        """
        for idx, card in enumerate(cards)
    )


def experiment_setup_rows(manifest: dict[str, Any], index_stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"Item": "Dataset", "Value": f"AA/AB/AC/AD x {manifest.get('per_category', 25)} = {manifest.get('total_selected', 100)} patents"},
        {"Item": "Seed", "Value": manifest.get("seed", "—")},
        {"Item": "Prompt", "Value": index_stats.get("entity_type_prompt_file", "patent_ai_semiconductor.yml")},
        {"Item": "LLM", "Value": f"{index_stats.get('llm_model', 'gpt-5.5')} / reasoning={index_stats.get('reasoning_effort', 'xhigh')}"},
        {"Item": "Embedding", "Value": f"{index_stats.get('embedding_model', 'text-embedding-3-large')} / dim={index_stats.get('embedding_dim', '3072')}"},
        {"Item": "Queries", "Value": "15 fixed questions; 직접 설계"},
    ]


def file_link(label: str, path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return f"<a class='file-link' href='file://{html_escape(str(path))}'><strong>{html_escape(label)}</strong><span>{html_escape(str(path))}</span></a>"


def appendix_links(experiment_dir: Path, output: Path, print_output: Path | None = None) -> str:
    files = [
        ("Readable report", output),
        ("Readable print report", print_output or DEFAULT_PRINT_OUTPUT),
        ("Detailed report", ROOT / "reports" / "rag_repro_100_comparison.html"),
        ("Detailed print report", ROOT / "reports" / "rag_repro_100_comparison_print.html"),
        ("Dataset manifest", experiment_dir / "dataset" / "patents_100_manifest.json"),
        ("LightRAG query results", experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"),
        ("Graph metrics", experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json"),
        ("Judge summary", experiment_dir / "evaluation" / "judge_summary.json"),
        ("Length-control judge", experiment_dir / "evaluation_length_control" / "judge_length_control_summary.json"),
    ]
    return "".join(file_link(label, path) for label, path in files)


def css() -> str:
    return """
:root{
  --ink:#17202a; --muted:#667085; --paper:#f6f8fb; --panel:#ffffff; --line:#d8e1ec;
  --blue:#2563eb; --green:#0f766e; --orange:#d97706; --red:#dc2626; --slate:#334155;
  --soft-blue:#e8f0ff; --soft-green:#e7f6f2; --soft-orange:#fff3df; --soft-red:#ffecec;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
a{color:inherit}
.topbar{position:sticky;top:0;z-index:10;background:rgba(246,248,251,.92);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.topbar-inner{max-width:1220px;margin:auto;padding:12px 22px;display:flex;align-items:center;justify-content:space-between;gap:18px}
.brand{font-weight:800;color:#0f172a}
.nav{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.nav a{text-decoration:none;color:#475569;font-size:13px;font-weight:700;padding:7px 10px;border-radius:999px}
.nav a:hover{background:#e2e8f0;color:#0f172a}
main{max-width:1220px;margin:auto;padding:26px 22px 80px}
.hero{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr);gap:18px;align-items:stretch;margin:18px 0 24px}
.hero-copy,.panel,.metric-card,.mini-card,.flow-step,.result-card,.mode-card,.callout{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 12px 32px rgba(15,23,42,.06)}
.hero-copy{padding:34px}
.eyebrow{display:inline-block;color:var(--blue);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:0;margin-bottom:8px}
h1{font-size:48px;line-height:1.08;margin:8px 0 14px;letter-spacing:0;word-break:keep-all}
h2{font-size:28px;line-height:1.15;margin:0;letter-spacing:0;word-break:keep-all}
h3{letter-spacing:0;word-break:keep-all}
.lead{font-size:18px;color:#475569;margin:0;max-width:780px}
.hero-aside{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.metric-card{padding:18px;min-height:122px}
.metric-card span{display:block;color:#64748b;font-size:12px;font-weight:800;text-transform:uppercase}
.metric-card strong{display:block;margin:8px 0 6px;font-size:30px;line-height:1;color:#0f172a}
.metric-card small{color:#64748b}
.compact strong{font-size:24px}
.section{padding:30px 0}
.section-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:16px}
.section-head p{max-width:520px;margin:0;color:#64748b}
.panel{padding:22px;margin-bottom:16px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.split{display:grid;grid-template-columns:1.35fr .65fr;gap:16px}
.implementation-grid{display:grid;grid-template-columns:minmax(0,.88fr) minmax(360px,1.12fr);gap:16px;align-items:start}
.mini-card,.mode-card,.result-card,.callout{padding:18px}
.mini-card h3,.mode-card h3,.result-card h3,.callout h3{margin:0 0 8px;font-size:18px}
.mini-card p,.mode-card p,.result-card p,.callout p{margin:0;color:#64748b}
.rq-card h3{font-size:17px;line-height:1.45;color:#0f172a}
.rq-answer{margin-top:10px!important;color:#334155!important;line-height:1.6}
.rq-answer strong{color:#0f172a;margin-right:4px}
.rq-points{margin:12px 0 0;padding-left:18px;color:#475569;line-height:1.65}
.rq-points li{margin:5px 0}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.implementation-grid .flow{grid-template-columns:1fr}
.flow-step{padding:18px;position:relative;min-height:150px}
.flow-step:not(:last-child)::after{content:"";position:absolute;right:-11px;top:50%;width:10px;height:2px;background:var(--blue)}
.implementation-grid .flow-step{min-height:auto}
.implementation-grid .flow-step:not(:last-child)::after{left:28px;right:auto;top:auto;bottom:-11px;width:2px;height:10px;background:var(--blue)}
.flow-step b{display:inline-grid;place-items:center;width:31px;height:31px;border-radius:50%;background:#0f172a;color:white;font-size:12px;margin-bottom:10px}
.flow-step h3{margin:0 0 7px;font-size:17px}.flow-step p{margin:0;color:#64748b;font-size:13px}
.actual-run{border:1px solid var(--line);border-radius:8px;background:linear-gradient(135deg,#fff,#f8fafc);padding:18px}
.actual-run-head h3{margin:0 0 6px}.actual-run-head p{margin:0 0 14px;color:#64748b}
.actual-flow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;position:relative}
.actual-node{position:relative;border:1px solid var(--line);border-radius:8px;background:white;padding:13px;min-height:104px}
.actual-node span{display:block;color:#2563eb;font-size:11px;font-weight:900;text-transform:uppercase}
.actual-node strong{display:block;margin:7px 0 5px;font-size:20px;line-height:1.1;color:#0f172a}
.actual-node small{display:block;color:#64748b;line-height:1.45}
.actual-sample{margin-top:12px;padding:13px;border-radius:8px;background:#eff6ff;border:1px solid #bfdbfe}
.actual-sample strong{display:block;margin-bottom:6px}.actual-sample p{margin:5px 0;color:#334155}.actual-sample span{font-weight:900;color:#2563eb}
.prompt-excerpts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
.prompt-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:16px;min-width:0}
.prompt-card h3{margin:0 0 6px;font-size:17px}.prompt-card p{margin:0 0 10px;color:#64748b;line-height:1.55}
.prompt-card .code{max-height:430px}
.prompt-card.translation{background:#f8fafc}
.callout{background:linear-gradient(135deg,#fff,#f8fafc)}
.callout.accent{border-color:#93c5fd;background:linear-gradient(135deg,#fff,#eff6ff)}
.code{background:#111827;color:#e5e7eb;border-radius:8px;padding:14px;overflow:auto;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;background:white;margin-top:10px}
.table,.relation-table{width:100%;border-collapse:separate;border-spacing:0;min-width:680px}
.table th,.table td,.relation-table th,.relation-table td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;overflow-wrap:anywhere;line-height:1.55}
.table th,.relation-table th{background:#eef2f7;font-size:12px;color:#475569;white-space:nowrap}.table tr:last-child td,.relation-table tr:last-child td{border-bottom:0}
.table tbody tr:nth-child(even),.relation-table tbody tr:nth-child(even){background:#f8fafc}
.compact-table{min-width:640px}
.compact-table th,.compact-table td{font-size:13px;padding:9px 12px}
.relation-table{table-layout:fixed;min-width:1220px}
.relation-table th,.relation-table td{padding:10px 12px}
.relation-table th:nth-child(1),.relation-table td:nth-child(1){width:14%}
.relation-table th:nth-child(2),.relation-table td:nth-child(2){width:18%}
.relation-table th:nth-child(3),.relation-table td:nth-child(3){width:14%}
.relation-table th:nth-child(4),.relation-table td:nth-child(4){width:36%}
.relation-table th:nth-child(5),.relation-table td:nth-child(5){width:12%;white-space:pre-line}
.relation-table th:nth-child(6),.relation-table td:nth-child(6){width:6%;text-align:center}
.bar-chart{display:grid;gap:10px}
.bar-row{display:grid;grid-template-columns:170px 1fr 70px;gap:10px;align-items:center}
.bar-label strong{display:block}.bar-label span{color:#64748b;font-size:12px}
.bar-track{height:12px;border-radius:999px;background:#e2e8f0;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--green),var(--blue));border-radius:999px}
.bar-value{font-weight:800;text-align:right}
.graph-layout{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(320px,.9fr);gap:16px}
.graph-viz-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:16px;align-items:start}
.viz h3{margin-top:0}
.viz p{color:#64748b;margin:0 0 12px}
.viz svg{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fbfdff}
.edge{stroke:#94a3b8;stroke-width:1.5;opacity:.72}
.multi-edge{stroke:#2563eb;stroke-width:2.6;opacity:.92}
.edge-label{font-size:9px;fill:#334155;stroke:white;stroke-width:3px;paint-order:stroke}
.viz text{font-size:10px;fill:#0f172a;pointer-events:none}
.graph-legend{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin:10px 0 14px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:#fbfdff;font-size:12px;color:#334155}
.graph-legend strong{margin-right:4px;color:#0f172a}.legend-item{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}.legend-dot{width:11px;height:11px;border-radius:50%;display:inline-block;box-shadow:0 0 0 1px rgba(15,23,42,.12)}.legend-note{color:#64748b;margin-left:auto}
.graph-svg{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fbfdff}
.graph-svg line{stroke:#94a3b8;stroke-width:1.5;opacity:.75}.graph-svg text{font-size:11px;fill:#334155}.node{fill:#64748b}.node-techcomponent{fill:#2563eb}.node-architecture{fill:#0f766e}.node-operation{fill:#9333ea}.node-material{fill:#16a34a}.node-method{fill:#d97706}
.mode-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.mode-card strong{font-size:24px;color:#0f172a}
.qa-test-grid{display:grid;gap:16px}
.qa-test-card{border:1px solid var(--line);border-radius:8px;background:white;padding:18px}
.qa-test-card h3{margin:0 0 8px;font-size:20px}.qa-question{margin:0 0 12px;color:#334155;font-weight:700}
.qa-mode-table{table-layout:fixed;min-width:980px}
.qa-mode-table th:nth-child(1),.qa-mode-table td:nth-child(1){width:9%}
.qa-mode-table th:nth-child(2),.qa-mode-table td:nth-child(2){width:24%}
.qa-mode-table th:nth-child(3),.qa-mode-table td:nth-child(3){width:42%}
.qa-mode-table th:nth-child(4),.qa-mode-table td:nth-child(4){width:14%}
.qa-mode-table th:nth-child(5),.qa-mode-table td:nth-child(5){width:11%;text-align:center}
.qa-hybrid-answer{margin-top:12px;padding:13px;border-radius:8px;background:#f8fafc;border:1px solid var(--line)}
.qa-hybrid-answer > strong{display:block;margin-bottom:6px}.qa-hybrid-answer p{margin:0;color:#475569}.qa-hybrid-answer p strong{display:inline;margin:0;color:#0f172a;font-weight:800}
.result-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.result-lines{display:grid;gap:6px;margin:10px 0}.result-lines p{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid var(--line)}.result-lines span{font-weight:800;color:#475569}
.comparison-table{table-layout:fixed;min-width:1040px}
.comparison-table th:nth-child(1),.comparison-table td:nth-child(1){width:24%}
.comparison-table th:nth-child(2),.comparison-table td:nth-child(2){width:18%}
.comparison-table th:nth-child(3),.comparison-table td:nth-child(3){width:20%}
.comparison-table th:nth-child(4),.comparison-table td:nth-child(4){width:38%}
.paper-table{width:100%;border-collapse:collapse;min-width:980px;background:#fff;font-size:13px;line-height:1.45}
.paper-table caption{caption-side:top;margin:0 0 10px;font-weight:900;font-size:16px;color:#0f172a;text-align:left}
.paper-table th,.paper-table td{border:1px solid #cbd5e1;padding:9px 10px;vertical-align:top;text-align:left}
.paper-table th{background:#f1f5f9;color:#334155;font-weight:900}
.paper-table .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.paper-table .emph{font-weight:900;color:#0f172a}
.paper-win-table th{text-align:center}.paper-win-table td:first-child{font-weight:800;color:#334155}
.paper-ablation-table{min-width:1080px}.paper-ablation-table th:nth-child(1),.paper-ablation-table td:nth-child(1){width:15%;font-weight:800}.paper-ablation-table th:nth-child(2),.paper-ablation-table td:nth-child(2){width:33%}.paper-ablation-table th:nth-child(3),.paper-ablation-table td:nth-child(3){width:18%}
.paper-case-table{min-width:980px}.paper-case-table th{width:16%}.paper-case-table p{margin:0 0 8px}.paper-case-table p:last-child{margin-bottom:0}.paper-case-table strong{color:#0f172a}
.file-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.file-link{display:flex;justify-content:space-between;gap:16px;padding:12px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;text-decoration:none}.file-link span{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#64748b;text-align:right;overflow-wrap:anywhere}
.muted{color:#64748b}.note{color:#64748b;margin:8px 0 0}
.pill-row{display:flex;flex-wrap:wrap;gap:8px}.pill{padding:6px 9px;border-radius:999px;background:#e2e8f0;color:#334155;font-size:12px;font-weight:800}.pill.done{background:#dcfce7;color:#166534}.pill.blank{background:#fef3c7;color:#92400e}
details{border:1px solid var(--line);border-radius:8px;background:#fff;padding:12px;margin-top:10px}summary{cursor:pointer;font-weight:800}
@media(max-width:980px){.hero,.graph-layout,.graph-viz-grid,.grid-2,.grid-3,.grid-4,.split,.implementation-grid,.flow,.mode-grid,.result-grid,.file-grid,.prompt-excerpts{grid-template-columns:1fr}.hero-aside{grid-template-columns:1fr 1fr}.flow-step:not(:last-child)::after,.implementation-grid .flow-step:not(:last-child)::after{display:none}.nav{display:none}h1{font-size:36px}.section-head{display:block}.section-head p{margin-top:8px}.bar-row{grid-template-columns:1fr}.actual-flow{grid-template-columns:1fr}}
@media print{.topbar{display:none}body{background:white;font-size:11px}main{max-width:none;padding:0}.section{break-before:page}.hero{break-before:auto}.panel,.metric-card,.mini-card,.flow-step,.result-card,.mode-card,.callout,.actual-run,.actual-node{box-shadow:none;break-inside:avoid}h1{font-size:32px}h2{font-size:20px}.table th,.table td,.relation-table th,.relation-table td{padding:6px;font-size:9px}.code{font-size:8px}.graph-svg,.viz svg{max-height:130mm}.hero,.flow,.grid-2,.grid-3,.grid-4,.split,.implementation-grid,.graph-layout,.graph-viz-grid,.mode-grid,.result-grid{gap:8px}}
"""


def readable_print_css() -> str:
    return """
body.print-version{background:#fff;color:#111827;font-size:13.5px;line-height:1.52}
body.print-version .topbar{display:none}
body.print-version main{max-width:none;padding:20px 24px 38px}
body.print-version .hero{grid-template-columns:1.25fr .75fr;margin:0 0 14px;gap:10px}
body.print-version .hero-copy{padding:18px}
body.print-version h1{font-size:32px;line-height:1.15;margin:6px 0 9px}
body.print-version h2{font-size:23px}
body.print-version h3{font-size:16px}
body.print-version .lead{font-size:14.5px}
body.print-version .section{padding:16px 0}
body.print-version .section-head{margin-bottom:8px}
body.print-version .section-head p{font-size:12.5px}
body.print-version .panel,
body.print-version .metric-card,
body.print-version .mini-card,
body.print-version .flow-step,
body.print-version .result-card,
body.print-version .mode-card,
body.print-version .callout,
body.print-version .actual-run,
body.print-version .actual-node,
body.print-version .prompt-card,
body.print-version .qa-test-card{box-shadow:none;break-inside:avoid;page-break-inside:avoid}
body.print-version .panel{padding:14px;margin-bottom:10px}
body.print-version .metric-card{min-height:74px;padding:10px}
body.print-version .metric-card strong{font-size:22px}
body.print-version .flow-step{padding:10px;min-height:auto}
body.print-version .flow-step b{width:24px;height:24px;margin-bottom:6px}
body.print-version .flow-step p{font-size:11.5px}
body.print-version .implementation-grid{grid-template-columns:.8fr 1.2fr;gap:10px}
body.print-version .actual-node{padding:8px;min-height:76px}
body.print-version .actual-node strong{font-size:15.5px}
body.print-version .actual-node small{font-size:10.5px;line-height:1.38}
body.print-version .prompt-excerpts{gap:8px;margin-top:10px}
body.print-version .prompt-card{padding:10px}
body.print-version .prompt-card h3{font-size:14.5px}
body.print-version .prompt-card p{font-size:11.2px;margin-bottom:7px}
body.print-version .prompt-card .code{max-height:none}
body.print-version .code{font-size:8.6px;line-height:1.32;padding:9px;background:#fff;color:#111827;border:1px solid #cbd5e1}
body.print-version .table-wrap{overflow:visible;margin-top:6px}
body.print-version table{min-width:0!important;width:100%!important;table-layout:fixed}
body.print-version th,
body.print-version td{padding:6px 7px!important;font-size:9.6px!important;line-height:1.34!important;word-break:normal;overflow-wrap:anywhere}
body.print-version .relation-table{min-width:0}
body.print-version .qa-mode-table{min-width:0}
body.print-version .paper-table{min-width:0;font-size:9.6px}
body.print-version .paper-table caption{font-size:13.5px;margin-bottom:6px}
body.print-version .file-grid{grid-template-columns:1fr 1fr}
body.print-version .file-link{padding:7px 8px;gap:8px}
body.print-version .file-link strong{font-size:11px}
body.print-version .file-link span{font-size:8.2px}
body.print-version details{break-inside:avoid;page-break-inside:avoid}
body.print-version details:not([open]) > :not(summary){display:none}
body.print-version .viz svg,
body.print-version .graph-svg{max-height:145mm}
body.print-version .graph-layout,
body.print-version .graph-viz-grid{gap:8px}
body.print-version .graph-legend{font-size:9.2px;padding:6px 7px;gap:5px 8px}
body.print-version .qa-hybrid-answer{padding:8px}
body.print-version .qa-hybrid-answer p{font-size:10.4px;line-height:1.4}
body.print-version .rq-card{padding:10px}
body.print-version .rq-card h3{font-size:13.5px}
body.print-version .rq-answer,
body.print-version .rq-points{font-size:10.4px;line-height:1.42}
body.print-version .rq-points{padding-left:14px;margin-top:6px}
@page{size:A4 landscape;margin:10mm}
@media print{
  body.print-version{font-size:11.5px}
  body.print-version main{padding:0}
  body.print-version .section{break-before:page;page-break-before:always}
  body.print-version .hero{break-before:auto;page-break-before:auto}
  body.print-version #implementation{break-before:page;page-break-before:always}
  body.print-version #appendix details:not([open]){display:none}
  body.print-version a{text-decoration:none;color:inherit}
}
"""


def build_print_html(html: str) -> str:
    print_css = readable_print_css()
    html = html.replace("<title>Patent-100 LightRAG Implementation Report</title>", "<title>Patent-100 LightRAG Implementation Report - Print</title>")
    html = html.replace("</style>", f"\n{print_css}</style>", 1)
    html = html.replace("<body>", "<body class=\"print-version\">", 1)
    html = html.replace("Readable version · 기존 상세 보고서 보존", "Readable print version · 인쇄용")
    html = html.replace("이 버전은 구현 흐름을 먼저 보여주고, 뒤의 실험 결과는 핵심 결론만 남긴 요약 보고서다. raw JSON과 복구 로그는 상세 보고서와 appendix 링크로 분리했다.", "이 버전은 readable 보고서를 그대로 인쇄할 수 있도록 표, 그래프, 프롬프트 블록 간격을 조정한 출력용 문서다.")
    return html


def build_html(experiment_dir: Path, output: Path, print_output: Path | None = None) -> str:
    manifest = read_json(experiment_dir / "dataset" / "patents_100_manifest.json", {}) or {}
    index_stats = read_json(experiment_dir / "lightrag_patent_prompt_100" / "index_stats.json", {}) or {}
    graph_metrics = read_json(experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json", {}) or {}
    auto_metrics = read_json(experiment_dir / "evaluation" / "auto_metrics.json", {}) or {}
    judge_summary = read_json(experiment_dir / "evaluation" / "judge_summary.json", {}) or {}
    length_summary = read_json(experiment_dir / "evaluation_length_control" / "judge_length_control_summary.json", {}) or {}
    query_type_summary = read_json(experiment_dir / "evaluation_length_control" / "query_type_breakdown_length_control.json", {}) or {}
    queries = read_jsonl(experiment_dir / "queries" / "eval_queries_15.jsonl")
    light_rows = read_jsonl(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl")
    graph_rows = read_jsonl(experiment_dir / "graphrag_full_100_fresh" / "query_results_15_methods.jsonl")
    graph_stats = read_json(experiment_dir / "graphrag_full_100_fresh" / "output" / "stats.json", {}) or {}

    working_dir = Path(str(index_stats.get("working_dir") or experiment_dir / "lightrag_patent_prompt_100" / "storage"))
    chunks = read_json(working_dir / "kv_store_text_chunks.json", {}) or {}
    full_entities = read_json(working_dir / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(working_dir / "kv_store_full_relations.json", {}) or {}
    entity_chunks = read_json(working_dir / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(working_dir / "kv_store_relation_chunks.json", {}) or {}
    llm_cache = read_json(working_dir / "kv_store_llm_response_cache.json", {}) or {}
    vdb_entities = load_vdb_rows(working_dir / "vdb_entities.json")
    vdb_relationships = load_vdb_rows(working_dir / "vdb_relationships.json")
    stats = chunk_stats(chunks if isinstance(chunks, dict) else {})
    prompt_path = LIGHTRAG_ROOT / "prompts" / "entity_type" / "patent_ai_semiconductor.yml"
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    single_doc_id, single_patent_graph = patent_graph_example(working_dir)

    light_success = sum(1 for row in light_rows if row.get("status") == "success" and row.get("answer"))
    graph_baseline_rows = [row for row in graph_rows if row.get("method") == "global"]
    graph_baseline_success = sum(1 for row in graph_baseline_rows if row.get("status") == "success" and clean_answer(row.get("answer")))
    judge_total = judge_summary.get("total_judgments", 0)
    query_type_count = Counter(str(row.get("type", "unknown")) for row in queries)
    not_run_rows = [
        {"Baseline": "HyDE", "Status": "미실행", "Reason": "이번 범위 밖"},
        {"Baseline": "RQ-RAG", "Status": "미실행", "Reason": "GPU/구현 비용 문제로 제외"},
        {"Baseline": "Independent vector-only RAG", "Status": "미실행", "Reason": "LightRAG naive와 중복"},
    ]
    query_rows = [
        {"Type": key, "Count": count}
        for key, count in sorted(query_type_count.items(), key=lambda item: item[0])
    ]

    snippet = f"""# .env
PROMPT_DIR={LIGHTRAG_ROOT / "prompts"}
ENTITY_TYPE_PROMPT_FILE=patent_ai_semiconductor.yml
ENTITY_EXTRACTION_USE_JSON=false

# LightRAG addon_params
addon_params={{
    "language": "Korean",
    "entity_type_prompt_file": os.getenv("ENTITY_TYPE_PROMPT_FILE"),
}}"""

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Patent-100 LightRAG Implementation Report</title>
  <style>{css()}</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">Patent-100 LightRAG Report</div>
      <nav class="nav">
        <a href="#overview">Overview</a>
        <a href="#implementation">Implementation</a>
        <a href="#qa-test">QA Test</a>
        <a href="#comparison">RAG Comparison</a>
        <a href="#discussion">Discussion</a>
        <a href="#appendix">Appendix</a>
      </nav>
    </div>
  </header>
  <main>
    <section id="overview" class="hero">
      <div class="hero-copy">
        <span class="eyebrow">Readable version · 기존 상세 보고서 보존</span>
        <h1>AI 반도체 특허 100건으로 구현한 LightRAG 흐름</h1>
        <p class="lead">이 버전은 구현 흐름을 먼저 보여주고, 뒤의 실험 결과는 핵심 결론만 남긴 요약 보고서다. raw JSON과 복구 로그는 상세 보고서와 appendix 링크로 분리했다.</p>
      </div>
      <div class="hero-aside">
        <article class="metric-card"><span>Dataset</span><strong>{html_escape(manifest.get("total_selected", 100))}</strong><small>AA/AB/AC/AD x {html_escape(manifest.get("per_category", 25))}, seed={html_escape(manifest.get("seed", "—"))}</small></article>
        <article class="metric-card"><span>LightRAG graph</span><strong>{html_escape(graph_metrics.get("graph_nodes", "—"))}</strong><small>{html_escape(graph_metrics.get("graph_edges", "—"))} relations</small></article>
        <article class="metric-card"><span>Queries</span><strong>{len(queries)}</strong><small>LightRAG {light_success}/{len(light_rows)}, GraphRAG baseline {graph_baseline_success}/{len(graph_baseline_rows)}</small></article>
        <article class="metric-card"><span>Judge</span><strong>{html_escape(judge_total)}</strong><small>Gemini 3.5 Flash pairwise judgments</small></article>
      </div>
    </section>

    <section id="implementation" class="section">
      <div class="section-head">
        <h2>1. LightRAG Implementation</h2>
        <p>중요한 부분은 여기다. 특허가 어떤 텍스트로 들어가고, R/P/D를 거쳐 어떤 graph와 vector store가 되는지 실제 값을 붙여 설명한다.</p>
      </div>

      <div class="panel">
        <h3>Implementation flow</h3>
        <div class="implementation-grid">
          <div class="flow">{pipeline_html()}</div>
          {actual_data_flow_visual(manifest, index_stats, graph_metrics, working_dir, chunks, full_entities, full_relations, entity_chunks, relation_chunks, light_rows)}
        </div>
        {rpd_prompt_excerpts(prompt_text)}
      </div>

      <div class="grid-3">
        {model_cards(index_stats)}
      </div>

      <div class="panel">
        <h3>Patent schema and prompt wiring</h3>
        <div class="grid-2">
          <div>
            {table([
                {"Field group": "Identifier/source", "Fields": "patent_id, 출원번호, 공개번호, 등록번호"},
                {"Field group": "Technical text", "Fields": "제목, 요약, AI 목적/솔루션, 대표청구항"},
                {"Field group": "Classification", "Fields": "중/소분류, IPC/CPC"},
                {"Field group": "Ownership/time", "Fields": "출원인/권리자, 국가, 출원연도"},
            ], "table compact-table")}
          </div>
          <div>{code_block(snippet, 1200)}</div>
        </div>
        <p class="note">특허번호와 출원번호는 source/reference로 보존하고, entity node로는 뽑지 않도록 prompt에서 제외했다.</p>
      </div>

      <div class="panel">
        {rpd_summary(working_dir, chunks, full_entities, full_relations, entity_chunks, relation_chunks, llm_cache, vdb_entities, vdb_relationships)}
      </div>

      <div class="panel">
        <h3>Graph quality gate</h3>
        <div class="grid-4">{quality_cards(graph_metrics)}</div>
      </div>

      <div class="panel">
        <h3>Graph visualization and relation table</h3>
        <p class="note">상세 보고서에서 만든 multi-patent relation cluster를 그대로 사용한다. 단순 hub snapshot보다 LightRAG가 실제로 병합한 entity relation과 source patent를 설명하기 쉽다.</p>
        {full_graph_visualization(working_dir)}
        <div class="single-graph-block">
          <h3>Single patent graph example</h3>
          <p class="note">아래는 특허 <code>{html_escape(single_doc_id)}</code> 하나에서 나온 entity/relation만 렌더링한 예시다. 전체 graph가 특허번호 중심 계층 구조가 아니라, source chunk에서 추출된 기술 entity 간 관계 graph라는 점을 보여준다.</p>
          {single_patent_graph}
        </div>
      </div>

      <div class="panel">
        <h3>Entity type distribution</h3>
        {entity_type_chart(graph_metrics.get("entity_type_counts", {}) if isinstance(graph_metrics, dict) else {})}
      </div>

      <div class="panel">
        <h3>Retrieval mode infographic</h3>
        <div class="mode-grid">{mode_metric_cards(auto_metrics)}</div>
        <h3>AA-1 actual retrieval trace</h3>
        {table(retrieval_mode_rows(light_rows, "AA-1"), "table compact-table")}
      </div>
    </section>

    <section id="qa-test" class="section">
      <div class="section-head">
        <h2>2. QA Retrieval Test</h2>
        <p>질문이 들어왔을 때 LightRAG가 keyword를 나누고, local/global retrieval을 따로 수행한 뒤 hybrid 답변으로 합치는지 실제 query 결과로 확인한다.</p>
      </div>
      <div class="panel">
        <h3>Local / Global / Hybrid split on real questions</h3>
        {qa_behavior_cards(light_rows)}
      </div>
    </section>

    <section id="comparison" class="section">
      <div class="section-head">
        <h2>3. LightRAG vs GraphRAG Comparison</h2>
        <p>비교 결과는 논문 형식에 맞춘 세 개 표로만 정리한다. raw judge JSON과 복구 내역은 appendix 링크에 보존했다.</p>
      </div>
      <div class="panel">
        {paper_win_rate_table(length_summary)}
        {paper_ablation_table(auto_metrics)}
        {paper_case_study_table(experiment_dir)}
      </div>
    </section>

    <section id="discussion" class="section">
      <div class="section-head">
        <h2>4. Discussion</h2>
        <p>논문의 평가 질문 구조를 따라 Patent-100 결과를 RQ1~RQ4로 다시 정리한다. 수행하지 않은 조건은 결론에서 분리해 표시했다.</p>
      </div>
      <div class="grid-2">
        {rq_discussion_cards(graph_metrics, auto_metrics, length_summary, index_stats)}
      </div>
    </section>

    <section id="appendix" class="section">
      <div class="section-head">
        <h2>5. Appendix</h2>
        <p>읽기용 본문에서는 뺀 원문과 파일 경로다. 재현 근거가 필요할 때만 펼쳐 보면 된다.</p>
      </div>
      <div class="panel">
        <h3>Reproduction files</h3>
        <div class="file-grid">{appendix_links(experiment_dir, output, print_output)}</div>
      </div>
      <div class="panel">
        <details>
          <summary>Experiment setup and not-run baselines</summary>
          {table(experiment_setup_rows(manifest, index_stats), "table compact-table")}
          {table(query_rows, "table compact-table")}
          {table(not_run_rows, "table compact-table")}
        </details>
        <details>
          <summary>Patent prompt excerpt</summary>
          {code_block(prompt_text[:2600], 3000)}
        </details>
        <details>
          <summary>15 query set</summary>
          {table([{"query_id": q.get("query_id"), "type": q.get("type"), "question": q.get("question")} for q in queries], "table compact-table")}
        </details>
        <details>
          <summary>GraphRAG runtime summary</summary>
          {table([
              {"System": "LightRAG indexing", "Elapsed": fmt_seconds(index_stats.get("elapsed_seconds")), "Note": "final successful attempt"},
              {"System": "GraphRAG indexing", "Elapsed": fmt_seconds(graph_stats.get("total_runtime")), "Note": "full fresh index"},
              {"System": "GraphRAG extract_graph", "Elapsed": fmt_seconds((graph_stats.get("workflows") or {}).get("extract_graph", {}).get("overall")), "Note": "LLM extraction"},
              {"System": "GraphRAG community_reports", "Elapsed": fmt_seconds((graph_stats.get("workflows") or {}).get("create_community_reports", {}).get("overall")), "Note": "community summary generation"},
          ], "table compact-table")}
        </details>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output = Path(args.output)
    print_output = Path(args.print_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print_output.parent.mkdir(parents=True, exist_ok=True)
    html = strip_trailing_whitespace(build_html(experiment_dir, output, print_output))
    output.write_text(html, encoding="utf-8")
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
