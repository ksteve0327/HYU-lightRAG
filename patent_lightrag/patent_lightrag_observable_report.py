from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from patent_lightrag.common import ROOT, html_escape, read_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_OUTPUT = ROOT / "reports" / "patent_lightrag_observable_report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an observable LightRAG report for Patent-100.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def short(value: Any, limit: int = 520) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + " ...[truncated]"


def table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>미실행 또는 관측 값 없음</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html_escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in keys) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def pre(value: Any, limit: int = 2400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"<pre>{html_escape(short(text, limit))}</pre>"


def load_vdb_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {}) or {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def chunk_stats(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    token_values = [int(row.get("tokens", 0) or 0) for row in chunks.values() if isinstance(row, dict)]
    doc_counter = Counter(str(row.get("full_doc_id") or row.get("file_path") or "") for row in chunks.values() if isinstance(row, dict))
    return {
        "count": len(chunks),
        "min": min(token_values) if token_values else 0,
        "max": max(token_values) if token_values else 0,
        "avg": round(statistics.mean(token_values), 1) if token_values else 0,
        "multi_chunk_docs": sum(1 for value in doc_counter.values() if value > 1),
    }


def multi_source_rows(store: dict[str, Any], label: str, limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(
        store.items(),
        key=lambda item: int(item[1].get("count", 0)) if isinstance(item[1], dict) else 0,
        reverse=True,
    ):
        if not isinstance(value, dict) or int(value.get("count", 0) or 0) < 2:
            continue
        chunk_ids = value.get("chunk_ids", [])
        if "<SEP>" in key:
            left, right = key.split("<SEP>", 1)
            display = f"{left} -> {right}"
        else:
            display = key
        rows.append(
            {
                label: display,
                "source chunks": value.get("count", "—"),
                "chunk_ids": ", ".join(map(str, chunk_ids[:5])) if isinstance(chunk_ids, list) else "—",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def context_body(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data", {})
    body = data.get("data", {}) if isinstance(data, dict) else {}
    return body if isinstance(body, dict) else {}


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    data = row.get("data", {})
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return meta if isinstance(meta, dict) else {}


def source_patent_count(row: dict[str, Any]) -> int:
    patents: set[str] = set()
    for section in ["entities", "relationships", "chunks"]:
        for item in context_body(row).get(section, []) or []:
            if not isinstance(item, dict):
                continue
            for key in ["file_path", "source_id", "chunk_id"]:
                for part in str(item.get(key, "")).split("<SEP>"):
                    part = part.strip()
                    if part:
                        patents.add(part.replace("-chunk-000", ""))
    return len(patents)


def mode_trace_rows(light_rows: list[dict[str, Any]], query_id: str = "AA-1") -> list[dict[str, Any]]:
    rows = []
    for mode in ["naive", "local", "global", "hybrid"]:
        row = next((item for item in light_rows if item.get("query_id") == query_id and item.get("mode") == mode), {})
        body = context_body(row)
        meta = metadata(row)
        keywords = meta.get("keywords", {}) if isinstance(meta, dict) else {}
        rows.append(
            {
                "mode": mode,
                "elapsed": f"{float(row.get('elapsed_seconds') or 0):.1f}s" if row else "—",
                "entities": len(body.get("entities", []) or []),
                "relations": len(body.get("relationships", []) or []),
                "chunks": len(body.get("chunks", []) or []),
                "source patents": source_patent_count(row) if row else "—",
                "high-level": ", ".join(map(str, (keywords.get("high_level") or [])[:4])) if isinstance(keywords, dict) else "—",
                "low-level": ", ".join(map(str, (keywords.get("low_level") or [])[:5])) if isinstance(keywords, dict) else "—",
            }
        )
    return rows


def mode_summary_rows(auto_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in auto_metrics.get("system_summary", []) if isinstance(auto_metrics, dict) else []:
        system = str(row.get("system", ""))
        if not system.startswith("lightrag_"):
            continue
        rows.append(
            {
                "mode": system.replace("lightrag_", ""),
                "success": f"{row.get('success')}/{row.get('queries')}",
                "avg entities": row.get("avg_retrieved_entities", "—"),
                "avg relations": row.get("avg_retrieved_relations", "—"),
                "avg chunks": row.get("avg_retrieved_chunks", "—"),
                "avg source patents": row.get("avg_unique_source_patents", "—"),
                "avg latency": f"{float(row.get('avg_latency_seconds') or 0):.1f}s",
            }
        )
    order = {"naive": 0, "local": 1, "global": 2, "hybrid": 3}
    return sorted(rows, key=lambda item: order.get(str(item["mode"]), 99))


def entity_type_rows(type_counts: dict[str, int]) -> list[dict[str, Any]]:
    total = sum(type_counts.values()) or 1
    return [
        {"type": key, "count": value, "share": f"{value / total * 100:.1f}%"}
        for key, value in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
    ]


def final_answer_row(light_rows: list[dict[str, Any]], query_id: str = "AA-1") -> dict[str, Any]:
    return next((row for row in light_rows if row.get("query_id") == query_id and row.get("mode") == "hybrid"), {})


def retrieval_evidence_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    body = context_body(row)
    rows = []
    for index, chunk in enumerate((body.get("chunks") or [])[:5], start=1):
        if not isinstance(chunk, dict):
            continue
        rows.append(
            {
                "rank": index,
                "patent": chunk.get("file_path") or chunk.get("chunk_id") or "—",
                "chunk_id": chunk.get("chunk_id", "—"),
                "content excerpt": short(chunk.get("content", ""), 260),
            }
        )
    return rows


def vector_card(title: str, rows: list[dict[str, Any]], fields: list[str]) -> str:
    row = rows[0] if rows else {}
    visible = {field: ("[vector omitted]" if field == "vector" else short(row.get(field, ""), 360)) for field in fields}
    return f"<article class='mini-card'><span class='eyebrow'>{html_escape(title)}</span><strong>{len(rows)} vectors</strong>{pre(visible, 1100)}</article>"


def css() -> str:
    return """
:root{--ink:#17202a;--muted:#65717d;--paper:#f5f1e8;--card:#fffdf8;--line:#ded7ca;--blue:#2667ff;--orange:#ef8354;--green:#2a9d8f}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 "Malgun Gothic","Noto Sans KR",sans-serif}
main{max-width:1180px;margin:auto;padding:40px 24px 90px}
h1{font-size:64px;line-height:1.08;letter-spacing:0;margin:12px 0 20px;max-width:900px;word-break:keep-all}
h2{font-size:29px;letter-spacing:0;margin:0 0 14px;word-break:keep-all}
h3{margin:0 0 8px;letter-spacing:0}
p{margin:7px 0 14px}
code,pre{font-family:Consolas,monospace}
code{background:#eee9df;border-radius:5px;padding:2px 5px;font-size:12px;overflow-wrap:anywhere}
pre{white-space:pre-wrap;background:#171b22;color:#e9eef5;padding:18px;border-radius:12px;max-height:420px;overflow:auto}
.hero{padding:42px;border:1px solid var(--line);background:linear-gradient(135deg,#fffdf8 55%,#dbe6ff);border-radius:28px;margin-bottom:26px}
.eyebrow{display:block;color:var(--blue);font-weight:800;text-transform:uppercase;letter-spacing:0;font-size:11px;margin-bottom:8px}
.lead{font-size:18px;max-width:830px}.muted,small{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}
.card{grid-column:span 12;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:25px;margin-top:18px;overflow:hidden}
.half{grid-column:span 6}.third{grid-column:span 4}.stat{font-size:36px;font-weight:900;line-height:1;color:var(--blue)}
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:22px 0}
.flow div{padding:15px 9px;border-radius:12px;text-align:center;background:#eef3ff;border:1px solid #cfdbfb;font-weight:700}
.flow div:not(:last-child)::after{content:"→";float:right;color:var(--blue);transform:translateX(14px)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:0}
.table-wrap{overflow:auto}.bar-row{display:grid;grid-template-columns:92px 1fr;gap:8px;margin:12px 0;align-items:center}
.bar{min-width:105px;color:white;padding:7px 10px;border-radius:7px;font-size:12px;font-weight:800}
.entity{background:var(--blue)}.relation{background:var(--orange)}
details{border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-weight:700}
.mini-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.mini-card{background:#f7f3ea;border-radius:14px;padding:16px;overflow:hidden}.mini-card strong{display:block;font-size:22px}
.callout{border-left:5px solid var(--orange);padding:13px 16px;background:#fff2e8;border-radius:0 12px 12px 0}.ok{border-color:var(--green);background:#eaf7f4}
.blank{opacity:.72;background:repeating-linear-gradient(135deg,#fffdf8,#fffdf8 10px,#f7f3ea 10px,#f7f3ea 20px)}
@media(max-width:800px){.half,.third{grid-column:span 12}.flow,.mini-grid{grid-template-columns:1fr}.flow div:not(:last-child)::after{display:none}.hero{padding:25px}h1{font-size:38px}}
"""


def build_html(experiment_dir: Path, output: Path) -> str:
    manifest = read_json(experiment_dir / "dataset" / "patents_100_manifest.json", {}) or {}
    index_stats = read_json(experiment_dir / "lightrag_patent_prompt_100" / "index_stats.json", {}) or {}
    graph_metrics = read_json(experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json", {}) or {}
    auto_metrics = read_json(experiment_dir / "evaluation" / "auto_metrics.json", {}) or {}
    judge_summary = read_json(experiment_dir / "evaluation" / "judge_summary.json", {}) or {}
    length_summary = read_json(experiment_dir / "evaluation_length_control" / "judge_length_control_summary.json", {}) or {}
    light_rows = read_jsonl(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl")

    pilot_stats = read_json(ROOT / "experiments" / "patent_prompt_pilot_20" / "index_stats.json", {}) or {}
    pilot_manifest = read_json(ROOT / "experiments" / "patent_prompt_pilot_20" / "pilot_docs_manifest.json", {}) or {}
    pilot_storage = Path(str(pilot_stats.get("working_dir") or ROOT / "experiments" / "patent_prompt_pilot_20" / "storage"))
    storage = Path(str(index_stats.get("working_dir") or experiment_dir / "lightrag_patent_prompt_100" / "storage"))

    chunks = read_json(storage / "kv_store_text_chunks.json", {}) or {}
    entities = read_json(storage / "kv_store_full_entities.json", {}) or {}
    relations = read_json(storage / "kv_store_full_relations.json", {}) or {}
    entity_chunks = read_json(storage / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(storage / "kv_store_relation_chunks.json", {}) or {}
    llm_cache = read_json(storage / "kv_store_llm_response_cache.json", {}) or {}
    vdb_chunks = load_vdb_rows(storage / "vdb_chunks.json")
    vdb_entities = load_vdb_rows(storage / "vdb_entities.json")
    vdb_relations = load_vdb_rows(storage / "vdb_relationships.json")
    cstats = chunk_stats(chunks if isinstance(chunks, dict) else {})

    pilot_chunks = read_json(pilot_storage / "kv_store_text_chunks.json", {}) or {}
    pilot_entity_chunks = read_json(pilot_storage / "kv_store_entity_chunks.json", {}) or {}
    pilot_relation_chunks = read_json(pilot_storage / "kv_store_relation_chunks.json", {}) or {}

    multi_entity_count = sum(1 for value in entity_chunks.values() if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1) if isinstance(entity_chunks, dict) else 0
    multi_relation_count = sum(1 for value in relation_chunks.values() if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1) if isinstance(relation_chunks, dict) else 0
    pilot_multi_entity_count = sum(1 for value in pilot_entity_chunks.values() if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1) if isinstance(pilot_entity_chunks, dict) else 0
    pilot_multi_relation_count = sum(1 for value in pilot_relation_chunks.values() if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1) if isinstance(pilot_relation_chunks, dict) else 0

    sample_chunk_key, sample_chunk = next(iter(chunks.items())) if isinstance(chunks, dict) and chunks else ("", {})
    sample_doc = str(sample_chunk.get("full_doc_id") or sample_chunk.get("file_path") or "")
    doc_entities = entities.get(sample_doc, {}).get("entity_names", []) if isinstance(entities.get(sample_doc), dict) else []
    doc_relations = relations.get(sample_doc, {}).get("relation_pairs", []) if isinstance(relations.get(sample_doc), dict) else []
    cache_return = ""
    for value in llm_cache.values() if isinstance(llm_cache, dict) else []:
        if isinstance(value, dict) and value.get("return"):
            cache_return = str(value.get("return"))
            break

    cumulative_rows = [
        {"stage": "5 docs", "documents": "—", "chunks": "—", "entities": "—", "relations": "—", "multi-source entities": "—", "multi-source relations": "—", "status": "미실행"},
        {"stage": "10 docs", "documents": "—", "chunks": "—", "entities": "—", "relations": "—", "multi-source entities": "—", "multi-source relations": "—", "status": "미실행"},
        {"stage": "20 pilot", "documents": pilot_manifest.get("document_count", 20), "chunks": len(pilot_chunks) if isinstance(pilot_chunks, dict) else "—", "entities": pilot_stats.get("graph_nodes", "—"), "relations": pilot_stats.get("graph_edges", "—"), "multi-source entities": pilot_multi_entity_count, "multi-source relations": pilot_multi_relation_count, "status": "완료"},
        {"stage": "25 docs", "documents": "—", "chunks": "—", "entities": "—", "relations": "—", "multi-source entities": "—", "multi-source relations": "—", "status": "미실행"},
        {"stage": "100 fresh", "documents": index_stats.get("document_count", 100), "chunks": cstats["count"], "entities": graph_metrics.get("graph_nodes", "—"), "relations": graph_metrics.get("graph_edges", "—"), "multi-source entities": multi_entity_count, "multi-source relations": multi_relation_count, "status": "완료"},
    ]
    max_entities = max(int(row["entities"]) for row in cumulative_rows if str(row["entities"]).isdigit())
    max_relations = max(int(row["relations"]) for row in cumulative_rows if str(row["relations"]).isdigit())
    bar_html = ""
    for row in cumulative_rows:
        if not str(row["entities"]).isdigit():
            continue
        e = int(row["entities"])
        r = int(row["relations"])
        bar_html += f"""
        <div class="bar-row"><span>{html_escape(row["stage"])}</span>
          <div class="bar entity" style="width:{max(12, e / max_entities * 100):.1f}%">{e} entities</div>
          <div class="bar relation" style="width:{max(12, r / max_relations * 100):.1f}%">{r} relations</div>
        </div>
        """

    final = final_answer_row(light_rows, "AA-1")
    final_answer = final.get("answer", "미실행")
    length_after = (length_summary.get("lengths") or {}).get("after", {}) if isinstance(length_summary, dict) else {}
    pair_wins = judge_summary.get("pair_wins", {}) if isinstance(judge_summary, dict) else {}
    normalized_pair_wins = (length_summary.get("judge_normalized_verbosity_aware") or {}).get("pair_wins", {}) if isinstance(length_summary, dict) else {}

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patent-100 LightRAG 값 변화 관찰 보고서</title>
<style>{css()}</style>
</head>
<body><main>
<section class="hero"><span class="eyebrow">Observable LightRAG · Patent-100 · 2026-06-15</span>
<h1>AI 반도체 특허 100건이<br>그래프와 검색 값으로 바뀌는 과정</h1>
<p class="lead">AI 반도체 특허 100건을 공식 LightRAG에 넣었을 때 구조화 특허 문서가 청크, 엔티티, 관계, 그래프, 벡터 저장소, 검색 컨텍스트로 바뀌는 과정을 관찰한다. 이 보고서는 정답률 중심 보고서가 아니라 실제 저장 값과 검색 값의 이동을 보여주는 관찰 보고서다.</p>
<p class="muted">LLM: {html_escape(index_stats.get("llm_model", "gpt-5.5"))} via codex-proxy · Embedding: {html_escape(index_stats.get("embedding_model", "text-embedding-3-large"))} {html_escape(index_stats.get("embedding_dim", "3072"))} dimensions · chunk {html_escape(index_stats.get("chunk_token_size", "1200"))} tokens</p>
</section>

<div class="grid">
<section class="card third"><span class="eyebrow">Final graph</span><div class="stat">{html_escape(graph_metrics.get("graph_nodes", "—"))}</div><p>고유 엔티티 노드</p></section>
<section class="card third"><span class="eyebrow">Final graph</span><div class="stat">{html_escape(graph_metrics.get("graph_edges", "—"))}</div><p>고유 관계 엣지</p></section>
<section class="card third"><span class="eyebrow">Official merge</span><div class="stat">{multi_entity_count}</div><p>둘 이상의 청크에서 합쳐진 엔티티</p></section>

<section class="card"><h2>실험 목적과 관찰 질문</h2>
<p>목적은 Patent-100에서 LightRAG의 내부 값 변화를 확인하는 것이다. 특히 특허 수가 늘어난 뒤 기술 entity/relation 중심 graph가 만들어졌는지, 메타데이터 node가 억제됐는지, 검색 mode별 context가 어떻게 달라졌는지를 본다.</p>
{table([
  {"관찰 질문": "특허 문서는 LightRAG 안에서 어떤 저장 값으로 바뀌는가?", "의미": "chunk, entity, relation, vector store 생성 흐름 확인"},
  {"관찰 질문": "반복 등장하는 기술 구성요소는 어떻게 병합되는가?", "의미": "동일 entity/relation의 chunk 출처 누적 확인"},
  {"관찰 질문": "특허 특화 prompt는 메타데이터 hub를 줄였는가?", "의미": "metadata relation ratio와 excluded entity ratio 확인"},
  {"관찰 질문": "검색 mode에 따라 context 값은 어떻게 달라지는가?", "의미": "naive/local/global/hybrid 차이 확인"},
  {"관찰 질문": "최종 답변은 어떤 특허 chunk에 연결되는가?", "의미": "answer와 source evidence 연결성 확인"},
])}</section>

<section class="card"><h2>데이터가 움직인 경로</h2>
<div class="flow"><div>patent_rawdata</div><div>100 documents</div><div>{cstats["count"]} chunks</div><div>LLM extraction</div><div>graph + vectors</div><div>retrieval context</div></div>
<div class="callout ok"><strong>전처리 범위:</strong> 특허 1건은 <code>patent_id</code>, 출원번호, 제목, 요약, AI 목적/솔루션, 대표청구항, 중/소분류, IPC/CPC, 출원인/권리자, 국가, 출원연도를 포함한 구조화 텍스트로 변환했다.</div>
<details><summary>실제 첫 입력 chunk 보기: <code>{html_escape(sample_chunk_key)}</code></summary><p>{html_escape(short(sample_chunk.get("content", ""), 1800))}</p></details>
</section>

<section class="card half"><h2>AI 반도체 특허 데이터는 어떤 정보로 구성되는가</h2>
{table([
  {"특허 요소": "기본 식별자", "예시": "patent_id, 출원번호, 공개번호, 등록번호", "LightRAG 변환": "entity로 뽑지 않도록 prompt에서 제외"},
  {"특허 요소": "기술 제목/요약", "예시": "Hardware architecture for neural network", "LightRAG 변환": "chunk 원문과 기술 entity 후보"},
  {"특허 요소": "AI 목적/솔루션", "예시": "연산 병렬성, 데이터 이동 비용 절감", "LightRAG 변환": "operation, performance relation"},
  {"특허 요소": "대표청구항", "예시": "input register, multiplication block", "LightRAG 변환": "TechComponent와 relation"},
  {"특허 요소": "분류/국가/상태", "예시": "IPC/CPC, US, 등록", "LightRAG 변환": "metadata hub 방지를 위해 제외 대상"},
])}
<p class="muted">이 데이터는 NBA 기사보다 구조화 필드가 많다. 그래서 prompt에서 특허번호, IPC/CPC, 국가, 법적상태를 entity로 뽑지 말라고 명시해야 graph가 기술 중심으로 유지된다.</p></section>

<section class="card half"><h2>LightRAG 내부에서 데이터는 어떻게 변환되는가</h2>
{table([
  {"단계": "1. Document Insert", "입력": "구조화 특허 텍스트", "처리": "100건 insert", "출력": "doc status 100"},
  {"단계": "2. Chunking", "입력": "긴 특허 텍스트", "처리": "1200 token 단위 분할", "출력": f"{cstats['count']} chunks"},
  {"단계": "3. Entity Extraction", "입력": "chunk 텍스트", "처리": "특허 prompt 기반 LLM 추출", "출력": f"{graph_metrics.get('graph_nodes', '—')} entities"},
  {"단계": "4. Relation Extraction", "입력": "chunk + entity", "처리": "기술 관계 추출", "출력": f"{graph_metrics.get('graph_edges', '—')} relations"},
  {"단계": "5. Merge", "입력": "동일 이름/entity pair", "처리": "chunk_ids와 profile 병합", "출력": f"multi-source entity {multi_entity_count}"},
  {"단계": "6. Vectorization", "입력": "chunk/entity/relation", "처리": "OpenRouter embedding", "출력": "3종 vector store"},
  {"단계": "7. Retrieval", "입력": "15개 query", "처리": "naive/local/global/hybrid", "출력": "60/60 successful answers"},
])}</section>

<section class="card"><h2>20 pilot → 100 fresh 변화</h2>
<p class="muted">NBA 템플릿의 5→10→25 누적 실행은 우리 Patent-100 실험에서는 수행하지 않았다. 대신 실제로 존재하는 20건 prompt pilot과 100건 fresh index를 표시하고, 수행하지 않은 5/10/25 row는 빈칸으로 둔다.</p>
{table(cumulative_rows)}
<div style="margin-top:22px">{bar_html}</div>
<p class="muted">20건 pilot은 prompt 검증 목적이고, 100건 fresh는 최종 비교 실험용 index다. 두 실행은 같은 데이터 누적 run이 아니라 별도 working dir에서 실행됐다.</p></section>

<section class="card half"><h2>청크 실제 값</h2>
<details open><summary><code>{html_escape(sample_chunk_key)}</code> · {html_escape(sample_chunk.get("tokens", "—"))} tokens</summary>
<p>{html_escape(short(sample_chunk.get("content", ""), 1600))}</p>
<p class="muted">full_doc_id: <code>{html_escape(sample_chunk.get("full_doc_id", ""))}</code> · source: <code>{html_escape(sample_chunk.get("file_path", ""))}</code></p>
</details></section>

<section class="card half"><h2>엔티티 유형 분포 · 100건</h2>{table(entity_type_rows(graph_metrics.get("entity_type_counts", {}) if isinstance(graph_metrics, dict) else {}))}</section>

<section class="card"><h2>하나의 특허는 LightRAG 안에서 어떻게 바뀌는가</h2>
<p><code>{html_escape(sample_doc)}</code> 특허 chunk에서 entity와 relation이 추출되고, 동일 이름/entity pair는 전체 graph의 기존 값과 병합된다.</p>
{table([
  {"단계": "원문", "실제 변화": short(sample_chunk.get("content", ""), 260)},
  {"단계": "청크", "실제 변화": sample_chunk_key},
  {"단계": "추출 엔티티", "실제 변화": ", ".join(map(str, doc_entities[:12]))},
  {"단계": "추출 관계", "실제 변화": "; ".join(f"{pair[0]} -> {pair[1]}" for pair in doc_relations[:8] if isinstance(pair, list) and len(pair) >= 2)},
  {"단계": "병합", "실제 변화": "동일 entity/relation pair의 chunk_ids가 누적됨"},
  {"단계": "검색 활용", "실제 변화": "query context의 entities, relationships, chunks 중 하나로 선택될 수 있음"},
])}
<div class="callout ok"><strong>핵심 해석:</strong> 특허번호는 source/reference로 남고, graph node는 기술 구성요소와 연산/구조/공정 entity 중심으로 구성된다.</div></section>

<section class="card"><h2>중복은 실제로 어떻게 제거됐나</h2>
<p>이번 실행의 중복 제거는 이름과 relation pair 기준 병합이다. 프롬프트로 metadata entity를 줄였기 때문에 <code>US</code>, <code>등록</code>, IPC/CPC 같은 hub가 top hub에서 사라졌다.</p>
<div class="callout"><strong>실제 변화:</strong> 100건 fresh storage에서 multi-source entity는 {multi_entity_count}개, multi-source relation은 {multi_relation_count}개다.</div>
<h3 style="margin-top:22px">병합된 엔티티 예시</h3>{table(multi_source_rows(entity_chunks if isinstance(entity_chunks, dict) else {}, "entity"))}
<h3 style="margin-top:24px">병합된 관계 예시</h3>{table(multi_source_rows(relation_chunks if isinstance(relation_chunks, dict) else {}, "edge"))}</section>

<section class="card"><h2>텍스트에서 그래프로 바뀐 구조</h2>
<div class="mini-grid">
<article class="mini-card"><span class="eyebrow">Top hubs</span>{table([{"entity": item.get("entity"), "degree": item.get("degree"), "type": item.get("entity_type")} for item in (graph_metrics.get("top_hubs", []) or [])[:8] if isinstance(item, dict)])}</article>
<article class="mini-card"><span class="eyebrow">Graph quality</span>{table([
  {"metric": "metadata relation ratio", "value": graph_metrics.get("metadata_relation_ratio", "—")},
  {"metric": "technical relation ratio", "value": graph_metrics.get("technical_relation_ratio", "—")},
  {"metric": "excluded entity ratio", "value": graph_metrics.get("excluded_entity_ratio", "—")},
  {"metric": "degree >= 50 hub count", "value": graph_metrics.get("degree_ge_50_hub_count", "—")},
])}</article>
<article class="mini-card"><span class="eyebrow">Why graph matters</span><p>특허 문장 안의 회로, 메모리 구조, 연산 방식, 성능 지표가 연결되므로 chunk-only 검색보다 기술 관계를 명시적으로 관찰할 수 있다.</p></article>
</div></section>

<section class="card"><h2>LLM 추출 전후</h2>
<p>각 chunk에 대해 entity/relation 추출 응답을 캐시하고, 그 결과가 full_entities, full_relations, graphml, vector DB에 반영된다.</p>
<details open><summary><code>kv_store_llm_response_cache</code> 추출 응답 일부</summary>
<p><strong>입력 청크</strong><br>{html_escape(short(sample_chunk.get("content", ""), 360))}</p>
{pre(cache_return, 2600)}
</details></section>

<section class="card"><h2>벡터 값의 이동</h2>
<div class="mini-grid">
{vector_card("chunks", vdb_chunks, ["content", "full_doc_id", "file_path", "vector"])}
{vector_card("entities", vdb_entities, ["entity_name", "content", "file_path", "vector"])}
{vector_card("relationships", vdb_relations, ["src_id", "tgt_id", "content", "file_path", "vector"])}
</div>
<p class="muted">원시 3072차원 벡터는 보고서에서 생략했다. 관찰 목적은 어떤 대상이 vector record가 되는지 확인하는 것이다.</p></section>

<section class="card"><h2>검색 모드별 값 변화</h2>
<p>동일 질문: <code>AI 코어 및 가속기 특허에서 신경망 연산 가속기 구조는 어떤 방식으로 연산 병렬성을 높이는가?</code></p>
{table(mode_trace_rows(light_rows, "AA-1"))}
<p class="muted">naive는 graph context 없이 chunk만 가져온다. local/global/hybrid는 keyword extraction 이후 entity/relation/chunk context를 구성한다.</p></section>

<section class="card"><h2>검색 모드별 의미</h2>
{table([
  {"검색 모드": "naive", "의미": "청크 벡터 중심 검색", "Patent-100 질문에서의 역할": "원문과 직접 비슷한 특허 찾기"},
  {"검색 모드": "local", "의미": "엔티티 주변 관계 중심 검색", "Patent-100 질문에서의 역할": "특정 기술 구성요소 주변 관계 확인"},
  {"검색 모드": "global", "의미": "관계/theme 중심 검색", "Patent-100 질문에서의 역할": "여러 특허의 공통 기술 패턴 확인"},
  {"검색 모드": "hybrid", "의미": "local + global 결합", "Patent-100 질문에서의 역할": "기술 entity와 broader relation을 함께 사용"},
])}</section>

<section class="card"><h2>질문 해석과 Keyword·검색 모드의 관계</h2>
<div class="callout ok"><strong>질문의 의미:</strong> 신경망 연산 가속기 구조가 어떤 하드웨어 구성과 데이터 흐름으로 병렬성을 높이는지 묻는다.</div>
{table([
  {"구분": "High-Level Keywords", "의미": "질문의 전체 주제", "이번 질문의 예시": "AI 코어, 가속기 특허, 신경망 연산 가속기 구조"},
  {"구분": "Low-Level Keywords", "의미": "직접 탐색할 구체 기술 대상", "이번 질문의 예시": "신경망 연산, 가속기 구조, 연산 병렬성"},
])}
<pre>Low-Level Keywords  -> Local 검색
High-Level Keywords -> Global 검색
두 검색 결과 결합   -> Hybrid 검색</pre></section>

<section class="card"><h2>왜 Local 관계는 많고 Global 관계는 적은가</h2>
{table([
  {"검색 모드": "Local", "시작 Keyword": "Low-Level", "Top-K 대상": "엔티티", "AA-1 결과": "10 entities -> 69 relations -> 10 chunks"},
  {"검색 모드": "Global", "시작 Keyword": "High-Level", "Top-K 대상": "관계", "AA-1 결과": "10 relations -> 18 entities -> 10 chunks"},
  {"검색 모드": "Hybrid", "시작 Keyword": "Both", "Top-K 대상": "병합 context", "AA-1 결과": "24 entities -> 36 relations -> 10 chunks"},
])}
<div class="callout"><strong>핵심 비교:</strong> Local은 선택된 entity의 주변 관계를 확장하므로 relation 수가 커질 수 있고, Global은 relation 자체를 top-k로 고르기 때문에 relation 수가 작게 유지된다.</div></section>

<section class="card"><h2>최종 Hybrid 답변</h2>{pre(final_answer, 3600)}</section>

<section class="card"><h2>최종 Hybrid 답변은 근거와 잘 연결되는가</h2>
<p>정성 평가는 별도 judge에서 수행했고, 여기서는 답변에 들어간 source chunk가 무엇인지 관찰한다.</p>
{table(retrieval_evidence_rows(final))}
<div class="callout"><strong>해석:</strong> AA-1 hybrid 답변은 retrieved chunks 10개, source patents {source_patent_count(final)}개를 사용했다. 다만 답변 품질은 judge 결과상 naive 대비 명확한 우위가 크지 않았다.</div></section>

<section class="card"><h2>관찰된 장점과 한계</h2>
<div class="grid">
<div class="card half" style="margin-top:0"><h3>LightRAG가 잘한 점</h3>{table([
  {"항목": "기술 중심 graph", "설명": f"technical relation ratio={graph_metrics.get('technical_relation_ratio', '—')}"},
  {"항목": "metadata hub 억제", "설명": f"metadata relation ratio={graph_metrics.get('metadata_relation_ratio', '—')}, excluded entity ratio={graph_metrics.get('excluded_entity_ratio', '—')}"},
  {"항목": "검색 context 관측 가능", "설명": "mode별 entity/relation/chunk 수를 추적할 수 있음"},
  {"항목": "source 보존", "설명": "patent_id는 graph node가 아니라 file_path/source_id로 남음"},
])}</div>
<div class="card half" style="margin-top:0"><h3>현재 실행의 한계</h3>{table([
  {"항목": "5/10/25 누적 실험", "문제": "NBA 템플릿과 달리 수행하지 않아 빈칸으로 둠"},
  {"항목": "Hybrid vs naive", "문제": "judge 결과에서 hybrid가 naive를 명확히 압도하지 않음"},
  {"항목": "Length bias", "문제": f"GraphRAG global normalized avg={((length_after.get('graphrag_global') or {}).get('avg', '—'))} chars"},
  {"항목": "정규 스키마", "문제": "relation type이 표준 ontology relation으로 강제되지는 않음"},
])}</div>
</div></section>

<section class="card"><h2>관찰 결론과 주의점</h2>
<p><strong>1.</strong> Patent-100은 LightRAG 안에서 {cstats["count"]} chunks, {graph_metrics.get("graph_nodes", "—")} entities, {graph_metrics.get("graph_edges", "—")} relations로 변환됐다.</p>
<p><strong>2.</strong> 특허 특화 prompt 적용 후 metadata relation ratio는 {graph_metrics.get("metadata_relation_ratio", "—")}로 낮아졌고, technical relation ratio는 {graph_metrics.get("technical_relation_ratio", "—")}로 높게 유지됐다.</p>
<p><strong>3.</strong> 검색 단계에서는 mode별 context가 크게 달라진다. AA-1에서 local은 69 relations, global은 10 relations, hybrid는 36 relations를 사용했다.</p>
<p><strong>4.</strong> 답변 평가에서는 GraphRAG global이 강했지만 길이 편향이 남아 있고, LightRAG hybrid는 naive 대비 뚜렷한 우위를 보이지 않았다.</p>
<p><strong>5.</strong> 이 보고서는 값 이동 관찰 보고서이며, 수행하지 않은 HyDE/RQ-RAG/5→10→25 누적 실험은 결과를 채우지 않았다.</p></section>

<section class="card"><h2>최종 결론</h2>
<p class="lead">AI 반도체 특허 100건은 LightRAG 내부에서 단순 텍스트 묶음이 아니라 기술 entity, relation, source chunk, vector record로 분해되어 관찰 가능한 graph-retrieval 구조로 변환됐다.</p>
<p>다만 이 구조가 항상 chunk-only retrieval보다 나은 답변을 만든다고 단정할 수는 없다. 이번 실험의 보수적 결론은 “그래프 구축 품질은 성공했지만, 답변 품질의 이득은 질문 유형과 judge bias를 분리해서 해석해야 한다”이다.</p></section>

<section class="card"><h2>재현 파일</h2>
<p><code>{html_escape(str(ROOT / "patent_lightrag" / "patent_lightrag_observable_report.py"))}</code> · 이 보고서 생성기</p>
<p><code>{html_escape(str(experiment_dir / "lightrag_patent_prompt_100" / "storage"))}</code> · LightRAG raw storage</p>
<p><code>{html_escape(str(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"))}</code> · mode별 query 결과</p>
<p><code>{html_escape(str(output))}</code> · 생성된 observable HTML</p>
</section>
</div></main></body></html>"""


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strip_trailing_whitespace(build_html(experiment_dir, output)), encoding="utf-8")
    print(json.dumps({"output": str(output), "exists": output.exists()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
