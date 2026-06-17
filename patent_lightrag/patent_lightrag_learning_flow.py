from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from patent_lightrag.common import ROOT, html_escape, read_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "rag_repro_100_seed20260609"
DEFAULT_OUTPUT = ROOT / "reports" / "patent_lightrag_learning_flow.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Peter Pan style LightRAG learning-flow report for Patent-100.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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


def short(value: Any, limit: int = 420) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + " ...[truncated]"


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def table(rows: list[dict[str, Any]], class_name: str = "table") -> str:
    if not rows:
        return "<p class='empty'>미실행 또는 데이터 없음</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html_escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_escape(row.get(key, ''))}</td>" for key in keys) + "</tr>")
    return f"<table class='{class_name}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def pre_block(value: Any, limit: int = 1800) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"<pre class='code'>{html_escape(short(text, limit))}</pre>"


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


def result_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("query_id")), str(row.get("mode") or row.get("method"))): row for row in rows}


def load_vdb_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {}) or {}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def source_patent_count(row: dict[str, Any]) -> int:
    patents: set[str] = set()
    for section in ["entities", "relationships", "chunks"]:
        for item in context_body(row).get(section, []) or []:
            if not isinstance(item, dict):
                continue
            for key in ["file_path", "source_id", "chunk_id"]:
                raw = str(item.get(key, ""))
                for part in raw.split("<SEP>"):
                    part = part.strip()
                    if part:
                        patents.add(part.replace("-chunk-000", ""))
    return len(patents)


def type_percentages(entity_type_counts: dict[str, int]) -> list[dict[str, str]]:
    total = sum(entity_type_counts.values()) or 1
    return [
        {
            "Entity type": key,
            "Count": str(value),
            "Share": f"{value / total * 100:.1f}%",
        }
        for key, value in sorted(entity_type_counts.items(), key=lambda item: item[1], reverse=True)
    ]


def chunk_stats(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tokens = [int(row.get("tokens", 0) or 0) for row in chunks.values() if isinstance(row, dict)]
    doc_counts = Counter(str(row.get("full_doc_id") or row.get("file_path") or "") for row in chunks.values() if isinstance(row, dict))
    return {
        "count": len(chunks),
        "min": min(tokens) if tokens else 0,
        "max": max(tokens) if tokens else 0,
        "avg": statistics.mean(tokens) if tokens else 0,
        "multi_chunk_docs": sum(1 for count in doc_counts.values() if count > 1),
    }


def select_doc_examples(full_entities: dict[str, Any], full_relations: dict[str, Any]) -> tuple[str, list[str], list[list[Any]]]:
    candidates = []
    for doc_id, rel_record in full_relations.items():
        pairs = rel_record.get("relation_pairs", []) if isinstance(rel_record, dict) else []
        ents = full_entities.get(doc_id, {}).get("entity_names", []) if isinstance(full_entities.get(doc_id), dict) else []
        if pairs and ents:
            candidates.append((len(pairs), str(doc_id), ents, pairs))
    if not candidates:
        return "", [], []
    _, doc_id, ents, pairs = sorted(candidates, reverse=True)[0]
    return doc_id, list(ents), list(pairs)


def sample_chunk_for_doc(chunks: dict[str, dict[str, Any]], doc_id: str) -> dict[str, Any]:
    for key, value in chunks.items():
        if key.startswith(f"{doc_id}-chunk-") and isinstance(value, dict):
            return value
    return next((value for value in chunks.values() if isinstance(value, dict)), {})


def retrieval_trace_cards(light_rows: list[dict[str, Any]]) -> str:
    lookup = result_lookup(light_rows)
    trace_specs = [
        ("AA-1", "AI 가속기 병렬성"),
        ("X-2", "PIM vs NPU"),
        ("F-1", "특정 특허 구조 확인"),
    ]
    cards = []
    for query_id, title in trace_specs:
        row = lookup.get((query_id, "hybrid")) or lookup.get((query_id, "local")) or {}
        body = context_body(row)
        metadata = query_metadata(row)
        keywords = metadata.get("keywords", {}) if isinstance(metadata, dict) else {}
        processing = metadata.get("processing_info", {}) if isinstance(metadata, dict) else {}
        high = ", ".join(map(str, (keywords.get("high_level") or [])[:4])) if isinstance(keywords, dict) else "—"
        low = ", ".join(map(str, (keywords.get("low_level") or [])[:5])) if isinstance(keywords, dict) else "—"
        first_rel = (body.get("relationships") or [{}])[0] if isinstance(body, dict) else {}
        first_chunk = (body.get("chunks") or [{}])[0] if isinstance(body, dict) else {}
        cards.append(
            f"""
            <article class="trace-card">
              <h3>{html_escape(title)}</h3>
              <p class="trace-question">{html_escape(row.get("question", "미실행"))}</p>
              <div class="trace-block">
                <b>keywords</b>
                <ul>
                  <li>high: {html_escape(high or "—")}</li>
                  <li>low: {html_escape(low or "—")}</li>
                </ul>
              </div>
              <div class="trace-block">
                <b>retrieved context</b>
                <ul>
                  <li>entities: {len(body.get("entities", []) or [])}</li>
                  <li>relations: {len(body.get("relationships", []) or [])}</li>
                  <li>chunks: {len(body.get("chunks", []) or [])}</li>
                  <li>source patents: {source_patent_count(row)}</li>
                </ul>
              </div>
              <p class="trace-chunk"><strong>relation</strong> {html_escape(short(first_rel, 240))}</p>
              <p class="trace-chunk"><strong>chunk</strong> {html_escape(short(first_chunk.get("content", ""), 260))}</p>
              <p class="trace-chunk"><strong>answer</strong> {html_escape(short(row.get("answer", ""), 300))}</p>
              <p class="trace-chunk"><strong>processing</strong> {html_escape(short(processing, 220))}</p>
            </article>
            """
        )
    return "\n".join(cards)


def mode_ablation_rows(auto_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in auto_metrics.get("system_summary", []) if isinstance(auto_metrics, dict) else []:
        system = str(row.get("system", ""))
        if not system.startswith("lightrag_"):
            continue
        rows.append(
            {
                "Mode": system.replace("lightrag_", ""),
                "Success": f"{row.get('success', '—')}/{row.get('queries', '—')}",
                "Avg answer chars": row.get("avg_answer_chars", "—"),
                "Avg entities": row.get("avg_retrieved_entities", "—"),
                "Avg relations": row.get("avg_retrieved_relations", "—"),
                "Avg chunks": row.get("avg_retrieved_chunks", "—"),
                "Avg latency(s)": row.get("avg_latency_seconds", "—"),
            }
        )
    order = {"naive": 0, "local": 1, "global": 2, "hybrid": 3}
    return sorted(rows, key=lambda item: order.get(str(item["Mode"]), 99))


def count_multi_source(store: dict[str, Any]) -> int:
    return sum(
        1 for value in store.values()
        if isinstance(value, dict) and int(value.get("count", 0) or 0) > 1
    )


def mode_context_counts(light_rows: list[dict[str, Any]], query_id: str, mode: str) -> str:
    row = result_lookup(light_rows).get((query_id, mode), {})
    body = context_body(row)
    return (
        f"E={len(body.get('entities', []) or [])}, "
        f"R={len(body.get('relationships', []) or [])}, "
        f"C={len(body.get('chunks', []) or [])}"
    )


def lightrag_build_cards(
    manifest: dict[str, Any],
    index_stats: dict[str, Any],
    graph_metrics: dict[str, Any],
    cstats: dict[str, Any],
    doc_id: str,
    sample_chunk: dict[str, Any],
    entity_names: list[str],
    relation_pairs: list[list[Any]],
    entity_chunks: dict[str, Any],
    relation_chunks: dict[str, Any],
    chunk_vdb: list[dict[str, Any]],
    entity_vdb: list[dict[str, Any]],
    relation_vdb: list[dict[str, Any]],
    light_rows: list[dict[str, Any]],
) -> str:
    doc_count = manifest.get("doc_count") or manifest.get("total_selected") or index_stats.get("document_count", 100)
    entity_tags = "".join(f"<span class='tag green'>{html_escape(name)}</span>" for name in entity_names[:6])
    relation_tags = "".join(
        f"<span class='tag gold'>{html_escape(pair[0])} → {html_escape(pair[1])}</span>"
        for pair in relation_pairs[:4]
        if isinstance(pair, list) and len(pair) >= 2
    )
    retrieval_rows = [
        {"Mode": "local", "Context": mode_context_counts(light_rows, "AA-1", "local"), "Meaning": "entity 주변 관계 확장"},
        {"Mode": "global", "Context": mode_context_counts(light_rows, "AA-1", "global"), "Meaning": "relation/theme top-k"},
        {"Mode": "hybrid", "Context": mode_context_counts(light_rows, "AA-1", "hybrid"), "Meaning": "local + global 병합"},
    ]
    storage_rows = [
        {"순서": "입력", "실제 값": f"{doc_count} patents", "생성/사용 파일": "dataset/patents_100.jsonl"},
        {"순서": "Chunk", "실제 값": f"{cstats['count']} chunks, avg {round(cstats['avg'], 1)} tokens", "생성/사용 파일": "kv_store_text_chunks.json"},
        {"순서": "R 추출", "실제 값": f"sample {doc_id}: {len(entity_names)} entities / {len(relation_pairs)} relations", "생성/사용 파일": "kv_store_full_entities.json, kv_store_full_relations.json"},
        {"순서": "P/D 병합", "실제 값": f"{count_multi_source(entity_chunks)} merged entities, {count_multi_source(relation_chunks)} multi-source edges", "생성/사용 파일": "kv_store_entity_chunks.json, kv_store_relation_chunks.json"},
        {"순서": "Graph", "실제 값": f"{graph_metrics.get('graph_nodes', '—')} nodes / {graph_metrics.get('graph_edges', '—')} edges", "생성/사용 파일": "graph_chunk_entity_relation.graphml"},
        {"순서": "Vector", "실제 값": f"{len(chunk_vdb) + len(entity_vdb) + len(relation_vdb)} records", "생성/사용 파일": "vdb_chunks.json, vdb_entities.json, vdb_relationships.json"},
        {"순서": "QA", "실제 값": f"AA-1 hybrid: {mode_context_counts(light_rows, 'AA-1', 'hybrid')}", "생성/사용 파일": "query_results_15_modes.jsonl"},
    ]
    return f"""
    <div class="grid-3" style="margin-top:16px">
      <article class="mini-card">
        <h3>데이터가 들어가는 모습</h3>
        <p>구조화된 특허 100건이 chunk로 나뉘고, 이후 모든 graph/vector storage의 출처가 됩니다.</p>
        <div class="tag-row">
          <span class="tag green">{html_escape(doc_count)} patents</span>
          <span class="tag">{html_escape(cstats["count"])} chunks</span>
          <span class="tag gold">chunk_token_size={html_escape(index_stats.get("chunk_token_size", 1200))}</span>
        </div>
      </article>
      <article class="mini-card">
        <h3>Graph로 바뀌는 모습</h3>
        <p>R 함수가 entity/relation을 만들고, P/D 단계에서 같은 기술 표현을 병합해 graph node와 edge로 저장합니다.</p>
        <div class="tag-row">
          <span class="tag green">{html_escape(graph_metrics.get("graph_nodes", "—"))} nodes</span>
          <span class="tag gold">{html_escape(graph_metrics.get("graph_edges", "—"))} edges</span>
          <span class="tag">{count_multi_source(entity_chunks)} merged entities</span>
        </div>
      </article>
      <article class="mini-card">
        <h3>질문에 쓰이는 모습</h3>
        <p>chunk/entity/relation vector가 query와 비교되고, local/global/hybrid mode에 따라 다른 근거 묶음이 답변으로 들어갑니다.</p>
        <div class="tag-row">
          <span class="tag">{len(chunk_vdb) + len(entity_vdb) + len(relation_vdb)} vector records</span>
          <span class="tag green">AA-1 hybrid</span>
          <span class="tag gold">{html_escape(mode_context_counts(light_rows, "AA-1", "hybrid"))}</span>
        </div>
      </article>
    </div>
    <div class="grid-2" style="margin-top:16px">
        <article class="mini-card">
          <h3>실제 R output 예시</h3>
          <p><code>{html_escape(doc_id)}</code> chunk에서 만들어진 entity/relation 일부입니다.</p>
          <div class="tag-row">{entity_tags}</div>
          <div class="tag-row">{relation_tags}</div>
        </article>
        <article class="mini-card">
          <h3>local/global/hybrid 분기</h3>
          <p>같은 AA-1 질문이 local/global/hybrid에서 서로 다른 context 수로 분기됩니다.</p>
          {table(retrieval_rows)}
        </article>
    </div>
    <div class="panel" style="margin-top:16px">
      <h3>실제 생성 파일 기준 흐름</h3>
      {table(storage_rows)}
    </div>
    """


def baseline_cards(judge_summary: dict[str, Any], length_summary: dict[str, Any]) -> str:
    pair_wins = judge_summary.get("pair_wins", {}) if isinstance(judge_summary, dict) else {}
    normalized = (length_summary.get("judge_normalized_verbosity_aware") or {}).get("pair_wins", {}) if isinstance(length_summary, dict) else {}

    def overall(pair_id: str, source: dict[str, Any]) -> str:
        block = (source.get(pair_id) or {}).get("Overall", {})
        if not isinstance(block, dict) or not block:
            return "—"
        return ", ".join(f"{k}: {v}" for k, v in block.items())

    cards = [
        ("완료", "NaiveRAG", "LightRAG mode='naive'로 graph 없이 chunk embedding 검색을 수행했다.", overall("lightrag_hybrid__vs__lightrag_naive", pair_wins)),
        ("완료", "LightRAG Hybrid", "local entity/relation context와 global relation/theme context를 병합했다.", overall("lightrag_hybrid__vs__lightrag_naive", normalized)),
        ("완료", "GraphRAG-style", "official Microsoft GraphRAG basic/local/global을 별도 실행했다.", overall("lightrag_hybrid__vs__graphrag_global", pair_wins)),
        ("미실행", "HyDE", "이번 Patent-100 실험에서는 hypothetical document expansion을 돌리지 않았다.", "—"),
        ("미실행", "RQ-RAG", "GPU/구현 비용 이슈로 제외했다. 결과 표는 비워둔다.", "—"),
        ("미실행", "독립 vector-only RAG", "LightRAG naive와 중복되므로 별도 구현하지 않았다.", "—"),
    ]
    return "\n".join(
        f"""
        <article class="rq-method {'blank-method' if status == '미실행' else ''}">
          <span>{html_escape(status)}</span>
          <h3>{html_escape(name)}</h3>
          <p>{html_escape(desc)}</p>
          <div class="tag-row"><span class="tag {'gold' if status == '미실행' else 'green'}">{html_escape(result)}</span></div>
        </article>
        """
        for status, name, desc, result in cards
    )


def question_cards(queries: list[dict[str, Any]]) -> str:
    cards = []
    for row in queries[:15]:
        cards.append(
            f"""
            <article class="qa-card">
              <h3>{html_escape(row.get("query_id", ""))}</h3>
              <p>{html_escape(row.get("question", ""))}</p>
              <div class="qa-answer">
                <b>{html_escape(row.get("type", ""))}</b>
                <span>{html_escape(row.get("expected_focus", ""))}</span>
              </div>
            </article>
            """
        )
    return "\n".join(cards)


def file_cards(experiment_dir: Path, output: Path) -> str:
    files = [
        ("LightRAG query results", experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl"),
        ("LightRAG graph metrics", experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json"),
        ("LightRAG storage", experiment_dir / "lightrag_patent_prompt_100" / "storage"),
        ("GraphRAG results", experiment_dir / "graphrag_full_100_fresh" / "query_results_15_methods.jsonl"),
        ("Judge summary", experiment_dir / "evaluation" / "judge_summary.json"),
        ("Length-control judge", experiment_dir / "evaluation_length_control" / "judge_length_control_summary.json"),
        ("This learning-flow report", output),
    ]
    return "\n".join(
        f"<a class='file-link' href='file://{html_escape(str(path))}'><strong>{html_escape(label)}</strong><span>{html_escape(str(path))}</span></a>"
        for label, path in files
    )


def css() -> str:
    return """
    :root {
      --ink: #17130d;
      --muted: #6f6556;
      --paper: #f7efe0;
      --paper-2: #fffaf0;
      --line: rgba(56, 42, 23, .16);
      --green: #315f4f;
      --green-2: #d9eadb;
      --blue: #1e536c;
      --blue-2: #dcecf4;
      --gold: #c68a22;
      --gold-2: #fbedd1;
      --red: #9b3d2e;
      --shadow: 0 24px 70px rgba(42, 31, 16, .14);
      --radius-xl: 30px;
      --radius-lg: 22px;
      --mono: "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, monospace;
      --sans: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
      --serif: "Iowan Old Style", "Palatino", "Book Antiqua", Georgia, serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 12% 8%, rgba(198, 138, 34, .24), transparent 28rem),
        radial-gradient(circle at 88% 22%, rgba(49, 95, 79, .24), transparent 30rem),
        linear-gradient(135deg, #f8f1e5 0%, #efe2cc 45%, #f6ead9 100%);
      min-height: 100vh;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .26;
      background-image:
        linear-gradient(rgba(49, 95, 79, .12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(49, 95, 79, .12) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, #000, transparent 85%);
    }
    a { color: inherit; }
    .shell { width: min(1180px, calc(100% - 36px)); margin: 0 auto; }
    .topbar {
      position: sticky;
      top: 14px;
      z-index: 10;
      margin: 14px auto 0;
      width: min(1180px, calc(100% - 36px));
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(56, 42, 23, .14);
      border-radius: 999px;
      background: rgba(255, 250, 240, .75);
      backdrop-filter: blur(18px);
      box-shadow: 0 12px 40px rgba(42, 31, 16, .09);
    }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 800; }
    .brand-mark {
      width: 32px; height: 32px; display: grid; place-items: center;
      border-radius: 50%; color: #fffaf0;
      background: conic-gradient(from 220deg, var(--green), var(--gold), var(--blue), var(--green));
      box-shadow: inset 0 0 0 3px rgba(255, 250, 240, .45);
    }
    .nav { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
    .nav a {
      text-decoration: none; font-size: 13px; font-weight: 700; color: var(--muted);
      padding: 9px 12px; border-radius: 999px;
    }
    .nav a:hover { color: var(--ink); background: rgba(49, 95, 79, .1); }
    .hero { position: relative; padding: 78px 0 44px; }
    .hero-grid {
      display: grid; grid-template-columns: minmax(0, 1.12fr) minmax(300px, .88fr);
      gap: 28px; align-items: stretch;
    }
    .hero-card, .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      background: rgba(255, 250, 240, .7);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero-card { padding: clamp(28px, 5vw, 56px); overflow: hidden; position: relative; }
    .eyebrow {
      display: inline-flex; gap: 8px; align-items: center; padding: 8px 12px;
      border-radius: 999px; background: rgba(49, 95, 79, .12); color: var(--green);
      font-size: 12px; font-weight: 900; text-transform: uppercase; letter-spacing: 0;
    }
    h1 {
      margin: 22px 0 18px; font-family: var(--serif);
      font-size: 72px; line-height: 1.02; letter-spacing: 0;
      max-width: 880px; word-break: keep-all;
    }
    .hero-copy { margin: 0; color: var(--muted); font-size: 20px; line-height: 1.72; max-width: 790px; }
    .hero-copy strong { color: var(--ink); }
    .stats-panel { padding: 22px; display: grid; gap: 14px; }
    .stat {
      position: relative; overflow: hidden; min-height: 120px; padding: 20px;
      border: 1px solid var(--line); border-radius: 24px;
      background: linear-gradient(135deg, rgba(255, 250, 240, .92), rgba(255, 250, 240, .55));
    }
    .stat span { display: block; color: var(--muted); font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; }
    .stat strong { display: block; margin-top: 9px; font-size: 40px; line-height: 1; letter-spacing: 0; }
    .stat small { display: block; margin-top: 10px; color: var(--muted); line-height: 1.5; }
    .section { padding: 38px 0; }
    .section-head { display: grid; grid-template-columns: minmax(0, .7fr) minmax(280px, .3fr); gap: 24px; align-items: end; margin-bottom: 18px; }
    .section h2 { margin: 0; font-family: var(--serif); font-size: 48px; line-height: 1.05; letter-spacing: 0; word-break: keep-all; }
    .section-intro { margin: 0; color: var(--muted); line-height: 1.7; font-size: 16px; }
    .panel { padding: clamp(20px, 3vw, 30px); }
    .pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
    .rq2-pipeline { grid-template-columns: repeat(4, 1fr); }
    .pipe-step, .mini-card, .qa-card, .trace-card, .rq-method {
      border: 1px solid var(--line); border-radius: 22px; background: rgba(255, 250, 240, .72);
    }
    .pipe-step { padding: 18px; position: relative; }
    .pipe-step:not(:last-child)::after {
      content: ""; position: absolute; top: 50%; right: -13px; width: 13px; height: 2px; background: var(--gold);
    }
    .pipe-step b {
      display: grid; place-items: center; width: 34px; height: 34px; margin-bottom: 12px;
      border-radius: 50%; background: var(--ink); color: var(--paper-2); font-family: var(--mono); font-size: 14px;
    }
    .pipe-step h3, .mini-card h3, .callout h3 { margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }
    .pipe-step p, .mini-card p, .callout p { margin: 0; color: var(--muted); line-height: 1.6; font-size: 14px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
    .mini-card { padding: 20px; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .tag {
      display: inline-flex; align-items: center; min-height: 28px; padding: 5px 9px; border-radius: 999px;
      background: rgba(30, 83, 108, .1); color: var(--blue); font-size: 12px; font-weight: 800;
    }
    .tag.green { background: rgba(49, 95, 79, .12); color: var(--green); }
    .tag.gold { background: rgba(198, 138, 34, .16); color: #7b5211; }
    .code {
      overflow: auto; border-radius: 20px; padding: 18px; background: #1f1a13; color: #f8ead0;
      font-family: var(--mono); font-size: 13px; line-height: 1.65; border: 1px solid rgba(255, 250, 240, .12);
      white-space: pre-wrap;
    }
    .chunk-strip { display: grid; grid-template-columns: repeat(9, minmax(34px, 1fr)); gap: 7px; margin: 20px 0 8px; }
    .chunk {
      min-height: 70px; border-radius: 12px; background: linear-gradient(180deg, var(--green-2), rgba(49, 95, 79, .1));
      border: 1px solid rgba(49, 95, 79, .22); display: flex; align-items: end; justify-content: center;
      padding: 7px 4px; color: var(--green); font-family: var(--mono); font-size: 11px; font-weight: 900;
    }
    .chunk:nth-child(3n) { background: linear-gradient(180deg, var(--gold-2), rgba(198, 138, 34, .13)); color: #7b5211; border-color: rgba(198, 138, 34, .28); }
    .chunk:nth-child(5n) { background: linear-gradient(180deg, var(--blue-2), rgba(30, 83, 108, .12)); color: var(--blue); border-color: rgba(30, 83, 108, .24); }
    .callout { border-radius: 24px; padding: 22px; border: 1px solid rgba(198, 138, 34, .32); background: linear-gradient(135deg, rgba(251, 237, 209, .85), rgba(255, 250, 240, .72)); }
    .table {
      width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 18px; background: rgba(255, 250, 240, .6);
    }
    .table th, .table td {
      padding: 13px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top;
      font-size: 14px; line-height: 1.5; overflow-wrap: anywhere;
    }
    .table th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
    .table tr:last-child td { border-bottom: 0; }
    .graph-cloud { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .node-card {
      min-height: 96px; display: grid; place-items: center; text-align: center; padding: 14px;
      border-radius: 24px; border: 1px solid var(--line);
      background: radial-gradient(circle at 50% 20%, rgba(217, 234, 219, .9), rgba(255, 250, 240, .68));
      font-weight: 900;
    }
    .node-card small { display: block; margin-top: 6px; color: var(--muted); font-family: var(--mono); font-size: 11px; font-weight: 700; }
    .trace-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .trace-card { display: grid; gap: 14px; padding: 20px; }
    .trace-card h3 { margin: 0; font-family: var(--serif); font-size: 24px; line-height: 1.08; letter-spacing: 0; }
    .trace-question { margin: 0; color: var(--ink); font-weight: 800; line-height: 1.45; word-break: keep-all; }
    .trace-block { padding: 14px; border-radius: 18px; background: rgba(255, 250, 240, .62); border: 1px solid var(--line); }
    .trace-block b { display: block; margin-bottom: 8px; color: var(--muted); font-size: 12px; letter-spacing: 0; text-transform: uppercase; }
    .trace-block ul { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .trace-chunk { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.58; }
    .trace-chunk strong { color: var(--green); font-family: var(--mono); font-size: 12px; }
    .qa-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
    .qa-card { display: grid; gap: 12px; padding: 18px; background: linear-gradient(145deg, rgba(255, 250, 240, .78), rgba(217, 234, 219, .38)); }
    .qa-card h3 { margin: 0; font-family: var(--serif); font-size: 22px; line-height: 1.08; letter-spacing: 0; }
    .qa-card p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .qa-answer { padding: 13px; border: 1px solid var(--line); border-radius: 16px; background: rgba(255, 250, 240, .62); }
    .qa-answer b { display: block; margin-bottom: 6px; color: var(--green); font-family: var(--mono); font-size: 11px; letter-spacing: 0; text-transform: uppercase; }
    .qa-answer span { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .rq-methods { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
    .rq-method {
      min-height: 190px; padding: 18px; background: linear-gradient(145deg, rgba(255, 250, 240, .74), rgba(220, 236, 244, .38));
      position: relative; overflow: hidden;
    }
    .rq-method.blank-method { opacity: .72; background: repeating-linear-gradient(135deg, rgba(255,250,240,.72), rgba(255,250,240,.72) 10px, rgba(251,237,209,.62) 10px, rgba(251,237,209,.62) 20px); }
    .rq-method span { display: inline-flex; margin-bottom: 12px; padding: 6px 9px; border-radius: 999px; color: var(--green); background: rgba(49, 95, 79, .12); font-family: var(--mono); font-size: 11px; font-weight: 900; }
    .rq-method h3 { margin: 0 0 8px; font-family: var(--serif); font-size: 25px; line-height: 1.05; letter-spacing: 0; }
    .rq-method p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.58; }
    .file-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .file-link {
      display: flex; justify-content: space-between; align-items: center; gap: 14px; padding: 15px 16px;
      border-radius: 18px; border: 1px solid var(--line); background: rgba(255, 250, 240, .65); text-decoration: none; font-weight: 800;
    }
    .file-link span { color: var(--muted); font-size: 12px; font-family: var(--mono); font-weight: 700; overflow-wrap: anywhere; text-align: right; }
    .empty { color: var(--muted); font-style: italic; }
    footer { padding: 42px 0 70px; color: var(--muted); text-align: center; line-height: 1.7; }
    @media (max-width: 920px) {
      .hero-grid, .section-head, .grid-2, .grid-3, .grid-4, .trace-grid, .qa-grid, .rq-methods, .file-grid, .pipeline, .graph-cloud { grid-template-columns: 1fr; }
      .pipe-step:not(:last-child)::after { display: none; }
      .nav { display: none; }
      .chunk-strip { grid-template-columns: repeat(4, 1fr); }
      h1 { font-size: 44px; }
      .section h2 { font-size: 34px; }
    }
    """


def build_html(experiment_dir: Path, output: Path) -> str:
    manifest = read_json(experiment_dir / "dataset" / "patents_100_manifest.json", {}) or {}
    index_stats = read_json(experiment_dir / "lightrag_patent_prompt_100" / "index_stats.json", {}) or {}
    graph_metrics = read_json(experiment_dir / "lightrag_patent_prompt_100" / "graph_metrics.json", {}) or {}
    auto_metrics = read_json(experiment_dir / "evaluation" / "auto_metrics.json", {}) or {}
    judge_summary = read_json(experiment_dir / "evaluation" / "judge_summary.json", {}) or {}
    length_summary = read_json(experiment_dir / "evaluation_length_control" / "judge_length_control_summary.json", {}) or {}
    queries = read_jsonl(experiment_dir / "queries" / "eval_queries_15.jsonl")
    light_rows = read_jsonl(experiment_dir / "lightrag_patent_prompt_100" / "query_results_15_modes.jsonl")

    storage = Path(str(index_stats.get("working_dir") or experiment_dir / "lightrag_patent_prompt_100" / "storage"))
    chunks = read_json(storage / "kv_store_text_chunks.json", {}) or {}
    full_entities = read_json(storage / "kv_store_full_entities.json", {}) or {}
    full_relations = read_json(storage / "kv_store_full_relations.json", {}) or {}
    entity_chunks = read_json(storage / "kv_store_entity_chunks.json", {}) or {}
    relation_chunks = read_json(storage / "kv_store_relation_chunks.json", {}) or {}
    entity_vdb = load_vdb_rows(storage / "vdb_entities.json")
    relation_vdb = load_vdb_rows(storage / "vdb_relationships.json")
    chunk_vdb = load_vdb_rows(storage / "vdb_chunks.json")

    cstats = chunk_stats(chunks if isinstance(chunks, dict) else {})
    doc_id, entity_names, relation_pairs = select_doc_examples(full_entities, full_relations)
    sample_chunk = sample_chunk_for_doc(chunks if isinstance(chunks, dict) else {}, doc_id)
    entity_profile = next((row for row in entity_vdb if row.get("entity_name") in entity_names), entity_vdb[0] if entity_vdb else {})
    relation_profile = relation_vdb[0] if relation_vdb else {}
    chunk_profile = chunk_vdb[0] if chunk_vdb else {}
    top_hubs = graph_metrics.get("top_hubs", []) if isinstance(graph_metrics, dict) else []
    node_cards = "\n".join(
        f"<div class='node-card'>{html_escape(item.get('entity', ''))}<small>degree {html_escape(item.get('degree', ''))} · {html_escape(item.get('entity_type', ''))}</small></div>"
        for item in top_hubs[:12]
        if isinstance(item, dict)
    )

    chunk_boxes = "".join(f"<div class='chunk'>{i:03d}</div>" for i in range(min(cstats["count"], 26))) + "<div class='chunk'>...</div>"
    example_rel_rows = [
        {"Source entity": pair[0], "Target entity": pair[1]}
        for pair in relation_pairs[:8]
        if isinstance(pair, list) and len(pair) >= 2
    ]
    paper_rows = [
        {"논문 흐름": "Document segmentation", "우리 실험": f"{manifest.get('doc_count', 100)} patents -> {cstats['count']} chunks", "상태": "완료"},
        {"논문 흐름": "Entity & relation extraction", "우리 실험": f"{graph_metrics.get('graph_nodes', '—')} nodes / {graph_metrics.get('graph_edges', '—')} edges", "상태": "완료"},
        {"논문 흐름": "Entity/relation/chunk indexing", "우리 실험": f"3 vector stores, dim={index_stats.get('embedding_dim', '—')}", "상태": "완료"},
        {"논문 흐름": "Dual-level retrieval", "우리 실험": "local/global/hybrid + naive", "상태": "완료"},
        {"논문 흐름": "HyDE baseline", "우리 실험": "—", "상태": "미실행"},
        {"논문 흐름": "RQ-RAG baseline", "우리 실험": "—", "상태": "미실행"},
    ]

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Patent-100 LightRAG Learning Flow</title>
  <style>{css()}</style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><span class="brand-mark">L</span>Patent-100 LightRAG Lab</div>
    <nav class="nav" aria-label="page navigation">
      <a href="#paper">Paper</a>
      <a href="#map">Map</a>
      <a href="#chunking">Chunking</a>
      <a href="#graph">Graph</a>
      <a href="#index">Index</a>
      <a href="#retrieval">Retrieval</a>
      <a href="#trace">Trace</a>
      <a href="#qa">Q&A</a>
      <a href="#rq1">RQ1</a>
      <a href="#rq2">RQ2</a>
      <a href="#files">Files</a>
    </nav>
  </header>

  <main>
    <section class="hero shell">
      <div class="hero-grid">
        <div class="hero-card">
          <span class="eyebrow">learning artifact, Patent-100</span>
          <h1>AI 반도체 특허의<br>LightRAG 변환 과정</h1>
          <p class="hero-copy">
            이 문서는 최종 성능표만 보여주는 보고서가 아니라, <strong>특허 100건이 구조화 텍스트로 바뀌고,
            chunk, entity/relation graph, vector index, dual-level retrieval을 거쳐 답변 생성에 쓰이는 과정</strong>을
            Peter Pan 템플릿 형식으로 다시 정리한 학습용 보고서입니다. 우리가 실제로 돌리지 않은 실험은 빈칸으로 남겼습니다.
          </p>
        </div>
        <aside class="stats-panel panel" aria-label="actual index summary">
          <div class="stat">
            <span>patent documents</span>
            <strong>{html_escape(manifest.get("doc_count", 100))}</strong>
            <small>AA/AB/AC/AD 각 25건, seed={html_escape(manifest.get("seed", "20260609"))}</small>
          </div>
          <div class="stat">
            <span>knowledge graph</span>
            <strong>{html_escape(graph_metrics.get("graph_nodes", "—"))} / {html_escape(graph_metrics.get("graph_edges", "—"))}</strong>
            <small>특허 특화 prompt로 추출된 entity node / relation edge</small>
          </div>
          <div class="stat">
            <span>vector indexes</span>
            <strong>3 x {html_escape(index_stats.get("embedding_dim", "—"))}</strong>
            <small>chunks, entities, relationships를 text-embedding-3-large로 embedding</small>
          </div>
        </aside>
      </div>
    </section>

    <section id="paper" class="section shell">
      <div class="section-head">
        <h2>논문 흐름과의 대응</h2>
        <p class="section-intro">
          이 보고서는 논문식 LightRAG의 핵심 흐름 중 <strong>3.1 graph-based indexing, 3.2 dual-level retrieval,
          3.3 answer generation, 3.4 complexity 관찰</strong>을 Patent-100 산출물로 설명합니다.
          HyDE/RQ-RAG처럼 이번에 실행하지 않은 baseline은 결과를 채우지 않았습니다.
        </p>
      </div>
      <div class="panel">{table(paper_rows)}</div>
    </section>

    <section id="map" class="section shell">
      <div class="section-head">
        <h2>1. 전체 지도</h2>
        <p class="section-intro">
          LightRAG는 특허 문서를 단순 chunk vector DB로만 보지 않고, 기술 entity와 relation을 추출해
          graph storage와 vector storage를 함께 만듭니다.
        </p>
      </div>
      <div class="panel pipeline">
        <article class="pipe-step"><b>01</b><h3>Patent 입력</h3><p>특허 100건을 patent_id, 출원번호, 제목, 요약, 청구항, 분류, 출원인, 국가, 연도 필드가 있는 구조화 텍스트로 넣습니다.</p></article>
        <article class="pipe-step"><b>02</b><h3>Chunking</h3><p>chunk_token_size={html_escape(index_stats.get("chunk_token_size", "1200"))} 기준으로 {cstats["count"]}개 chunk가 생성됐습니다.</p></article>
        <article class="pipe-step"><b>03</b><h3>Graph Build</h3><p>LLM이 각 chunk에서 기술 entity/relation을 추출하고 D 단계에서 중복을 병합합니다.</p></article>
        <article class="pipe-step"><b>04</b><h3>Indexing</h3><p>chunk, entity, relationship 각각을 embedding해 검색 가능한 key-value/vector record로 저장합니다.</p></article>
        <article class="pipe-step"><b>05</b><h3>Retrieval</h3><p>질문이 들어오면 naive/local/global/hybrid mode로 근거를 가져오고 답변을 생성합니다.</p></article>
      </div>
      {lightrag_build_cards(
          manifest,
          index_stats,
          graph_metrics,
          cstats,
          doc_id,
          sample_chunk,
          entity_names,
          relation_pairs,
          entity_chunks if isinstance(entity_chunks, dict) else {},
          relation_chunks if isinstance(relation_chunks, dict) else {},
          chunk_vdb,
          entity_vdb,
          relation_vdb,
          light_rows,
      )}
    </section>

    <section id="chunking" class="section shell">
      <div class="section-head">
        <h2>2. Chunking</h2>
        <p class="section-intro">
          특허 한 건은 긴 구조화 텍스트입니다. 대부분 1개 chunk로 들어가지만, 청구항과 분류 설명이 긴 문서는 여러 chunk로 나뉩니다.
        </p>
      </div>
      <div class="grid-2">
        <div class="panel">
          <h3>실제 설정과 결과</h3>
          {pre_block({
              "chunk_token_size": index_stats.get("chunk_token_size", 1200),
              "documents": manifest.get("doc_count", 100),
              "chunks": cstats["count"],
              "token_min": cstats["min"],
              "token_max": cstats["max"],
              "token_avg": round(cstats["avg"], 1),
              "multi_chunk_documents": cstats["multi_chunk_docs"],
          }, 1200)}
          <div class="chunk-strip" aria-label="first chunks">{chunk_boxes}</div>
        </div>
        <div class="panel">
          <h3>chunk 하나는 이렇게 저장됨</h3>
          {pre_block({
              "chunk_id": sample_chunk.get("_id", ""),
              "tokens": sample_chunk.get("tokens", ""),
              "chunk_order_index": sample_chunk.get("chunk_order_index", ""),
              "full_doc_id": sample_chunk.get("full_doc_id", ""),
              "file_path": sample_chunk.get("file_path", ""),
              "content": short(sample_chunk.get("content", ""), 700),
          }, 1400)}
          <div class="callout"><h3>여기서 배울 점</h3><p>chunk는 나중에 answer prompt에 직접 들어가는 원문 근거입니다. graph는 구조를 주고, chunk는 실제 특허 문장과 청구항 근거를 제공합니다.</p></div>
        </div>
      </div>
    </section>

    <section id="graph" class="section shell">
      <div class="section-head">
        <h2>3. Graph가 만들어지는 과정</h2>
        <p class="section-intro">
          R 함수가 entity/relation tuple을 만들고, P 함수가 key-value profile을 만들며, D 함수가 중복 entity/relation을 병합합니다.
        </p>
      </div>
      <div class="grid-3">
        <article class="mini-card"><h3>R: Entity 추출</h3><p>특허 {html_escape(doc_id)}에서 추출된 entity 예시입니다.</p><div class="tag-row">{"".join(f"<span class='tag green'>{html_escape(name)}</span>" for name in entity_names[:8])}</div></article>
        <article class="mini-card"><h3>R: Relation 추출</h3><p>기술 구성요소 간 relation pair가 edge 후보가 됩니다.</p><div class="tag-row">{"".join(f"<span class='tag gold'>{html_escape(pair[0])} → {html_escape(pair[1])}</span>" for pair in relation_pairs[:5] if isinstance(pair, list) and len(pair) >= 2)}</div></article>
        <article class="mini-card"><h3>D: Deduplication</h3><p>같은 이름의 entity와 같은 source-target relation이 병합되고 source chunk가 누적됩니다.</p><div class="tag-row"><span class="tag">same name merge</span><span class="tag">source_id union</span><span class="tag">description profile</span></div></article>
      </div>
      <div class="grid-2" style="margin-top:16px">
        <div class="panel">
          <h3>Entity type 분포</h3>
          {table(type_percentages(graph_metrics.get("entity_type_counts", {}) if isinstance(graph_metrics, dict) else {}))}
        </div>
        <div class="panel">
          <h3>Top connected entities</h3>
          <div class="graph-cloud">{node_cards}</div>
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <h3>Relation tuple 예시</h3>
        {table(example_rel_rows)}
      </div>
    </section>

    <section id="index" class="section shell">
      <div class="section-head">
        <h2>4. Index가 만들어지는 과정</h2>
        <p class="section-intro">
          LightRAG는 graph만 저장하지 않습니다. chunk, entity, relationship을 각각 embedding해서 query embedding과 비교할 수 있게 만듭니다.
        </p>
      </div>
      <div class="grid-3">
        <div class="panel"><h3>Chunk vector record</h3>{pre_block({k: ("[vector omitted]" if k == "vector" else short(v, 420)) for k, v in chunk_profile.items() if k in ["content", "full_doc_id", "file_path", "vector"]}, 1400)}</div>
        <div class="panel"><h3>Entity vector record</h3>{pre_block({k: ("[vector omitted]" if k == "vector" else short(v, 420)) for k, v in entity_profile.items() if k in ["entity_name", "content", "source_id", "file_path", "vector"]}, 1400)}</div>
        <div class="panel"><h3>Relationship vector record</h3>{pre_block({k: ("[vector omitted]" if k == "vector" else short(v, 420)) for k, v in relation_profile.items() if k in ["src_id", "tgt_id", "content", "source_id", "file_path", "vector"]}, 1400)}</div>
      </div>
    </section>

    <section id="retrieval" class="section shell">
      <div class="section-head">
        <h2>5. Retrieval이 진행되는 과정</h2>
        <p class="section-intro">
          같은 질문이라도 mode에 따라 가져오는 근거가 달라집니다. naive는 chunk만, local/global/hybrid는 graph context를 함께 씁니다.
        </p>
      </div>
      <div class="panel">
        {table(mode_ablation_rows(auto_metrics))}
      </div>
      <div class="grid-4" style="margin-top:16px">
        <article class="mini-card"><h3>Naive</h3><p>graph 없이 chunk embedding만 사용합니다. 독립 vector-only baseline은 별도 구현하지 않았고 이 mode로 대체했습니다.</p></article>
        <article class="mini-card"><h3>Local</h3><p>low-level keyword로 관련 entity와 주변 relationship을 찾습니다.</p></article>
        <article class="mini-card"><h3>Global</h3><p>high-level keyword로 relation/theme 중심의 넓은 context를 찾습니다.</p></article>
        <article class="mini-card"><h3>Hybrid</h3><p>local과 global 결과를 병합해 entity/relation/chunk 근거를 구성합니다.</p></article>
      </div>
    </section>

    <section id="trace" class="section shell">
      <div class="section-head">
        <h2>6. 실제 Retrieval Trace</h2>
        <p class="section-intro">
          아래 trace는 실제 <code>query_results_15_modes.jsonl</code>의 hybrid 결과에서 keyword, retrieved context, answer 일부를 뽑은 것입니다.
        </p>
      </div>
      <div class="trace-grid">{retrieval_trace_cards(light_rows)}</div>
    </section>

    <section id="qa" class="section shell">
      <div class="section-head">
        <h2>8. RQ에 사용한 질문</h2>
        <p class="section-intro">15개 쿼리는 모델이 만든 것이 아니라 직접 설계한 고정 평가셋입니다.</p>
      </div>
      <div class="qa-grid">{question_cards(queries)}</div>
    </section>

    <section id="rq1" class="section shell">
      <div class="section-head">
        <h2>9. RQ1 baseline 비교</h2>
        <p class="section-intro">
          이 섹션은 템플릿 구조를 유지하되, 실제로 돌린 것만 채웠습니다. HyDE/RQ-RAG/별도 vector-only RAG는 미실행으로 비워두었습니다.
        </p>
      </div>
      <div class="rq-methods">{baseline_cards(judge_summary, length_summary)}</div>
    </section>

    <section id="rq2" class="section shell">
      <div class="section-head">
        <h2>10. RQ2 ablation 설계</h2>
        <p class="section-intro">
          LightRAG 내부 ablation은 naive/local/global/hybrid 네 mode로 실행했습니다. 결론은 hybrid가 naive를 압도하지는 않았고, 질문 유형별로 효용이 달랐다는 쪽입니다.
        </p>
      </div>
      <div class="panel rq2-pipeline pipeline">
        <article class="pipe-step"><b>N</b><h3>Naive</h3><p>chunk-only retrieval. graph 없이 동작.</p></article>
        <article class="pipe-step"><b>L</b><h3>Local</h3><p>entity 중심 low-level retrieval.</p></article>
        <article class="pipe-step"><b>G</b><h3>Global</h3><p>relation/theme 중심 high-level retrieval.</p></article>
        <article class="pipe-step"><b>H</b><h3>Hybrid</h3><p>local + global 병합.</p></article>
      </div>
      <div class="grid-2" style="margin-top:16px">
        <div class="panel">
          <h3>Graph indexing 효과</h3>
          <p class="section-intro">그래프 품질 지표는 목표를 통과했습니다. metadata relation ratio={html_escape(graph_metrics.get("metadata_relation_ratio", "—"))}, technical relation ratio={html_escape(graph_metrics.get("technical_relation_ratio", "—"))}, excluded entity ratio={html_escape(graph_metrics.get("excluded_entity_ratio", "—"))}.</p>
        </div>
        <div class="panel">
          <h3>Dual-level retrieval 효과</h3>
          <p class="section-intro">Judge 기준으로 hybrid가 naive를 명확히 이기지는 못했습니다. 다만 cross-category 질문에서는 hybrid graph context가 더 유효하게 작동했습니다.</p>
        </div>
      </div>
    </section>

    <section id="files" class="section shell">
      <div class="section-head">
        <h2>11. 실제 생성 파일</h2>
        <p class="section-intro">보고서에 사용한 실제 산출물 경로입니다.</p>
      </div>
      <div class="file-grid">{file_cards(experiment_dir, output)}</div>
    </section>
  </main>

  <footer class="shell">
    Patent-100 LightRAG learning-flow report. 기존 비교 리포트는 보존하고, 이 파일은 Peter Pan 템플릿 형식을 따른 별도 학습용 HTML입니다.
  </footer>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_text = strip_trailing_whitespace(build_html(experiment_dir, output))
    output.write_text(html_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "exists": output.exists()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
