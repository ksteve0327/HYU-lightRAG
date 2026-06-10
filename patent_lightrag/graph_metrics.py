from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from patent_lightrag.common import write_json


EXCLUDED_ENTITY_PATTERNS = [
    re.compile(r"^\d{2}-\d{6}$"),
    re.compile(r"^\d{4}-\d{7}$"),
    re.compile(r"^\d{4}\.\d{2}\.\d{2}$"),
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d{7,}$"),
    re.compile(r"^[A-Z]\d{2}[A-Z]-\d{3,4}/\d+(?:\.\d+)?$"),
    re.compile(r"^[A-Z]\d{2}[A-Z]-\d{4}/\d+(?:\.\d+)?$"),
    re.compile(r"^[A-Z]{2,3}$"),
    re.compile(r"^A[ABCD][A-Z]?$"),
]
EXCLUDED_ENTITY_VALUES = {
    "US",
    "KR",
    "JP",
    "CN",
    "EP",
    "WO",
    "등록",
    "공개",
    "거절",
    "소멸",
    "AI 반도체 특허",
}
METADATA_KEYWORDS = {
    "patent_id",
    "출원번호",
    "공개번호",
    "등록번호",
    "특허 식별",
    "특허번호",
    "ipc",
    "cpc",
    "분류",
    "중분류",
    "소분류",
    "국가",
    "국적",
    "법적",
    "등록일",
    "공개일",
    "출원일",
    "출원인",
    "권리자",
    "application",
    "publication",
    "registration",
    "classification",
}
TECHNICAL_KEYWORDS = {
    "구성",
    "구성요소",
    "포함",
    "연결",
    "저장",
    "판독",
    "제어",
    "생성",
    "수행",
    "최적화",
    "가속",
    "융합",
    "컴파일",
    "데이터 전송",
    "아키텍처",
    "회로",
    "메모리",
    "연산",
    "처리",
    "read",
    "write",
    "control",
    "compute",
    "operation",
    "architecture",
}
TECHNICAL_TYPES = {
    "techcomponent",
    "architecture",
    "operation",
    "method",
    "material",
    "performancemetric",
    "applicationdomain",
    "artifact",
    "concept",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LightRAG graph quality metrics.")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def graphml_path(working_dir: Path) -> Path:
    return working_dir / "graph_chunk_entity_relation.graphml"


def load_graphml(working_dir: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    path = graphml_path(working_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing graphml file: {path}")

    key_names: dict[str, str] = {}
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []

    for event, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "key":
            key_id = elem.attrib.get("id", "")
            key_name = elem.attrib.get("attr.name", "")
            if key_id and key_name:
                key_names[key_id] = key_name
        elif tag == "node":
            node_id = elem.attrib.get("id", "")
            payload = {"id": node_id}
            for data in elem:
                data_tag = data.tag.rsplit("}", 1)[-1]
                if data_tag == "data":
                    payload[key_names.get(data.attrib.get("key", ""), "")] = data.text or ""
            if node_id:
                nodes[node_id] = payload
            elem.clear()
        elif tag == "edge":
            payload = {
                "source": elem.attrib.get("source", ""),
                "target": elem.attrib.get("target", ""),
            }
            for data in elem:
                data_tag = data.tag.rsplit("}", 1)[-1]
                if data_tag == "data":
                    payload[key_names.get(data.attrib.get("key", ""), "")] = data.text or ""
            if payload["source"] and payload["target"]:
                edges.append(payload)
            elem.clear()
    return nodes, edges


def is_excluded_entity(name: str) -> bool:
    normalized = name.strip()
    if normalized in EXCLUDED_ENTITY_VALUES:
        return True
    if "특허 " in normalized and re.search(r"\d", normalized):
        return True
    if "청구항" in normalized and re.search(r"\d", normalized):
        return True
    return any(pattern.match(normalized) for pattern in EXCLUDED_ENTITY_PATTERNS)


def contains_any(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def classify_relation(edge: dict[str, str], nodes: dict[str, dict[str, str]]) -> str:
    source = edge.get("source", "")
    target = edge.get("target", "")
    combined = "\n".join(
        [
            source,
            target,
            edge.get("keywords", ""),
            edge.get("description", ""),
        ]
    )
    if is_excluded_entity(source) or is_excluded_entity(target):
        return "metadata"
    if contains_any(combined, METADATA_KEYWORDS):
        return "metadata"

    source_type = str(nodes.get(source, {}).get("entity_type", "")).lower()
    target_type = str(nodes.get(target, {}).get("entity_type", "")).lower()
    if contains_any(combined, TECHNICAL_KEYWORDS):
        return "technical"
    if source_type in TECHNICAL_TYPES and target_type in TECHNICAL_TYPES:
        return "technical"
    return "other"


def shannon_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def main() -> None:
    args = parse_args()
    working_dir = Path(args.working_dir)
    nodes, edges = load_graphml(working_dir)
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    type_counts = Counter(
        str(node.get("entity_type", "unknown") or "unknown").lower()
        for node in nodes.values()
    )
    excluded_nodes = [name for name in nodes if is_excluded_entity(name)]
    relation_counts = Counter(classify_relation(edge, nodes) for edge in edges)
    total_edges = len(edges)
    total_nodes = len(nodes)
    hub_nodes = [
        {
            "entity": name,
            "degree": count,
            "entity_type": nodes.get(name, {}).get("entity_type", ""),
            "excluded_entity": is_excluded_entity(name),
        }
        for name, count in degree.most_common(20)
    ]
    metadata_hubs = [
        row
        for row in hub_nodes
        if row["excluded_entity"] or contains_any(row["entity"], METADATA_KEYWORDS)
    ]

    metrics = {
        "label": args.label,
        "working_dir": str(working_dir.resolve()),
        "graph_nodes": total_nodes,
        "graph_edges": total_edges,
        "relation_counts": dict(relation_counts),
        "metadata_relation_ratio": round(relation_counts.get("metadata", 0) / total_edges, 4)
        if total_edges
        else 0,
        "technical_relation_ratio": round(relation_counts.get("technical", 0) / total_edges, 4)
        if total_edges
        else 0,
        "other_relation_ratio": round(relation_counts.get("other", 0) / total_edges, 4)
        if total_edges
        else 0,
        "excluded_entity_count": len(excluded_nodes),
        "excluded_entity_ratio": round(len(excluded_nodes) / total_nodes, 4)
        if total_nodes
        else 0,
        "entity_type_counts": dict(type_counts),
        "entity_type_entropy": shannon_entropy(type_counts),
        "degree_ge_50_hub_count": sum(1 for count in degree.values() if count >= 50),
        "top_hubs": hub_nodes,
        "metadata_hubs_in_top20": metadata_hubs,
        "targets": {
            "metadata_relation_ratio": "<= 0.25",
            "technical_relation_ratio": ">= 0.40",
            "excluded_entity_ratio": "<= 0.10",
        },
    }
    write_json(Path(args.output), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
