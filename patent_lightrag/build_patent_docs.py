from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_lightrag.common import (
    PATHS,
    build_doc_record,
    ensure_dirs,
    estimate_tokens,
    read_csv_rows,
    summarize_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured LightRAG documents from sampled patents.")
    parser.add_argument("--input", default=str(PATHS.sample_csv))
    parser.add_argument("--output", default=str(PATHS.docs_jsonl))
    parser.add_argument("--manifest", default=str(PATHS.docs_manifest))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    _, rows = read_csv_rows(Path(args.input))
    docs = [build_doc_record(row) for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    token_estimates = [estimate_tokens(str(doc["text"])) for doc in docs]
    manifest = {
        "document_count": len(docs),
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "summary": summarize_rows(rows),
        "token_estimate": {
            "total": sum(token_estimates),
            "min": min(token_estimates) if token_estimates else 0,
            "max": max(token_estimates) if token_estimates else 0,
            "avg": round(sum(token_estimates) / len(token_estimates), 2) if token_estimates else 0,
        },
        "sample_document": docs[0] if docs else None,
    }
    write_json(Path(args.manifest), manifest)
    print(f"docs={len(docs)} output={args.output} manifest={args.manifest}")


if __name__ == "__main__":
    main()
