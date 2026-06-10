from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from patent_lightrag.common import DEFAULT_SEED, PATHS, ROOT, estimate_tokens, write_json


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "patent_prompt_pilot_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced pilot JSONL from the 200 patent docs.")
    parser.add_argument("--docs", default=str(PATHS.docs_jsonl))
    parser.add_argument("--output", default=str(DEFAULT_EXPERIMENT_DIR / "patent_docs_pilot_20.jsonl"))
    parser.add_argument("--manifest", default=str(DEFAULT_EXPERIMENT_DIR / "pilot_docs_manifest.json"))
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_docs(path: Path) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def main() -> None:
    args = parse_args()
    docs = load_docs(Path(args.docs))
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for doc in docs:
        category = str(doc.get("category", "")).strip()
        if category:
            by_category[category].append(doc)

    rng = random.Random(args.seed)
    selected: list[dict[str, object]] = []
    selected_by_category: dict[str, list[str]] = {}
    for category in sorted(by_category):
        candidates = sorted(by_category[category], key=lambda item: str(item.get("id", "")))
        if len(candidates) < args.per_category:
            raise ValueError(
                f"Category {category} has only {len(candidates)} docs, "
                f"but {args.per_category} are required."
            )
        category_selected = sorted(
            rng.sample(candidates, args.per_category),
            key=lambda item: str(item.get("id", "")),
        )
        selected.extend(category_selected)
        selected_by_category[category] = [str(item.get("id", "")) for item in category_selected]

    selected = sorted(
        selected,
        key=lambda item: (
            str(item.get("category", "")),
            str(item.get("sub_category", "")),
            str(item.get("id", "")),
        ),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for doc in selected:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    token_estimates = [estimate_tokens(str(doc.get("text", ""))) for doc in selected]
    manifest = {
        "input_docs": str(Path(args.docs).resolve()),
        "output_docs": str(output.resolve()),
        "seed": args.seed,
        "per_category": args.per_category,
        "document_count": len(selected),
        "category_counts": dict(Counter(str(doc.get("category", "")) for doc in selected)),
        "sub_category_counts": dict(
            Counter(
                f"{doc.get('category', '')}/{doc.get('sub_category', '')}"
                for doc in selected
            )
        ),
        "selected_ids_by_category": selected_by_category,
        "token_estimate": {
            "total": sum(token_estimates),
            "min": min(token_estimates) if token_estimates else 0,
            "max": max(token_estimates) if token_estimates else 0,
            "avg": round(sum(token_estimates) / len(token_estimates), 2)
            if token_estimates
            else 0,
        },
        "sample_document": selected[0] if selected else {},
    }
    write_json(Path(args.manifest), manifest)
    print(json.dumps({"output": str(output), "document_count": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
