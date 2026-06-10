from __future__ import annotations

import argparse

from pathlib import Path

from patent_lightrag.common import (
    DEFAULT_SEED,
    DEFAULT_TARGET_PER_CATEGORY,
    PATHS,
    RAW_CSV,
    ensure_dirs,
    read_csv_rows,
    stratified_sample,
    summarize_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample 200 AI-semiconductor patents for LightRAG.")
    parser.add_argument("--input", default=str(RAW_CSV))
    parser.add_argument("--output", default=str(PATHS.sample_csv))
    parser.add_argument("--manifest", default=str(PATHS.sample_manifest))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-per-category", type=int, default=DEFAULT_TARGET_PER_CATEGORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    fieldnames, rows = read_csv_rows(Path(args.input))
    sampled, manifest = stratified_sample(
        rows,
        seed=args.seed,
        target_per_category=args.target_per_category,
    )
    manifest["input_summary"] = summarize_rows(rows)
    manifest["sample_summary"] = summarize_rows(sampled)
    write_csv(Path(args.output), fieldnames, sampled)
    write_json(Path(args.manifest), manifest)
    print(f"sampled={len(sampled)} output={args.output} manifest={args.manifest}")


if __name__ == "__main__":
    main()
