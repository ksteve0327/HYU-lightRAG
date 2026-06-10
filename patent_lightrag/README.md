# Patent LightRAG 3.1-3.4 Workflow

This folder contains the HYU patent-data workflow for observing the LightRAG paper flow through section 3.4.

## 1. Add OpenRouter Key

Edit:

```text
LightRAG-main/.env
```

Replace:

```env
EMBEDDING_BINDING_API_KEY=<OPENROUTER_API_KEY>
```

with your OpenRouter key.

## 2. Preflight

```bash
python3 -m patent_lightrag.preflight
python3 -m patent_lightrag.preflight --check-openrouter
```

The first command checks Codex proxy and local Python modules. The second command makes one OpenRouter embedding request, so run it only after adding the API key.

## 3. Data Prep

```bash
python3 -m patent_lightrag.sample_patents
python3 -m patent_lightrag.build_patent_docs
```

Outputs:

```text
data/patents/patent_sample_200.csv
data/patents/patent_docs.jsonl
data/patents/sampling_manifest.json
data/patents/docs_manifest.json
```

## 4. Index and Query

Run these from an environment where LightRAG dependencies are installed.

```bash
python3 -m patent_lightrag.index_patents --limit 20
python3 -m patent_lightrag.query_flow
python3 -m patent_lightrag.flow_report
```

Remove `--limit 20` for the full 200-patent run.

Final report:

```text
reports/lightrag_flow_3_1_3_4.html
```
