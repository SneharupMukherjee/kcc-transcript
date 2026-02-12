# Intent + Answer Clustering (Local)

## Purpose
Batch-generate intent clusters and answer modes for KCC transcripts so Streamlit can show representative and descriptive Q/A without expensive runtime computation.

## Inputs
Default input: `data/derived/kcc_merged_2024_2025_up.parquet`

## Model + Backend
- llama.cpp binary: `tools/llama.cpp/build/bin/llama-embedding`
- GGUF embedding model: `data/derived/models/nomic-embed-text-v2-moe.Q4_K_M.gguf`

## Run
```bash
/home/sneharup/KCC/apt/.venv/bin/python scripts/build_intent_clusters.py
```

Optional flags:
```bash
/home/sneharup/KCC/apt/.venv/bin/python scripts/build_intent_clusters.py \
  --limit 20000 \
  --min-cluster-size 25 \
  --answer-min-cluster-size 10
```

## Outputs (data/derived)
- `kcc_intent_questions.parquet`
- `kcc_intent_answers.parquet`
- `kcc_intent_summaries.parquet`
- `kcc_intent_embeddings_questions.npy`
- `kcc_intent_metadata.json`

## Notes
- Output is deterministic for the same data and embedding model.
- Update thresholds if clusters are too coarse or too fragmented.
