import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from kcc_text_norm import normalize_text  # noqa: E402
from llama_embed import LlamaEmbedConfig, LlamaEmbedder  # noqa: E402


def token_stats(texts: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in texts:
        if not t:
            continue
        for tok in str(t).split():
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def specificity_score(text: str, token_counts: dict[str, int], n_docs: int) -> float:
    if not text:
        return 0.0
    toks = text.split()
    if not toks:
        return 0.0
    length = min(len(text), 400)
    digits = sum(ch.isdigit() for ch in text)
    rare = sum(1 for t in toks if token_counts.get(t, 0) <= max(2, int(0.0005 * n_docs)))
    score = (length / 400.0) + (digits / 10.0) + (rare / max(len(toks), 1))
    if len(toks) < 4:
        score *= 0.4
    return float(score)


def answer_score(text: str, token_counts: dict[str, int], n_docs: int) -> float:
    if not text:
        return 0.0
    length = min(len(text), 800)
    digits = sum(ch.isdigit() for ch in text)
    sentences = sum(1 for ch in text if ch in ".?!।")
    separators = sum(1 for ch in text if ch in "/:-")
    toks = text.split()
    rare = sum(1 for t in toks if token_counts.get(t, 0) <= max(2, int(0.0005 * n_docs)))
    score = (length / 800.0) + (digits / 12.0) + (sentences / 5.0) + (separators / 8.0)
    score += (rare / max(len(toks), 1))
    if length < 80:
        score *= 0.5
    return float(score)


def cluster_hdbscan(emb: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan

    if emb.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(emb)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/derived/kcc_merged_2024_2025_up.parquet")
    parser.add_argument("--out-dir", default="data/derived")
    parser.add_argument("--model", default="data/derived/models/nomic-embed-text-v2-moe.Q4_K_M.gguf")
    parser.add_argument("--llama-bin", default="tools/llama.cpp/build/bin/llama-embedding")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--answer-min-cluster-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--embed-chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = pl.scan_parquet(str(input_path)).select(["QueryText", "KccAns"])
    if args.limit:
        scan = scan.head(args.limit)
    df = scan.collect().to_pandas()

    df["q_clean"] = df["QueryText"].map(normalize_text)
    df["a_clean"] = df["KccAns"].map(normalize_text)
    df = df[(df["q_clean"] != "") & df["QueryText"].notna()]

    question_table = df[["QueryText", "q_clean"]].drop_duplicates("q_clean").reset_index(drop=True)
    question_table.insert(0, "question_id", range(len(question_table)))
    q_texts = question_table["q_clean"].tolist()

    embedder = LlamaEmbedder(
        LlamaEmbedConfig(
            llama_bin=args.llama_bin,
            model_path=args.model,
            threads=args.threads,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
        )
    )

    q_emb = embedder.embed(q_texts, chunk_size=args.embed_chunk_size)
    np.save(out_dir / "kcc_intent_embeddings_questions.npy", q_emb)

    q_labels = cluster_hdbscan(q_emb, args.min_cluster_size)
    question_table["question_cluster_id"] = q_labels

    # assign question clusters back to original rows
    q_lookup = dict(zip(question_table["q_clean"], question_table["question_cluster_id"]))
    df["question_cluster_id"] = df["q_clean"].map(q_lookup)

    # scores
    q_token_counts = token_stats(question_table["q_clean"])
    q_docs = max(len(question_table), 1)
    question_table["spec_score"] = question_table["q_clean"].map(lambda t: specificity_score(t, q_token_counts, q_docs))

    # representative by centroid similarity
    rep_scores = np.zeros(len(question_table))
    for cluster_id in sorted(set(q_labels)):
        if cluster_id < 0:
            continue
        idx = np.where(q_labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        centroid = q_emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(q_emb[idx], centroid).reshape(-1)
        rep_scores[idx] = sims
    question_table["rep_score"] = rep_scores
    question_table["desc_score"] = question_table["rep_score"] + question_table["spec_score"]

    # answer clustering per question cluster
    answers_rows = []
    summaries = []
    a_token_counts = token_stats(df["a_clean"].fillna(""))
    a_docs = max(len(df), 1)

    for cluster_id in tqdm(sorted(set(q_labels)), desc="Answer clustering"):
        if cluster_id < 0:
            continue
        sub = df[df["question_cluster_id"] == cluster_id]
        if sub.empty:
            continue
        answers = sub[["KccAns", "a_clean"]].drop_duplicates("a_clean").reset_index(drop=True)
        answers = answers[answers["a_clean"] != ""]
        if answers.empty:
            continue

        a_emb = embedder.embed(answers["a_clean"].tolist(), chunk_size=args.embed_chunk_size)
        labels = cluster_hdbscan(a_emb, args.answer_min_cluster_size)
        answers["answer_cluster_id"] = labels
        answers["desc_score"] = answers["a_clean"].map(lambda t: answer_score(t, a_token_counts, a_docs))
        answers["question_cluster_id"] = cluster_id
        answers_rows.append(answers)

        # representative and descriptive questions
        q_sub = question_table[question_table["question_cluster_id"] == cluster_id]
        rep_q = q_sub.sort_values("rep_score", ascending=False).head(1)
        desc_q = q_sub.sort_values("desc_score", ascending=False).head(1)

        # best answers overall + per answer cluster
        best_overall = answers.sort_values("desc_score", ascending=False).head(1)
        best_overall_text = best_overall["KccAns"].iloc[0]

        best_per_cluster = []
        for a_cluster in sorted(set(labels)):
            if a_cluster < 0:
                continue
            a_sub = answers[answers["answer_cluster_id"] == a_cluster]
            if a_sub.empty:
                continue
            best = a_sub.sort_values("desc_score", ascending=False).head(1)
            best_per_cluster.append(
                {
                    "answer_cluster_id": int(a_cluster),
                    "KccAns": best["KccAns"].iloc[0],
                }
            )

        summaries.append(
            {
                "question_cluster_id": int(cluster_id),
                "rep_question": rep_q["QueryText"].iloc[0],
                "desc_question": desc_q["QueryText"].iloc[0],
                "best_answer_overall": best_overall_text,
                "best_answer_per_answer_cluster": json.dumps(best_per_cluster, ensure_ascii=False),
            }
        )

    # save outputs
    question_table["cluster_size"] = question_table.groupby("question_cluster_id")["q_clean"].transform("count")
    question_table.to_parquet(out_dir / "kcc_intent_questions.parquet", index=False)

    if answers_rows:
        answers_all = pd.concat(answers_rows, ignore_index=True)
        answers_all.insert(0, "answer_id", range(len(answers_all)))
        answers_all.to_parquet(out_dir / "kcc_intent_answers.parquet", index=False)

    if summaries:
        pd.DataFrame(summaries).to_parquet(out_dir / "kcc_intent_summaries.parquet", index=False)

    meta = {
        "input": str(input_path),
        "model": args.model,
        "llama_bin": args.llama_bin,
        "min_cluster_size": args.min_cluster_size,
        "answer_min_cluster_size": args.answer_min_cluster_size,
        "limit": args.limit,
        "embed_chunk_size": args.embed_chunk_size,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
    }
    (out_dir / "kcc_intent_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
