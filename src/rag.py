"""Bagian A (Pertemuan 26) — RAG untuk dokumen kebijakan kredit.

Alur: dokumen -> chunk per pasal -> dense + BM25 -> hybrid (RRF) -> rerank -> jawaban + sitasi / fallback.
"""
from __future__ import annotations

import functools
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "rag" / "docs"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FALLBACK_THRESHOLD = 0.35   # ambang top_score dense; sesuaikan dengan datamu dan jelaskan di README
RRF_K = 60
CANDIDATES = 8              # jumlah kandidat hybrid yang dikirim ke reranker
FALLBACK_TEXT = "Maaf, informasi tersebut tidak ada di dokumen kebijakan kredit."


# ---------------------------------------------------------------- 1. chunking
def load_chunks(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Potong setiap dokumen .md per baris 'Pasal n: ...' (document-aware chunking)."""
    chunks = []
    for md_file in sorted(Path(docs_dir).glob("K*.md")):
        doc_id = md_file.stem
        content = md_file.read_text(encoding="utf-8")

        first_line = content.splitlines()[0] if content else ""
        title = first_line.lstrip("# ").strip()

        for match in re.finditer(r"Pasal\s+(\d+):\s*(.+)", content):
            pasal = int(match.group(1))
            text = match.group(2).strip()
            
            chunks.append({
                "chunk_id": f"{doc_id}-P{pasal}",
                "doc_id":   doc_id,
                "pasal":    pasal,
                "title":    title,
                "text":     text,
            })
    return chunks


def tokenize(text: str) -> list[str]:
    """Untuk BM25: huruf kecil, buang tanda baca, pecah per kata."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ---------------------------------------------------------------- index (dibuat sekali)
@functools.lru_cache(maxsize=1)
def _index():
    """Bangun semua komponen pencarian SEKALI saja."""
    chunks = load_chunks()
    texts = [f"{c['title']}. {c['text']}" for c in chunks]
    
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(texts, normalize_embeddings=True)
    
    tokenized_texts = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    reranker = CrossEncoder(RERANK_MODEL)
    
    return chunks, texts, embedder, vectors, bm25, reranker


# ---------------------------------------------------------------- 2-3. dense dan BM25
def dense_search(question: str) -> tuple[list[int], list[float]]:
    """Index chunk urut dari paling mirip + skor cosine-nya (urutan sama)."""
    chunks, texts, embedder, vectors, bm25, reranker = _index()

    q_vec = embedder.encode(question, normalize_embeddings=True)
    scores = vectors @ q_vec

    indices = np.argsort(scores)[::-1].tolist()
    scores_sorted = scores[indices].tolist()

    return indices, scores_sorted


def bm25_search(question: str) -> list[int]:
    """Index chunk urut dari skor BM25 tertinggi."""
    chunks, texts, embedder, vectors, bm25, reranker = _index()

    q_tokens = tokenize(question)
    scores = bm25.get_scores(q_tokens)

    indices = np.argsort(scores)[::-1].tolist()

    return indices


# ---------------------------------------------------------------- 4. hybrid (RRF)
def rrf(rankings: list[list], k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion."""
    rrf_scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if item not in rrf_scores:
                rrf_scores[item] = 0.0
            rrf_scores[item] += 1.0 / (k + rank)
            
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)


# ---------------------------------------------------------------- 5. rerank (FUNGSI YANG HILANG)
def rerank(question: str, candidates: list[int]) -> list[int]:
    """Urutkan ulang kandidat dengan cross-encoder."""
    chunks, texts, embedder, vectors, bm25, reranker = _index()

    pairs = [[question, texts[idx]] for idx in candidates]
    scores = reranker.predict(pairs)

    sorted_pairs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    
    return [idx for score, idx in sorted_pairs]


# ---------------------------------------------------------------- 6. jawaban + sitasi + fallback
def answer(question: str, mode: str = "hybrid_rerank", top_k: int = 3) -> dict:
    """Jawab satu pertanyaan."""
    t0 = time.time()

    # 1. Pertanyaan kosong langsung fallback TANPA memuat model
    if not question.strip():
        return {
            "answer": FALLBACK_TEXT,
            "sources": [],
            "top_score": 0.0,
            "fallback": True,
            "latency_ms": (time.time() - t0) * 1000
        }

    # 2. Load index
    chunks, texts, embedder, vectors, bm25, reranker = _index()

    # 3. Dense search (untuk top_score fallback)
    dense_indices, dense_scores = dense_search(question)
    top_score = float(dense_scores[0]) if dense_scores else 0.0

    # 4. Tentukan urutan akhir
    if mode == "dense":
        final_indices = dense_indices
    elif mode == "bm25":
        final_indices = bm25_search(question)
    elif mode == "hybrid":
        bm25_indices = bm25_search(question)
        final_indices = rrf([dense_indices, bm25_indices])
    elif mode == "hybrid_rerank":
        bm25_indices = bm25_search(question)
        hybrid_indices = rrf([dense_indices, bm25_indices])
        kandidat = hybrid_indices[:CANDIDATES]
        final_indices = rerank(question, kandidat)
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")

    # 5. Cek ambang batas fallback
    if top_score < FALLBACK_THRESHOLD:
        return {
            "answer": FALLBACK_TEXT,
            "sources": [],
            "top_score": top_score,
            "fallback": True,
            "latency_ms": (time.time() - t0) * 1000
        }

    # 6. Ambil top_k (3 teratas)
    top_indices = final_indices[:top_k]
    best_idx = top_indices[0]
    best_chunk = chunks[best_idx]

    # 7. Susun jawaban + sitasi
    ans_text = f"{best_chunk['text']} [Sumber: {best_chunk['doc_id']}, Pasal {best_chunk['pasal']}]"

    # 8. Ambil doc_id untuk sources
    sources = [chunks[idx]["doc_id"] for idx in top_indices]

    return {
        "answer": ans_text,
        "sources": sources,
        "top_score": top_score,
        "fallback": False,
        "latency_ms": (time.time() - t0) * 1000
    }


if __name__ == "__main__":
    for q in ["Berapa denda kalau telat bayar?", "Siapa presiden pertama Indonesia?", ""]:
        print(q or "<kosong>", "->", answer(q))