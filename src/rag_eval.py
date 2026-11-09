"""Bagian A — evaluasi RAG pada data/rag/eval.csv untuk keempat mode.

Jalankan: python src/rag_eval.py
Output : reports/rag_eval_detail.csv  (satu baris per mode x pertanyaan)
         reports/rag_eval_summary.csv (satu baris per mode)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import ROOT, answer  # noqa: E402

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def hit_at_k(expected: str, sources: list[str], k: int = 3) -> bool:
    """True kalau dokumen yang benar ada di k sumber teratas."""
    if expected == "NONE":
        return False
    return expected in sources[:k]


def summarize(detail: pd.DataFrame) -> dict:
    """Ringkas hasil satu mode."""
    # Pisahkan pertanyaan dalam topik dan luar topik
    in_topic = detail[detail["expected_doc"] != "NONE"]
    out_topic = detail[detail["expected_doc"] == "NONE"]

    # 1. Hit@3 (rata-rata hit3 untuk pertanyaan dalam topik)
    hit_at_3 = in_topic["hit3"].mean() if not in_topic.empty else 0.0

    # 2. Citation Accuracy (top1 tepat DAN tidak fallback)
    citation_correct = (
        (in_topic["top1"] == in_topic["expected_doc"]) & (~in_topic["fallback"])
    ).sum()
    citation_acc = citation_correct / len(in_topic) if not in_topic.empty else 0.0

    # 3. Fallback Accuracy (proporsi luar topik yang fallback)
    fallback_correct = out_topic["fallback"].sum()
    fallback_acc = fallback_correct / len(out_topic) if not out_topic.empty else 0.0

    # 4. Rata-rata Latency
    avg_latency_ms = detail["latency_ms"].mean()

    return {
        "hit_at_3": round(hit_at_3, 4),
        "citation_acc": round(citation_acc, 4),
        "fallback_acc": round(fallback_acc, 4),
        "avg_latency_ms": round(avg_latency_ms, 2)
    }


def main() -> None:
    """Langkah eksekusi evaluasi."""
    # 1. Baca eval.csv
    eval_path = ROOT / "data" / "rag" / "eval.csv"
    if not eval_path.exists():
        print(f"Error: File tidak ditemukan di {eval_path}")
        return
        
    df_eval = pd.read_csv(eval_path)
    
    # Siapkan folder reports
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 2. Pemanasan (warm-up) agar waktu muat model tidak masuk ke latency query pertama
    print("Melakukan pemanasan model (warm-up)...")
    answer("Berapa denda kalau telat bayar?", mode="hybrid_rerank")

    detail_records = []

    # 3. Evaluasi setiap mode dan pertanyaan
    for mode in MODES:
        print(f"Memproses mode: {mode}...")
        for _, row in df_eval.iterrows():
            q = row["question"]
            expected = row["expected_doc"]
            
            # Panggil answer()
            res = answer(q, mode=mode, top_k=3)
            
            sources = res["sources"]
            # top1 adalah sumber pertama (jika ada)
            top1 = sources[0] if sources else ""
            
            is_hit = hit_at_k(expected, sources, k=3)
            
            detail_records.append({
                "mode": mode,
                "question": q,
                "expected_doc": expected,
                "top1": top1,
                "sources": sources,
                "hit3": is_hit,
                "top_score": round(res["top_score"], 4),
                "fallback": res["fallback"],
                "latency_ms": round(res["latency_ms"], 2)
            })
            
    # 4. Simpan detail ke CSV
    df_detail = pd.DataFrame(detail_records)
    detail_file = reports_dir / "rag_eval_detail.csv"
    df_detail.to_csv(detail_file, index=False)
    print(f"\n✅ Detail evaluasi disimpan ke: {detail_file}")

    # 5. Hitung summary untuk setiap mode dan simpan
    summary_records = []
    for mode in MODES:
        df_mode = df_detail[df_detail["mode"] == mode]
        summ = summarize(df_mode)
        summ["mode"] = mode
        summary_records.append(summ)

    df_summary = pd.DataFrame(summary_records)
    # Susun ulang urutan kolom agar rapi
    df_summary = df_summary[["mode", "hit_at_3", "citation_acc", "fallback_acc", "avg_latency_ms"]]
    
    summary_file = reports_dir / "rag_eval_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"✅ Summary evaluasi disimpan ke: {summary_file}\n")

    # Cetak tabel summary ke terminal/notebook
    print("=" * 70)
    print("HASIL EVALUASI RAG")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()