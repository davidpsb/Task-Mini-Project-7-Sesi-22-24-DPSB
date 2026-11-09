"""Bagian B — quality gate: model hanya boleh dipromosikan kalau lulus batas minimal.

Jalankan sendiri: python src/quality_gate.py   (membaca reports/best_metrics.json)
Exit code 0 = lulus, 1 = gagal.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = {"roc_auc": 0.70, "recall_bad": 0.50}


def check(metrics: dict, gates: dict = GATES) -> tuple[bool, list[str]]:
    """Bandingkan setiap metrik dengan batasnya.

    Output: (lulus, alasan). alasan berisi satu pesan per metrik di bawah batas,
    misalnya "recall_bad 0.356 < 0.5". Lulus = tidak ada alasan.
    """
    alasan = []
    
    for metric_name, threshold in gates.items():
        # Ambil nilai metrik (default 0 jika metrik tidak ada di dictionary)
        value = metrics.get(metric_name, 0)
        
        if value < threshold:
            alasan.append(f"{metric_name} {value} < {threshold}")
            
    lulus = len(alasan) == 0
    return lulus, alasan


def main() -> int:
    """Baca reports/best_metrics.json, jalankan check, cetak hasil, kembalikan exit code."""
    metrics_file = ROOT / "reports" / "best_metrics.json"
    
    # Jika file tidak ada, kita tidak bisa mengecek gate
    if not metrics_file.exists():
        print(f"QUALITY GATE: GAGAL -> File {metrics_file} tidak ditemukan.")
        return 1
        
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"QUALITY GATE: GAGAL -> Gagal membaca file JSON: {e}")
        return 1

    # Jalankan pengecekan
    lulus, alasan = check(metrics)
    
    if lulus:
        print("QUALITY GATE: LULUS")
        return 0
    else:
        # Gabungkan semua alasan dengan pemisah koma
        alasan_str = ", ".join(alasan)
        print(f"QUALITY GATE: GAGAL -> {alasan_str}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())