"""Bagian C — simulasi trafik dua minggu ke model champion.

minggu_1 : X_test apa adanya (mirip data training), 300 request
minggu_2 : X_test yang digeser, 300 request:
           - age dikurangi 10, minimal 19
           - credit_amount dikali 1,8 (dibulatkan ke int)
           - duration ditambah 12, maksimal 72
           - 35% baris (acak, numpy default_rng(SEED)) purpose diganti "business"
           - 6 input rusak: baris 10 dan 110 age=None, baris 20 dan 120 credit_amount=-500,
             baris 30 duration="dua belas", baris 40 purpose=""
request_id : "1-0000" ... untuk minggu_1, "2-0000" ... untuk minggu_2
Label asli disimpan terpisah di logs/labels.csv (request_id, bad) — anggap label baru datang belakangan.

Jalankan: python src/simulate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, SEED, load_data, split  # noqa: E402
from serve import CreditScorer, LOG_PATH  # noqa: E402


def make_week2(X: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan salinan X yang sudah digeser sesuai aturan minggu_2."""
    X2 = X.copy()
    
    # 1. age dikurangi 10, minimal 19
    X2["age"] = (X2["age"] - 10).clip(lower=19).astype(int)
    
    # 2. credit_amount dikali 1,8, dibulatkan ke int
    X2["credit_amount"] = (X2["credit_amount"] * 1.8).round().astype(int)
    
    # 3. duration ditambah 12, maksimal 72
    X2["duration"] = (X2["duration"] + 12).clip(upper=72).astype(int)
    
    # 4. 35% baris (acak, numpy default_rng(SEED)) purpose diganti "business"
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(X2)) < 0.35
    X2.loc[mask, "purpose"] = "business"
    
    return X2


def main() -> None:
    """Jalankan simulasi 2 minggu."""
    print("Menyiapkan data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    
    # Hapus log lama jika ada
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        
    # Buat folder logs jika belum ada
    (ROOT / "logs").mkdir(exist_ok=True)
    
    # Inisialisasi Scorer
    print("Memuat model champion...")
    scorer = CreditScorer()
    
    labels = []
    
    # ─────────────────────────────────────────────────────────
    # MINGGU 1 (Data Normal)
    # ─────────────────────────────────────────────────────────
    print("Mengirim 300 request Minggu 1 (Normal)...")
    for i, (idx, row) in enumerate(X_test.iterrows()):
        request_id = f"1-{i:04d}"
        
        # Konversi ke dict Python murni (hindari tipe numpy)
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()}
        
        scorer.predict(record, batch="minggu_1", request_id=request_id)
        labels.append({"request_id": request_id, "bad": int(y_test.loc[idx])})

    # ─────────────────────────────────────────────────────────
    # MINGGU 2 (Data Drift & Input Rusak)
    # ─────────────────────────────────────────────────────────
    print("Mengirim 300 request Minggu 2 (Drift + 6 Rusak)...")
    X_week2 = make_week2(X_test)
    
    for i, (idx, row) in enumerate(X_week2.iterrows()):
        request_id = f"2-{i:04d}"
        
        # Konversi ke dict Python murni
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()}
        
        # Sisipkan 6 input rusak sesuai instruksi
        if i in (10, 110):
            record["age"] = None
        elif i in (20, 120):
            record["credit_amount"] = -500
        elif i == 30:
            record["duration"] = "dua belas"
        elif i == 40:
            record["purpose"] = ""
            
        scorer.predict(record, batch="minggu_2", request_id=request_id)
        labels.append({"request_id": request_id, "bad": int(y_test.loc[idx])})

    # ─────────────────────────────────────────────────────────
    # Simpan Label
    # ─────────────────────────────────────────────────────────
    labels_file = ROOT / "logs" / "labels.csv"
    pd.DataFrame(labels).to_csv(labels_file, index=False)
    print(f"\n✅ Label asli disimpan di {labels_file}")
    
    # Cetak Ringkasan
    df_log = pd.read_csv(LOG_PATH)
    print("\n=== RINGKASAN SIMULASI ===")
    summary = df_log.groupby(["batch", "status"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print("==========================")


if __name__ == "__main__":
    main()