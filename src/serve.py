"""Bagian C (Pertemuan 28) — serving model champion yang sudah di-instrument.

Setiap request dicatat ke logs/predictions.csv: waktu, latency, status ok/error, probabilitas, keputusan, dan input.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd

from data import FEATURES, ROOT, validate_record

LOG_PATH = ROOT / "logs" / "predictions.csv"
LOG_COLUMNS = ["timestamp", "request_id", "batch", "status", "error", "latency_ms", "prob_bad", "decision"] + FEATURES
MODEL_URI = "models:/credit-risk@champion"


def decide(prob_bad: float) -> str:
    """Kebijakan K05 Pasal 2: < 0,3 -> "setuju", 0,3 sampai 0,6 -> "review", > 0,6 -> "tolak"."""
    if prob_bad < 0.3:
        return "setuju"
    elif prob_bad <= 0.6:
        return "review"
    else:
        return "tolak"


class CreditScorer:
    def __init__(self, model_uri: str = MODEL_URI):
        """Muat model dan siapkan file log."""
        # 1. Set tracking URI MLflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        
        # 2. Muat model dari registry
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # 3. Buat folder logs/ dan file predictions.csv jika belum ada
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_PATH.exists():
            df = pd.DataFrame(columns=LOG_COLUMNS)
            df.to_csv(LOG_PATH, index=False)

    def predict(self, record: dict, batch: str = "live", request_id: str | None = None) -> dict:
        """Layani satu pemohon dan catat log."""
        # 1. Catat waktu mulai
        start_time = time.time()
        
        # Beri ID unik jika tidak ada request_id
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
            
        timestamp = datetime.now().isoformat()
        status = "ok"
        error_msg = ""
        prob_bad = None
        decision = None

        try:
            # 2. Validasi input
            validate_record(record)
            
            # 3. Lakukan prediksi
            # Ubah dict menjadi DataFrame 1 baris agar sesuai dengan format sklearn
            input_df = pd.DataFrame([record])[FEATURES]
            
            # predict_proba mengembalikan array [[prob_good, prob_bad]]
            probs = self.model.predict_proba(input_df)
            prob_bad = round(float(probs[0][1]), 4)
            
            # 4. Tentukan keputusan
            decision = decide(prob_bad)
            
        except ValueError as e:
            # Tangkap error validasi (misal umur kosong)
            status = "error"
            error_msg = str(e)
        except Exception as e:
            # Tangkap error tak terduga lainnya
            status = "error"
            error_msg = f"Unexpected error: {str(e)}"

        # 5. Hitung latency
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # 6. Susun baris log
        log_row = {
            "timestamp": timestamp,
            "request_id": request_id,
            "batch": batch,
            "status": status,
            "error": error_msg,
            "latency_ms": latency_ms,
            "prob_bad": prob_bad if prob_bad is not None else "",
            "decision": decision if decision is not None else ""
        }
        
        # Tambahkan fitur ke baris log (tangani jika ada key yang hilang)
        for f in FEATURES:
            log_row[f] = record.get(f, "")

        # 7. Simpan ke CSV (mode 'a' = append/tambahkan di bawah)
        df_row = pd.DataFrame([log_row])[LOG_COLUMNS]
        df_row.to_csv(LOG_PATH, mode='a', header=False, index=False)

        # 8. Return output (tanpa kolom fitur, sesuai docstring)
        return {
            "timestamp": timestamp,
            "request_id": request_id,
            "batch": batch,
            "status": status,
            "error": error_msg,
            "latency_ms": latency_ms,
            "prob_bad": prob_bad,
            "decision": decision
        }