"""Data German Credit (credit risk) — dipakai Bagian B dan C.

Sumber: https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk
Simpan file aslinya sebagai data/credit/german_credit_data.csv
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit" / "german_credit_data.csv"
SEED = 42

NUMERIC = ["age", "credit_amount", "duration"]
CATEGORICAL = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "bad"  # 1 = risiko buruk (gagal bayar), 0 = baik


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Baca CSV dan rapikan."""
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan di {path}.\n"
            "Silakan unduh dataset dari Kaggle: "
            "https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk "
            "dan simpan sebagai data/credit/german_credit_data.csv"
        )

    # 1. Baca CSV (kolom pertama adalah index)
    df = pd.read_csv(path, index_col=0)

    # 2. Nama kolom: huruf kecil, spasi -> underscore
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 3. job diubah ke str (karena 0-3 adalah kode kategori, bukan angka)
    df["job"] = df["job"].astype(str)

    # 4. Isi NA di saving_accounts dan checking_account dengan "unknown"
    df["saving_accounts"] = df["saving_accounts"].fillna("unknown")
    df["checking_account"] = df["checking_account"].fillna("unknown")

    # 5. Buat kolom target 'bad'
    # (1 jika risk == "bad", 0 jika sebaliknya)
    df[TARGET] = (df["risk"] == "bad").astype(int)

    # 6. Return hanya fitur yang dipakai + target
    return df[FEATURES + [TARGET]]


def split(df: pd.DataFrame):
    """Bagi data: 70% train, 30% test, stratified pada TARGET, random_state=SEED."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def data_version(path: Path = DATA_PATH) -> str:
    """Sidik jari isi file data: SHA-256 dari bytes file, ambil 12 karakter pertama."""
    content = path.read_bytes()
    full_hash = hashlib.sha256(content).hexdigest()
    return full_hash[:12]


def validate_record(record: dict) -> None:
    """Dipakai saat serving. Lempar ValueError kalau ada input tidak valid."""
    # 1. Cek field kosong (None atau "")
    missing = [f for f in FEATURES if record.get(f) is None or record.get(f) == ""]
    if missing:
        raise ValueError(f"field kosong: {missing}")

    # 2. Cek field numerik harus int/float dan > 0
    for f in NUMERIC:
        val = record[f]
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"{f} harus angka positif, dapat {val}")