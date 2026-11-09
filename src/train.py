"""Bagian B — definisi model dan evaluasinya (dipanggil oleh pipeline.py).

Library: sklearn (ColumnTransformer, Pipeline, StandardScaler, SimpleImputer, OneHotEncoder,
LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, metrics).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, SEED  # noqa: F401

# Tiga kandidat yang dibandingkan di MLflow.
CANDIDATES = {
    "logreg": ("LogisticRegression", {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}),
    "random_forest": ("RandomForestClassifier", {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5,
                                                 "class_weight": "balanced", "random_state": SEED}),
    "gradient_boosting": ("GradientBoostingClassifier", {"n_estimators": 150, "max_depth": 2,
                                                         "learning_rate": 0.05, "random_state": SEED}),
}


def build_model(name: str) -> Pipeline:
    """Kembalikan sklearn Pipeline untuk kandidat `name`."""
    if name not in CANDIDATES:
        raise ValueError(f"Model tidak dikenal: {name}")

    # 1. Preprocessing
    # - Numerik: StandardScaler
    # - Kategori: SimpleImputer (isi "unknown") + OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), CATEGORICAL),
        ]
    )

    # 2. Ambil nama kelas dan parameter dari CANDIDATES
    class_name, params = CANDIDATES[name]

    # 3. Inisialisasi model berdasarkan namanya
    if class_name == "LogisticRegression":
        clf = LogisticRegression(**params)
    elif class_name == "RandomForestClassifier":
        clf = RandomForestClassifier(**params)
    elif class_name == "GradientBoostingClassifier":
        clf = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Kelas model tidak dikenali: {class_name}")

    # 4. Gabungkan preprocessing + model
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf)
    ])


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Metrik untuk kelas bad (1). prob_bad >= threshold dianggap prediksi bad."""
    # 1. Dapatkan probabilitas prediksi untuk kelas 'bad' (kolom ke-1)
    prob_bad = model.predict_proba(X)[:, 1]

    # 2. Ubah probabilitas menjadi prediksi 0/1 berdasarkan threshold
    y_pred = (prob_bad >= threshold).astype(int)

    # 3. Hitung metrik
    return {
        "roc_auc": round(roc_auc_score(y, prob_bad), 4),
        "recall_bad": round(recall_score(y, y_pred, zero_division=0), 4),
        "precision_bad": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1_bad": round(f1_score(y, y_pred, zero_division=0), 4)
    }