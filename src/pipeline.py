"""Bagian B (Pertemuan 27) — pipeline commit-to-deploy versi lokal.

Urutan: test -> train 3 kandidat (MLflow) -> quality gate -> pilih terbaik yang lulus -> register -> promote "champion".
Jalankan: python src/pipeline.py --reason "commit baru"     (tambah --skip-tests untuk melewati pytest)
Lihat hasil: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, data_version, load_data, split  # noqa: E402
from quality_gate import check  # noqa: E402
from train import CANDIDATES, build_model, evaluate  # noqa: E402

EXPERIMENT = "credit-risk"
MODEL_NAME = "credit-risk"
ALIAS = "champion"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
# MLflow 3 menyimpan model sklearn dengan format skops. Tipe di bawah dibuat sendiri, jadi aman dipercaya;
# kirim sebagai skops_trusted_types=TRUSTED_TYPES saat mlflow.sklearn.log_model.
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def setup_mlflow() -> None:
    """Set tracking URI ke TRACKING_URI. Kalau experiment EXPERIMENT belum ada, buat dengan
    artifact_location = (ROOT / "mlruns").as_uri(). Lalu set experiment aktif."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    
    # Cek apakah experiment sudah ada
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        artifact_location = (ROOT / "mlruns").as_uri()
        client.create_experiment(EXPERIMENT, artifact_location=artifact_location)
        
    mlflow.set_experiment(EXPERIMENT)


def step_test() -> None:
    """[1/5] Jalankan `python -m pytest -q tests` (subprocess). Kalau gagal, hentikan pipeline (sys.exit)."""
    print("\n[1/5] Menjalankan test...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Test gagal:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("✅ Test lulus.")


def step_train(reason: str) -> list[dict]:
    """[2/5] Untuk setiap kandidat di train.CANDIDATES, dalam satu MLflow run (run_name = nama kandidat)"""
    print("\n[2/5] Melatih kandidat model...")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    d_version = data_version()
    
    results = []
    
    for name, (class_name, params) in CANDIDATES.items():
        print(f"  Melatih {name}...")
        with mlflow.start_run(run_name=name) as run:
            # 1. Latih
            model = build_model(name)
            model.fit(X_train, y_train)
            
            # 2. Evaluasi
            metrics = evaluate(model, X_test, y_test)
            
            # 3. Log params
            mlflow.log_params({
                "model": name,
                "data_version": d_version,
                "n_train": len(X_train),
            })
            # Hyperparameter dengan prefix clf__
            mlflow.log_params({f"clf__{k}": v for k, v in params.items()})
            
            # 4. Log metrics
            mlflow.log_metrics(metrics)
            
            # 5. Tag
            mlflow.set_tag("reason", reason)
            
            # 6. Log model
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train.head(5)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=TRUSTED_TYPES
            )
            
            results.append({
                "run_id": run.info.run_id,
                "model": name,
                "data_version": d_version,
                **metrics
            })
            
    return results


def step_gate(results: list[dict]) -> list[dict]:
    """[3/5] Jalankan quality_gate.check untuk setiap kandidat."""
    print("\n[3/5] Mengevaluasi Quality Gate...")
    client = MlflowClient()
    passed = []
    
    for res in results:
        lulus, alasan = check(res)
        status = "lulus" if lulus else "gagal"
        
        # Set tag di MLflow
        client.set_tag(res["run_id"], "quality_gate", status)
        
        # Simpan status ke dict
        res["quality_gate"] = status
        res["alasan_gagal"] = alasan
        
        if lulus:
            passed.append(res)
            print(f"  ✅ {res['model']} LULUS.")
        else:
            print(f"  ❌ {res['model']} GAGAL: {', '.join(alasan)}")
            
    # Simpan semua hasil ke reports/candidates.json
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "candidates.json", "w") as f:
        json.dump(results, f, indent=2)
        
    if not passed:
        print("❌ Tidak ada model yang lulus Quality Gate. Pipeline dihentikan.")
        sys.exit(1)
        
    return passed


def step_select(passed: list[dict]) -> dict:
    """[4/5] Pilih kandidat dengan roc_auc tertinggi dari yang lulus. Simpan ke reports/best_metrics.json."""
    print("\n[4/5] Memilih model terbaik...")
    # Urutkan berdasarkan roc_auc descending
    best = max(passed, key=lambda x: x["roc_auc"])
    print(f"  🏆 Model terbaik: {best['model']} (AUC: {best['roc_auc']})")
    
    # Simpan ke reports/best_metrics.json
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "best_metrics.json", "w") as f:
        json.dump(best, f, indent=2)
        
    return best


def step_register_promote(best: dict) -> None:
    """[5/5] Daftarkan model run terpilih ke registry MODEL_NAME (versi baru)."""
    print("\n[5/5] Mendaftarkan model ke Registry & Promosi...")
    client = MlflowClient()
    
    # Daftarkan model
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"  Model terdaftar dengan versi: {mv.version}")
    
    # Cek apakah alias champion sudah ada
    try:
        current_champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except mlflow.exceptions.MlflowException:
        current_champion = None
        
    if current_champion is None:
        print("  Belum ada champion. Mempromosikan model ini sebagai champion baru.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
        return
        
    # Jika sudah ada champion, bandingkan AUC
    old_run = client.get_run(current_champion.run_id)
    old_auc = old_run.data.metrics.get("roc_auc", 0.0)
    new_auc = best["roc_auc"]
    
    print(f"  AUC Champion lama (v{current_champion.version}): {old_auc}")
    print(f"  AUC Model baru (v{mv.version}): {new_auc}")
    
    if new_auc > old_auc:
        print(f"  ✅ Model baru lebih baik. Mempromosikan v{mv.version} sebagai champion baru.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    else:
        print(f"  ⏸️  Model baru tidak lebih baik. Tetap menggunakan champion lama (v{current_champion.version}).")


def main() -> None:
    """Parse argumen --reason (default "manual") dan --skip-tests, lalu jalankan langkah 1-5 berurutan."""
    parser = argparse.ArgumentParser(description="Pipeline MLOps Credit Risk")
    parser.add_argument("--reason", type=str, default="manual", help="Alasan commit / run ini")
    parser.add_argument("--skip-tests", action="store_true", help="Lewati step pytest")
    args = parser.parse_args()
    
    setup_mlflow()
    
    if not args.skip_tests:
        step_test()
    else:
        print("\n[1/5] Melewati test (--skip-tests).")
        
    results = step_train(args.reason)
    passed = step_gate(results)
    best = step_select(passed)
    step_register_promote(best)
    
    print("\n🎉 Pipeline selesai!")


if __name__ == "__main__":
    main()