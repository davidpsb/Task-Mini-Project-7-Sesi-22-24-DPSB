"""Bagian C (Pertemuan 28) — metrik operasional, kualitas, drift (KS, PSI, Jensen-Shannon), alert, dashboard.

Referensi = data training (X_train). Dibandingkan dengan setiap batch di logs/predictions.csv.
Jalankan: python src/monitor.py
Output : reports/monitoring_report.md, reports/monitoring_dashboard.png, reports/drift_metrics.csv,
         reports/retrain_trigger.json

Library: numpy, pandas, scipy.stats.ks_2samp, scipy.spatial.distance.jensenshannon,
sklearn.metrics.roc_auc_score, matplotlib.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CATEGORICAL, NUMERIC, ROOT, load_data, split  # noqa: E402,F401

# aturan alert
PSI_LIMIT, KS_P_LIMIT, JS_LIMIT = 0.2, 0.05, 0.1
ERROR_RATE_LIMIT, P95_LATENCY_MS_LIMIT, AUC_LIMIT = 0.01, 200, 0.70
EPS = 1e-4


# ---------------------------------------------------------------- drift
def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index."""
    # Paksa menjadi numerik, teks/string akan menjadi NaN dan dibuang
    ref = pd.to_numeric(pd.Series(reference), errors='coerce').dropna().values
    cur = pd.to_numeric(pd.Series(current), errors='coerce').dropna().values
    
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
        
    # 1. Batas bin = kuantil referensi
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(ref, quantiles)
    
    # Buang batas ganda, ujung kiri -inf, ujung kanan +inf
    breakpoints = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # 2. Hitung proporsi
    p_counts, _ = np.histogram(ref, bins=breakpoints)
    q_counts, _ = np.histogram(cur, bins=breakpoints)
    
    p = p_counts / len(ref) + EPS
    q = q_counts / len(cur) + EPS
    
    # 3. PSI = sum((q-p) * ln(q/p))
    return float(np.sum((q - p) * np.log(q / p)))












def js_distance(reference, current) -> float:
    """Jensen-Shannon distance antar distribusi kategori."""
    # Paksa semua data menjadi string agar tidak error saat di-sort (mixed types)
    ref = pd.Series(reference).astype(str)
    cur = pd.Series(current).astype(str)
    
    # Gabungkan semua kategori dari kedua data
    categories = pd.Index(np.unique(np.concatenate([ref.unique(), cur.unique()])))
    
    # Hitung proporsi dengan urutan kategori yang sama
    p = ref.value_counts(normalize=True).reindex(categories, fill_value=0).values
    q = cur.value_counts(normalize=True).reindex(categories, fill_value=0).values
    
    return float(jensenshannon(p, q, base=2))






def drift_table(reference, current, batch: str) -> pd.DataFrame:
    """Satu baris per fitur. Kolom: batch, feature, type, psi, ks_stat, ks_p, js, drift."""
    rows = []
    
    # Fitur Numerik
    for col in NUMERIC:
        # Paksa menjadi numerik & buang NaN
        ref_col = pd.to_numeric(reference[col], errors='coerce').dropna()
        cur_col = pd.to_numeric(current[col], errors='coerce').dropna()
        
        p = psi(ref_col, cur_col)
        ks_stat, ks_p = ks_2samp(ref_col, cur_col)
        is_drift = (p > PSI_LIMIT) or (ks_p < KS_P_LIMIT)
        
        rows.append({
            "batch": batch, "feature": col, "type": "numerik",
            "psi": p, "ks_stat": ks_stat, "ks_p": ks_p, "js": np.nan, "drift": is_drift
        })
        
    # Fitur Kategori
    for col in CATEGORICAL:
        js = js_distance(reference[col], current[col])
        is_drift = js > JS_LIMIT
        
        rows.append({
            "batch": batch, "feature": col, "type": "kategori",
            "psi": np.nan, "ks_stat": np.nan, "ks_p": np.nan, "js": js, "drift": is_drift
        })
        
    return pd.DataFrame(rows)











# ---------------------------------------------------------------- operasional & kualitas
def ops_metrics(log: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per batch. Menghitung metrik operasional dan AUC."""
    log['timestamp'] = pd.to_datetime(log['timestamp'])
    merged = log.merge(labels, on='request_id', how='left')
    
    results = []
    for batch in log['batch'].unique():
        batch_log = merged[merged['batch'] == batch]
        if len(batch_log) == 0:
            continue
            
        reqs = len(batch_log)
        err_rate = (batch_log['status'] == 'error').mean()
        p50 = batch_log['latency_ms'].quantile(0.5)
        p95 = batch_log['latency_ms'].quantile(0.95)
        
        # Throughput: requests / detik
        time_diff = (batch_log['timestamp'].max() - batch_log['timestamp'].min()).total_seconds()
        throughput = reqs / time_diff if time_diff > 0 else 0.0
        
        mean_prob = batch_log['prob_bad'].mean()
        share_tolak = (batch_log['decision'] == 'tolak').mean()
        
        # AUC (hanya untuk status ok yang memiliki label)
        ok_batch = batch_log[batch_log['status'] == 'ok'].dropna(subset=['bad'])
        auc = np.nan
        if len(ok_batch) > 0 and ok_batch['bad'].nunique() == 2:
            try:
                auc = roc_auc_score(ok_batch['bad'], ok_batch['prob_bad'])
            except Exception:
                pass
                
        results.append({
            "batch": batch, "requests": reqs, 
            "error_rate": round(err_rate, 4),
            "p50_latency_ms": round(p50, 2), 
            "p95_latency_ms": round(p95, 2),
            "throughput_rps": round(throughput, 2), 
            "mean_prob_bad": round(mean_prob, 4),
            "share_tolak": round(share_tolak, 4), 
            "auc": round(auc, 4) if pd.notna(auc) else np.nan
        })
        
    return pd.DataFrame(results)


def alerts(ops: pd.DataFrame, drift: pd.DataFrame) -> list[str]:
    """Daftar pesan ALERT."""
    msgs = []
    
    for _, row in ops.iterrows():
        batch = row['batch']
        if row['error_rate'] > ERROR_RATE_LIMIT:
            msgs.append(f"{batch}: error rate {row['error_rate']*100:.1f}% > {ERROR_RATE_LIMIT*100}%")
        if row['p95_latency_ms'] > P95_LATENCY_MS_LIMIT:
            msgs.append(f"{batch}: p95 latency {row['p95_latency_ms']} ms > {P95_LATENCY_MS_LIMIT} ms")
        if pd.notna(row['auc']) and row['auc'] < AUC_LIMIT:
            msgs.append(f"{batch}: AUC {row['auc']} < {AUC_LIMIT}")
            
    if not drift.empty:
        for batch in drift['batch'].unique():
            drifted = drift[(drift['batch'] == batch) & (drift['drift'])]['feature'].tolist()
            if drifted:
                msgs.append(f"{batch}: drift pada {', '.join(drifted)}")
                
    return msgs


# ---------------------------------------------------------------- dashboard & laporan
def dashboard(reference: pd.DataFrame, log: pd.DataFrame, ops: pd.DataFrame, drift: pd.DataFrame, path: Path) -> None:
    """Simpan gambar grid 2x3."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Distribusi credit_amount
    for batch in ['Referensi'] + list(log['batch'].unique()):
        data = reference['credit_amount'] if batch == 'Referensi' else log[(log['batch']==batch) & (log['status']=='ok')]['credit_amount'].dropna()
        if not data.empty:
            axes[0,0].hist(data, bins=30, alpha=0.5, density=True, label=batch)
    axes[0,0].set_title('Distribusi Credit Amount')
    axes[0,0].legend()

    # 2. Distribusi age
    for batch in ['Referensi'] + list(log['batch'].unique()):
        data = reference['age'] if batch == 'Referensi' else log[(log['batch']==batch) & (log['status']=='ok')]['age'].dropna()
        if not data.empty:
            axes[0,1].hist(data, bins=30, alpha=0.5, density=True, label=batch)
    axes[0,1].set_title('Distribusi Age')
    axes[0,1].legend()

    # 3. PSI Fitur Numerik
    drift_num = drift[drift['type']=='numerik']
    if not drift_num.empty:
        pivot_psi = drift_num.pivot(index='feature', columns='batch', values='psi')
        pivot_psi.plot(kind='bar', ax=axes[0,2])
        axes[0,2].axhline(PSI_LIMIT, color='red', linestyle='--', label='Limit')
        axes[0,2].set_title('PSI Fitur Numerik')
        axes[0,2].legend()

    # 4. JS Fitur Kategori
    drift_cat = drift[drift['type']=='kategori']
    if not drift_cat.empty:
        pivot_js = drift_cat.pivot(index='feature', columns='batch', values='js')
        pivot_js.plot(kind='bar', ax=axes[1,0])
        axes[1,0].axhline(JS_LIMIT, color='red', linestyle='--', label='Limit')
        axes[1,0].set_title('Jensen-Shannon Fitur Kategori')
        axes[1,0].legend()

    # 5. Error Rate
    axes[1,1].bar(ops['batch'], ops['error_rate'])
    axes[1,1].axhline(ERROR_RATE_LIMIT, color='red', linestyle='--', label='Limit')
    axes[1,1].set_title('Error Rate per Batch')
    axes[1,1].legend()

    # 6. Share Tolak
    axes[1,2].bar(ops['batch'], ops['share_tolak'], color='orange')
    axes[1,2].set_title('Porsi Keputusan Tolak per Batch')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    """Jalankan seluruh proses monitoring."""
    print("Menyiapkan data referensi...")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    reference = X_train
    
    log_file = ROOT / "logs" / "predictions.csv"
    label_file = ROOT / "logs" / "labels.csv"
    
    if not log_file.exists() or not label_file.exists():
        print("❌ File log tidak ditemukan. Jalankan `python src/simulate.py` terlebih dahulu.")
        return
        
    log = pd.read_csv(log_file)
    labels = pd.read_csv(label_file)
    
    print("Menghitung drift & metrik...")
    drift_results = []
    for batch in log['batch'].unique():
        batch_log = log[(log['batch'] == batch) & (log['status'] == 'ok')]
        if len(batch_log) == 0:
            continue
        
        current = batch_log[NUMERIC + CATEGORICAL]
        dt = drift_table(reference, current, batch)
        drift_results.append(dt)
        
    drift_df = pd.concat(drift_results, ignore_index=True) if drift_results else pd.DataFrame()
    ops_df = ops_metrics(log, labels)
    alert_msgs = alerts(ops_df, drift_df)
    
    # Simpan Laporan
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    drift_df.to_csv(reports_dir / "drift_metrics.csv", index=False)
    dashboard(reference, log, ops_df, drift_df, reports_dir / "monitoring_dashboard.png")
    
    # Markdown Report
    with open(reports_dir / "monitoring_report.md", "w") as f:
        f.write("# Laporan Monitoring\n\n")
        f.write("## Metrik Operasional\n\n")
        f.write(ops_df.to_markdown(index=False))
        f.write("\n\n## Drift\n\n")
        f.write(drift_df.to_markdown(index=False))
        f.write("\n\n## Alerts\n\n")
        if alert_msgs:
            for msg in alert_msgs:
                f.write(f"- {msg}\n")
        else:
            f.write("Tidak ada alert.\n")
            
        f.write("\n\n## Contoh Error\n\n")
        errors = log[log['status'] == 'error']
        if not errors.empty:
            f.write(errors[['timestamp', 'request_id', 'error']].to_markdown(index=False))
        else:
            f.write("Tidak ada error.")
            
    # Retrain Trigger
    has_drift = drift_df['drift'].any() if not drift_df.empty else False
    trigger = {
        "retrain": bool(has_drift),
        "alerts": alert_msgs,
        "next_step": "Retrain model dengan data terbaru" if has_drift else "Tidak perlu retrain"
    }
    with open(reports_dir / "retrain_trigger.json", "w") as f:
        json.dump(trigger, f, indent=4)
        
    # Print ke layar
    print("\n=== METRIK OPERASIONAL ===")
    print(ops_df.to_string(index=False))
    print("\n=== DRIFT ===")
    print(drift_df.to_string(index=False))
    print("\n=== ALERTS ===")
    if alert_msgs:
        for msg in alert_msgs:
            print(f"ALERT: {msg}")
    else:
        print("Tidak ada alert.")
        
    print("\n✅ Laporan berhasil disimpan di folder reports/")


if __name__ == "__main__":
    main()