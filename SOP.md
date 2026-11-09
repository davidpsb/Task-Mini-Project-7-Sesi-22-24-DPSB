# SOP — Alert monitoring model credit-risk

> Maksimal 1 halaman. Tulis langkah yang bisa langsung diikuti orang lain saat alert menyala.

## 1. Pemicu
SOP ini dijalankan ketika sistem monitoring (`src/monitor.py`) menembakkan ALERT karena salah satu kondisi berikut:
- Error rate > 1%
- p95 Latency > 200 ms
- AUC model < 0.70
- Terjadi Data Drift (PSI > 0.2, p-value KS < 0.05, atau Jensen-Shannon > 0.1)

## 2. Pemeriksaan pertama (15 menit pertama)
1. Buka file `reports/monitoring_report.md` dan `reports/monitoring_dashboard.png`.
2. Cek volume request di batch tersebut. Jika request sangat sedikit (misal < 10), alert bisa jadi *false alarm* secara statistik.
3. Buka `logs/predictions.csv`. Filter berdasarkan batch yang bermasalah.
4. Periksa apakah alert menyala karena *error rate* (kolom `status` = `error`) atau karena *drift* (fitur bergeser).

## 3. Membedakan penyebab
- **Input Rusak:** Lihat `logs/predictions.csv` pada baris `status=error`. Jika pesan errornya "field kosong" atau "harus angka positif", berarti masalah ada di validasi API, bukan di model.
- **Populasi Berubah (Drift):** Lihat `reports/drift_metrics.csv`. Jika nilai PSI/JS melewati batas, cek apakah perubahan ini masuk akal secara bisnis (misal: tren pinjaman naik karena musim liburan). Jika AUC masih > 0.70, model masih bisa diandalkan.
- **Model Memburuk:** Jika AUC turun di bawah 0.70 (bisa dicek di `monitoring_report.md`), ini menandakan model sudah tidak relevan dengan pola data terbaru.

## 4. Tindakan
- **Jika Input Rusak:** Perbaiki validasi di *frontend/API* (misal: pastikan umur tidak boleh kosong). Tidak perlu retrain model.
- **Jika Hanya Drift (AUC Aman):** Tidak perlu retrain sekarang. Catat sebagai bahan evaluasi. Pantau 1-2 minggu ke depan.
- **Jika Model Memburuk (AUC < 0.70):** Lakukan **Rollback** ke model versi sebelumnya (lewat MLflow UI) untuk menghentikan kerugian, lalu segera siapkan **Retrain**.

## 5. Retrain dan verifikasi
1. **Data:** Kumpulkan data dari populasi baru (minggu_2 dan seterusnya) yang **sudah memiliki label** (nasabah yang sudah lunas/macet).
2. **Perintah:** Jalankan `python src/pipeline.py --reason "retrain drift minggu_2"`.
3. **Quality Gate:** Pastikan model baru lulus gate (AUC ≥ 0.70, Recall ≥ 0.50).
4. **Promosi:** Pipeline hanya akan mempromosikan model baru jika AUC-nya *strictly greater* (lebih tinggi) dari AUC champion lama. Jika tidak, champion lama tetap dipertahankan.

## 6. Pencatatan
- Semua request dan error otomatis tercatat di `logs/predictions.csv`.
- Semua metrik *drift* dan *alert* otomatis tercatat di folder `reports/`.
- Insiden (tanggal, penyebab, tindakan yang diambil) wajib dicatat manual di *Issue Tracker* (Jira/Notion) dan diinformasikan ke tim Risk Management serta Data Science.