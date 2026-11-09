# Laporan Monitoring

## Metrik Operasional

| batch    |   requests |   error_rate |   p50_latency_ms |   p95_latency_ms |   throughput_rps |   mean_prob_bad |   share_tolak |    auc |
|:---------|-----------:|-------------:|-----------------:|-----------------:|-----------------:|----------------:|--------------:|-------:|
| minggu_1 |        300 |         0    |            16.66 |            19.51 |            52.21 |          0.4524 |        0.2733 | 0.7629 |
| minggu_2 |        300 |         0.02 |            16.06 |            19.37 |            55.5  |          0.5313 |        0.4133 | 0.7387 |

## Drift

| batch    | feature          | type     |         psi |     ks_stat |          ks_p |           js | drift   |
|:---------|:-----------------|:---------|------------:|------------:|--------------:|-------------:|:--------|
| minggu_1 | age              | numerik  |   0.0640554 |   0.0680952 |   0.272736    | nan          | False   |
| minggu_1 | credit_amount    | numerik  |   0.03805   |   0.0409524 |   0.859126    | nan          | False   |
| minggu_1 | duration         | numerik  |   0.0123362 |   0.0414286 |   0.84971     | nan          | False   |
| minggu_1 | sex              | kategori | nan         | nan         | nan           |   0.0347571  | False   |
| minggu_1 | job              | kategori | nan         | nan         | nan           |   0.0538095  | False   |
| minggu_1 | housing          | kategori | nan         | nan         | nan           |   0.0134681  | False   |
| minggu_1 | saving_accounts  | kategori | nan         | nan         | nan           |   0.0752198  | False   |
| minggu_1 | checking_account | kategori | nan         | nan         | nan           |   0.00173999 | False   |
| minggu_1 | purpose          | kategori | nan         | nan         | nan           |   0.0769657  | False   |
| minggu_2 | age              | numerik  |   1.03292   |   0.447687  |   5.60984e-38 | nan          | True    |
| minggu_2 | credit_amount    | numerik  |   0.641752  |   0.321429  |   1.89044e-19 | nan          | True    |
| minggu_2 | duration         | numerik  |   3.24638   |   0.468912  |   8.85452e-42 | nan          | True    |
| minggu_2 | sex              | kategori | nan         | nan         | nan           |   0.0348182  | False   |
| minggu_2 | job              | kategori | nan         | nan         | nan           |   0.0491819  | False   |
| minggu_2 | housing          | kategori | nan         | nan         | nan           |   0.0122341  | False   |
| minggu_2 | saving_accounts  | kategori | nan         | nan         | nan           |   0.0763215  | False   |
| minggu_2 | checking_account | kategori | nan         | nan         | nan           |   0.00346774 | False   |
| minggu_2 | purpose          | kategori | nan         | nan         | nan           |   0.342883   | True    |

## Alerts

- minggu_2: error rate 2.0% > 1.0%
- minggu_2: drift pada age, credit_amount, duration, purpose


## Contoh Error

| timestamp                  | request_id   | error                                         |
|:---------------------------|:-------------|:----------------------------------------------|
| 2026-11-09 17:57:02.467196 | 2-0010       | field kosong: ['age']                         |
| 2026-11-09 17:57:02.648007 | 2-0020       | credit_amount harus angka positif, dapat -500 |
| 2026-11-09 17:57:02.829757 | 2-0030       | duration harus angka positif, dapat dua belas |
| 2026-11-09 17:57:03.035734 | 2-0040       | field kosong: ['purpose']                     |
| 2026-11-09 17:57:04.282650 | 2-0110       | field kosong: ['age']                         |
| 2026-11-09 17:57:04.441511 | 2-0120       | credit_amount harus angka positif, dapat -500 |