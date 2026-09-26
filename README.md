# Deployment — Segmentasi Pelanggan (K-Means Clustering)

Paket ini adalah **Source Code Deployment** untuk menjalankan hasil notebook
`51423260_REAL MADRID HAIKAL PUTRA_KELAS F.ipynb` sebagai aplikasi web interaktif
menggunakan **Streamlit**.

Aplikasi menampilkan **seluruh proses CRISP-DM** pada notebook (Fase 1 Business
Understanding s.d. Fase 5 Evaluation) tanpa menambahkan proses di luar notebook.

## Isi Folder

| File | Fungsi |
|---|---|
| `app.py` | **Satu file** aplikasi Streamlit + seluruh sel notebook & kode pemodelan (EDA, Elbow, Silhouette, clustering, evaluasi) |
| `requirements.txt` | Daftar dependency Python yang dibutuhkan |
| `scaler.pkl` | `StandardScaler` yang sudah di-*fit* |
| `kmeans_model.pkl` | Model `KMeans` (k=4) yang sudah dilatih |
| `cluster_summary.csv` | Rata-rata fitur per cluster |
| `Customer_Transactions.csv` | Dataset (untuk menjalankan aplikasi / latih ulang) |

Seluruh kode pemodelan sudah menyatu di dalam `app.py`. Secara default model
dimuat dari `scaler.pkl` & `kmeans_model.pkl`; bila file tersebut tidak ada,
`app.py` otomatis melatih ulang dari dataset lalu menyimpannya kembali.

Fitur: `annual_income`, `spending_score`, `num_purchases` ·
`StandardScaler` · `KMeans(n_clusters=4, random_state=42, n_init=10)` ·
**Silhouette Score = 0.2767** (sama dengan notebook).

---

## 1. Menjalankan di Lokal (opsional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

Jika ingin melatih ulang model dari dataset, cukup hapus/tidak sertakan file
`scaler.pkl` dan `kmeans_model.pkl` — `app.py` akan melatih ulang secara otomatis
saat dijalankan lalu menyimpan artefak baru.

---

## 2. Deploy ke Streamlit Community Cloud

1. **Buat repository GitHub baru** (public), misalnya `customer-segmentation-kmeans`.
2. **Upload semua file** dalam folder ini ke repository tersebut:
   `app.py`, `requirements.txt`, `scaler.pkl`,
   `kmeans_model.pkl`, `cluster_summary.csv`, `Customer_Transactions.csv`.
3. Buka **https://share.streamlit.io** dan login dengan akun GitHub Anda.
4. Klik **"New app"**, lalu pilih:
   - **Repository**: repo yang baru dibuat
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **"Deploy"**. Tunggu proses build (biasanya 1–3 menit).
6. Setelah selesai, Anda akan mendapatkan URL publik seperti:
   `https://<nama-app>-<username>.streamlit.app`
7. **Salin URL tersebut** — itulah yang dicantumkan sebagai **Link Deployment**.

**Tips troubleshooting:**

- Jika error `ModuleNotFoundError`, pastikan `requirements.txt` ikut ter-upload.
- Jika error file `scaler.pkl` / `kmeans_model.pkl` tidak ditemukan, pastikan
  file tersebut ikut di-*commit* ke GitHub.
- Versi `scikit-learn` pada `requirements.txt` **harus sama** dengan versi yang
  membuat file `.pkl` (saat ini `1.8.0`) agar tidak muncul peringatan/error
  `InconsistentVersionWarning`.
