"""
app.py — Aplikasi Streamlit (Deployment) CRISP-DM K-Means Clustering
--------------------------------------------------------------------
Menampilkan seluruh proses pada notebook
51423260_REAL MADRID HAIKAL PUTRA_KELAS F.ipynb (Fase 1 s.d. Fase 5).

Kode pemodelan (training) sudah DIPISAH ke `train_model.py`. Aplikasi ini
memuat artefak hasil pelatihan (`scaler.pkl`, `kmeans_model.pkl`,
`cluster_summary.csv`) dan menggunakannya untuk segmentasi pelanggan.

Fitur interaktif untuk pengguna:
    1. Input Manual  — masukkan nilai fitur secara langsung
    2. Upload CSV    — unggah data pelanggan untuk diprediksi
    3. Latih Ulang   — latih ulang model dari Customer_Transactions.csv

Cara menjalankan:
    streamlit run app.py
"""

import io
import os

# Silencing warning joblib/loky "Could not find the number of physical cores" (Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import train_model
from train_model import FEATURES

# --- Import & pengaturan gaya plot (notebook: sel [8]) ---
sns.set(style="whitegrid")

st.set_page_config(page_title="Segmentasi Pelanggan - K-Means", layout="wide")

if "model_version" not in st.session_state:
    st.session_state.model_version = 0

VERSION = st.session_state.model_version


# ---------------------------------------------------------
# Memuat data & artefak model (di-cache dengan versi)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(version):
    # notebook: sel [10]
    return train_model.load_dataset()


@st.cache_resource(show_spinner=False)
def load_artifacts(version):
    # Memuat scaler & model; melatih ulang otomatis bila artefak belum ada.
    return train_model.load_artifacts()


@st.cache_data(show_spinner=False)
def load_cluster_summary(version):
    if os.path.exists(train_model.SUMMARY_PATH):
        return pd.read_csv(train_model.SUMMARY_PATH, index_col="Cluster")
    return None


@st.cache_data(show_spinner=False)
def elbow_analysis_cached(X_scaled):
    # notebook: sel [24]
    return train_model.elbow_analysis(X_scaled)


@st.cache_data(show_spinner=False)
def silhouette_analysis_cached(X_scaled):
    # notebook: sel [26]
    return train_model.silhouette_analysis(X_scaled)


# =========================================================
# FASE 1 — BUSINESS UNDERSTANDING (notebook: sel [2]-[5])
# =========================================================
st.title("Segmentasi Pelanggan menggunakan Metode K-Means Clustering")
st.caption("Tugas Mandiri — REAL MADRID HAIKAL PUTRA : 51423260 · Kerangka Kerja CRISP-DM")

st.header("🔵 Fase 1 — Business Understanding")
st.markdown(
    """
Memahami tujuan bisnis, metode, dan rumusan masalah sebelum masuk ke tahap teknis.

**1. Tujuan**

Tujuan dari proses clustering pada dataset *Customers Transactions* adalah untuk melakukan
segmentasi pelanggan berdasarkan karakteristik dan perilaku transaksi mereka. Segmentasi
dilakukan dengan mengelompokkan pelanggan yang memiliki karakteristik serupa ke dalam
kelompok (cluster) yang sama. Dengan adanya proses clustering, diharapkan dapat diketahui
pola dan karakteristik dari setiap kelompok pelanggan berdasarkan pendapatan, tingkat
pengeluaran, dan jumlah pembelian.

**2. Metode yang Digunakan**

Metode clustering yang digunakan adalah **K-Means Clustering**. K-Means merupakan metode
*unsupervised learning* yang digunakan untuk mengelompokkan data ke dalam beberapa cluster
berdasarkan tingkat kemiripan karakteristik antar data. Pada proses ini, pelanggan akan
dikelompokkan berdasarkan tiga fitur numerik, yaitu:

* **Annual Income** – menunjukkan pendapatan tahunan pelanggan.
* **Spending Score** – menunjukkan tingkat atau pola pengeluaran pelanggan.
* **Number of Purchases** – menunjukkan jumlah pembelian yang dilakukan pelanggan.

Sebelum dilakukan proses K-Means, data numerik akan melalui proses standardisasi agar
setiap fitur memiliki skala yang sebanding dan tidak ada fitur yang mendominasi proses
clustering hanya karena memiliki nilai yang lebih besar.

**3. Hal yang Dicari**

1. Jumlah cluster yang optimal untuk mengelompokkan pelanggan.
2. Karakteristik masing-masing cluster berdasarkan Annual Income, Spending Score, dan Number of Purchases.
3. Pola perilaku pelanggan yang terdapat pada masing-masing cluster.
4. Perbedaan karakteristik antar-cluster yang terbentuk.
5. Seberapa baik hasil pengelompokan yang diperoleh dari metode K-Means.

Untuk menentukan jumlah cluster yang optimal, digunakan **Elbow Method** dan
**Silhouette Score**.
"""
)

# =========================================================
# FASE 2 — DATA UNDERSTANDING (notebook: sel [6]-[14])
# =========================================================
st.header("🟢 Fase 2 — Data Understanding")
st.markdown(
    "Memuat dataset dan melakukan eksplorasi awal untuk memahami struktur, kualitas, "
    "dan karakteristik data sebelum diproses lebih lanjut."
)

df = load_dataset(VERSION)

# notebook: sel [10] — memuat dataset
st.subheader("Memuat Dataset")
st.dataframe(df.head(), use_container_width=True)

# notebook: sel [11] — informasi umum dataset
st.subheader("Informasi Umum Dataset")
buf = io.StringIO()
df.info(buf=buf)
st.text(buf.getvalue())

# notebook: sel [12] — cek missing value
st.subheader("Cek Missing Value")
st.write(df.isnull().sum())

# notebook: sel [13] — statistik deskriptif
st.subheader("Statistik Deskriptif")
st.dataframe(df.describe(), use_container_width=True)

st.markdown(
    "Berdasarkan hasil di atas, dataset tidak memiliki *missing value* sehingga dapat "
    "langsung digunakan pada tahap selanjutnya tanpa proses *imputation*."
)

# =========================================================
# FASE 3 — DATA PREPARATION (notebook: sel [15]-[20])
# =========================================================
st.header("🟡 Fase 3 — Data Preparation")
st.markdown(
    "Menyiapkan data agar siap digunakan pada tahap pemodelan, meliputi seleksi fitur "
    "relevan dan standardisasi skala data."
)

# notebook: sel [17] — seleksi fitur
st.subheader("Seleksi Fitur untuk Clustering")
X = train_model.select_features(df)
st.dataframe(X.head(), use_container_width=True)

# notebook: sel [20] — standardisasi data
st.subheader("Standardisasi Data")
st.markdown(
    "Standardisasi dilakukan agar setiap fitur memiliki skala yang sebanding, sehingga "
    "tidak ada fitur yang mendominasi proses clustering hanya karena memiliki rentang "
    "nilai yang lebih besar."
)

scaler, model = load_artifacts(VERSION)
X_scaled = scaler.transform(X)
st.write(X_scaled[:5])

# =========================================================
# FASE 4 — MODELING (notebook: sel [21]-[35])
# =========================================================
st.header("🟠 Fase 4 — Modeling")
st.markdown(
    "Menentukan jumlah cluster optimal (Elbow Method & Silhouette Score), melatih model "
    "K-Means, lalu memvisualisasikan hasil pengelompokan."
)

# notebook: sel [24] — Elbow Method
st.subheader("Menentukan Jumlah Cluster Optimal — Elbow Method")
K_range, inertia = elbow_analysis_cached(X_scaled)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertia, marker="o")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method untuk Menentukan Jumlah Cluster Optimal")
ax.set_xticks(K_range)
st.pyplot(fig)

# notebook: sel [26] — Silhouette Score
st.subheader("Silhouette Score")
K_range_sil, silhouette_scores = silhouette_analysis_cached(X_scaled)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range_sil, silhouette_scores, marker="o", color="green")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score untuk Setiap Jumlah Cluster")
ax.set_xticks(K_range_sil)
st.pyplot(fig)

# notebook: sel [27] — menampilkan nilai Silhouette Score tiap k
for k, score in zip(K_range_sil, silhouette_scores):
    st.write(f"k = {k} -> Silhouette Score = {score:.4f}")

st.markdown(
    "**Analisis:** Grafik Elbow Method tidak menunjukkan siku (*elbow*) yang terlalu tajam, "
    "tetapi penurunan inertia mulai melandai di sekitar k = 4 sampai k = 7. Hal ini sejalan "
    "dengan hasil Silhouette Score, di mana skor tertinggi diperoleh pada **k = 7 (0.2958)**, "
    "namun perbedaannya sangat tipis dibandingkan k = 4 hingga k = 8 (berkisar 0.27–0.30). "
    "Karena seluruh nilai Silhouette Score berada pada rentang yang relatif rendah dan "
    "cenderung datar, dapat disimpulkan bahwa struktur cluster pada dataset ini tidak "
    "terpisah secara sangat tegas.\n\n"
    "Dengan mempertimbangkan kemudahan interpretasi bisnis serta nilai Silhouette Score "
    "k = 4 (0.2767) yang tidak jauh berbeda dari nilai tertinggi, maka jumlah cluster yang "
    "dipilih untuk proses selanjutnya adalah **k = 4**."
)

# notebook: sel [30] — melatih model K-Means (memakai model hasil train_model.py)
st.subheader(f"Melatih Model K-Means (k = {model.n_clusters})")
df_clustered = df.copy()
df_clustered["Cluster"] = model.predict(X_scaled)
st.dataframe(df_clustered.head(), use_container_width=True)

# notebook: sel [31] — Silhouette Score model final
final_score = silhouette_scores[model.n_clusters - 2] if model.n_clusters >= 2 else None
st.write(
    f"Silhouette Score (k={model.n_clusters}) : {final_score:.4f}"
    if final_score is not None
    else "Silhouette Score tidak tersedia."
)

# notebook: sel [32] — jumlah pelanggan tiap cluster
st.write("Jumlah pelanggan pada masing-masing cluster:")
st.write(df_clustered["Cluster"].value_counts().sort_index())

# notebook: sel [34] — visualisasi scatter
st.subheader("Visualisasi Hasil Clustering")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df_clustered,
    x="annual_income",
    y="spending_score",
    hue="Cluster",
    palette="viridis",
    s=60,
    ax=ax,
)
ax.set_title("Segmentasi Pelanggan Berdasarkan Annual Income dan Spending Score")
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.legend(title="Cluster")
st.pyplot(fig)

# notebook: sel [35] — pairplot
g = sns.pairplot(df_clustered, vars=FEATURES, hue="Cluster", palette="viridis")
st.pyplot(g.fig)

# =========================================================
# FASE 5 — EVALUATION (notebook: sel [36]-[42])
# =========================================================
st.header("🔴 Fase 5 — Evaluation")
st.markdown(
    "Mengevaluasi kualitas hasil clustering (Silhouette Score) dan menganalisis "
    "karakteristik masing-masing cluster untuk memastikan hasil model dapat "
    "diinterpretasikan secara bisnis."
)

# notebook: sel [38] — karakteristik masing-masing cluster
st.subheader("Karakteristik Masing-Masing Cluster")
cluster_summary = df_clustered.groupby("Cluster")[FEATURES].mean().round(2)
st.dataframe(cluster_summary, use_container_width=True)

# notebook: sel [39] — visualisasi karakteristik rata-rata tiap cluster
cluster_summary_norm = (cluster_summary - cluster_summary.min()) / (
    cluster_summary.max() - cluster_summary.min()
)
fig, ax = plt.subplots(figsize=(9, 5))
cluster_summary_norm.plot(kind="bar", ax=ax)
ax.set_title("Perbandingan Relatif Karakteristik Setiap Cluster (Dinormalisasi)")
ax.set_xlabel("Cluster")
ax.set_ylabel("Nilai Ternormalisasi (0-1)")
ax.set_xticklabels(cluster_summary_norm.index, rotation=0)
ax.legend(title="Fitur")
st.pyplot(fig)

# notebook: sel [40] — analisis karakteristik cluster
st.markdown(
    """
**Analisis Karakteristik Cluster:**

Berdasarkan tabel `cluster_summary`, keempat cluster menunjukkan karakteristik pelanggan
yang berbeda:

| Cluster | Annual Income | Spending Score | Num Purchases | Jumlah Pelanggan |
|---|---|---|---|---|
| 0 | 71.977,55 (sedang) | 67,63 (tinggi) | 34,14 (**tertinggi**) | 2.174 |
| 1 | 140.371,16 (**tertinggi**) | 50,91 (sedang) | 27,04 (sedang-tinggi) | 2.363 |
| 2 | 71.032,50 (sedang) | 21,33 (**terendah**) | 15,43 (**terendah**) | 3.054 |
| 3 | 64.577,53 (**terendah**) | 73,39 (**tertinggi**) | 16,82 (rendah) | 2.409 |

* **Cluster 0 – Pelanggan Aktif (Annual Income sedang, Spending Score tinggi, jumlah pembelian tertinggi).** Kelompok ini paling sering bertransaksi dan cukup royal berbelanja meski pendapatannya tidak paling tinggi. Cluster ini paling potensial untuk program loyalitas karena frekuensi transaksinya paling banyak.
* **Cluster 1 – Pelanggan Berpendapatan Tinggi (Annual Income tertinggi, Spending Score & jumlah pembelian sedang).** Pelanggan pada cluster ini memiliki daya beli paling besar, namun belanjanya belum semaksimal Cluster 0. Cluster ini berpeluang ditingkatkan nilainya (*upselling*) melalui promosi yang lebih personal.
* **Cluster 2 – Pelanggan Kurang Aktif (Spending Score dan jumlah pembelian paling rendah, Annual Income sedang).** Meskipun pendapatannya tidak rendah, kelompok ini paling jarang berbelanja dan menghabiskan skor pengeluaran paling kecil. Cluster ini merupakan yang terbesar (3.054 pelanggan) sehingga penting menjadi target strategi re-engagement.
* **Cluster 3 – Pelanggan Royal Berpendapatan Rendah (Annual Income terendah, Spending Score tertinggi, jumlah pembelian rendah).** Pelanggan pada cluster ini memiliki pendapatan paling terbatas, tetapi ketika berbelanja cenderung menghabiskan proporsi pengeluaran yang besar meski frekuensi pembeliannya tidak tinggi — mengindikasikan pembelian bernilai besar namun jarang.
"""
)

# notebook: sel [42] — evaluasi hasil clustering
st.subheader("Evaluasi Hasil Clustering")
st.markdown(
    "**Analisis Evaluasi:** Model final dengan k = 4 menghasilkan Silhouette Score sebesar "
    "**0.2767**. Nilai ini berada pada rentang 0 sampai 1, dan secara umum nilai di bawah "
    "0.3 mengindikasikan struktur cluster yang **cukup lemah hingga sedang** — artinya antar "
    "cluster masih terdapat sedikit tumpang tindih (*overlap*) dan batas antar kelompok "
    "tidak terlalu tegas.\n\n"
    "Hal ini wajar terjadi karena fitur yang digunakan (Annual Income, Spending Score, "
    "Number of Purchases) berasal dari data transaksi yang bersifat kontinu dan tidak "
    "memiliki pemisahan alami yang tajam antar kelompok pelanggan. Meskipun demikian, hasil "
    "clustering tetap dapat digunakan sebagai dasar segmentasi karena perbedaan rata-rata "
    "karakteristik antar cluster (lihat tabel `cluster_summary`) masih cukup jelas terlihat "
    "dan dapat diinterpretasikan secara bisnis."
)

# =========================================================
# FASE 6 — INTERAKTIF: SEGMENTASI PELANGGAN BARU
# =========================================================
st.header("🟣 Segmentasi Pelanggan Baru")
st.markdown(
    "Gunakan model yang telah dilatih untuk menentukan segmen pelanggan baru. "
    "Pilih salah satu metode di bawah ini."
)

summary_ref = load_cluster_summary(VERSION)


def show_prediction(result_df):
    """Menampilkan ringkasan hasil prediksi cluster."""
    counts = result_df["Cluster"].value_counts().sort_index()
    st.write("Jumlah data per cluster:")
    st.write(counts)


tab_input, tab_upload, tab_retrain = st.tabs(
    ["✍️ Input Manual", "📁 Upload CSV", "🔄 Latih Ulang Model"]
)

# --- Tab 1: Input Manual -------------------------------------------------
with tab_input:
    st.subheader("Masukkan Data Pelanggan")
    with st.form("form_input_manual"):
        col1, col2, col3 = st.columns(3)
        annual_income = col1.number_input(
            "Annual Income", min_value=0.0, value=70000.0, step=1000.0
        )
        spending_score = col2.number_input(
            "Spending Score", min_value=0.0, value=50.0, step=1.0
        )
        num_purchases = col3.number_input(
            "Number of Purchases", min_value=0.0, value=20.0, step=1.0
        )
        submitted = st.form_submit_button("Prediksi Cluster")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "annual_income": annual_income,
                    "spending_score": spending_score,
                    "num_purchases": num_purchases,
                }
            ]
        )
        result = train_model.predict(input_df, scaler=scaler, model=model)
        cluster = int(result["Cluster"].iloc[0])
        st.success(f"Pelanggan berada pada **Cluster {cluster}** 🎯")
        st.dataframe(result, use_container_width=True)

        if summary_ref is not None and cluster in summary_ref.index:
            st.markdown("**Karakteristik rata-rata Cluster ini:**")
            st.dataframe(summary_ref.loc[[cluster]], use_container_width=True)

# --- Tab 2: Upload CSV ---------------------------------------------------
with tab_upload:
    st.subheader("Unggah Data Pelanggan (CSV)")
    st.caption(
        "File CSV harus memuat kolom: " + ", ".join(f"`{f}`" for f in FEATURES)
    )
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        missing = [f for f in FEATURES if f not in upload_df.columns]
        if missing:
            st.error("Kolom berikut tidak ditemukan: " + ", ".join(missing))
        else:
            st.markdown("**Pratinjau data:**")
            st.dataframe(upload_df.head(), use_container_width=True)

            result = train_model.predict(upload_df, scaler=scaler, model=model)
            st.markdown("**Hasil segmentasi:**")
            st.dataframe(result, use_container_width=True)
            show_prediction(result)

            st.download_button(
                "⬇️ Unduh hasil (CSV)",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="hasil_segmentasi.csv",
                mime="text/csv",
            )

# --- Tab 3: Latih Ulang Model -------------------------------------------
with tab_retrain:
    st.subheader("Latih Ulang Model dari Dataset")
    st.markdown(
        "Melatih ulang model K-Means menggunakan dataset yang tersedia "
        f"(`{train_model.DATASET_PATH}`, {len(df):,} baris). "
        "Artefak `scaler.pkl`, `kmeans_model.pkl`, dan `cluster_summary.csv` akan "
        "diperbarui."
    )
    st.caption(f"Jumlah cluster (k) tetap = {train_model.K_OPTIMAL}.")

    if st.button("🔄 Latih Ulang Sekarang"):
        with st.spinner("Melatih ulang model..."):
            _, _, metrics = train_model.train_and_save()
        st.session_state.model_version += 1
        st.success(
            "Pelatihan ulang selesai! "
            f"Silhouette Score = {metrics['silhouette_score']:.4f} · "
            f"{metrics['n_samples']:,} sampel · k = {metrics['n_clusters']}."
        )
        st.rerun()
