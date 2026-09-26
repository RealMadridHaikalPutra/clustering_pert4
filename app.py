"""
app.py — Deployment Streamlit dari notebook CRISP-DM (satu file)
----------------------------------------------------------------
Aplikasi ini menampilkan seluruh proses pada notebook
51423260_REAL MADRID HAIKAL PUTRA_KELAS F.ipynb (Fase 1 s.d. Fase 5)
tanpa menambahkan proses di luar notebook.

Seluruh kode notebook — termasuk kode pemodelan (Fase 3 s.d. Fase 5) —
berada di file ini. Secara default scaler & model dimuat dari artefak:
    - scaler.pkl
    - kmeans_model.pkl
Bila artefak tersebut belum ada, aplikasi akan melatih model otomatis
dari dataset (kode yang sama dengan notebook) lalu menyimpannya.

Cara menjalankan:
    streamlit run app.py
"""

import io
import os

# Silencing warning joblib/loky "Could not find the number of physical cores" (Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- Import & pengaturan gaya plot (notebook: sel [8]) ---
sns.set(style="whitegrid")

st.set_page_config(page_title="Segmentasi Pelanggan - K-Means", layout="wide")


# ---------------------------------------------------------
# Memuat data & artefak model
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    # notebook: sel [10]
    return pd.read_csv("Customer_Transactions.csv")


FEATURES = ["annual_income", "spending_score", "num_purchases"]
K_OPTIMAL = 4


@st.cache_resource
def load_artifacts(_X):
    # Utamakan artefak .pkl bila tersedia (hasil pemodelan notebook).
    if os.path.exists("scaler.pkl") and os.path.exists("kmeans_model.pkl"):
        scaler = joblib.load("scaler.pkl")
        return scaler, joblib.load("kmeans_model.pkl")

    # Cadangan: latih dari dataset memakai kode yang sama dengan notebook.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_X)

    # Elbow Method (notebook: sel [24])
    for k in range(1, 11):
        KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10).fit(X_scaled)

    # Silhouette Score (notebook: sel [26])
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10).fit_predict(X_scaled)
        silhouette_score(X_scaled, labels)

    # Melatih model K-Means (notebook: sel [30])
    model = KMeans(n_clusters=K_OPTIMAL, init="k-means++", random_state=42, n_init=10)
    model.fit(X_scaled)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(model, "kmeans_model.pkl")
    return scaler, model


@st.cache_data
def elbow_analysis(X_scaled):
    # notebook: sel [24]
    inertia = []
    K_range = list(range(1, 11))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    return K_range, inertia


@st.cache_data
def silhouette_analysis(X_scaled):
    # notebook: sel [26]
    silhouette_scores = []
    K_range_sil = list(range(2, 11))
    for k in K_range_sil:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    return K_range_sil, silhouette_scores


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

df = load_dataset()

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
features = FEATURES
X = df[features]
st.dataframe(X.head(), use_container_width=True)

# notebook: sel [20] — standardisasi data
st.subheader("Standardisasi Data")
st.markdown(
    "Standardisasi dilakukan agar setiap fitur memiliki skala yang sebanding, sehingga "
    "tidak ada fitur yang mendominasi proses clustering hanya karena memiliki rentang "
    "nilai yang lebih besar."
)

scaler, model = load_artifacts(X)
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
K_range, inertia = elbow_analysis(X_scaled)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertia, marker="o")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method untuk Menentukan Jumlah Cluster Optimal")
ax.set_xticks(K_range)
st.pyplot(fig)

# notebook: sel [26] — Silhouette Score
st.subheader("Silhouette Score")
K_range_sil, silhouette_scores = silhouette_analysis(X_scaled)
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

# notebook: sel [30] — melatih model K-Means
st.subheader("Melatih Model K-Means (k = 4)")
df["Cluster"] = model.predict(X_scaled)
st.dataframe(df.head(), use_container_width=True)

# notebook: sel [31] — Silhouette Score model final
k_optimal = K_OPTIMAL
final_score = silhouette_score(X_scaled, df["Cluster"])
st.write(f"Silhouette Score (k={k_optimal}) : {final_score:.4f}")

# notebook: sel [32] — jumlah pelanggan tiap cluster
st.write("Jumlah pelanggan pada masing-masing cluster:")
st.write(df["Cluster"].value_counts().sort_index())

# notebook: sel [34] — visualisasi scatter
st.subheader("Visualisasi Hasil Clustering")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df,
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
g = sns.pairplot(df, vars=features, hue="Cluster", palette="viridis")
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
cluster_summary = df.groupby("Cluster")[features].mean().round(2)
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
