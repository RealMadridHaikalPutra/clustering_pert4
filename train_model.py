"""
train_model.py — Pelatihan model K-Means (dipisah dari app.py)
--------------------------------------------------------------
Modul ini berisi SELURUH kode pemodelan notebook
51423260_REAL MADRID HAIKAL PUTRA_KELAS F.ipynb (Fase 3 s.d. Fase 5):
seleksi fitur, standardisasi, Elbow Method, Silhouette Score,
pelatihan K-Means (k=4), dan evaluasi.

Model & scaler yang dihasilkan disimpan sebagai artefak:
    - scaler.pkl
    - kmeans_model.pkl
    - cluster_summary.csv

Cara melatih ulang dari dataset (Customer_Transactions.csv):
    python train_model.py

Modul ini juga dapat di-import dari app.py, mis.:
    import train_model
    scaler, model = train_model.load_artifacts()
"""

import os

# Silencing warning joblib/loky "Could not find the number of physical cores" (Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --- Konstanta yang dipakai bersama app.py -------------------------------
FEATURES = ["annual_income", "spending_score", "num_purchases"]
K_OPTIMAL = 4
RANDOM_STATE = 42

DATASET_PATH = "Customer_Transactions.csv"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "kmeans_model.pkl"
SUMMARY_PATH = "cluster_summary.csv"


# ---------------------------------------------------------
# Memuat data
# ---------------------------------------------------------
def load_dataset(path=DATASET_PATH):
    """Memuat dataset transaksi pelanggan (notebook: sel [10])."""
    return pd.read_csv(path)


def select_features(df, features=FEATURES):
    """Seleksi fitur numerik untuk clustering (notebook: sel [17])."""
    return df[features]


# ---------------------------------------------------------
# Analisis jumlah cluster optimal
# ---------------------------------------------------------
def elbow_analysis(X_scaled, k_max=10):
    """Elbow Method — inertia tiap k (notebook: sel [24])."""
    inertia = []
    K_range = list(range(1, k_max + 1))
    for k in K_range:
        kmeans = KMeans(
            n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10
        )
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    return K_range, inertia


def silhouette_analysis(X_scaled, k_min=2, k_max=10):
    """Silhouette Score tiap k (notebook: sel [26])."""
    scores = []
    K_range = list(range(k_min, k_max + 1))
    for k in K_range:
        labels = KMeans(
            n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10
        ).fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    return K_range, scores


# ---------------------------------------------------------
# Pelatihan model
# ---------------------------------------------------------
def train_model(df, features=FEATURES, k=K_OPTIMAL):
    """Melatih StandardScaler + KMeans dan mengembalikan (scaler, model, metrics).

    Metrik berisi silhouette score, jumlah cluster, jumlah sampel,
    dan tabel rata-rata fitur per cluster.
    """
    X = df[features]

    # Standardisasi data (notebook: sel [20])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Melatih model K-Means (notebook: sel [30])
    model = KMeans(
        n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10
    )
    labels = model.fit_predict(X_scaled)

    # Evaluasi (notebook: sel [31]) + karakteristik cluster (sel [38])
    score = silhouette_score(X_scaled, labels)
    cluster_summary = (
        df.assign(Cluster=labels).groupby("Cluster")[features].mean().round(2)
    )

    metrics = {
        "silhouette_score": score,
        "n_clusters": k,
        "n_samples": len(df),
        "cluster_summary": cluster_summary,
    }
    return scaler, model, metrics


def save_artifacts(scaler, model, cluster_summary):
    """Menyimpan scaler, model, dan ringkasan cluster ke disk."""
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    cluster_summary.to_csv(SUMMARY_PATH)
    return SCALER_PATH, MODEL_PATH, SUMMARY_PATH


def train_and_save(path=DATASET_PATH, k=K_OPTIMAL, features=FEATURES):
    """Memuat dataset, melatih model, lalu menyimpan seluruh artefak."""
    df = load_dataset(path)
    scaler, model, metrics = train_model(df, features=features, k=k)
    save_artifacts(scaler, model, metrics["cluster_summary"])
    return scaler, model, metrics


def load_artifacts():
    """Memuat scaler & model dari artefak .pkl (melatih ulang bila belum ada)."""
    if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
        return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)

    scaler, model, _ = train_and_save()
    return scaler, model


def predict(df, scaler=None, model=None, features=FEATURES):
    """Memberi label cluster pada dataframe baru (mengembalikan salinan + 'Cluster')."""
    if scaler is None or model is None:
        scaler, model = load_artifacts()
    X_scaled = scaler.transform(df[features])
    result = df.copy()
    result["Cluster"] = model.predict(X_scaled)
    return result


# ---------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------
def main():
    scaler, model, metrics = train_and_save()
    print("Pelatihan model selesai.")
    print(f"  Sampel          : {metrics['n_samples']}")
    print(f"  Jumlah cluster  : {metrics['n_clusters']}")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Artefak tersimpan: {SCALER_PATH}, {MODEL_PATH}, {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
