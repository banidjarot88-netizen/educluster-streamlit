import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="EduCluster",
    page_icon="🎓",
    layout="wide"
)

# =============================
# LOAD MODEL (DILATIH SEKALI)
# =============================
@st.cache_resource
def load_model():
    # Data referensi (contoh / data latih)
    reference_data = pd.DataFrame({
        "Daily_Usage_Hours": [0.5, 1, 2, 3, 4, 5, 6],
        "Trust_in_AI_Tools": [2, 3, 3, 4, 4, 5, 5],
        "Impact_on_Grades": [-1, 0, 0, 1, 1, 1, 1]
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(reference_data)

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)

    return scaler, model

scaler, kmeans_model = load_model()

# =============================
# SESSION STATE
# =============================
defaults = {
    "logged_in": False,
    "role": None,
    "user_data": None,
    "cluster_result": None,
    "teacher_df": None,
    "silhouette": None,
    "dbi": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================
# FUNGSI SIMPAN DATA SISWA
# =============================
def save_student_data(df):
    file_path = "student_data.csv"
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", header=header, index=False)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🎓 EduCluster")

if st.session_state.logged_in:
    st.sidebar.divider()
    if st.sidebar.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.rerun()

if not st.session_state.logged_in:
    menu = st.sidebar.radio("Menu", ["Home", "Login"])
else:
    if st.session_state.role == "Siswa":
        menu = st.sidebar.radio(
            "Menu",
            ["Home", "Input Data Siswa", "Dashboard Siswa"]
        )
    else:
        menu = st.sidebar.radio(
            "Menu",
            ["Home", "Upload Dataset", "Dashboard Guru"]
        )

# =============================
# HOME
# =============================
if menu == "Home":
    st.title("🎓 EduCluster")
    st.subheader("Analisis Dampak AI terhadap Performa Akademik")

    st.write("""
    **EduCluster** adalah sistem berbasis data science untuk menganalisis
    dampak penggunaan Artificial Intelligence (AI) terhadap performa akademik
    siswa menggunakan algoritma **K-Means Clustering**.
    """)

    if st.session_state.logged_in:
        st.success(f"Login sebagai **{st.session_state.role}**")
    else:
        st.info("Silakan login terlebih dahulu")

# =============================
# LOGIN
# =============================
elif menu == "Login":
    st.title("🔐 Login EduCluster")

    role = st.selectbox("Login sebagai:", ["Siswa", "Guru"])
    name = st.text_input("Nama")

    if st.button("Login"):
        if not name.strip():
            st.warning("Nama wajib diisi")
        else:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.success("Login berhasil")
            st.rerun()

# =============================
# INPUT DATA SISWA
# =============================
elif menu == "Input Data Siswa":
    st.title("📝 Input Data Siswa")

    with st.form("form_siswa"):
        q1 = st.radio(
            "Durasi penggunaan AI per hari",
            ["< 1 jam", "1–3 jam", "3–5 jam", "> 5 jam"]
        )

        q2 = st.radio(
            "Tingkat kepercayaan terhadap AI",
            ["Sangat Tidak Percaya", "Tidak Percaya", "Netral", "Percaya", "Sangat Percaya"]
        )

        q3 = st.radio(
            "Dampak AI terhadap nilai",
            ["Menurun", "Tidak Berubah", "Meningkat"]
        )

        submit = st.form_submit_button("Proses")

    if submit:
        usage_map = {"< 1 jam": 0.5, "1–3 jam": 2, "3–5 jam": 4, "> 5 jam": 6}
        trust_map = {
            "Sangat Tidak Percaya": 1,
            "Tidak Percaya": 2,
            "Netral": 3,
            "Percaya": 4,
            "Sangat Percaya": 5
        }
        impact_map = {"Menurun": -1, "Tidak Berubah": 0, "Meningkat": 1}

        df = pd.DataFrame([{
            "Daily_Usage_Hours": usage_map[q1],
            "Trust_in_AI_Tools": trust_map[q2],
            "Impact_on_Grades": impact_map[q3]
        }])

        # Prediksi cluster (TANPA FIT ULANG)
        X_user_scaled = scaler.transform(df)
        cluster = kmeans_model.predict(X_user_scaled)[0]

        st.session_state.user_data = df
        st.session_state.cluster_result = cluster

        # Simpan data siswa ke CSV
        save_student_data(df)

        st.success("Data berhasil diproses & disimpan")
        st.rerun()

# =============================
# DASHBOARD SISWA
# =============================
elif menu == "Dashboard Siswa":
    st.title("📊 Dashboard Siswa")

    if st.session_state.cluster_result is None:
        st.warning("Silakan isi data terlebih dahulu")
    else:
        labels = ["🔵 Light User", "🟡 Moderate User", "🟢 Heavy User"]
        cluster = st.session_state.cluster_result

        st.subheader(f"Hasil Klaster: {labels[cluster]}")
        st.dataframe(st.session_state.user_data)

# =============================
# UPLOAD DATASET (GURU)
# =============================
elif menu == "Upload Dataset":
    st.title("📂 Upload Dataset (Guru)")

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        features = df[
            ["Daily_Usage_Hours", "Trust_in_AI_Tools", "Impact_on_Grades"]
        ]

        scaler_guru = StandardScaler()
        X_scaled = scaler_guru.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        df["Cluster"] = labels

        st.session_state.teacher_df = df
        st.session_state.silhouette = silhouette_score(X_scaled, labels)
        st.session_state.dbi = davies_bouldin_score(X_scaled, labels)

        st.success("Dataset guru berhasil diproses")
        st.dataframe(df.head(20))

# =============================
# DASHBOARD GURU
# =============================
elif menu == "Dashboard Guru":
    st.title("📈 Dashboard Guru")

    if st.session_state.teacher_df is None:
        st.warning("Upload dataset terlebih dahulu")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score", round(st.session_state.silhouette, 3))
        col2.metric("Davies-Bouldin Index", round(st.session_state.dbi, 3))

        st.subheader("Distribusi Cluster")
        st.bar_chart(
            st.session_state.teacher_df["Cluster"].value_counts()
        )

        st.subheader("Rata-rata Fitur per Cluster")
        st.dataframe(
            st.session_state.teacher_df.groupby("Cluster").mean()
        )

