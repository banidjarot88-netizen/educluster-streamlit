import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =============================
# KONFIGURASI AWAL
# =============================
st.set_page_config(
    page_title="EduCluster",
    page_icon="🎓",
    layout="wide"
)

# =============================
# SESSION STATE INIT
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
# SIDEBAR
# =============================
st.sidebar.title("🎓 EduCluster")
if st.session_state.logged_in:
    st.sidebar.divider()
    if st.sidebar.button("🔙 Kembali ke Login"):
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
    dampak penggunaan Artificial Intelligence (AI) terhadap performa
    akademik siswa menggunakan **K-Means Clustering**.
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
            st.success(f"Login berhasil sebagai {role}")
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
            "Sangat Tidak Percaya": 1, "Tidak Percaya": 2,
            "Netral": 3, "Percaya": 4, "Sangat Percaya": 5
        }
        impact_map = {"Menurun": -1, "Tidak Berubah": 0, "Meningkat": 1}

        df = pd.DataFrame([{
            "Daily_Usage_Hours": usage_map[q1],
            "Trust_in_AI_Tools": trust_map[q2],
            "Impact_on_Grades": impact_map[q3]
        }])

        X_scaled = StandardScaler().fit_transform(df)
        cluster = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)[0]

        st.session_state.user_data = df
        st.session_state.cluster_result = cluster

        st.success("Data berhasil diproses")
        st.rerun()

# =============================
# DASHBOARD SISWA
# =============================
elif menu == "Dashboard Siswa":
    st.title("📊 Dashboard Siswa")

    if st.session_state.cluster_result is None:
        st.warning("Isi data terlebih dahulu")
    else:
        cluster = st.session_state.cluster_result
        labels = ["🔵 Light User", "🟡 Moderate User", "🟢 Heavy User"]

        st.subheader(f"Hasil Klaster: {labels[cluster]}")
        st.dataframe(st.session_state.user_data)

# =============================
# UPLOAD DATASET (GURU)
# =============================
elif menu == "Upload Dataset":
    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        features = df[
            ["Daily_Usage_Hours", "Trust_in_AI_Tools", "Impact_on_Grades"]
        ]

        X_scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        df["Cluster"] = labels

        st.session_state.teacher_df = df
        st.session_state.silhouette = silhouette_score(X_scaled, labels)
        st.session_state.dbi = davies_bouldin_score(X_scaled, labels)

        st.success("Dataset berhasil diproses")
        st.dataframe(df.head())

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

        st.bar_chart(
            st.session_state.teacher_df["Cluster"].value_counts()
        )

        st.dataframe(
            st.session_state.teacher_df.groupby("Cluster").mean()
        )
