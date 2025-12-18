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
# MODEL REFERENSI (UNTUK SISWA)
# =============================
@st.cache_resource
def load_model():
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

scaler_siswa, model_siswa = load_model()

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
# SIDEBAR
# =============================
st.sidebar.title("🎓 EduCluster")

if st.session_state.logged_in:
    if st.sidebar.button("🔙 Logout"):
        st.session_state.clear()
        st.rerun()

if not st.session_state.logged_in:
    menu = st.sidebar.radio("Menu", ["Home", "Login"])
else:
    if st.session_state.role == "Siswa":
        menu = st.sidebar.radio("Menu", ["Home", "Input Data Siswa", "Dashboard Siswa"])
    else:
        menu = st.sidebar.radio("Menu", ["Home", "Upload Dataset", "Dashboard Guru"])

# =============================
# HOME
# =============================
if menu == "Home":
    st.title("🎓 EduCluster")
    st.write("""
    Sistem analisis dampak penggunaan AI terhadap performa akademik
    menggunakan **K-Means Clustering**.
    """)

# =============================
# LOGIN
# =============================
elif menu == "Login":
    st.title("🔐 Login")

    role = st.selectbox("Login sebagai", ["Siswa", "Guru"])
    name = st.text_input("Nama")

    if st.button("Login"):
        if name.strip() == "":
            st.warning("Nama wajib diisi")
        else:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.rerun()

# =============================
# INPUT DATA SISWA
# =============================
elif menu == "Input Data Siswa":
    st.title("📝 Input Data Siswa")

    with st.form("form_siswa"):
        usage = st.radio("Durasi penggunaan AI per hari",
                         ["< 1 jam", "1–3 jam", "3–5 jam", "> 5 jam"])
        trust = st.radio("Tingkat kepercayaan terhadap AI",
                         ["Sangat Tidak Percaya", "Tidak Percaya", "Netral", "Percaya", "Sangat Percaya"])
        impact = st.radio("Dampak AI terhadap nilai",
                          ["Menurun", "Tidak Berubah", "Meningkat"])
        submit = st.form_submit_button("Proses")

    if submit:
        usage_map = {"< 1 jam": 0.5, "1–3 jam": 2, "3–5 jam": 4, "> 5 jam": 6}
        trust_map = {"Sangat Tidak Percaya": 1, "Tidak Percaya": 2, "Netral": 3, "Percaya": 4, "Sangat Percaya": 5}
        impact_map = {"Menurun": -1, "Tidak Berubah": 0, "Meningkat": 1}

        df_user = pd.DataFrame([{
            "Daily_Usage_Hours": usage_map[usage],
            "Trust_in_AI_Tools": trust_map[trust],
            "Impact_on_Grades": impact_map[impact]
        }])

        X_scaled = scaler_siswa.transform(df_user)
        cluster = model_siswa.predict(X_scaled)[0]

        st.session_state.user_data = df_user
        st.session_state.cluster_result = cluster
        st.rerun()

# =============================
# DASHBOARD SISWA
# =============================
elif menu == "Dashboard Siswa":
    st.title("📊 Dashboard Siswa")

    if st.session_state.cluster_result is None:
        st.warning("Silakan isi data terlebih dahulu")
    else:
        label_map = {0: "🔵 Light User", 1: "🟡 Moderate User", 2: "🟢 Heavy User"}
        cluster = st.session_state.cluster_result

        st.subheader(f"Hasil Klaster: {label_map[cluster]}")
        st.dataframe(st.session_state.user_data)

# =============================
# UPLOAD DATASET GURU
# =============================
elif menu == "Upload Dataset":
    st.title("📂 Upload Dataset (Guru)")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        features = df[["Daily_Usage_Hours", "Trust_in_AI_Tools", "Impact_on_Grades"]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # ===== LABEL CLUSTER SECARA BENAR =====
        cluster_order = (
            df.groupby("Cluster")["Daily_Usage_Hours"]
            .mean()
            .sort_values()
            .index
        )

        label_names = ["Light User", "Moderate User", "Heavy User"]
        cluster_label_map = {cluster_order[i]: label_names[i] for i in range(3)}
        df["Cluster_Label"] = df["Cluster"].map(cluster_label_map)

        st.session_state.teacher_df = df
        st.session_state.silhouette = silhouette_score(X_scaled, df["Cluster"])
        st.session_state.dbi = davies_bouldin_score(X_scaled, df["Cluster"])

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
        st.metric("Silhouette Score", round(st.session_state.silhouette, 3))
        st.metric("Davies-Bouldin Index", round(st.session_state.dbi, 3))

        st.subheader("Distribusi Cluster")
        st.bar_chart(
            st.session_state.teacher_df["Cluster_Label"].value_counts()
        )

        st.subheader("Rata-rata per Cluster")
        st.dataframe(
            st.session_state.teacher_df
            .groupby("Cluster_Label")[["Daily_Usage_Hours", "Trust_in_AI_Tools", "Impact_on_Grades"]]
            .mean()
        )
