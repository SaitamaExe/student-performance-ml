import streamlit as st
import pandas as pd
import joblib

# ==========================
# 1. LOAD MODEL
# ==========================
# File encoder.pkl tidak dipakai lagi karena diganti pd.get_dummies yang lebih aman
model = joblib.load("student_model.pkl")

# ==========================
# 2. KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Prediksi Kelulusan Siswa",
    layout="wide"
)

st.title("🎓 Prediksi Kelulusan Siswa")
st.write("Sistem prediksi kelulusan siswa menggunakan Random Forest")
st.divider()

# ==========================
# 3. COMPONENT INPUT DATA
# ==========================
st.subheader("Data Akademik")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Umur", 10, 30, 17)
    studytime = st.selectbox("Waktu Belajar", [1, 2, 3, 4])
    failures = st.number_input("Jumlah Kegagalan", 0, 4, 0)

with col2:
    absences = st.number_input("Jumlah Absensi", 0, 100, 5)
    G1 = st.number_input("Nilai G1", 0, 20, 12)
    G2 = st.number_input("Nilai G2", 0, 20, 12)

with col3:
    sex = st.selectbox("Jenis Kelamin", ["F", "M"])
    school = st.selectbox("Sekolah", ["GP", "MS"])
    internet = st.selectbox("Internet", ["yes", "no"])

st.divider()

# ==========================
# 4. PROSES PREDIKSI
# ==========================
if st.button("🔍 Prediksi Kelulusan"):
    # Membuat DataFrame dari data input form
    data = pd.DataFrame({
        "school": [school],
        "sex": [sex],
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "internet": [internet],
        "G1": [G1],
        "G2": [G2]
    })

    # Mengubah data teks (kategori) menjadi angka biner secara otomatis
    data = pd.get_dummies(data)

    # Menyelaraskan struktur kolom input dengan kolom yang dipelajari model ML
    for col in model.feature_names_in_:
        if col not in data.columns:
            data[col] = 0

    # Mengurutkan susunan kolom agar presisi dengan model
    data = data[model.feature_names_in_]

    # Melakukan prediksi kelulusan
    hasil = model.predict(data)
    st.divider()

    # Menampilkan output hasil prediksi ke layar
    if hasil[0] == 1:
        st.success("✅ Prediksi: SISWA LULUS")
    else:
        st.error("⚠️ Prediksi: SISWA TIDAK LULUS")
