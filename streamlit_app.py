import streamlit as st

# Mengatur warna background halaman
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #d4edda;
}
[data-testid="stHeader"] {
    background-color: #155724;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# Menampilkan judul aplikasi
st.title("🌱 Good Prediction")

# Deskripsi aplikasi
st.write(
    "Selamat datang di aplikasi *Good Prediction*! "
    "Aplikasi ini dirancang untuk melakukan prediksi menggunakan model XGBoost dengan optimasi Grid Search. "
    "Gunakan fitur yang tersedia untuk mengunggah data, memprosesnya, dan mendapatkan hasil prediksi yang akurat."
)
