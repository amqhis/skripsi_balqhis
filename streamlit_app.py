import streamlit as st

# Set page configuration
st.set_page_config(page_title="Good Prediction", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Mengatur latar belakang utama */
        .main {
            background-color: #d4edda;
        }
        
        /* Mengatur bagian sidebar */
        .sidebar-container {
            background-color: #155724;
            padding: 20px;
            height: 100vh;
            border-radius: 10px;
        }

        /* Mengatur judul dan menu navigasi */
        .sidebar-container h1, .sidebar-container label {
            color: #c3e6cb;  /* Warna hijau muda yang nude */
            text-align: center;
        }

        /* Mengatur radio button */
        div[data-testid="stRadio"] {
            text-align: center;
        }

        div[data-testid="stRadio"] label {
            color: #c3e6cb !important; /* Warna hijau muda yang nude */
            font-size: 16px;
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# Layout with two columns
col1, col2 = st.columns([1, 3])

# Sidebar (Left Section)
with col1:
    st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    st.markdown("<h1>Good Prediction</h1>", unsafe_allow_html=True)
    
    option = st.radio(
        "Pilih Fitur:",
        ["Tentang Aplikasi", "Upload Data", "Preprocessing Data", "Visualisasi Data Historis", "Prediksi Masa Depan"],
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main Content (Right Section)
with col2:
    if option == "Tentang Aplikasi":
        st.header("Tentang Aplikasi")
        st.write("Aplikasi ini digunakan untuk melakukan prediksi penjualan menggunakan model XGBoost.")

    elif option == "Upload Data":
        st.header("Upload Data")
        st.write("Silakan upload file CSV untuk memulai analisis.")

    elif option == "Preprocessing Data":
        st.header("Preprocessing Data")
        st.write("Melakukan pembersihan dan transformasi data sebelum analisis.")

    elif option == "Visualisasi Data Historis":
        st.header("Visualisasi Data Historis")
        st.write("Menampilkan grafik berdasarkan data historis.")

    elif option == "Prediksi Masa Depan":
        st.header("Prediksi Masa Depan")
        st.write("Melakukan prediksi jumlah penjualan untuk periode mendatang.")
