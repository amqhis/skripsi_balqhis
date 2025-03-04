import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from streamlit_option_menu import option_menu

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Insight Predict",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat CSS kustom
def local_css():
    st.markdown(
        """
        <style>
            /* Mengatur sidebar */
            [data-testid="stSidebar"] {
                background-color: #cdd4b1;
            }
            
            /* Mengatur background main content */
            [data-testid="stAppViewContainer"] {
                background-color: #feecd0;
            }

            /* Mengatur logo agar lebih kecil dan di tengah */
            .logo-container {
                text-align: center;
                margin-top: -20px;
            }

            .logo-container img {
                width: 100px;
            }

            /* Styling untuk teks agar lebih kontras */
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #4a4a4a !important;
            }

            /* Styling untuk teks sambutan */
            .welcome-text {
                font-size: 22px;
                font-weight: bold;
                color: #4a4a4a;
                text-align: center;
                background-color: #f5deb3;
                padding: 15px;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi utama
def main():
    local_css()  # Memanggil CSS

    # Sidebar dengan menu navigasi
    with st.sidebar:
        st.markdown(
            """
            <div class='logo-container'>
                <img src="https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png" 
                     alt="Logo">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Menu navigasi
        selected = option_menu(
            None,
            ['🏠 Home', '📖 Tentang Aplikasi', '📂 Upload Data', 
             '⚙️ Preprocessing Data', '📊 Visualisasi Data Historis', '🔮 Prediksi Masa Depan'],
            menu_icon='cast',
            icons=['house', 'info-circle', 'cloud-upload', 'filter', 'bar-chart', 'line-chart'],
            default_index=0,
            styles={
                "container": {
                    "padding": "0px",
                    "background-color": "#cdd4b1"
                },
                "icon": {
                    "color": "#4a4a4a",
                    "font-size": "17px"
                },
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px",
                    "color": "#4a4a4a",
                    "--hover-color": "#b5c19a"
                },
                "nav-link-selected": {
                    "background-color": "#6b8e23",
                    "color": "white"
                },
            }
        )

    # **📌 Kata Sambutan Muncul di Homepage**
    if selected == '🏠 Home':
        st.markdown("<div class='welcome-text'>🎉 Hai, Selamat Datang di Insight Predict! 🎉</div>", unsafe_allow_html=True)
        st.write("""
        Insight Predict adalah aplikasi yang dirancang untuk membantu Anda menganalisis dan memprediksi tren data menggunakan metode berbasis **Machine Learning**.  
        Dengan fitur interaktif yang mudah digunakan, aplikasi ini memungkinkan Anda memahami pola data historis dan melakukan prediksi masa depan dengan lebih akurat.  
        📊 **Ayo mulai jelajahi fitur yang tersedia!** 🚀
        """)

    # **📌 Tentang Aplikasi**
    elif selected == '📖 Tentang Aplikasi':
        st.title('📊 Insight Predict - Analisis Data Cerdas')
        st.write("""
        Insight Predict adalah platform analisis berbasis data yang dirancang untuk membantu pengguna dalam memahami tren data dan membuat prediksi berdasarkan data historis.  
        Aplikasi ini menggunakan **model Machine Learning canggih** untuk memberikan hasil prediksi yang lebih akurat dan dapat diandalkan. Dengan visualisasi interaktif, pengguna dapat dengan mudah menginterpretasikan data, mengevaluasi hasil analisis, dan mengambil keputusan berbasis data dengan lebih baik.
        """)

        # **📌 Terms & Conditions**
        with st.expander("📜 Syarat & Ketentuan Penggunaan"):
            st.markdown("""
            **Jenis Data yang Dapat Digunakan:**  
            - Format **Excel (.xlsx)**
            - Harus memiliki kolom berikut:  
                - **Tanggal**  
                - **Jenis Produk**  
                - **Quantity**   
            """)

    # **📌 Fitur Upload Data**
    elif selected == '📂 Upload Data':
        st.title('📂 Upload Data Anda')
        uploaded_file = st.file_uploader("Pilih file Excel (.xlsx) untuk dianalisis", type=['xlsx'])

        if uploaded_file is not None:
            # Membaca file Excel
            df = pd.read_excel(uploaded_file)
            st.write("### 📊 Data yang Diupload")
            st.dataframe(df)
            # Menyimpan ke session state
            st.session_state['original_data'] = df
            st.success('✅ Data berhasil diunggah!')

    elif selected == '⚙️ Preprocessing Data':
        st.title("⚙️ Preprocessing Data")
        st.write("Fitur ini akan digunakan untuk membersihkan dan mempersiapkan data sebelum dilakukan analisis lebih lanjut.")
        st.warning("🚧 Fitur ini masih dalam tahap pengembangan.")

    elif selected == '📊 Visualisasi Data Historis':
        st.title("📈 Visualisasi Data Historis")
        st.write("Di sini, Anda akan melihat tren data berdasarkan histori yang telah diunggah.")
        st.warning("🚧 Fitur ini masih dalam tahap pengembangan.")

    elif selected == '🔮 Prediksi Masa Depan':
        st.title("🔮 Prediksi Masa Depan")
        st.write("Gunakan model Machine Learning untuk memprediksi tren di masa depan.")
        st.warning("🚧 Fitur ini masih dalam tahap pengembangan.")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
