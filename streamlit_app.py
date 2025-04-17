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
                background: linear-gradient(to bottom, #2E3B4E, #1E293B);
                border-right: 1px solid #3B4D61;
            }
            
            /* Mengatur background main content */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #F5F7FA, #E4E7EB);
            }
            
            /* Styling untuk header aplikasi */
            .header-container {
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                border-left: 5px solid #3366FF;
            }
            
            /* Mengatur logo agar lebih kecil dan di tengah */
            .logo-container {
                text-align: center;
                margin: 15px 0 30px 0;
                padding-bottom: 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .logo-container img {
                width: 120px;
                filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.2));
                transition: transform 0.3s ease;
            }
            
            .logo-container img:hover {
                transform: scale(1.05);
            }
            
            /* Styling untuk teks */
            h1 {
                color: #1E293B !important;
                font-size: 32px !important;
                font-weight: 700 !important;
                margin-bottom: 10px !important;
            }
            
            h2, h3 {
                color: #334155 !important;
                font-weight: 600 !important;
            }
            
            p, div, span {
                color: #475569 !important;
            }
            
            /* Styling untuk kartu konten */
            .content-card {
                background-color: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
                margin-bottom: 20px;
                border-top: 4px solid #3366FF;
            }
            
            /* Styling untuk teks sambutan */
            .welcome-text {
                font-size: 18px;
                line-height: 1.6;
                color: #475569;
                text-align: center;
                background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
                padding: 25px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #3366FF;
            }
            
            /* Styling untuk tombol */
            .stButton > button {
                background-color: #3366FF;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
                border: none;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background-color: #2952CC;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            /* Styling untuk metrik/KPI */
            .metric-container {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
            }
            
            /* Footer styling */
            .footer {
                text-align: center;
                padding: 20px;
                color: #94A3B8 !important;
                font-size: 14px;
                margin-top: 40px;
                border-top: 1px solid #E2E8F0;
            }
            
            /* Animation for page transitions */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .fade-in {
                animation: fadeIn 0.5s ease-in-out;
            }
            
            /* Styling untuk sidebar navigasi yang aktif */
            .nav-link-active {
                background-color: rgba(51, 102, 255, 0.8) !important;
                color: white !important;
                border-radius: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk menampilkan header
def display_header(title):
    st.markdown(f"""
        <div class="header-container fade-in">
            <h1>{title}</h1>
        </div>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan konten dalam kartu
def content_card(content, key=None):
    st.markdown(f"""
        <div class="content-card fade-in">
            {content}
        </div>
    """, unsafe_allow_html=True)

# Fungsi utama
def main():
    local_css()  # Memanggil CSS
    
    # Sidebar dengan menu navigasi
    with st.sidebar:
        st.markdown(
            """
            <div class='logo-container'>
                <img src="https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png" 
                     alt="Logo Insight Predict">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Menu navigasi
        selected = option_menu(
            None,
            ['🏠 Home', '📖 Tentang Aplikasi', '📂 Upload Data', 
             '⚙️ Preprocessing Data', '📊 Visualisasi Data', '🔮 Prediksi'],
            menu_icon='cast',
            icons=['house-fill', 'info-circle-fill', 'cloud-upload-fill', 
                   'gear-fill', 'bar-chart-fill', 'graph-up'],
            default_index=0,
            styles={
                "container": {
                    "padding": "10px",
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "18px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "8px 0",
                    "padding": "10px",
                    "border-radius": "6px",
                    "color": "rgba(255, 255, 255, 0.9)",
                    "--hover-color": "rgba(255, 255, 255, 0.1)"
                },
                "nav-link-selected": {
                    "background-color": "rgba(51, 102, 255, 0.8)",
                    "color": "white",
                    "font-weight": "600"
                },
            }
        )
        
        # Tambahkan info versi di bagian bawah sidebar
        st.sidebar.markdown("""
            <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; 
                        color: rgba(255, 255, 255, 0.5); font-size: 12px; text-align: center;">
                Insight Predict v1.0.0
            </div>
        """, unsafe_allow_html=True)

    # Konten halaman berdasarkan menu yang dipilih
    if selected == '🏠 Home':
        display_header("Selamat Datang di Insight Predict")
        
        st.markdown("""
            <div class="welcome-text fade-in">
                <span style="font-size: 24px; font-weight: 600;">👋 Hai!</span><br>
                Insight Predict membantu Anda menganalisis dan memprediksi data dengan mudah dan akurat.
                Mulai eksplorasi data Anda sekarang!
            </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan beberapa metrik contoh
        col1, col2, col3 = st.columns(3)
        with col1:
            content_card("""
                <h3 style="text-align: center; color: #3366FF !important;">📊 Visualisasi</h3>
                <p style="text-align: center;">
                    Lihat data Anda dalam bentuk visual yang interaktif dan informatif
                </p>
            """)
        
        with col2:
            content_card("""
                <h3 style="text-align: center; color: #3366FF !important;">⚙️ Preprocessing</h3>
                <p style="text-align: center;">
                    Siapkan data Anda dengan langkah-langkah preprocessing yang lengkap
                </p>
            """)
            
        with col3:
            content_card("""
                <h3 style="text-align: center; color: #3366FF !important;">🔮 Prediksi</h3>
                <p style="text-align: center;">
                    Gunakan model prediktif untuk melihat tren masa depan dari data Anda
                </p>
            """)
            
        # Langkah-langkah penggunaan
        content_card("""
            <h2>Cara Menggunakan Insight Predict</h2>
            <ol>
                <li>Upload data Anda melalui menu <b>Upload Data</b></li>
                <li>Lakukan preprocessing data pada menu <b>Preprocessing Data</b></li>
                <li>Eksplorasi visualisasi data di menu <b>Visualisasi Data</b></li>
                <li>Dapatkan prediksi masa depan di menu <b>Prediksi</b></li>
            </ol>
        """)
        
    elif selected == '📖 Tentang Aplikasi':
        display_header("Tentang Insight Predict")
        
        content_card("""
            <h2>Apa itu Insight Predict?</h2>
            <p>Insight Predict adalah aplikasi analisis data dan prediksi yang dirancang untuk membantu Anda mengolah data, 
            memvisualisasikan tren, dan membuat prediksi masa depan berdasarkan data historis.</p>
            
            <h3>Fitur Utama:</h3>
            <ul>
                <li>Upload dan kelola data dengan mudah</li>
                <li>Preprocessing data lengkap untuk persiapan analisis</li>
                <li>Visualisasi data interaktif untuk memudahkan analisis</li>
                <li>Model prediksi canggih untuk forecasting data masa depan</li>
                <li>Interface yang user-friendly dan mudah digunakan</li>
            </ul>
        """)
        
    elif selected == '📂 Upload Data':
        display_header("Upload Data")
        
        content_card("""
            <p>Upload file data Anda untuk memulai analisis.</p>
            <p>Format yang didukung: CSV, Excel, dan JSON.</p>
        """)
        
        uploaded_file = st.file_uploader("Pilih file data Anda", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            st.success("File berhasil diunggah!")
            # Di sini Anda dapat menambahkan kode untuk memproses file yang diunggah
            
    elif selected == '⚙️ Preprocessing Data':
        display_header("Preprocessing Data")
        
        content_card("""
            <p>Lakukan preprocessing pada data Anda sebelum dianalisis lebih lanjut.</p>
        """)
        
        # Tambahkan kode preprocessing data di sini
        
    elif selected == '📊 Visualisasi Data':
        display_header("Visualisasi Data")
        
        content_card("""
            <p>Eksplorasi data Anda dengan visualisasi interaktif.</p>
        """)
        
        # Tambahkan kode visualisasi data di sini
        
    elif selected == '🔮 Prediksi':
        display_header("Prediksi Data")
        
        content_card("""
            <p>Gunakan model prediktif untuk memprediksi data masa depan.</p>
        """)
        
        # Tambahkan kode prediksi data di sini
    
    # Footer
    st.markdown("""
        <div class="footer">
            © 2025 Insight Predict | Dibuat dengan ❤️ menggunakan Streamlit
        </div>
    """, unsafe_allow_html=True)

# Jalankan aplikasi
if __name__ == "__main__":
    main()






