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
                background: linear-gradient(180deg, #2C6E49, #4C956C);
                border-right: 1px solid rgba(255,255,255,0.1);
            }
            
            /* Mengatur background main content */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(to right, #FEFAE0, #FAEDCD);
            }
            
            /* Atur container dasar */
            .main-container {
                background-color: rgba(255, 255, 255, 0.85);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
                margin-bottom: 20px;
                border-left: 5px solid #2C6E49;
                animation: slideIn 0.5s ease-out;
            }
            
            /* Animasi untuk konten */
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Mengatur logo */
            .logo-container {
                text-align: center;
                margin: 10px auto 30px auto;
                padding-bottom: 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            .logo-container img {
                width: 140px;
                filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
                transition: all 0.3s ease;
            }
            
            .logo-container img:hover {
                transform: scale(1.05);
                filter: drop-shadow(0 6px 8px rgba(0, 0, 0, 0.2));
            }
            
            /* Styling untuk heading */
            h1 {
                color: #2C6E49 !important;
                font-size: 36px !important;
                font-weight: 700 !important;
                margin-bottom: 20px !important;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
            }
            
            h2 {
                color: #3E5C50 !important;
                font-size: 28px !important;
                font-weight: 600 !important;
                margin-top: 30px !important;
                margin-bottom: 15px !important;
            }
            
            h3, h4, h5, h6 {
                color: #445E50 !important;
                font-weight: 600 !important;
                margin-top: 20px !important;
            }
            
            /* Styling untuk teks */
            p, div, span {
                color: #1F2937 !important;
                line-height: 1.7 !important;
                font-size: 16px !important;
            }
            
            /* Styling untuk cards */
            .info-card {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                margin-bottom: 20px;
                border-top: 4px solid #4C956C;
                transition: transform 0.3s ease;
            }
            
            .info-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }
            
            /* Styling untuk teks sambutan */
            .welcome-text {
                font-size: 24px !important;
                font-weight: bold;
                color: #2C6E49 !important;
                text-align: center;
                background: linear-gradient(to right, #E9EDC9, #CCD5AE);
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                border-left: 5px solid #2C6E49;
            }
            
            /* Buttons styling */
            .stButton > button {
                background-color: #4C956C !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 10px 20px !important;
                font-weight: 500 !important;
                border: none !important;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background-color: #2C6E49 !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
                transform: translateY(-2px);
            }
            
            /* File uploader styling */
            [data-testid="stFileUploader"] {
                border: 2px dashed #4C956C;
                border-radius: 10px;
                padding: 20px;
                background-color: rgba(204, 213, 174, 0.3);
            }
            
            /* Success/warning message styling */
            .st-emotion-cache-16idsys {
                border-radius: 10px;
                padding: 20px !important;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            }
            
            /* Stat container styling */
            .stat-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                height: 100%;
                border-bottom: 4px solid #4C956C;
            }
            
            .stat-number {
                font-size: 28px !important;
                font-weight: 700 !important;
                color: #2C6E49 !important;
            }
            
            .stat-label {
                font-size: 16px !important;
                color: #4a4a4a !important;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #F9F7F0;
                border-radius: 8px 8px 0 0;
                padding: 10px 20px;
                color: #4a4a4a;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #4C956C !important;
                color: white !important;
            }
            
            /* Footer styling */
            .footer {
                text-align: center;
                padding: 20px;
                color: #4a4a4a;
                border-top: 1px solid #e0e0e0;
                margin-top: 50px;
                font-size: 14px;
            }
            
            /* Chart containers */
            .chart-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                margin: 20px 0;
            }
            
            /* Table styling */
            .dataframe {
                font-size: 14px !important;
            }

            /* Expander styling */
            .streamlit-expanderHeader {
                background-color: #F9F7F0;
                border-radius: 8px;
                padding: 10px !important;
                font-weight: 600 !important;
                color: #3E5C50 !important;
            }
            
            .streamlit-expanderContent {
                background-color: white;
                border-radius: 0 0 8px 8px;
                padding: 15px !important;
                border: 1px solid #F0F0F0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk membuat container utama
def main_container(content, key=None):
    st.markdown(f'<div class="main-container">{content}</div>', unsafe_allow_html=True)

# Fungsi untuk membuat info card
def info_card(title, content, icon):
    card_html = f"""
    <div class="info-card">
        <h3 style="font-size: 20px; margin-top: 0;"><span style="font-size: 24px; margin-right: 10px;">{icon}</span> {title}</h3>
        <p>{content}</p>
    </div>
    """
    return card_html

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
             '⚙️ Preprocessing Data', '📊 Visualisasi Data', '🔮 Prediksi'],
            menu_icon='cast',
            icons=['house-fill', 'info-circle-fill', 'cloud-upload-fill', 
                  'gear-fill', 'bar-chart-fill', 'graph-up-arrow'],
            default_index=0,
            styles={
                "container": {
                    "padding": "10px",
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "white",
                    "font-size": "18px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "8px 0",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "color": "white",
                    "--hover-color": "rgba(255, 255, 255, 0.2)"
                },
                "nav-link-selected": {
                    "background-color": "rgba(255, 255, 255, 0.2)",
                    "color": "white",
                    "font-weight": "600"
                },
            }
        )
        
        # Tambahkan versi aplikasi
        st.sidebar.markdown("""
            <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; 
                        color: rgba(255, 255, 255, 0.7); font-size: 14px; text-align: center;">
                Insight Predict v1.0.0
            </div>
        """, unsafe_allow_html=True)

    # **📌 Kata Sambutan Muncul di Homepage**
    if selected == '🏠 Home':
        st.markdown("<h1 style='text-align: center;'>Insight Predict</h1>", unsafe_allow_html=True)
        st.markdown("<div class='welcome-text'>🎉 Hai, Selamat Datang di Insight Predict! 🎉</div>", unsafe_allow_html=True)
        
        # Penjelasan singkat dalam container utama
        main_container("""
            <p style='text-align: center; font-size: 18px !important;'>
                Aplikasi berbasis data untuk analisis dan prediksi yang dirancang untuk membantu Anda 
                memahami pola data historis dan melakukan prediksi masa depan dengan akurat
                menggunakan teknologi <b>Machine Learning</b>.
            </p>
        """)
        
        # Fitur utama dalam cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(info_card(
                "Analisis Data", 
                "Upload dan analisis data bisnis Anda dengan mudah dan cepat.",
                "📊"
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown(info_card(
                "Visualisasi Interaktif", 
                "Lihat tren dan pola data Anda melalui visualisasi yang informatif.",
                "📈"
            ), unsafe_allow_html=True)
            
        with col3:
            st.markdown(info_card(
                "Prediksi Akurat", 
                "Gunakan model Machine Learning untuk memprediksi tren masa depan.",
                "🔮"
            ), unsafe_allow_html=True)
        
        # Langkah-langkah penggunaan
        main_container("""
            <h2>Cara Menggunakan Insight Predict</h2>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px; margin: 10px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <div style="font-size: 36px; color: #2C6E49; text-align: center;">1</div>
                    <p style="text-align: center;"><b>Upload Data</b><br>Unggah data Excel yang akan dianalisis</p>
                </div>
                <div style="flex: 1; min-width: 200px; margin: 10px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <div style="font-size: 36px; color: #2C6E49; text-align: center;">2</div>
                    <p style="text-align: center;"><b>Preprocessing</b><br>Siapkan data untuk analisis lebih lanjut</p>
                </div>
                <div style="flex: 1; min-width: 200px; margin: 10px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <div style="font-size: 36px; color: #2C6E49; text-align: center;">3</div>
                    <p style="text-align: center;"><b>Visualisasi</b><br>Eksplorasi tren data historis Anda</p>
                </div>
                <div style="flex: 1; min-width: 200px; margin: 10px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <div style="font-size: 36px; color: #2C6E49; text-align: center;">4</div>
                    <p style="text-align: center;"><b>Prediksi</b><br>Dapatkan prediksi untuk masa depan</p>
                </div>
            </div>
        """)

    # **📌 Tentang Aplikasi**
    elif selected == '📖 Tentang Aplikasi':
        st.title('Tentang Insight Predict')
        
        main_container("""
            <p style='font-size: 18px !important;'>
                Insight Predict adalah platform analisis data dan prediksi yang dirancang untuk membantu Anda memahami
                tren bisnis dan membuat prediksi berdasarkan data historis.
            </p>
            <p>
                Aplikasi ini menggunakan <b>model Machine Learning canggih</b> untuk memberikan hasil prediksi yang 
                akurat dan dapat diandalkan. Dengan visualisasi interaktif, Anda dapat dengan mudah menginterpretasikan
                data, mengevaluasi hasil analisis, dan mengambil keputusan berbasis data dengan lebih baik.
            </p>
        """)
        
        # Fitur utama
        st.subheader("Fitur Utama")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(info_card(
                "Data Processing", 
                "Penanganan data otomatis dengan validasi, pembersihan dan normalisasi untuk hasil yang akurat.",
                "⚙️"
            ), unsafe_allow_html=True)
            
            st.markdown(info_card(
                "Visualisasi Data", 
                "Grafik dan diagram interaktif untuk membantu memahami tren dan pola dalam data Anda.",
                "📊"
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown(info_card(
                "Prediksi Machine Learning", 
                "Model XGBoost yang canggih untuk memprediksi tren masa depan berdasarkan data historis.",
                "🤖"
            ), unsafe_allow_html=True)
            
            st.markdown(info_card(
                "Insight Bisnis", 
                "Dapatkan wawasan tentang performa bisnis Anda dan prediksi untuk perencanaan masa depan.",
                "💡"
            ), unsafe_allow_html=True)

        # Terms & Conditions
        with st.expander("📜 Syarat & Ketentuan Penggunaan"):
            st.markdown("""
            #### Jenis Data yang Dapat Digunakan:
            - Format **Excel (.xlsx)**
            - Harus memiliki kolom berikut:
                - **Tanggal Pembelian**
                - **Jenis Strapping Band**
                - **Quantity**
            
            #### Cara Menggunakan Data:
            1. Data harus berisi informasi penjualan historis
            2. Format tanggal harus konsisten
            3. Data numerik harus valid
            
            #### Batasan Aplikasi:
            - Optimal untuk dataset hingga 10.000 baris
            - Prediksi berfokus pada 2 tahun ke depan
            """)

    # **📌 Fitur Upload Data**
    elif selected == '📂 Upload Data':
        st.title('Upload Data Anda')
        
        main_container("""
            <p>
                Upload file Excel (.xlsx) yang berisi data historis untuk dianalisis. Pastikan file Anda memiliki
                kolom <b>Tanggal Pembelian</b>, <b>Jenis Strapping Band</b>, dan <b>Quantity</b>.
            </p>
        """)
        
        # Upload widget dengan styling
        uploaded_file = st.file_uploader("Pilih file Excel (.xlsx) untuk dianalisis", type=['xlsx'])

        if uploaded_file is not None:
            # Membaca file Excel
            df = pd.read_excel(uploaded_file)
            
            # Preview data dalam container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("### 📊 Data yang Diupload")
            st.dataframe(df)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Menyimpan ke session state
            st.session_state['original_data'] = df
            
            # Konfirmasi dengan animasi dan styling
            st.success('✅ Data berhasil diunggah dan siap untuk dianalisis!')
            
            # Tampilkan ringkasan data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="stat-container">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Total Baris Data</div>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="stat-container">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Kolom Data</div>
                </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
                
            with col3:
                if 'Tanggal Pembelian' in df.columns:
                    date_range = "{} hingga {}".format(
                        df['Tanggal Pembelian'].min().strftime('%d %b %Y'),
                        df['Tanggal Pembelian'].max().strftime('%d %b %Y')
                    )
                else:
                    date_range = "N/A"
                    
                st.markdown("""
                <div class="stat-container">
                    <div class="stat-number">📆</div>
                    <div class="stat-label">Rentang Data:<br>{}</div>
                </div>
                """.format(date_range), unsafe_allow_html=True)
            
            # Tampilkan langkah selanjutnya
            st.info('📌 Langkah selanjutnya: Lakukan preprocessing data di menu "⚙️ Preprocessing Data"')

    elif selected == '⚙️ Preprocessing Data':
        st.title("Preprocessing Data")
        
        if 'original_data' in st.session_state:
            df = st.session_state['original_data'].copy()
            
            # Data sebelum preprocessing dalam container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.write("### 📌 Data Sebelum Preprocessing")
            st.dataframe(df)
            st.markdown('</div>', unsafe_allow_html=True)
    
            # Cek kolom yang dibutuhkan
            required_columns = ['Tanggal Pembelian', 'Quantity']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"⚠️ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
            else:
                # Buat tabs untuk setiap langkah preprocessing
                preprocessing_tabs = st.tabs(["1️⃣ Data Cleaning", "2️⃣ Feature Engineering", "3️⃣ Normalisasi", "4️⃣ Hasil Akhir"])
                
                with preprocessing_tabs[0]:
                    st.subheader("Data Cleaning")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(info_card(
                            "Konversi Tanggal", 
                            "Mengubah kolom 'Tanggal Pembelian' ke format datetime untuk analisis temporal.",
                            "📅"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(info_card(
                            "Penghapusan Missing Values", 
                            "Menghapus baris dengan nilai yang hilang untuk memastikan kualitas data.",
                            "🧹"
                        ), unsafe_allow_html=True)
                    
                    # Proses data cleaning
                    df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
                    df.dropna(subset=['Tanggal Pembelian'], inplace=True)
                    df.dropna(inplace=True)
                    
                    st.success("✅ Data cleaning berhasil dilakukan!")
                
                with preprocessing_tabs[1]:
                    st.subheader("Feature Engineering")
                    
                    # Proses feature engineering
                    df['Year'] = df['Tanggal Pembelian'].dt.year
                    df['Month'] = df['Tanggal Pembelian'].dt.month
                    df['Quarter'] = df['Tanggal Pembelian'].dt.quarter
                    df['WeekOfYear'] = df['Tanggal Pembelian'].dt.isocalendar().week
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.write("Data dengan fitur tambahan:")
                    st.dataframe(df[['Tanggal Pembelian', 'Year', 'Month', 'Quarter', 'WeekOfYear', 'Quantity']].head(10))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(info_card(
                            "Ekstraksi Waktu", 
                            "Mengekstrak tahun, bulan, kuartal, dan minggu dari tanggal untuk analisis musiman.",
                            "🗓️"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(info_card(
                            "Persiapan Fitur", 
                            "Menyiapkan fitur untuk model machine learning dengan ekstraksi tanggal dan agregasi data.",
                            "🔧"
                        ), unsafe_allow_html=True)
                    
                    st.success("✅ Feature engineering berhasil dilakukan!")
                
                with preprocessing_tabs[2]:
                    st.subheader("Normalisasi Data")
                    
                    # Visualisasi sebelum normalisasi
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df['Tanggal Pembelian'], df['Quantity'], label="Quantity (Original)", color="blue")
                    ax.set_title("Data Sebelum Normalisasi")
                    ax.set_xlabel("Tanggal Pembelian")
                    ax.set_ylabel("Quantity (Original)")
                    ax.grid(True, alpha=0.3)
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Proses normalisasi
                    scaler = MinMaxScaler()
                    numeric_cols = ['Quantity']
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    
                    # Visualisasi setelah normalisasi
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df['Tanggal Pembelian'], df['Quantity'], label="Quantity (Normalized)", color="green")
                    ax.set_title("Data Setelah Normalisasi")
                    ax.set_xlabel("Tanggal Pembelian")
                    ax.set_ylabel("Quantity (Normalized)")
                    ax.grid(True, alpha=0.3)
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success("✅ Normalisasi data berhasil dilakukan!")
                
                with preprocessing_tabs[3]:
                    st.subheader("Hasil Akhir Preprocessing")
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.write("### ✅ Data Setelah Preprocessing")
                    st.dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Simpan hasil preprocessing ke session state
                    st.session_state['processed_data'] = df
                    
                    st.success("✅ Preprocessing data selesai! Data siap untuk visualisasi dan prediksi.")
                    st.info("📌 Langkah selanjutnya: Lihat visualisasi data di menu '📊 Visualisasi Data' atau buat prediksi di menu '🔮 Prediksi'")
        else:
            st.warning("⚠️ Harap unggah data terlebih dahulu di bagian '📂 Upload Data'.")
        
    elif selected == '📊 Visualisasi Data Historis':
        st.title("Visualisasi Data Historis")
        if 'original_data' in st.session_state:
            df = st.session_state['original_data']
            df['Month'] = df['Tanggal Pembelian'].dt.to_period('M').astype(str)
            df_monthly = df.groupby('Month')['Quantity'].sum().reset_index()
            max_month = df_monthly.loc[df_monthly['Quantity'].idxmax()]
            min_month = df_monthly.loc[df_monthly['Quantity'].idxmin()]
            st.write(f"Penjualan tertinggi terjadi pada {max_month['Month']} sebanyak {max_month['Quantity']} unit")
            st.write(f"Penjualan terendah terjadi pada {min_month['Month']} sebanyak {min_month['Quantity']} unit")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_monthly['Month'], df_monthly['Quantity'], marker='o', linestyle='-', color='b')
            plt.xticks(rotation=45)
            plt.title('Tren Penjualan Per Bulan')
            st.pyplot(fig)
            
            df_yearly = df.groupby(df['Tanggal Pembelian'].dt.year)['Quantity'].sum()
            fig, ax = plt.subplots()
            df_yearly.plot(kind='bar', color='skyblue', ax=ax)
            plt.title('Total Penjualan per Tahun')
            st.pyplot(fig)
            
            sales_by_type = df.groupby('Jenis Strapping Band')['Quantity'].sum()
            fig, ax = plt.subplots()
            sales_by_type.plot(kind='bar', color='green', ax=ax)
            plt.title('Penjualan Berdasarkan Jenis Strapping Band')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Upload data terlebih dahulu!")

    
    elif selected == '🔮 Prediksi Masa Depan':
        st.title("🔮 Prediksi Masa Depan")
    
        if 'processed_data' in st.session_state and not st.session_state['processed_data'].empty:
            df = st.session_state['processed_data']
    
            # Pastikan ada data historis yang benar
            if df.empty:
                st.warning("⚠️ Data historis kosong. Pastikan Anda telah melakukan preprocessing.")
            else:
                # Agregasi Data Bulanan
                df_monthly = df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
    
                # Pastikan ada data setelah preprocessing
                if df_monthly.empty:
                    st.warning("⚠️ Data setelah preprocessing kosong. Coba ulangi preprocessing.")
                else:
                    # Buat kolom 'Date' agar bisa dipakai di visualisasi
                    df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
    
                    # Normalisasi Data
                    scaler = MinMaxScaler()
                    df_monthly['Quantity_Scaled'] = scaler.fit_transform(df_monthly[['Quantity']])
    
                    # Persiapan Data untuk Model
                    X = df_monthly[['Year', 'Month']]
                    y = df_monthly['Quantity_Scaled']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
                    # Training Model XGBoost
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                    model.fit(X_train, y_train)
    
                    # Prediksi Masa Depan
                    future_dates = pd.DataFrame({
                        'Year': np.repeat(range(2024, 2026), 12),
                        'Month': list(range(1, 13)) * 2
                    })
    
                    future_dates['Date'] = pd.to_datetime(future_dates[['Year', 'Month']].assign(day=1))
    
                    # Prediksi
                    future_pred_scaled = model.predict(future_dates[['Year', 'Month']])
                    future_pred_actual = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
    
                    future_results = pd.DataFrame({
                        'Year': future_dates['Year'].values,
                        'Month': future_dates['Month'].values,
                        'Predicted_Quantity': future_pred_actual,
                        'Date': future_dates['Date']
                    })
    
                    # **Visualisasi Prediksi**
                    st.subheader("📈 Prediksi Penjualan 2024-2025")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df_monthly['Date'], df_monthly['Quantity'], label="Data Historis", marker='o', color='blue')
                    ax.plot(future_results['Date'], future_results['Predicted_Quantity'], label="Prediksi 2024-2025", marker='s', color='red')
                    ax.set_xlabel("Bulan")
                    ax.set_ylabel("Total Quantity")
                    ax.set_title("Prediksi Kuantitas Januari 2024 - Desember 2025")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)
    
                    # **Tampilkan Tabel Hasil Prediksi**
                    st.write("### 📋 Tabel Hasil Prediksi")
                    st.dataframe(future_results)
    
        else:
            st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu!")

  

# Menjalankan aplikasi
if __name__ == "__main__":
    main()






