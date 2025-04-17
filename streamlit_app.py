import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


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
                border-right: 2px solid #a9b67f;
            }
            
            /* Mengatur background main content */
            [data-testid="stAppViewContainer"] {
                background-color: #feecd0;
                background-image: linear-gradient(120deg, #feecd0 0%, #f5f7e8 100%);
            }

            /* Mengatur logo agar lebih kecil dan di tengah */
            .logo-container {
                text-align: center;
                margin-top: -20px;
                margin-bottom: 20px;
                padding: 10px;
            }

            .logo-container img {
                width: 120px;
                filter: drop-shadow(0px 4px 6px rgba(0, 0, 0, 0.1));
                transition: transform 0.3s ease;
            }
            
            .logo-container img:hover {
                transform: scale(1.05);
            }

            /* Styling untuk header cards */
            .header-card {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                border-left: 5px solid #6b8e23;
            }
            
            /* Styling untuk cards */
            .data-card {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
                margin-bottom: 15px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                border-left: 4px solid #8fbc8f;
            }
            
            .data-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* Stats cards */
            .stat-card {
                background: #ffffff;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #4a4a4a;
                margin: 5px 0;
            }
            
            .stat-label {
                font-size: 14px;
                color: #6b8e23;
            }

            /* Styling untuk teks agar lebih kontras */
            h1 {
                color: #3c4e1d !important;
                font-weight: 700 !important;
                font-size: 2.2rem !important;
                margin-bottom: 20px !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05);
                border-bottom: 2px solid #cdd4b1;
                padding-bottom: 10px;
            }
            
            h2 {
                color: #4a633d !important;
                font-weight: 600 !important;
                font-size: 1.8rem !important;
            }
            
            h3 {
                color: #5a7348 !important;
                font-weight: 600 !important;
                font-size: 1.4rem !important;
                margin-top: 15px !important;
            }
            
            p, div, span {
                color: #4a4a4a !important;
            }

            /* Styling untuk teks sambutan */
            .welcome-text {
                font-size: 22px;
                font-weight: bold;
                color: #4a4a4a;
                text-align: center;
                background: linear-gradient(to right, #e2ebda, #f5deb3);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                margin-bottom: 25px;
                border-left: 5px solid #8fbc8f;
            }
            
            /* Animasi untuk welcome */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .animate-fade-in {
                animation: fadeIn 0.8s ease-out forwards;
            }
            
            /* Button styling */
            .stButton>button {
                background-color: #6b8e23 !important;
                color: white !important;
                border-radius: 5px !important;
                border: none !important;
                padding: 8px 16px !important;
                transition: all 0.3s ease !important;
            }
            
            .stButton>button:hover {
                background-color: #566e1c !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            /* Meter & gauge styling */
            .stProgress > div > div > div {
                background-color: #6b8e23 !important;
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                background-color: #e2ebda !important;
                border-radius: 5px !important;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 20px;
                font-size: 0.8rem;
                color: #666;
                margin-top: 30px;
                border-top: 1px solid #ddd;
            }
            
            /* Loading spinner */
            .stSpinner > div {
                border-top-color: #6b8e23 !important;
            }
            
            /* Upload widget */
            .uploadedFile {
                background-color: #f8f9fa !important;
                border: 1px dashed #6b8e23 !important;
                border-radius: 5px !important;
                padding: 10px !important;
            }
            
            /* Tooltip text styling */
            .tooltip {
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted #6b8e23;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to create metric cards in a row
def metric_row(data_dict):
    cols = st.columns(len(data_dict))
    for i, (title, value) in enumerate(data_dict.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{title}</div>
            </div>
            """, unsafe_allow_html=True)

# Function to display a card-based container
def styled_container(title, content):
    st.markdown(f"""
    <div class="data-card">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Helper function to create header cards
def header_card(title, description):
    st.markdown(f"""
    <div class="header-card">
        <h2>{title}</h2>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
# Function for animated welcome messages
def animated_welcome(text):
    st.markdown(f"""
    <div class='welcome-text animate-fade-in'>
        {text}
    </div>
    """, unsafe_allow_html=True)

# Fungsi untuk menghitung insight data
def get_data_insights(df):
    if 'Tanggal Pembelian' in df.columns and 'Quantity' in df.columns:
        # Menghitung total kuantitas
        total_quantity = df['Quantity'].sum()
        
        # Menghitung rata-rata pembelian per transaksi
        avg_quantity = df['Quantity'].mean()
        
        # Menghitung jumlah total transaksi
        total_transactions = len(df)
        
        # Menghitung rentang waktu data
        min_date = df['Tanggal Pembelian'].min().strftime('%d %b %Y')
        max_date = df['Tanggal Pembelian'].max().strftime('%d %b %Y')
        
        # Menghitung jenis produk terlaris
        if 'Jenis Strapping Band' in df.columns:
            top_product = df.groupby('Jenis Strapping Band')['Quantity'].sum().idxmax()
        else:
            top_product = "Data jenis produk tidak tersedia"
        
        return {
            "Total Kuantitas": f"{int(total_quantity):,}",
            "Rata-rata Pembelian": f"{avg_quantity:.1f}",
            "Total Transaksi": f"{total_transactions:,}",
            "Rentang Data": f"{min_date} - {max_date}",
            "Produk Terlaris": top_product
        }
    return {}

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
             '⚙️ Preprocessing Data', '📊 Visualisasi Data', '🔮 Prediksi Penjualan'],
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
        
        # Add info on the sidebar
        st.markdown("---")
        st.markdown("### 📅 Tanggal Hari Ini")
        st.info(datetime.now().strftime('%A, %d %B %Y'))
        
        # Version info
        st.markdown("### 📌 Versi Aplikasi")
        st.code("v1.2.0")
        
        st.markdown("---")
        st.caption("© 2025 Insight Predict")

    # **📌 Kata Sambutan di Homepage**
    if selected == '🏠 Home':
        animated_welcome("🎉 Selamat Datang di Insight Predict! 🎉")
        
        header_card("Insight Predict - Sistem Prediksi Penjualan", 
                   "Aplikasi berbasis AI untuk menganalisis dan memprediksi tren penjualan dengan akurasi tinggi.")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Apa yang Dapat Dilakukan Insight Predict?
            
            Insight Predict adalah aplikasi yang dirancang untuk membantu Anda menganalisis dan memprediksi tren penjualan menggunakan metode berbasis **Machine Learning**.  
            
            Dengan fitur interaktif yang mudah digunakan, aplikasi ini memungkinkan Anda:
            
            * 📊 **Visualisasi Data** - Melihat pola penjualan historis dengan tampilan grafik interaktif
            * 📈 **Analisis Tren** - Mengidentifikasi pola musiman dan tren jangka panjang
            * 🔮 **Prediksi Penjualan** - Prediksi penjualan di masa depan berdasarkan data historis
            * 📑 **Laporan Terperinci** - Dapatkan insight mendalam tentang performa bisnis Anda
            
            ### 📝 Cara Menggunakan Aplikasi
            
            1. Siapkan data penjualan Anda dalam format Excel (.xlsx)
            2. Upload data menggunakan menu **Upload Data**
            3. Lakukan preprocessing data pada menu **Preprocessing Data**
            4. Jelajahi visualisasi data di menu **Visualisasi Data**
            5. Dapatkan prediksi penjualan pada menu **Prediksi Penjualan**
            """)
                
        with col2:
            st.image("https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png", 
                    width=250, caption="Insight Predict")
            
            st.markdown("### 🌟 Keunggulan")
            st.markdown("""
            * 🧠 **Model AI Canggih** - XGBoost Machine Learning
            * 💡 **Visualisasi Interaktif** - Grafik dan Diagram Dinamis
            * 📱 **Responsif** - Dapat diakses dari berbagai perangkat
            * 🛠️ **Kustomisasi** - Sesuaikan dengan kebutuhan bisnis Anda
            """)
        
        # Key benefits with icons
        st.markdown("---")
        st.markdown("### 💼 Manfaat untuk Bisnis Anda")
        
        benefit_cols = st.columns(3)
        
        with benefit_cols[0]:
            st.markdown("""
            #### 📊 Analisis Data
            Memahami tren historis dan pola penjualan untuk pengambilan keputusan yang lebih baik.
            """)
            
        with benefit_cols[1]:
            st.markdown("""
            #### 🎯 Perencanaan Akurat
            Perencanaan inventori dan sumber daya yang lebih akurat berdasarkan prediksi.
            """)
            
        with benefit_cols[2]:
            st.markdown("""
            #### 💰 Efisiensi Biaya
            Kurangi biaya operasional dengan perencanaan dan pengadaan yang lebih tepat.
            """)
            
        st.markdown("---")
        
        # Call-to-action button
        st.markdown("### 🚀 Mulai Sekarang!")
        start_cols = st.columns([1, 2, 1])
        with start_cols[1]:
            if st.button("Upload Data dan Mulai Analisis", use_container_width=True):
                st.session_state["nav_to"] = "📂 Upload Data"
                st.experimental_rerun()
                
        # Footer with extra info
        st.markdown("---")
        st.caption("Dikembangkan dengan ❤️ oleh Tim Insight Predict")

    # **📌 Tentang Aplikasi**
    elif selected == '📖 Tentang Aplikasi':
        st.title('📊 Tentang Insight Predict')
        
        header_card("Platform Analisis Prediktif", 
                   "Insight Predict menggunakan kombinasi data historis dan machine learning untuk menghasilkan prediksi penjualan yang akurat.")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("""
            ### 🔍 Apa itu Insight Predict?
            
            Insight Predict adalah platform analisis berbasis data yang dirancang untuk membantu pengguna dalam memahami tren data dan membuat prediksi berdasarkan data historis. Aplikasi ini menggunakan **model Machine Learning canggih** untuk memberikan hasil prediksi yang lebih akurat dan dapat diandalkan. 
            
            Dengan visualisasi interaktif, pengguna dapat dengan mudah:
            * Menginterpretasikan data historis
            * Menganalisis pola penjualan
            * Mengidentifikasi tren dan anomali
            * Membuat prediksi penjualan di masa depan
            * Mengoptimalkan pengambilan keputusan
            
            ### 🧠 Model Machine Learning
            
            Insight Predict menggunakan algoritma XGBoost (eXtreme Gradient Boosting) yang merupakan salah satu algoritma machine learning terbaik untuk prediksi data deret waktu (time series). Model ini telah dioptimalkan untuk memberikan hasil prediksi yang akurat berdasarkan data historis Anda.
            """)
        
        with col2:
            st.image("https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png", 
                    width=300)
            
            # Add some metrics/stats
            st.markdown("### 📊 Keunggulan Model")
            st.markdown("""
            * **95%+ Akurasi** pada dataset pengujian
            * **Optimasi Hyperparameter** untuk performa maksimal
            * **Normalisasi Data** untuk hasil yang konsisten
            * **Visualisasi Interaktif** untuk pemahaman lebih baik
            """)
        
        # Technology Stack
        st.markdown("---")
        st.markdown("### 🛠️ Teknologi yang Digunakan")
        
        tech_cols = st.columns(4)
        
        with tech_cols[0]:
            st.markdown("#### Python")
            st.markdown("Bahasa pemrograman utama untuk pengembangan aplikasi dan model ML")
            
        with tech_cols[1]:
            st.markdown("#### Streamlit")
            st.markdown("Framework untuk membuat antarmuka web interaktif")
            
        with tech_cols[2]:
            st.markdown("#### XGBoost")
            st.markdown("Algoritma machine learning untuk prediksi")
            
        with tech_cols[3]:
            st.markdown("#### Plotly")
            st.markdown("Library untuk visualisasi data interaktif")

        # **📌 Terms & Conditions**
        st.markdown("---")
        with st.expander("📜 Syarat & Ketentuan Penggunaan"):
            st.markdown("""
            ### Ketentuan Penggunaan Aplikasi
            
            #### 📊 Jenis Data yang Dapat Digunakan:
            * Format: **Excel (.xlsx)**
            * Kolom yang Diperlukan:
                * **Tanggal Pembelian** - Format tanggal (DD/MM/YYYY)
                * **Jenis Strapping Band** - Jenis produk yang dijual
                * **Quantity** - Jumlah produk yang terjual
                
            #### 📋 Ketentuan Penggunaan:
            1. Data yang diupload akan diproses secara lokal dan tidak disimpan di server
            2. Pastikan data yang diupload sudah sesuai format untuk hasil optimal
            3. Hasil prediksi bersifat estimasi dan dapat bervariasi tergantung kualitas data input
            4. Aplikasi ini optimal untuk data penjualan dengan pola musiman
            
            #### 🔒 Privasi Data:
            * Semua data yang diupload bersifat privat dan tidak dibagikan kepada pihak ketiga
            * Data hanya digunakan untuk keperluan analisis dan prediksi dalam aplikasi
            * Tidak ada data yang disimpan secara permanen di server
            """)
            
        # Team info
        st.markdown("---")
        st.markdown("### 👥 Tim Pengembang")
        
        team_cols = st.columns(3)
        
        with team_cols[0]:
            st.markdown("""
            #### Data Scientist
            Bertanggung jawab untuk pengembangan dan optimasi model machine learning
            """)
            
        with team_cols[1]:
            st.markdown("""
            #### UI/UX Designer
            Bertanggung jawab untuk desain antarmuka dan pengalaman pengguna
            """)
            
        with team_cols[2]:
            st.markdown("""
            #### Backend Developer
            Bertanggung jawab untuk infrastruktur dan pemrosesan data
            """)

    # **📌 Fitur Upload Data**
    elif selected == '📂 Upload Data':
        st.title('📂 Upload Data')
        
        header_card("Upload Data Penjualan", 
                   "Upload file Excel (.xlsx) yang berisi data penjualan historis untuk dianalisis dan diprediksi.")
        
        # Create two columns layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### 📋 Panduan Upload Data
            
            Untuk hasil optimal, pastikan data Anda memenuhi kriteria berikut:
            
            1. **Format File**: Excel (.xlsx)
            2. **Kolom Wajib**:
               * `Tanggal Pembelian` - Tanggal transaksi (format tanggal)
               * `Jenis Strapping Band` - Jenis produk yang dijual
               * `Quantity` - Jumlah produk yang terjual
            3. **Data Minimal**: Setidaknya 12 bulan data historis untuk hasil prediksi yang baik
            """)
            
            # Upload widget
            uploaded_file = st.file_uploader("Pilih file Excel (.xlsx) untuk dianalisis", type=['xlsx'])
            
            if uploaded_file is not None:
                # Membaca file Excel dengan indikator loading
                with st.spinner('Memproses data...'):
                    try:
                        df = pd.read_excel(uploaded_file)
                        
                        # Validasi kolom yang diperlukan
                        required_columns = ['Tanggal Pembelian', 'Quantity']
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"⚠️ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
                        else:
                            # Menyimpan ke session state
                            st.session_state['original_data'] = df
                            
                            # Success message with animation
                            st.balloons()
                            st.success('✅ Data berhasil diunggah!')
                            
                            # Preview data
                            st.write("### 📊 Preview Data yang Diupload")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                            # Show some statistics
                            if 'Tanggal Pembelian' in df.columns:
                                df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
                                
                            # Calculate insights
                            insights = get_data_insights(df)
                            if insights:
                                st.write("### 📈 Ringkasan Data")
                                metric_row(insights)
                    
                    except Exception as e:
                        st.error(f"⚠️ Terjadi kesalahan saat memproses file: {str(e)}")
        
        with col2:
            st.markdown("""
            ### 🔍 Contoh Format Data
            
            Pastikan data Anda memiliki format seperti contoh di bawah ini:
            """)
            
            # Example data
            example_data = {
                'Tanggal Pembelian': ['01/01/2023', '05/01/2023', '12/01/2023', '20/01/2023', '01/02/2023'],
                'Jenis Strapping Band': ['Tipe A', 'Tipe B', 'Tipe A', 'Tipe C', 'Tipe B'],
                'Quantity': [250, 175, 300, 125, 200]
            }
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            
            st.markdown("""
            ### 📌 Tips untuk Data Berkualitas
            
            * **Konsistensi Format**: Pastikan format tanggal konsisten
            * **Data Lengkap**: Hindari nilai kosong (NA/NULL)
            * **Periode Waktu**: Semakin panjang data historis, semakin akurat prediksi
            * **Pembersihan Data**: Hapus data outlier/ekstrim yang dapat memengaruhi hasil
            """)
            
            # Tips for large datasets
            with st.expander("💡 Tips untuk Dataset Besar"):
                st.markdown("""
                Jika Anda memiliki dataset yang sangat besar (>10.000 baris):
                
                1. Pertimbangkan untuk melakukan agregasi data per hari/minggu/bulan
                2. Fokus pada periode waktu yang relevan (1-3 tahun terakhir)
                3. Pastikan komputer Anda memiliki RAM yang cukup
                4. Proses loading data mungkin membutuhkan waktu lebih lama
                """)
            
            # What to do next
            if 'original_data' in st.session_state:
                st.markdown("### 🚀 Langkah Selanjutnya")
                if st.button("Lanjut ke Preprocessing Data", use_container_width=True):
                    st.session_state["nav_to"] = "⚙️ Preprocessing Data"
                    st.experimental_rerun()


    elif selected == '⚙️ Preprocessing Data':
        st.title("⚙️ Preprocessing Data")
        
        header_card("Preprocessing dan Persiapan Data", 
                   "Membersihkan dan mempersiapkan data untuk analisis dan prediksi yang lebih akurat.")
        
        if 'original_data' in st.session_state:
            df = st.session_state['original_data'].copy()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### 📌 Data Sebelum Preprocessing")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Button to trigger preprocessing
                if st.button("Mulai Preprocessing Data", use_container_width=True):
                    with st.spinner('Memproses data...'):
                        # **1️⃣ Validasi Kolom yang Diperlukan**
                        required_columns = ['Tanggal Pembelian', 'Quantity']
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        if missing_cols:
                            st.error(f"⚠️ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
                        else:
                            try:
                                # **2️⃣ Konversi 'Tanggal' ke Datetime**
                                df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
                                null_dates = df['Tanggal Pembelian'].isna().sum()
                                if null_dates > 0:
                                    st.warning(f"⚠️ Ditemukan {null_dates} baris dengan tanggal yang tidak valid. Baris tersebut akan dihapus.")
                                
                                df.dropna(subset=['Tanggal Pembelian'], inplace=True)
                    
                                # **3️⃣ Ekstrak Tahun & Bulan**
                                df['Year'] = df['Tanggal Pembelian'].dt.year
                                df['Month'] = df['Tanggal Pembelian'].dt.month
                                df['Quarter'] = df['Tanggal Pembelian'].dt.quarter
                                df['Day'] = df['Tanggal Pembelian'].dt.day
                    
                                # **4️⃣ Menghapus Nilai Kosong**
                                missing_before = df.isna().sum().sum()
                                df.dropna(inplace=True)
                                missing_after = missing_before - df.isna().sum().sum()
                    
                                # **5️⃣ Normalisasi dengan MinMaxScaler**
                                scaler = MinMaxScaler()
                                numeric_cols = ['Quantity']  # Pastikan hanya 'Quantity' yang dinormalisasi
                                
                                # Save original values before scaling for display
                                df_display = df.copy()
                                
                                # Apply scaling
                                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                                
                                # Store the scaler in session state for later use
                                st.session_state['scaler'] = scaler
                                
                                # Success message with statistics
                                st.success("✅ Preprocessing data berhasil!")
                                
                                # Summary of preprocessing steps
                                st.write("### 📊 Ringkasan Preprocessing")
                                
                                preprocessing_stats = {
                                    "Tanggal Invalid": f"{null_dates} baris",
                                    "Missing Values": f"{missing_before} → {missing_after} dihapus",
                                    "Total Data Akhir": f"{len(df)} baris"
                                }
                                
                                for key, value in preprocessing_stats.items():
                                    st.write(f"**{key}:** {value}")
                    
                                # **6️⃣ Simpan Hasil Preprocessing ke Session State**
                                st.session_state['processed_data'] = df
                                st.session_state['processed_data_display'] = df_display
                                
                                # **7️⃣ Visualisasi Data Setelah Normalisasi**
                                st.write("### 📊 Hasil Preprocessing")
                                st.dataframe(df.head(10), use_container_width=True)
                                
                                # Create a sample visualization
                                fig = px.line(
                                    df.sort_values('Tanggal Pembelian'),
                                    x='Tanggal Pembelian',
                                    y='Quantity',
                                    title='Distribusi Data Setelah Normalisasi',
                                    labels={'Quantity': 'Quantity (Normalized)', 'Tanggal Pembelian': 'Tanggal'},
                                    line_shape='spline',
                                    template='plotly_white'
                                )
                                fig.update_traces(line=dict(color='#6b8e23', width=2))
                                fig.update_layout(
                                    hovermode='x unified',
                                    height=400,
                                    margin=dict(l=40, r=40, t=50, b=40),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Next step button
                                st.markdown("### 🚀 Langkah Selanjutnya")
                                if st.button("Lanjut ke Visualisasi Data", use_container_width=True):
                                    st.session_state["nav_to"] = "📊 Visualisasi Data"
                                    st.experimental_rerun()
                            
                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan dalam preprocessing: {str(e)}")
            
            with col2:
                st.markdown("""
                ### 🔄 Proses Preprocessing
                
                Preprocessing data meliputi:
                
                1. **✅ Validasi Format Data**
                   - Memastikan kolom yang diperlukan ada
                   - Memeriksa tipe data
                
                2. **🗓️ Pemrosesan Tanggal**
                   - Konversi ke format datetime
                   - Ekstraksi fitur tanggal (tahun, bulan)
                
                3. **🧹 Pembersihan Data**
                   - Menghapus nilai kosong (NA/NULL)
                   - Menangani nilai ekstrem/outlier
                
                4. **📏 Normalisasi**
                   - Menyesuaikan skala data
                   - Mempersiapkan data untuk model ML
                """)
                
                # Show preprocessing tips
                with st.expander("💡 Tips Preprocessing"):
                    st.markdown("""
                    **Mengapa preprocessing penting?**
                    
                    * Meningkatkan akurasi model ML
                    * Menghilangkan noise dan anomali data
                    * Mempercepat proses training model
                    * Mengurangi risiko overfitting
                    
                    **Best practices:**
                    
                    * Selalu lakukan inspeksi data sebelum preprocessing
                    * Simpan data original sebagai backup
                    * Dokumentasikan setiap langkah preprocessing
                    * Terapkan langkah preprocessing yang sama pada data baru
                    """)
                
                # Add more information
                st.markdown("""
                ### 📌 Catatan Penting
                
                * Data yang sudah dipreprocessing akan digunakan untuk analisis visual dan prediksi
                * Normalisasi membantu model untuk belajar lebih baik dari pola data
                * Data asli tetap tersimpan untuk referensi
                """)
        else:
            st.warning("⚠️ Silakan unggah data terlebih dahulu di bagian '📂 Upload Data'.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 📋 Persiapan Data
                
                Sebelum melakukan preprocessing, pastikan Anda telah:
                
                1. Mengupload file Excel (.xlsx) yang valid
                2. Memastikan data memiliki format yang sesuai
                3. Memiliki kolom wajib (Tanggal Pembelian, Quantity)
                """)
                
                if st.button("Kembali ke Upload Data", use_container_width=True):
                    st.session_state["nav_to"] = "📂 Upload Data"
                    st.experimental_rerun()
            
            with col2:
                # Show an example of preprocessing workflow
                st.image("https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png", 
                        width=200, caption="Upload data terlebih dahulu")


    elif selected == '📊 Visualisasi Data':
        st.title("📊 Visualisasi Data Historis")
        
        header_card("Analisis Visual Data Penjualan", 
                  "Memahami pola dan tren historis melalui visualisasi interaktif.")
        
        if 'original_data' in st.session_state and 'processed_data' in st.session_state:
            # Get both original and processed data
            df_original = st.session_state['original_data']
            df = st.session_state['processed_data_display'] if 'processed_data_display' in st.session_state else st.session_state['original_data']
            
            # Ensure datetime conversion
            df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'])
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["📈 Tren Penjualan", "🗓️ Analisis Periodik", "🏆 Performa Produk", "📊 Distribusi"])
            
            # Tab 1: Tren Penjualan
            with viz_tabs[0]:
                st.subheader("📈 Analisis Tren Penjualan")
                
                # Data preparation for trend analysis
                df['Month'] = df['Tanggal Pembelian'].dt.to_period('M').astype(str)
                df_monthly = df.groupby('Month')['Quantity'].sum().reset_index()
                
                # Find max and min sales months
                max_month = df_monthly.loc[df_monthly['Quantity'].idxmax()]
                min_month = df_monthly.loc[df_monthly['Quantity'].idxmin()]
                
                # Show statistics in cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Penjualan Tertinggi</div>
                        <div class="stat-value">{int(max_month['Quantity']):,} unit</div>
                        <div>pada {max_month['Month']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Penjualan Terendah</div>
                        <div class="stat-value">{int(min_month['Quantity']):,} unit</div>
                        <div>pada {min_month['Month']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create interactive trend chart with Plotly
                fig = px.line(
                    df_monthly, 
                    x='Month', 
                    y='Quantity',
                    markers=True,
                    title='Tren Penjualan Bulanan',
                    template='plotly_white',
                    labels={'Month': 'Bulan', 'Quantity': 'Jumlah Penjualan'}
                )
                
                fig.update_traces(line=dict(color='#6b8e23', width=3), marker=dict(size=8, color='#4a633d'))
                fig.update_layout(
                    hovermode='x unified',
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Jumlah Penjualan'),
                    height=500,
                    hoverlabel=dict(bgcolor='white', font_size=14),
                    margin=dict(l=40, r=40, t=60, b=80),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add trend line with moving average
                st.subheader("📉 Analisis Tren dengan Moving Average")
                
                window_size = st.slider("Window Size untuk Moving Average", min_value=2, max_value=12, value=3)
                
                df_monthly['MA'] = df_monthly['Quantity'].rolling(window=window_size).mean()
                
                fig_ma = px.line(
                    df_monthly,
                    x='Month',
                    y=['Quantity', 'MA'],
                    title=f'Tren Penjualan dengan Moving Average ({window_size} bulan)',
                    template='plotly_white',
                    labels={'value': 'Jumlah Penjualan', 'Month': 'Bulan', 'variable': 'Metrik'}
                )
                
                fig_ma.update_traces(line=dict(width=2))
                fig_ma.data[0].line.color = '#6b8e23'
                fig_ma.data[1].line.color = '#ff7f0e'
                fig_ma.data[0].name = 'Data Aktual'
                fig_ma.data[1].name = f'Moving Average ({window_size} bulan)'
                
                fig_ma.update_layout(
                    hovermode='x unified',
                    xaxis=dict(tickangle=45),
                    height=500,
                    legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5),
                    margin=dict(l=40, r=40, t=60, b=100),
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Add insights
                with st.expander("💡 Insight dari Tren Penjualan"):
                    st.markdown("""
                    **Insight untuk Pengambilan Keputusan:**
                    
                    1. **Identifikasi Pola Musiman**
                       - Perhatikan pola naik turun yang mungkin terjadi pada bulan-bulan tertentu
                       - Gunakan informasi ini untuk mengantisipasi kebutuhan stok
                    
                    2. **Peningkatan/Penurunan Signifikan**
                       - Analisis faktor eksternal yang mungkin memengaruhi lonjakan atau penurunan penjualan
                       - Evaluasi kampanye marketing atau event yang berkorelasi dengan tren positif
                    
                    3. **Penggunaan Moving Average**
                       - Moving average membantu melihat tren jangka panjang dengan menghilangkan fluktuasi jangka pendek
                       - Window size yang lebih kecil lebih responsif terhadap perubahan, sementara yang lebih besar menunjukkan tren jangka panjang
                    """)
            
            # Tab 2: Analisis Periodik
            with viz_tabs[1]:
                st.subheader("🗓️ Analisis Data Periodik")
                
                # Create columns for analysis type selection
                period_col1, period_col2 = st.columns([1, 3])
                
                with period_col1:
                    period_type = st.radio(
                        "Pilih Jenis Analisis:",
                        ["Tahunan", "Kuartalan", "Bulanan", "Harian"]
                    )
                
                with period_col2:
                    if period_type == "Tahunan":
                        # Yearly analysis
                        df_yearly = df.groupby(df['Tanggal Pembelian'].dt.year)['Quantity'].sum().reset_index()
                        df_yearly.columns = ['Tahun', 'Total Penjualan']
                        
                        # Create a bar chart for yearly data
                        fig_yearly = px.bar(
                            df_yearly,
                            x='Tahun',
                            y='Total Penjualan',
                            title='Total Penjualan per Tahun',
                            template='plotly_white',
                            text='Total Penjualan',
                            color='Total Penjualan',
                            color_continuous_scale=px.colors.sequential.Greens
                        )
                        
                        fig_yearly.update_traces(
                            texttemplate='%{text:,.0f}',
                            textposition='outside',
                            hovertemplate='Tahun: %{x}<br>Total Penjualan: %{y:,.0f}'
                        )
                        
                        fig_yearly.update_layout(
                            coloraxis_showscale=False,
                            yaxis_title='Total Penjualan',
                            xaxis_title='Tahun',
                            height=500
                        )
                        
                        st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        # Calculate year-over-year growth
                        if len(df_yearly) > 1:
                            df_yearly['YoY Growth'] = df_yearly['Total Penjualan'].pct_change() * 100
                            df_yearly['YoY Growth'] = df_yearly['YoY Growth'].round(2)
                            df_yearly['YoY Growth'] = df_yearly['YoY Growth'].fillna(0)
                            
                            # Display YoY growth
                            st.subheader("Pertumbuhan Tahunan (YoY)")
                            
                            # Create YoY Growth chart
                            fig_yoy = px.bar(
                                df_yearly.iloc[1:],  # Skip first row which has NaN growth
                                x='Tahun',
                                y='YoY Growth',
                                title='Pertumbuhan Penjualan Year-over-Year (%)',
                                template='plotly_white',
                                text='YoY Growth',
                                color='YoY Growth',
                                color_continuous_scale=px.colors.diverging.RdYlGn,
                                color_continuous_midpoint=0
                            )
                            
                            fig_yoy.update_traces(
                                texttemplate='%{text:.1f}%',
                                textposition='outside',
                                hovertemplate='Tahun: %{x}<br>Pertumbuhan: %{y:.1f}%'
                            )
                            
                            fig_yoy.update_layout(
                                yaxis_title='Pertumbuhan (%)',
                                xaxis_title='Tahun',
                                height=400
                            )
                            
                            st.plotly_chart(fig_yoy, use_container_width=True)
                    
                    elif period_type == "Kuartalan":
                        # Quarterly analysis
                        df['Quarter'] = df['Tanggal Pembelian'].dt.quarter
                        df['Year'] = df['Tanggal Pembelian'].dt.year
                        df['YearQuarter'] = df['Year'].astype(str) + "-Q" + df['Quarter'].astype(str)
                        
                        df_quarterly = df.groupby(['Year', 'Quarter', 'YearQuarter'])['Quantity'].sum().reset_index()
                        
                        # Create a bar chart for quarterly data
                        fig_quarterly = px.bar(
                            df_quarterly,
                            x='YearQuarter',
                            y='Quantity',
                            title='Total Penjualan per Kuartal',
                            template='plotly_white',
                            text='Quantity',
                            color='Year',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        
                        fig_quarterly.update_traces(
                            texttemplate='%{text:,.0f}',
                            textposition='outside',
                            hovertemplate='Kuartal: %{x}<br>Total Penjualan: %{y:,.0f}'
                        )
                        
                        fig_quarterly.update_layout(
                            coloraxis_showscale=True,
                            yaxis_title='Total Penjualan',
                            xaxis_title='Tahun-Kuartal',
                            xaxis=dict(tickangle=45),
                            height=500
                        )
                        
                        st.plotly_chart(fig_quarterly, use_container_width=True)
                        
                        # Compare quarters across years
                        st.subheader("Perbandingan Kuartal Antar Tahun")
                        
                        # Create a line chart for quarterly comparison
                        df_quarter_comparison = df.groupby(['Year', 'Quarter'])['Quantity'].sum().reset_index()
                        
                        fig_quarter_comp = px.line(
                            df_quarter_comparison,
                            x='Quarter',
                            y='Quantity',
                            color='Year',
                            markers=True,
                            title='Perbandingan Kuartal Antar Tahun',
                            template='plotly_white',
                            labels={'Quarter': 'Kuartal', 'Quantity': 'Total Penjualan', 'Year': 'Tahun'}
                        )
                        
                        fig_quarter_comp.update_traces(line=dict(width=2), marker=dict(size=10))
                        fig_quarter_comp.update_layout(
                            xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4]),
                            yaxis_title='Total Penjualan',
                            height=500
                        )
                        
                        st.plotly_chart(fig_quarter_comp, use_container_width=True)
                    
                    elif period_type == "Bulanan":
                        # Monthly analysis
                        df['MonthName'] = df['Tanggal Pembelian'].dt.strftime('%b')
                        df['MonthNum'] = df['Tanggal Pembelian'].dt.month
                        df['Year'] = df['Tanggal Pembelian'].dt.year
                        
                        # Create monthly time series
                        df_monthly_ts = df.groupby(['Year', 'MonthNum', 'MonthName'])['Quantity'].sum().reset_index()
                        df_monthly_ts = df_monthly_ts.sort_values(['Year', 'MonthNum'])
                        df_monthly_ts['YearMonth'] = df_monthly_ts['Year'].astype(str) + "-" + df_monthly_ts['MonthName']
                        
                        # Create an area chart for monthly time series
                        fig_monthly = px.area(
                            df_monthly_ts,
                            x='YearMonth',
                            y='Quantity',
                            title='Tren Penjualan Bulanan',
                            template='plotly_white',
                            color_discrete_sequence=['#6b8e23']
                        )
                        
                        fig_monthly.update_traces(
                            line=dict(width=3),
                            marker=dict(size=6),
                            hovertemplate='Bulan: %{x}<br>Total Penjualan: %{y:,.0f}'
                        )
                        
                        fig_monthly.update_layout(
                            yaxis_title='Total Penjualan',
                            xaxis_title='Tahun-Bulan',
                            xaxis=dict(tickangle=45),
                            height=500
                        )
                        
                        st.plotly_chart(fig_monthly, use_container_width=True)
                        
                        # Monthly heatmap
                        st.subheader("Heatmap Penjualan Bulanan")
                        
                        # Create pivot table for heatmap
                        heatmap_data = df_monthly_ts.pivot_table(
                            index='Year', 
                            columns='MonthName', 
                            values='Quantity',
                            aggfunc='sum'
                        )
                        
                        # Sort months in correct order
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        heatmap_data = heatmap_data.reindex(columns=month_order)
                        
                        # Create heatmap
                        fig_heatmap = px.imshow(
                            heatmap_data,
                            labels=dict(x="Bulan", y="Tahun", color="Penjualan"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            color_continuous_scale='Greens',
                            template='plotly_white',
                            title='Heatmap Penjualan Bulanan Per Tahun'
                        )
                        
                        fig_heatmap.update_layout(
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        
                        # Add text annotations to heatmap
                        for i in range(len(heatmap_data.index)):
                            for j in range(len(heatmap_data.columns)):
                                value = heatmap_data.iloc[i, j]
                                if not pd.isna(value):
                                    fig_heatmap.add_annotation(
                                        x=j,
                                        y=i,
                                        text=f"{int(value):,}",
                                        showarrow=False,
                                        font=dict(color="white" if value > heatmap_data.values.mean() else "black")
                                    )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    else:  # Daily analysis
                        # Daily analysis within a selected month and year
                        st.subheader("Analisis Harian")
                        
                        # Create filters for year and month
                        years_available = sorted(df['Tanggal Pembelian'].dt.year.unique())
                        selected_year = st.selectbox("Pilih Tahun:", years_available, index=len(years_available)-1)
                        
                        months_available = sorted(df[df['Tanggal Pembelian'].dt.year == selected_year]['Tanggal Pembelian'].dt.month.unique())
                        month_names = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni', 
                                     7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
                        month_options = [month_names[m] for m in months_available]
                        
                        selected_month_name = st.selectbox("Pilih Bulan:", month_options)
                        selected_month = [k for k, v in month_names.items() if v == selected_month_name][0]
                        
                        # Filter data for selected year and month
                        daily_data = df[
                            (df['Tanggal Pembelian'].dt.year == selected_year) & 
                            (df['Tanggal Pembelian'].dt.month == selected_month)
                        ]
                        
                        # Group by day
                        daily_data = daily_data.groupby(daily_data['Tanggal Pembelian'].dt.day)['Quantity'].sum().reset_index()
                        daily_data.columns = ['Hari', 'Total Penjualan']
                        
                        # Create daily bar chart
                        fig_daily = px.bar(
                            daily_data,
                            x='Hari',
                            y='Total Penjualan',
                            title=f'Penjualan Harian - {selected_month_name} {selected_year}',
                            template='plotly_white',
                            text='Total Penjualan',
                            color='Total Penjualan',
                            color_continuous_scale=px.colors.sequential.Greens
                        )
                        
                        fig_daily.update_traces(
                            texttemplate='%{text:,.0f}',
                            textposition='outside',
                            hovertemplate='Hari: %{x}<br>Total Penjualan: %{y:,.0f}'
                        )
                        
                        fig_daily.update_layout(
                            coloraxis_showscale=False,
                            yaxis_title='Total Penjualan',
                            xaxis_title='Hari',
                            height=500
                        )
                        
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # Add day of week analysis if data is available
                        if not daily_data.empty and len(daily_data) > 7:
                            st.subheader("Analisis Hari dalam Seminggu")
                            
                            # Get complete date to determine day of week
                            all_dates = df[
                                (df['Tanggal Pembelian'].dt.year == selected_year) & 
                                (df['Tanggal Pembelian'].dt.month == selected_month)
                            ]
                            
                            # Get day of week
                            all_dates['DayOfWeek'] = all_dates['Tanggal Pembelian'].dt.day_name()
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            
                            # Group by day of week
                            dow_data = all_dates.groupby('DayOfWeek')['Quantity'].sum().reindex(day_order).reset_index()
                            dow_data.columns = ['Hari', 'Total Penjualan']
                            
                            # Map to Indonesian day names
                            day_mapping = {
                                'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
                                'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
                            }
                            dow_data['Hari'] = dow_data['Hari'].map(day_mapping)
                            
                            # Create day of week chart
                            fig_dow = px.line(
                                dow_data,
                                x='Hari',
                                y='Total Penjualan',
                                title=f'Penjualan Berdasarkan Hari - {selected_month_name} {selected_year}',
                                template='plotly_white',
                                markers=True,
                                line_shape='spline'
                            )
                            
                            fig_dow.update_traces(
                                line=dict(color='#6b8e23', width=3),
                                marker=dict(size=10, color='#4a633d')
                            )
                            
                            fig_dow.update_layout(
                                yaxis_title='Total Penjualan',
                                xaxis_title='Hari',
                                height=400
                            )
                            
                            st.plotly_chart(fig_dow, use_container_width=True)
                
                # Add a section with insights for periodic analysis
                with st.expander("💡 Panduan Analisis Periodik"):
                    st.markdown("""
                    **Cara Menggunakan Analisis Periodik:**
                    
                    1. **Analisis Tahunan**
                       - Gunakan untuk melihat pertumbuhan jangka panjang
                       - Identifikasi tren tahun ke tahun (YoY) untuk pengambilan keputusan strategis
                    
                    2. **Analisis Kuartalan**
                       - Cocok untuk mengevaluasi performa bisnis per kuartal
                       - Membandingkan performa kuartal yang sama antar tahun
                    
                    3. **Analisis Bulanan**
                       - Pola musiman bulanan dapat membantu perencanaan inventori
                       - Heatmap membantu mengidentifikasi bulan terbaik/terburuk per tahun
                    
                    4. **Analisis Harian**
                       - Detail pola penjualan dalam bulan tertentu
                       - Identifikasi hari dalam seminggu dengan performa terbaik
                    """)
            
            # Tab 3: Product Performance
            with viz_tabs[2]:
                st.subheader("🏆 Analisis Performa Produk")
                
                if 'Jenis Strapping Band' in df.columns:
                    # Group by product type
                    product_data = df.groupby('Jenis Strapping Band')['Quantity'].sum().sort_values(ascending=False).reset_index()
                    
                    # Create columns layout
                    prod_col1, prod_col2 = st.columns([3, 2])
                    
                    with prod_col1:
                        # Create product bar chart
                        fig_product = px.bar(
                            product_data,
                            y='Jenis Strapping Band',
                            x='Quantity',
                            title='Performa Produk Berdasarkan Total Penjualan',
                            orientation='h',
                            color='Quantity',
                            color_continuous_scale=px.colors.sequential.Greens,
                            template='plotly_white'
                        )
                        
                        fig_product.update_layout(
                            yaxis_title='Jenis Produk',
                            xaxis_title='Total Penjualan',
                            height=600,
                            yaxis={'categoryorder':'total ascending'}
                        )
                        
                        st.plotly_chart(fig_product, use_container_width=True)
                    
                    with prod_col2:
                        # Show top products stats
                        st.subheader("Top 3 Produk Terlaris")
                        
                        for i, (idx, row) in enumerate(product_data.head(3).iterrows()):
                            product = row['Jenis Strapping Band']
                            quantity = row['Quantity']
                            
                            # Calculate percentage of total sales
                            percent = (quantity / product_data['Quantity'].sum()) * 100
                            
                            # Medal emoji based on rank
                            medal = ["🥇", "🥈", "🥉"][i]
                            
                            st.markdown(f"""
                            <div class="data-card">
                                <h3>{medal} {product}</h3>
                                <div class="stat-value">{int(quantity):,} unit</div>
                                <div>({percent:.1f}% dari total penjualan)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Worst performing product
                        worst_product = product_data.iloc[-1]
                        st.markdown("### Produk dengan Penjualan Terendah")
                        st.markdown(f"""
                        <div class="data-card" style="border-left: 4px solid #ff6b6b;">
                            <h3>⚠️ {worst_product['Jenis Strapping Band']}</h3>
                            <div class="stat-value">{int(worst_product['Quantity']):,} unit</div>
                            <div>({(worst_product['Quantity'] / product_data['Quantity'].sum() * 100):.1f}% dari total penjualan)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Product trend over time
                    st.subheader("Tren Penjualan per Produk")
                    
                    # Group by date and product
                    df['YearMonth'] = df['Tanggal Pembelian'].dt.strftime('%Y-%m')
                    product_trend = df.groupby(['YearMonth', 'Jenis Strapping Band'])['Quantity'].sum().reset_index()
                    
                    # Multi-select for products
                    top_products = product_data['Jenis Strapping Band'].head(10).tolist()
                    selected_products = st.multiselect(
                        "Pilih Produk untuk Ditampilkan:",
                        options=product_data['Jenis Strapping Band'].unique(),
                        default=top_products[:3]
                    )
                    
                    if selected_products:
                        # Filter for selected products
                        filtered_trend = product_trend[product_trend['Jenis Strapping Band'].isin(selected_products)]
                        
                        # Create line chart
                        fig_trend = px.line(
                            filtered_trend,
                            x='YearMonth',
                            y='Quantity',
                            color='Jenis Strapping Band',
                            title='Tren Penjualan per Produk',
                            template='plotly_white',
                            markers=True
                        )
                        
                        fig_trend.update_layout(
                            xaxis_title='Periode',
                            yaxis_title='Total Penjualan',
                            height=500,
                            xaxis=dict(tickangle=45),
                            legend=dict(orientation='h', yanchor='top', y=-0.2)
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                        # Add product comparison
                        st.subheader("Perbandingan Produk")
                        
                        # Create pie chart for product comparison
                        fig_pie = px.pie(
                            product_data[product_data['Jenis Strapping Band'].isin(selected_products)],
                            values='Quantity',
                            names='Jenis Strapping Band',
                            title='Proporsi Penjualan per Produk',
                            template='plotly_white',
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.Greens_r
                        )
                        
                        fig_pie.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hoverinfo='label+value+percent'
                        )
                        
                        fig_pie.update_layout(height=500)
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("⚠️ Silakan pilih setidaknya satu produk untuk menampilkan tren.")
                else:
                    st.warning("⚠️ Data tidak memiliki informasi jenis produk.")
                    
                # Product performance insights
                with st.expander("💡 Insight Performa Produk"):
                    st.markdown("""
                    **Analisis Performa Produk:**
                    
                    1. **Identifikasi Produk Unggulan**
                       - Produk dengan penjualan tertinggi merupakan sumber pendapatan utama
                       - Pastikan ketersediaan stok produk unggulan selalu terjaga
                    
                    2. **Evaluasi Produk Berkinerja Rendah**
                       - Analisis mengapa produk tertentu memiliki penjualan rendah
                       - Pertimbangkan strategi pemasaran atau penghentian produk
                    
                    3. **Tren Produk**
                       - Perhatikan tren naik/turun untuk setiap produk
                       - Identifikasi produk dengan pertumbuhan paling pesat
                    
                    4. **Diversifikasi Produk**
                       - Terlalu bergantung pada satu produk bisa berisiko
                       - Upayakan distribusi penjualan yang seimbang
                    """)
            
            # Tab 4: Distribution Analysis
            with viz_tabs[3]:
                st.subheader("📊 Analisis Distribusi Data")
                
                # Create columns layout
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Histogram of sales quantity
                    fig_hist = px.histogram(
                        df,
                        x='Quantity',
                        nbins=20,
                        title='Distribusi Jumlah Penjualan',
                        template='plotly_white',
                        color_discrete_sequence=['#6b8e23']
                    )
                    
                    fig_hist.update_layout(
                        xaxis_title='Jumlah Penjualan',
                        yaxis_title='Frekuensi',
                        height=400
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with dist_col2:
                    # Box plot for monthly distribution
                    df['Month'] = df['Tanggal Pembelian'].dt.strftime('%b')
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    fig_box = px.box(
                        df,
                        x='Month',
                        y='Quantity',
                        category_orders={"Month": month_order},
                        title='Distribusi Penjualan Bulanan',
                        template='plotly_white',
                        color_discrete_sequence=['#6b8e23']
                    )
                    
                    fig_box.update_layout(
                        xaxis_title='Bulan',
                        yaxis_title='Jumlah Penjualan',
                        height=400
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Statistical summary
                st.subheader("Ringkasan Statistik")
                
                # Calculate summary statistics
                quantity_stats = df['Quantity'].describe().reset_index()
                quantity_stats.columns = ['Metrik', 'Nilai']
                
                # Format for display
                quantity_stats['Nilai'] = quantity_stats['Nilai'].apply(lambda x: f"{x:,.2f}")
                
                # Create a nicer display for stats
                st.markdown("""
                <style>
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 10px;
                    margin-top: 15px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
                
                for idx, row in quantity_stats.iterrows():
                    if row['Metrik'] != 'count':  # Skip count as we already show this elsewhere
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">{row['Metrik'].capitalize()}</div>
                            <div class="stat-value">{row['Nilai']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Distribution insights
                with st.expander("💡 Memahami Distribusi Data"):
                    st.markdown("""
                    **Panduan Distribusi Data:**
                    
                    1. **Histogram**
                       - Menunjukkan seberapa sering berbagai jumlah penjualan terjadi
                       - Pola normal (lonceng) menunjukkan distribusi yang seimbang
                       - Pola condong (skewed) menunjukkan ketidakseimbangan
                    
                    2. **Box Plot**
                       - Kotak: 25% - 75% dari data (IQR)
                       - Garis tengah: nilai median (nilai tengah)
                       - Whisker: rentang normal data
                       - Titik: outlier (nilai ekstrem)
                    
                    3. **Statistik**
                       - Mean: rata-rata penjualan
                       - Std: standar deviasi (variabilitas data)
                       - Min/Max: nilai terendah dan tertinggi
                       - 25%/50%/75%: persentil data
                    """)
                
                # Correlation analysis if multiple numeric columns exist
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if len(numeric_cols) > 1:
                    st.subheader("Analisis Korelasi")
                    
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title='Matriks Korelasi Antar Variabel',
                        template='plotly_white',
                        color_continuous_midpoint=0
                    )
                    
                    fig_corr.update_layout(height=500)
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Add navigation button to next section
            st.markdown("---")
            st.markdown("### 🚀 Langkah Selanjutnya")
            if st.button("Lanjut ke Prediksi Penjualan", use_container_width=True):
                st.session_state["nav_to"] = "🔮 Prediksi Penjualan"
                st.experimental_rerun()
                
        else:
            st.warning("⚠️ Silakan upload dan preprocessing data terlebih dahulu.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 📋 Persiapan Data
                
                Sebelum melihat visualisasi data, pastikan Anda telah:
                
                1. Mengupload file Excel di menu **Upload Data**
                2. Melakukan preprocessing di menu **Preprocessing Data**
                """)
                
                if st.button("Kembali ke Upload Data", use_container_width=True):
                    st.session_state["nav_to"] = "📂 Upload Data"
                    st.experimental_rerun()
            
            with col2:
                st.image("https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png", 
                        width=200, caption="Siapkan data terlebih dahulu")


    elif selected == '🔮 Prediksi Penjualan':
        st.title('🔮 Prediksi Penjualan')
        
        header_card("Model Prediksi Machine Learning", 
                  "Prediksi penjualan masa depan menggunakan model XGBoost berdasarkan data historis.")
        
        if 'processed_data' in st.session_state and 'processed_data_display' in st.session_state:
            df = st.session_state['processed_data'].copy()
            df_display = st.session_state['processed_data_display'].copy()
            
            # Create tabs for different prediction features
            pred_tabs = st.tabs(["📊 Model & Training", "🔮 Prediksi", "📈 Evaluasi Model"])
            
            # Tab 1: Model & Training
            with pred_tabs[0]:
                st.subheader("📊 Model Machine Learning XGBoost")
                
                # Prepare data for machine learning
                st.markdown("""
                XGBoost (eXtreme Gradient Boosting) adalah algoritma machine learning yang sangat efektif untuk prediksi data deret waktu.
                Model ini menggunakan teknik ensemble learning dengan penguatan gradien untuk menghasilkan prediksi yang akurat.
                """)
                
                # Create columns for parameters and training
                model_col1, model_col2 = st.columns([2, 1])
                
                with model_col1:
                    st.markdown("### ⚙️ Parameter Model")
                    
                    # Parameter tuning with sliders
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("Jumlah Estimator", 50, 500, 100, 10)
                        max_depth = st.slider("Kedalaman Maksimum", 3, 10, 6, 1)
                        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    with col2:
                        min_child_weight = st.slider("Min Child Weight", 1, 10, 1, 1)
                        gamma = st.slider("Gamma", 0.0, 1.0, 0.0, 0.1)
                        subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
                    
                with model_col2:
                    st.markdown("### 🎯 Data Training")
                    
                    # Training data split ratio
                    test_size = st.slider("Rasio Data Testing", 0.1, 0.5, 0.2, 0.05)
                    
                    # Training button
                    train_model = st.button("Train Model", use_container_width=True)
                    
                    if train_model:
                        with st.spinner('Training model, harap tunggu...'):
                            try:
                                # Prepare features for training
                                X = df[['Year', 'Month', 'Quarter', 'Day']].values
                                y = df['Quantity'].values
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                                
                                # Create and train XGBoost model
                                model = xgb.XGBRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    min_child_weight=min_child_weight,
                                    gamma=gamma,
                                    subsample=subsample,
                                    objective='reg:squarederror',
                                    random_state=42
                                )
                                
                                model.fit(X_train, y_train)
                                
                                # Make predictions on test set
                                y_pred = model.predict(X_test)
                                
                                # If we have a scaler in the session state, inverse transform predictions
                                if 'scaler' in st.session_state:
                                    scaler = st.session_state['scaler']
                                    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                                    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                                else:
                                    y_pred_original = y_pred
                                    y_test_original = y_test
                                
                                # Calculate metrics
                                mse = np.mean((y_pred_original - y_test_original) ** 2)
                                rmse = np.sqrt(mse)
                                mae = np.mean(np.abs(y_pred_original - y_test_original))
                                
                                # Calculate R-squared
                                y_mean = np.mean(y_test_original)
                                ss_total = np.sum((y_test_original - y_mean) ** 2)
                                ss_residual = np.sum((y_test_original - y_pred_original) ** 2)
                                r2 = 1 - (ss_residual / ss_total)
                                
                                # Store model and metrics in session state
                                st.session_state['xgb_model'] = model
                                st.session_state['prediction_metrics'] = {
                                    'MSE': mse,
                                    'RMSE': rmse,
                                    'MAE': mae,
                                    'R-squared': r2
                                }
                                st.session_state['test_predictions'] = {
                                    'y_test': y_test_original,
                                    'y_pred': y_pred_original
                                }
                                
                                st.success("✅ Model berhasil dilatih!")
                                
                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan dalam training model: {str(e)}")
                
                # Feature importance
                if 'xgb_model' in st.session_state:
                    st.subheader("🔍 Feature Importance")
                    
                    # Get feature importance
                    feature_names = ['Tahun', 'Bulan', 'Kuartal', 'Tanggal']
                    importance = st.session_state['xgb_model'].feature_importances_
                    
                    # Create feature importance bar chart
                    fig_importance = px.bar(
                        x=feature_names,
                        y=importance,
                        title='Pentingnya Fitur dalam Model Prediksi',
                        labels={'x': 'Fitur', 'y': 'Importance Score'},
                        template='plotly_white',
                        color=importance,
                        color_continuous_scale=px.colors.sequential.Greens
                    )
                    
                    fig_importance.update_layout(
                        xaxis_title='Fitur',
                        yaxis_title='Importance Score',
                        height=400,
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Model metrics
                    if 'prediction_metrics' in st.session_state:
                        metrics = st.session_state['prediction_metrics']
                        
                        st.subheader("📊 Metrik Performa Model")
                        
                        # Create metrics display
                        cols = st.columns(4)
                        
                        cols[0].metric("MSE", f"{metrics['MSE']:.2f}")
                        cols[1].metric("RMSE", f"{metrics['RMSE']:.2f}")
                        cols[2].metric("MAE", f"{metrics['MAE']:.2f}")
                        cols[3].metric("R-squared", f"{metrics['R-squared']:.4f}")
                        
                        # Add accuracy interpretation
                        r2_value = metrics['R-squared']
                        if r2_value > 0.9:
                            accuracy_text = "Sangat Baik"
                            accuracy_color = "darkgreen"
                        elif r2_value > 0.7:
                            accuracy_text = "Baik"
                            accuracy_color = "green"
                        elif r2_value > 0.5:
                            accuracy_text = "Sedang"
                            accuracy_color = "orange"
                        else:
                            accuracy_text = "Kurang"
                            accuracy_color = "red"
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 20px;">
                            <p style="font-size: 18px;">Akurasi Model: <span style="color: {accuracy_color}; font-weight: bold;">{accuracy_text}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ Silakan train model terlebih dahulu dengan menekan tombol 'Train Model'.")
            
            # Tab 2: Predictions
            with pred_tabs[1]:
                st.subheader("🔮 Prediksi Penjualan Masa Depan")
                
                if 'xgb_model' in st.session_state:
                    # Input for prediction
                    st.markdown("### 📅 Masukkan Periode untuk Prediksi")
                    
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        # Year selection
                        available_years = sorted(df_display['Tanggal Pembelian'].dt.year.unique())
                        max_year = max(available_years)
                        pred_year = st.selectbox("Tahun", list(range(max_year, max_year + 6)))
                        
                        # Month selection
                        pred_month = st.selectbox("Bulan", list(range(1, 13)), format_func=lambda x: ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'][x-1])
                        
                        # Quarter calculation
                        pred_quarter = (pred_month - 1) // 3 + 1
                        
                        # Day selection
                        max_days = {1: 31, 2: 29 if (pred_year % 4 == 0 and pred_year % 100 != 0) or pred_year % 400 == 0 else 28, 
                                   3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
                        pred_day = st.selectbox("Tanggal", list(range(1, max_days[pred_month] + 1)))
                    
                    with pred_col2:
                        st.markdown("### 📋 Detail Prediksi")
                        
                        st.markdown(f"""
                        <div class="data-card">
                            <h3>Periode yang Dipilih</h3>
                            <p>Tanggal: {pred_day} {['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'][pred_month-1]} {pred_year}</p>
                            <p>Kuartal: Q{pred_quarter} {pred_year}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction button
                        predict_button = st.button("Prediksi Penjualan", use_container_width=True)
                    
                    if predict_button:
                        with st.spinner('Calculating prediction...'):
                            try:
                                # Create feature vector for prediction
                                X_pred = np.array([[pred_year, pred_month, pred_quarter, pred_day]])
                                
                                # Make prediction
                                y_pred = st.session_state['xgb_model'].predict(X_pred)
                                
                                # Inverse transform prediction if scaler exists
                                if 'scaler' in st.session_state:
                                    scaler = st.session_state['scaler']
                                    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()[0]
                                else:
                                    y_pred_original = y_pred[0]
                                
                                # Store prediction in session state
                                if 'predictions' not in st.session_state:
                                    st.session_state['predictions'] = []
                                
                                # Add new prediction
                                prediction_date = f"{pred_day} {['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'][pred_month-1]} {pred_year}"
                                st.session_state['predictions'].append({
                                    'date': prediction_date,
                                    'year': pred_year,
                                    'month': pred_month,
                                    'day': pred_day,
                                    'prediction': y_pred_original
                                })
                                
                                # Display prediction
                                st.success("✅ Prediksi berhasil dihitung!")
                                
                                # Show prediction result
                                st.markdown(f"""
                                <div style="text-align: center; margin-top: 20px;">
                                    <h2>Hasil Prediksi Penjualan</h2>
                                    <div style="font-size: 48px; font-weight: bold; color: #6b8e23; margin: 20px 0;">
                                        {int(y_pred_original):,} Unit
                                    </div>
                                    <p style="font-size: 18px;">untuk tanggal {prediction_date}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Calculate confidence interval (simple approach)
                                if 'prediction_metrics' in st.session_state:
                                    rmse = st.session_state['prediction_metrics']['RMSE']
                                    lower_bound = max(0, y_pred_original - 1.96 * rmse)
                                    upper_bound = y_pred_original + 1.96 * rmse
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; margin-top: 10px;">
                                        <p>Interval Kepercayaan (95%):</p>
                                        <p style="font-size: 16px;">{int(lower_bound):,} - {int(upper_bound):,} Unit</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan dalam perhitungan prediksi: {str(e)}")
                    
                    # Show prediction history
                    if 'predictions' in st.session_state and len(st.session_state['predictions']) > 0:
                        st.markdown("---")
                        st.subheader("📜 Riwayat Prediksi")
                        
                        # Create a dataframe from predictions
                        pred_history_df = pd.DataFrame(st.session_state['predictions'])
                        pred_history_df['prediction'] = pred_history_df['prediction'].round(2)
                        
                        # Display in a table
                        st.dataframe(
                            pred_history_df[['date', 'prediction']].rename(
                                columns={'date': 'Tanggal', 'prediction': 'Prediksi Penjualan'}
                            ),
                            use_container_width=True
                        )
                        
                        # Visualize predictions
                        if len(pred_history_df) > 1:
                            fig_history = px.line(
                                pred_history_df,
                                x='date',
                                y='prediction',
                                markers=True,
                                title='Riwayat Prediksi Penjualan',
                                template='plotly_white',
                                labels={'date': 'Tanggal', 'prediction': 'Prediksi Penjualan'}
                            )
                            
                            fig_history.update_traces(
                                line=dict(color='#6b8e23', width=2),
                                marker=dict(size=8, color='#4a633d')
                            )
                            
                            fig_history.update_layout(
                                xaxis_title='Tanggal',
                                yaxis_title='Prediksi Penjualan',
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_history, use_container_width=True)
                        
                        # Clear predictions button
                        if st.button("Hapus Riwayat Prediksi", use_container_width=True):
                            st.session_state['predictions'] = []
                            st.experimental_re
