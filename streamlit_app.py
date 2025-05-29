import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from xgboost import XGBRegressor

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
        st.markdown("""
        <style>
        .welcome-container {
            display: flex;
            flex-direction: column;
            justify-content: center;  /* Vertikal tengah */
            align-items: center;      /* Horizontal tengah */
            height: 300px;            /* Atur tinggi container */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333333;
        }
        .welcome-text {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .welcome-subtext {
            font-size: 18px;
            line-height: 1.5;
            max-width: 600px;
            text-align: center;
            color: #555555;
        }
        </style>
    
        <div class="welcome-container">
            <div class="welcome-text">🎉 Selamat Datang di <span style="color:#FF6F61;">Insight Predict</span>! 🎉</div>
            <div class="welcome-subtext">
                Temukan pola tersembunyi dan prediksi masa depan dengan <strong>Machine Learning</strong> yang mudah digunakan.<br><br>
                📈 Jelajahi data historis, dapatkan insight, dan ambil keputusan lebih tepat dengan teknologi terkini.<br><br>
                🚀 <em>Mulai petualangan data Anda sekarang!</em> ✨
            </div>
        </div>
        """, unsafe_allow_html=True)


    elif selected == '📖 Tentang Aplikasi':
        st.title('📊 Insight Predict - Aplikasi Prediksi Penjualan')
        
        st.markdown("""
        <div style="max-width:700px; margin:auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color:#333;">
            <p style="font-size:18px; line-height:1.6;">
            Insight Predict adalah aplikasi analisis data yang dirancang untuk membantu perusahaan dalam memprediksi tren penjualan secara akurat dan efektif.
            Dengan menggunakan <strong>model XGBoost</strong> yang dioptimasi melalui <strong>Grid Search</strong>, aplikasi ini mampu mengenali pola data historis dan memberikan prediksi kuantitas penjualan di masa depan dengan tingkat keandalan tinggi.
            </p>
            
            <p style="font-size:18px; line-height:1.6;">
            Melalui visualisasi interaktif dan fitur preprocessing data, pengguna dapat dengan mudah mengolah data penjualan bulanan, mengidentifikasi lag terbaik, dan mengevaluasi performa model prediksi.  
            Aplikasi ini sangat berguna untuk mendukung pengambilan keputusan dalam manajemen persediaan, perencanaan produksi, dan strategi pemasaran yang lebih tepat sasaran.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📜 Syarat & Ketentuan Penggunaan"):
            st.markdown("""
            <ul style="max-width:700px; margin:auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color:#555; font-size:16px; line-height:1.5;">
                <li><strong>Format Data:</strong> File Excel (.xlsx)</li>
                <li><strong>Kolom Wajib:</strong>  
                    <ul>
                        <li>Tanggal Pembelian</li>
                        <li>Jenis Produk</li>
                        <li>Quantity (Jumlah Penjualan)</li>
                    </ul>
                </li>
                <li>Data harus bersih dari missing value agar hasil prediksi optimal.</li>
                <li>Disarankan memiliki data penjualan bulanan minimal 2 tahun agar model dapat belajar dengan baik.</li>
            </ul>
            """, unsafe_allow_html=True)


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
    
        if 'original_data' in st.session_state:
            df = st.session_state['original_data'].copy()
    
            st.write("### 📌 Data Sebelum Preprocessing")
            st.dataframe(df)
    
            # 1️⃣ Validasi Kolom
            required_columns = ['Tanggal Pembelian', 'Quantity']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"⚠️ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
            else:
                # 2️⃣ Konversi 'Tanggal Pembelian'
                df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
                df.dropna(subset=['Tanggal Pembelian'], inplace=True)
    
                # 3️⃣ Statistik Deskriptif per Tahun
                df['Tahun'] = df['Tanggal Pembelian'].dt.year
                deskripsi_per_tahun = df.groupby('Tahun')['Quantity'].describe()
                st.write("### 📊 Statistik Deskriptif per Tahun")
                st.dataframe(deskripsi_per_tahun)
    
                # 4️⃣ Cek Missing Value
                missing_values = df.isnull().sum().sum()
                if missing_values > 0:
                    st.warning(f"⚠️ Missing value ditemukan sebanyak {missing_values}! Membersihkan data...")
                    df.dropna(subset=['Tanggal Pembelian', 'Quantity'], inplace=True)
                    df = df[df['Quantity'] > 0]
                else:
                    st.success("✅ Tidak ada missing value dalam dataset.")
    
                # 5️⃣ Agregasi Bulanan
                df['Year'] = df['Tanggal Pembelian'].dt.year
                df['Month'] = df['Tanggal Pembelian'].dt.month
                df_monthly = df.groupby(['Year', 'Month'])[['Quantity']].sum().reset_index()
                st.write("### 📅 Data Agregasi Bulanan")
                st.dataframe(df_monthly)
    
                # 6️⃣ Visualisasi ACF dan PACF Manual (Seperti Google Colab)
                st.write("### 🔄 Visualisasi ACF dan PACF (Manual Style)")
                st.info("📌 Lag terbaik berdasarkan PACF adalah **18**")
    
                from statsmodels.tsa.stattools import acf, pacf
                import numpy as np
                import matplotlib.pyplot as plt
    
                lags = 20
                acf_vals = acf(df_monthly['Quantity'], nlags=lags)
                pacf_vals = pacf(df_monthly['Quantity'], nlags=lags)
                threshold = 1.96 / np.sqrt(len(df_monthly))
    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
                # Plot ACF
                ax1.stem(range(len(acf_vals)), acf_vals, markerfmt='.', basefmt=" ", linefmt='green')
                ax1.axhline(y=threshold, linestyle='-', color='red')
                ax1.axhline(y=-threshold, linestyle='-', color='red')
                ax1.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
                ax1.set_title("Autocorrelation Function (ACF)")
                ax1.set_xlabel("Lag")
                ax1.set_ylabel("ACF")
                ax1.grid(True)
    
                # Plot PACF
                ax2.stem(range(len(pacf_vals)), pacf_vals, markerfmt='.', basefmt=" ", linefmt='blue')
                ax2.axhline(y=threshold, linestyle='-', color='red')
                ax2.axhline(y=-threshold, linestyle='-', color='red')
                ax2.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
                ax2.set_title("Partial Autocorrelation Function (PACF)")
                ax2.set_xlabel("Lag")
                ax2.set_ylabel("PACF")
                ax2.grid(True)
    
                plt.tight_layout()
                st.pyplot(fig)
    
                # 7️⃣ Buat fitur lag 18 setelah visualisasi
                df_monthly['lag_18'] = df_monthly['Quantity'].shift(18)
                df_lag18 = df_monthly.dropna(subset=['lag_18'])
                st.write("### 🧾 Data dengan Fitur Lag 18 dan Quantity Asli")
                st.dataframe(df_lag18[['Year', 'Month', 'lag_18', 'Quantity']])
    
                # 8️⃣ Simpan ke Session State
                st.session_state['processed_data'] = df_monthly
    
        else:
            st.warning("⚠️ Harap unggah data terlebih dahulu di bagian '📂 Upload Data'.")



    
    elif selected == '📊 Visualisasi Data Historis':
        import matplotlib.pyplot as plt  # ✅ Tambahkan ini
    
        st.title("📊 Visualisasi Data Historis")
        if 'original_data' in st.session_state:
            df = st.session_state['original_data'].copy()
    
            # Konversi tanggal
            df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
            df.dropna(subset=['Tanggal Pembelian'], inplace=True)
    
            # Bulanan
            df['Month'] = df['Tanggal Pembelian'].dt.to_period('M').astype(str)
            df_monthly = df.groupby('Month')['Quantity'].sum().reset_index()
            max_month = df_monthly.loc[df_monthly['Quantity'].idxmax()]
            min_month = df_monthly.loc[df_monthly['Quantity'].idxmin()]
            st.write(f"📈 Penjualan tertinggi terjadi pada {max_month['Month']} sebanyak {max_month['Quantity']} unit")
            st.write(f"📉 Penjualan terendah terjadi pada {min_month['Month']} sebanyak {min_month['Quantity']} unit")
    
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_monthly['Month'], df_monthly['Quantity'], marker='o', linestyle='-', color='b')
            plt.xticks(rotation=45)
            plt.title('Tren Penjualan Per Bulan')
            st.pyplot(fig)
    
            # Tahunan
            df_yearly = df.groupby(df['Tanggal Pembelian'].dt.year)['Quantity'].sum()
            fig, ax = plt.subplots()
            df_yearly.plot(kind='bar', color='skyblue', ax=ax)
            plt.title('Total Penjualan per Tahun')
            st.pyplot(fig)
    
            # Berdasarkan Jenis (jika ada)
            if 'Jenis Strapping Band' in df.columns:
                sales_by_type = df.groupby('Jenis Strapping Band')['Quantity'].sum()
                fig, ax = plt.subplots()
                sales_by_type.plot(kind='bar', color='green', ax=ax)
                plt.title('Penjualan Berdasarkan Jenis Strapping Band')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("Kolom 'Jenis Strapping Band' tidak ditemukan dalam data.")
        else:
            st.warning("⚠️ Upload data terlebih dahulu di menu '📂 Upload Data'!")

    
    elif selected == '🔮 Prediksi Masa Depan':
        st.title("🔮 Prediksi Masa Depan")
    
        if 'processed_data' in st.session_state:
            df_monthly = st.session_state['processed_data']
    
            # Cek kolom yang dibutuhkan
            if {'Year', 'Month', 'Quantity'}.issubset(df_monthly.columns):
                best_lag = 18
    
                # Buat kolom datetime dan set sebagai index
                df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(DAY=1))
                df_monthly = df_monthly.set_index('Date')
    
                # Tambahkan fitur lag
                df_pred = df_monthly.copy()
                df_pred[f'lag_{best_lag}'] = df_pred['Quantity'].shift(best_lag)
    
                # Siapkan data training (drop baris dengan NaN)
                X_train = df_pred[[f'lag_{best_lag}']].dropna()
                y_train = df_pred['Quantity'].loc[X_train.index]
    
                # Grid search dengan parameter yang sudah diketahui terbaik
                from xgboost import XGBRegressor
                from sklearn.model_selection import GridSearchCV
    
                param_grid = {
                    'learning_rate': [0.01],
                    'max_depth': [1],
                    'min_child_weight': [13]
                }
    
                grid_search = GridSearchCV(
                    estimator=XGBRegressor(objective='reg:squarederror'),
                    param_grid=param_grid,
                    scoring='r2',
                    cv=3,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
    
                # Latih ulang model dengan seluruh data training
                best_model.fit(X_train, y_train)
    
                # Prediksi 4 bulan ke depan secara recursive
                future_preds = []
                last_known = df_monthly['Quantity'].tolist()
    
                for _ in range(4):
                    input_lag = last_known[-best_lag]
                    pred = best_model.predict(np.array([[input_lag]]))[0]
                    future_preds.append(pred)
                    last_known.append(pred)
    
                # Buat DataFrame hasil prediksi
                future_dates = pd.date_range(start=df_monthly.index[-1] + pd.DateOffset(months=1), periods=4, freq='MS')
                df_future = pd.DataFrame({
                    'Year': future_dates.year,
                    'Month': future_dates.month,
                    'Quantity': future_preds,
                    'Date': future_dates
                })
    
                st.subheader("Prediksi Kuantitas 4 Bulan ke Depan:")
                st.dataframe(df_future[['Year', 'Month', 'Quantity']])
    
                # Gabungkan data historis dan prediksi untuk plotting
                df_monthly_reset = df_monthly.reset_index()
                df_plot = pd.concat([df_monthly_reset[['Date', 'Quantity']], df_future[['Date', 'Quantity']]], ignore_index=True)
    
                # Plot visualisasi
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
    
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_plot['Date'][:len(df_monthly)], df_plot['Quantity'][:len(df_monthly)], label='Data Historis', color='blue')
                ax.plot(df_plot['Date'][len(df_monthly):], df_plot['Quantity'][len(df_monthly):], label='Prediksi 4 Bulan', color='orange')
                ax.axvline(x=df_monthly.index[-1], color='red', linestyle='--', label='Mulai Prediksi')
    
                ax.set_title("Prediksi Kuantitas 4 Bulan ke Depan (Lag 18)")
                ax.set_xlabel("Tanggal (Bulan-Tahun)")
                ax.set_ylabel("Kuantitas")
                ax.legend()
                ax.grid(False)
    
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
                plt.xticks(rotation=45)
                plt.tight_layout()
    
                st.pyplot(fig)
    
            else:
                st.error("Data bulanan harus memiliki kolom: Year, Month, dan Quantity")
        else:
            st.info("Silakan lakukan preprocessing data terlebih dahulu di menu '📊 Data Preprocessing'")



# Menjalankan aplikasi
if __name__ == "__main__":
    main()
