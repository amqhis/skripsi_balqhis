import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from streamlit_option_menu import option_menu
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV


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
        st.title('📊 Insight Predict')
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
    
                # 6️⃣ Normalisasi (dahulu)
                scaler = MinMaxScaler()
                df_monthly['Quantity_Scaled'] = scaler.fit_transform(df_monthly[['Quantity']])
                st.write("### ✅ Data Setelah Normalisasi")
                st.dataframe(df_monthly[['Year', 'Month', 'Quantity', 'Quantity_Scaled']])
    
                # 7️⃣ Visualisasi ACF & PACF setelah normalisasi
                st.write("### 🔁 Visualisasi ACF dan PACF (Quantity_Scaled)")
                lags = 20
                acf_vals = acf(df_monthly['Quantity_Scaled'], nlags=lags)
                pacf_vals = pacf(df_monthly['Quantity_Scaled'], nlags=lags)
                threshold = 1.96 / np.sqrt(len(df_monthly))
    
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
                axes[0].stem(range(len(acf_vals)), acf_vals, markerfmt='.', basefmt=" ", linefmt='green')
                axes[0].axhline(y=threshold, linestyle='-', color='red')
                axes[0].axhline(y=-threshold, linestyle='-', color='red')
                axes[0].axhline(y=0, linestyle='--', color='black')
                axes[0].set_title("Autocorrelation Function (ACF)")
    
                axes[1].stem(range(len(pacf_vals)), pacf_vals, markerfmt='.', basefmt=" ", linefmt='blue')
                axes[1].axhline(y=threshold, linestyle='-', color='red')
                axes[1].axhline(y=-threshold, linestyle='-', color='red')
                axes[1].axhline(y=0, linestyle='--', color='black')
                axes[1].set_title("Partial Autocorrelation Function (PACF)")
    
                st.pyplot(fig)
                st.info("📌 Dari visualisasi ACF dan PACF, lag terbaik yang disarankan adalah **lag 18**.")
    
                # 8️⃣ Tampilkan Isi Lag 18 (dari data asli Quantity, bukan Quantity_Scaled)
                df_monthly['lag_18'] = df_monthly['Quantity'].shift(18)
                df_lag18 = df_monthly.dropna(subset=['lag_18'])
                st.write("### 🧾 Data dengan Lag 18")
                st.dataframe(df_lag18[['Year', 'Month', 'lag_18', 'Quantity']])
    
                # 9️⃣ Simpan ke Session State
                st.session_state['processed_data'] = df_monthly
    
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
            df = st.session_state['processed_data'].copy()
    
            # Pastikan lag_18 ada
            if 'lag_18' not in df.columns:
                st.error("⚠️ Data belum mengandung kolom lag_18. Lakukan preprocessing dengan lag 18 dulu.")
                st.stop()
    
            # Buat data training (buang baris yang lag_18 = NaN)
            data_train = df.dropna(subset=['lag_18']).copy()
    
            # Normalisasi jika belum ada
            if 'Quantity_Scaled' not in data_train.columns:
                scaler = MinMaxScaler()
                data_train['Quantity_Scaled'] = scaler.fit_transform(data_train[['Quantity']])
                data_train['lag_18_Scaled'] = scaler.transform(data_train[['lag_18']])
            else:
                scaler = MinMaxScaler()
                scaler.fit(data_train[['Quantity']])
                data_train['lag_18_Scaled'] = scaler.transform(data_train[['lag_18']])
    
            # Fitur dan target
            X_train = data_train[['lag_18_Scaled']]
            y_train = data_train['Quantity_Scaled']
    
            # Training XGBoost
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)
    
            # Mulai recursive forecasting prediksi 24 bulan ke depan (2024-2025)
            n_future = 24
            last_18_values = df['Quantity'].tail(18).values  # ambil 18 data terakhir asli
    
            predictions_scaled = []
            for i in range(n_future):
                # Ambil lag_18 terbaru: 18 langkah sebelum prediksi saat ini
                lag_18_val = last_18_values[-18]
    
                # Normalisasi lag_18
                lag_18_scaled = scaler.transform(np.array([[lag_18_val]]))
    
                # Prediksi scaled
                pred_scaled = model.predict(lag_18_scaled.reshape(1, -1))[0]
    
                # Simpan hasil prediksi scaled
                predictions_scaled.append(pred_scaled)
    
                # Inverse transform prediksi dan tambahkan ke last_18_values
                pred_actual = scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]
                last_18_values = np.append(last_18_values, pred_actual)  # update array untuk prediksi berikutnya
    
            # Buat dataframe hasil prediksi
            future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=n_future, freq='MS')
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Year': future_dates.year,
                'Month': future_dates.month,
                'Predicted_Quantity': scaler.inverse_transform(np.array(predictions_scaled).reshape(-1,1)).flatten()
            })
    
            # Visualisasi historis + prediksi
            st.subheader("📈 Prediksi Penjualan dengan Lag 18 (Recursive Forecasting)")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df['Date'], df['Quantity'], label='Data Historis', marker='o')
            ax.plot(future_df['Date'], future_df['Predicted_Quantity'], label='Prediksi 2024-2025', marker='s', color='red')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Quantity')
            ax.set_title('Prediksi Penjualan Strapping Band')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
    
            # Tampilkan tabel prediksi
            st.write("### 📋 Tabel Hasil Prediksi")
            st.dataframe(future_df)
    
        else:
            st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu!")




  

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
