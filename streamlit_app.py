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
                st.write("### 📅 Data Agregasi Bulanan")
                st.dataframe(df_monthly)
    
                # 6️⃣ Buat fitur lag 18
                df_monthly['lag_18'] = df_monthly['Quantity'].shift(18)
                df_lag18 = df_monthly.dropna(subset=['lag_18'])
                st.write("### 🧾 Data dengan Fitur Lag 18 dan Quantity Asli")
                st.dataframe(df_lag18[['Year', 'Month', 'lag_18', 'Quantity']])
    
                # 7️⃣ Simpan ke Session State
                st.session_state['processed_data'] = df_monthly
    
                # 8️⃣ Visualisasi ACF dan PACF
                st.write("### 🔄 Autocorrelation Function (ACF) & Partial ACF (PACF)")
                
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                import matplotlib.pyplot as plt
                
                try:
                    if len(df_monthly['Quantity']) <= 20:
                        st.warning("📉 Jumlah data kurang dari 21 baris, tidak bisa menampilkan ACF/PACF dengan lag 20.")
                    else:
                        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                        plot_acf(df_monthly['Quantity'], lags=20, ax=ax_acf)
                        st.pyplot(fig_acf)
                
                        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                        plot_pacf(df_monthly['Quantity'], lags=20, method='ywm', ax=ax_pacf)
                        st.pyplot(fig_pacf)
                
                except Exception as e:
                    st.error(f"❌ Gagal menampilkan ACF/PACF: {str(e)}")


    
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
