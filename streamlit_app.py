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
                width: 120px;
            }

            /* Styling untuk menu navigasi */
            .css-1v0mbdj.eknhn3m4 {
                color: #2c3e50 !important;
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
            ['Tentang Aplikasi', 'Upload Data', 'Preprocessing Data', 
             'Visualisasi Data Historis', 'Prediksi Masa Depan'],
            menu_icon='cast',
            icons=['info-circle', 'cloud-upload', 'filter', 'bar-chart', 'line-chart'],
            default_index=0,
            styles={
                "container": {
                    "padding": "0px",
                    "background-color": "#cdd4b1"
                },
                "icon": {
                    "color": "#2c3e50",
                    "font-size": "17px"
                },
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px",
                    "color": "#2c3e50",
                    "--hover-color": "#b5c19a"
                },
                "nav-link-selected": {
                    "background-color": "#6b8e23",
                    "color": "white"
                },
            }
        )

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
