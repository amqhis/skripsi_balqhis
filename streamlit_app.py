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
            .logo-container img {
                display: block;
                margin: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi utama
def main():
    local_css()  # Memanggil CSS

    # Sidebar dengan menu
    with st.sidebar:
        st.markdown(
            """
            <div class='logo-container'>
                <img src="https://raw.githubusercontent.com/Auliaafitriani/SkripsiAulia/main/LogoPriorityAid.png" 
                     alt="Logo" 
                     width="250" 
                     style="margin-top: 0;">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Menu navigasi
        selected = option_menu(None, 
                               ['About', 'Upload Data', 'Preprocessing', 
                                'PSO and K-Medoids Results'],
                               menu_icon='cast',
                               icons=['house', 'cloud-upload', 'gear', 'graph-up'],
                               default_index=0,
                               styles={
                                   "container": {
                                       "padding": "0px", 
                                       "background-color": "#f8f9fa"
                                   },
                                   "icon": {
                                       "color": "#3498db", 
                                       "font-size": "17px"
                                   }, 
                                   "nav-link": {
                                       "font-size": "15px", 
                                       "text-align": "left", 
                                       "margin":"1px", 
                                       "color": "#2c3e50",
                                       "--hover-color": "#e9ecef"
                                   },
                                   "nav-link-selected": {
                                       "background-color": "#3498db", 
                                       "color": "white"
                                   },
                               })

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
