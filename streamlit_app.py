import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Tambahkan konfigurasi Streamlit untuk styling
st.set_page_config(
    page_title="Insight Predict",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Panggil fungsi CSS di awal main()
def main():
    local_css()  # Tambahkan ini di awal fungsi main

    # Sisanya tetap sama seperti kode sebelumnya
    with st.sidebar:
        # Tambahkan class 'logo-container' untuk logo
        st.markdown(
            """
            <div class='logo-container'>
                <img src="Logo Insight Predict.png" 
                     alt="Logo" 
                     width="250" 
                     style="margin-top: 0;">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Styling option menu dengan warna yang lebih menarik
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


