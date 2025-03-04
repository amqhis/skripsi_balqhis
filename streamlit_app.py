import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# About Page
if page == "About":
    st.title("📖 About This App")
    st.write(
        "This application is built using Streamlit to demonstrate a predictive model for sales forecasting using XGBoost. "
        "It allows users to upload sales data, preprocess it, train a model, and visualize predictions."
    )
    st.write("### Features:")
    st.markdown("- Upload and process sales data")
    st.markdown("- Train a model using XGBoost")
    st.markdown("- Visualize predictions")
    st.markdown("- Compare model performance")
    
# Home Page
elif page == "Home":
    st.title("🎈 My New App")
    st.write("Welcome to the sales forecasting app! Upload your dataset and start predicting.")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.write(df.head())
        
        # Preprocessing
        st.write("### Data Preprocessing")
        scaler = MinMaxScaler()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        st.write("Data has been normalized using MinMaxScaler.")
        
        # Splitting Data
        st.write("### Splitting Data")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Train Data: {X_train.shape}, Test Data: {X_test.shape}")
        
        # Training XGBoost Model
        st.write("### Training Model")
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Visualization
        st.write("### Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig)
        
        st.write("Model training and evaluation completed! 🎉")
