import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

st.title("💎 Diamond Price Prediction App")
st.write("""
Aplikasi ini memprediksi harga berlian berdasarkan karakteristik fisik dan kualitasnya menggunakan algoritma **XGBoost**.
""")

# 2. Fungsi Load Data (Pastikan file diamonds.csv ada di GitHub)
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("diamonds.csv")
        return data
    except FileNotFoundError:
        st.error("File 'diamonds.csv' tidak ditemukan. Pastikan file sudah diunggah ke GitHub.")
        return None

# 3. Fungsi Training Model & Evaluasi
@st.cache_resource
def train_best_model(df):
    df_train = df.copy()
    le_dict = {}
    
    # Encoding kolom kategorikal
    for col in ['cut', 'color', 'clarity']:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        le_dict[col] = le
    
    # Pisahkan Fitur dan Target
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    
    # Split 90:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Training XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung Metrik Evaluasi
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, le_dict, r2, rmse, mae

# Eksekusi Load Data & Training
df = load_data()

if df is not None:
    model, encoders, r2_val, rmse_val, mae_val = train_best_model(df)

    # 4. Form Input (Posisi di Tengah)
    st.subheader("📝 Input Karakteristik Berlian")
    with st.form("diamond_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            carat = st.number_input("Carat (Berat)", min_value=0.2, max_value=5.0, value=0.23, step=0.01)
            cut = st.selectbox("Cut (Kualitas)", df['cut'].unique())
            color = st.selectbox("Color (Warna)", df['color'].unique())

        with col2:
            clarity = st.selectbox("Clarity (Kejernihan)", df['clarity'].unique())
            depth = st.number_input("Depth %", min_value=43.0, max_value=79.0, value=61.5)
            table = st.number_input("Table Width", min_value=43.0, max_value=95.0, value=55.0)

        with col3:
            x = st.number_input("Panjang (x) mm", min_value=0.0, max_value=11.0, value=3.95)
            y = st.number_input("Lebar (y) mm", min_value=0.0, max_value=59.0, value=3.98)
            z = st.number_input("Kedalaman (z) mm", min_value=0.0, max_value=32.0, value=2.43)

        submitted = st.form_submit_button("Prediksi Harga Sekarang", use_container_width=True)

    # 5. Output Hasil & Metrik Error
    if submitted:
        # Menyiapkan data input untuk prediksi
        input_data = pd.DataFrame({
            'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
            'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z]
        })
        
        # Encoding input sesuai label encoder saat training
        for col in ['cut', 'color', 'clarity']:
            input_data[col] = encoders[col].transform(input_data[col])
        
        prediction = model.predict(input_data)
        
        st.markdown("---")
        st.success(f"### 💰 Estimasi Harga Berlian: **${prediction[0]:,.2f}**")
        
        # Tampilan Akurasi dan Error
        st.write("#### 📊 Hasil Evaluasi Model:")
        m1, m2, m3 = st.columns(3)
        m1.metric("Akurasi (R2)", f"{r2_val:.2f}%")
        m2.metric("RMSE (Error)", f"${rmse_val:.2f}")
        m3.metric("MAE (Error)", f"${mae_val:.2f}")
        
        st.caption("Nilai error menunjukkan rata-rata selisih prediksi dengan harga asli di dataset.")
