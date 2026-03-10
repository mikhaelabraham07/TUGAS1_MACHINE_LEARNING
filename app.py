import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")

st.title("💎 Diamond Price Prediction App")
st.write("""
Aplikasi ini memprediksi harga berlian berdasarkan karakteristik fisik dan kualitasnya.
Data yang dimasukkan akan diproses menggunakan algoritma **XGBoost**.
""")

# 2. Fungsi Load Data & Model Training
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")
    return df

@st.cache_resource
def train_best_model(df):
    # Preprocessing kilat agar sesuai dengan data training di Colab
    df_train = df.copy()
    le = LabelEncoder()
    for col in ['cut', 'color', 'clarity']:
        df_train[col] = le.fit_transform(df_train[col])
    
    # Fitur sesuai tabel: carat, cut, color, clarity, depth, table, x, y, z
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    
    # Menggunakan skenario terbaik (misal 90:10)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, le

# Load data dan model
df = load_data()
model, le = train_best_model(df)

# 3. Form Input di Bagian Tengah (Ganti bagian sidebar sebelumnya)
st.subheader("💎 Input Karakteristik Berlian")

# Gunakan form agar halaman tidak reload setiap kali input berubah
with st.form("diamond_form"):
    # Membagi input menjadi 3 kolom agar rapi di tengah
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

    # Tombol submit di dalam form
    submitted = st.form_submit_button("Prediksi Harga Sekarang", use_container_width=True)

# 4. Logika Prediksi
if submitted:
    input_data = pd.DataFrame({
        'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
        'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z]
    })
    
    # Preprocessing
    input_encoded = input_data.copy()
    for col in ['cut', 'color', 'clarity']:
        # Gunakan label encoder yang konsisten
        le.fit(df[col])
        input_encoded[col] = le.transform(input_encoded[col])
    
    # Jalankan Prediksi
    prediction = model.predict(input_encoded)
    
    st.markdown("---")
    st.success(f"### Estimasi Harga Berlian: **${prediction[0]:,.2f}**")
