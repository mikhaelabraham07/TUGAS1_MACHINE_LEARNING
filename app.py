import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    df_train = df.copy()
    le = LabelEncoder()
    for col in ['cut', 'color', 'clarity']:
        df_train[col] = le.fit_transform(df_train[col])
    
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    
    # Split data untuk mendapatkan nilai akurasi real-time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung akurasi (R2 Score)
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred) * 100
    
    return model, le, accuracy

# Load data dan model
df = load_data()
model, le, accuracy_val = train_best_model(df)

# 3. Form Input di Bagian Tengah
st.subheader("💎 Input Karakteristik Berlian")

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

# 4. Logika Prediksi & Tampilan Akurasi
if submitted:
    input_data = pd.DataFrame({
        'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
        'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z]
    })
    
    input_encoded = input_data.copy()
    for col in ['cut', 'color', 'clarity']:
        le.fit(df[col])
        input_encoded[col] = le.transform(input_encoded[col])
    
    prediction = model.predict(input_encoded)
    
    st.markdown("---")
    
    # Layout untuk hasil dan akurasi
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.success(f"### Estimasi Harga:\n**${prediction[0]:,.2f}**")
    
    with res_col2:
        # Menampilkan akurasi dalam bentuk metrik
        st.info(f"### Akurasi Model:\n**{accuracy_val:.2f}%**")

    # Catatan tambahan di bawah
    st.caption(f"Model dilatih menggunakan XGBoost dengan data training 90% dari dataset asli.")
