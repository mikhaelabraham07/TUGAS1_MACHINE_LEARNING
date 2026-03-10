import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ... (Bagian Konfigurasi & Load Data tetap sama) ...

@st.cache_resource
def train_best_model(df):
    df_train = df.copy()
    le = LabelEncoder()
    for col in ['cut', 'color', 'clarity']:
        df_train[col] = le.fit_transform(df_train[col])
    
    X = df_train.drop('price', axis=1)
    y = df_train['price']
    
    # Split data 90:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Hitung Metrik Evaluasi
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, le, r2, rmse, mae

# Load data dan model (mengambil 5 variabel hasil training)
df = load_data()
model, le, r2_val, rmse_val, mae_val = train_best_model(df)

# ... (Bagian Form Input tetap sama) ...

if submitted:
    # ... (Proses Prediksi tetap sama) ...
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
    
    # 1. Tampilkan Harga Prediksi Utama
    st.success(f"### 💰 Estimasi Harga Berlian: **${prediction[0]:,.2f}**")
    
    # 2. Tampilkan Metrik Evaluasi (Akurasi & Error) dalam 3 Kolom
    st.write("#### 📊 Performa & Evaluasi Model:")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label="Akurasi (R2 Score)", value=f"{r2_val:.2f}%")
    with m2:
        st.metric(label="RMSE (Akar Galat Kuadrat)", value=f"${rmse_val:.2f}")
    with m3:
        st.metric(label="MAE (Rata-rata Kesalahan)", value=f"${mae_val:.2f}")

    st.caption("Catatan: RMSE menunjukkan seberapa jauh prediksi meleset secara rata-rata dalam satuan Dollar ($).")
