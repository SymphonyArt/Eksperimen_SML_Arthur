
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

# Abaikan warnings
warnings.filterwarnings('ignore', category=UserWarning)

def preprocess_pipeline(file_path):
    """
    Melakukan pipeline preprocessing lengkap dan menyimpan hasilnya
    ke 4 file CSV:
    1. X_train_res.csv
    2. y_train_res.csv
    3. X_test.csv
    4. y_test.csv
    """
    
    # ---------------------------------
    # 1. Muat Data
    # ---------------------------------
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset dimuat. Bentuk awal: {df.shape}")
    except Exception as e:
        print(f"Error memuat data: {e}")
        return

    # ---------------------------------
    # 2. Cleaning Data
    # ---------------------------------
    
    # Hapus 'Unnamed: 0'
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    # Standarisasi 'Gender' (f -> F)
    if 'Gender' in df.columns:
        df['Gender'].replace('f', 'F', inplace=True)
    
    # ---------------------------------
    # 3. Handle Outlier (IQR)
    # ---------------------------------
    
    # Menggunakan kolom numerik (termasuk target) sesuai alur notebook
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if num_cols:
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        batas_bawah = Q1 - (1.5 * IQR)
        batas_atas = Q3 + (1.5 * IQR)
        
        # Buat mask untuk baris non-outlier
        mask = ~((df[num_cols] < batas_bawah) | (df[num_cols] > batas_atas)).any(axis=1)
        
        df_sebelum = df.shape[0]
        df = df[mask].copy()
        print(f"IQR: {df_sebelum - df.shape[0]} baris outlier dihapus. Data kini: {df.shape}")

    # ---------------------------------
    # 4. Encoding
    # ---------------------------------
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
    
    # ---------------------------------
    # 5. Pisahkan Fitur (X) & Target (y)
    # ---------------------------------
    if 'Diagnosis' not in df.columns:
        print("Error: Kolom target 'Diagnosis' tidak ditemukan.")
        return
        
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    # Simpan nama kolom
    feature_names = X.columns
    
    # ---------------------------------
    # 6. Standarisasi Fitur (X)
    # ---------------------------------
    
    # Scaling dilakukan pada X keseluruhan (sesuai alur notebook)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ubah X_scaled ke DataFrame (jaga kolom)
    X = pd.DataFrame(X_scaled, columns=feature_names)
    
    # ---------------------------------
    # 7. Split Data
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Data di-split: {X_train.shape[0]} train, {X_test.shape[0]} test.")

    # ---------------------------------
    # 8. Handle Imbalance (SMOTE)
    # ---------------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE diterapkan pada data training.")

    # ---------------------------------
    # 9. Simpan Hasil ke File CSV
    # ---------------------------------
    
    # Beri nama pada Series target agar CSV memiliki header
    y_train_res.name = 'Diagnosis'
    y_test.name = 'Diagnosis'
    
    # Simpan 4 file
    X_train_res.to_csv('X_train_res.csv', index=False)
    y_train_res.to_csv('y_train_res.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("\n--- Pipeline Selesai ---")
    print("4 file telah disimpan:")
    print("1. X_train_res.csv")
    print("2. y_train_res.csv")
    print("3. X_test.csv")
    print("4. y_test.csv")
    
    return

# --- CONTOH PENGGUNAAN ---
if __name__ == "__main__":
    
    # --- SESUAIKAN PATH INI ---
    PATH_DATA = r"..\Diabetes Classification.csv" 
    
    print(f"Memulai pipeline preprocessing untuk file: {PATH_DATA} \n")
    
    # Memanggil fungsi pipeline
    preprocess_pipeline(PATH_DATA)