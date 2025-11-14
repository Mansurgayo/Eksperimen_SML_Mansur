"""
Modelling - Mansur
--------------------------
Melatih model ML menggunakan MLflow (local tracking) - BASIC LEVEL.
Sesuai kriteria 2 poin.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# === 1. Set MLflow config lebih awal ===
# Menyimpan run secara lokal di folder ./mlruns
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("Eksperimen_Model_Mansur")

# === 2. Aktifkan autolog (harus setelah set experiment) ===
# Autolog sudah cukup untuk kriteria Basic
mlflow.sklearn.autolog()

# === 3. Load dataset hasil preprocessing ===
# Pastikan path ini sesuai dengan struktur folder Anda
try:
    data = pd.read_csv("namadataset_preprocessing/cleaned_dataset.csv")
except FileNotFoundError:
    print("Error: File 'namadataset_preprocessing/cleaned_dataset.csv' tidak ditemukan.")
    print("Pastikan file tersebut ada di dalam folder 'Membangun_model'.")
    exit()


# Sesuaikan nama target sesuai dataset kamu
if "Outcome" not in data.columns:
    print("Error: Kolom 'Outcome' tidak ditemukan di dataset.")
    exit()
    
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 5. Training model dengan MLflow ===
# 'run_name' akan muncul di dashboard MLflow
with mlflow.start_run(run_name="LogReg_Basic_Autolog"):
    
    print("🚀 Mulai training model Logistic Regression (Basic)...")
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Prediksi dan metrik (autolog akan mencatat ini secara otomatis)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Hasil Evaluasi Model (Basic) ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\n✅ Training (Basic) selesai dan tersimpan otomatis di MLflow (autolog aktif).")
print("📁 Jalankan perintah berikut untuk membuka tracking UI:")
print("👉 mlflow ui --port 5000")