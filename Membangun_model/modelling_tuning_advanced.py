"""
Fine-tuning Model - Mansur (ADVANCED)
-------------------------------
Hyperparameter tuning + MLflow (Manual Logging Online DagsHub).
Sesuai kriteria 4 poin.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# Import metrik tambahan
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# === 1. Setup MLflow untuk DagsHub (ADVANCED) ===
DAGSHUB_URI = "https://dagshub.com/Mansurgayo/Eksperimen_SML_Mansur.mlflow"
mlflow.set_tracking_uri(DAGSHUB_URI)
mlflow.set_experiment("Eksperimen_Tuning_Advanced")

# PENTING: Set environment variables untuk otentikasi DagsHub
# Anda bisa set ini di terminal sebelum menjalankan script:
# export MLFLOW_TRACKING_USERNAME=NAMA_USER_DAGSHUB
# export MLFLOW_TRACKING_PASSWORD=TOKEN_DAGSHUB_ANDA
# os.environ["MLFLOW_TRACKING_USERNAME"] = "NAMA_USER_DAGSHUB"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "TOKEN_DAGSHUB_ANDA"

# === 2. Hapus autolog() ===
# (Sama seperti skilled, manual log wajib)

# === 3. Load dataset ===
try:
    data = pd.read_csv("namadataset_preprocessing/cleaned_dataset.csv")
except FileNotFoundError:
    print("Error: File 'namadataset_preprocessing/cleaned_dataset.csv' tidak ditemukan.")
    exit()

if "Outcome" not in data.columns:
    print("Error: Kolom 'Outcome' tidak ditemukan di dataset.")
    exit()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 5. Mulai Run MLflow ===
with mlflow.start_run(run_name="LogReg_Advanced_DagsHub"):

    print("🚀 Mulai proses fine-tuning (Advanced - DagsHub)...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=500))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"]
    }
    
    # === 6. MANUAL LOGGING (Parameters) ===
    mlflow.log_params(param_grid)
    mlflow.log_param("cv", 5)
    mlflow.log_param("scoring", "f1")
    mlflow.log_param("dataset_shape", str(data.shape)) # Contoh tag/param tambahan

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Hitung semua metrik
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # === Metrik Tambahan (Syarat Advanced) ===
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n=== Hasil Fine-tuning (Advanced) ===")
    print("Best Params :", grid.best_params_)
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Precision   : {precision:.4f} (Metrik Tambahan 1)")
    print(f"Recall      : {recall:.4f} (Metrik Tambahan 2)")

    # === 7. MANUAL LOGGING (Metrics) ===
    # Log metrik utama
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score_f1", grid.best_score_)
    
    # Log metrik tambahan
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # === 8. MANUAL LOGGING (Artifacts) ===
    # a. Simpan model
    mlflow.sklearn.log_model(best_model, "best_estimator") 

    # b. Simpan confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot()
    plt.title("Confusion Matrix (Test Set)")
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    
    print(f"\n✅ Model dan artifak terkirim ke DagsHub (Run: LogReg_Advanced_DagsHub).")


print("\n✅ Fine-tuning (Advanced) selesai.")
print(f"👉 Cek hasilnya di DagsHub: {DAGSHUB_URI}")