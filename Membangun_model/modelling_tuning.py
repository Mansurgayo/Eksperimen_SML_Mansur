"""
Fine-tuning Model - Mansur (SKILLED)
-------------------------------
Hyperparameter tuning + MLflow (Manual Logging Lokal).
Sesuai kriteria 3 poin.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# === 1. Setup MLflow lebih awal ===
mlflow.set_tracking_uri("file:./mlruns")           # tracking lokal
mlflow.set_experiment("Eksperimen_Tuning_Mansur")  # Nama eksperimen di MLflow

# === 2. HAPUS autolog() ===
# mlflow.sklearn.autolog() <-- INI DIHAPUS/DI-COMMENT UNTUK SKILLED

# === 3. Load dataset ===
try:
    # Path disesuaikan, asumsi folder 'namadataset_preprocessing' ada di dalam 'Membangun_model'
    data = pd.read_csv("namadataset_preprocessing/cleaned_dataset.csv")
except FileNotFoundError:
    print("Error: File 'namadataset_preprocessing/cleaned_dataset.csv' tidak ditemukan.")
    print("Pastikan file tersebut ada di dalam folder 'Membangun_model'.")
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
with mlflow.start_run(run_name="LogReg_GridSearch_Manual"):

    print("🚀 Mulai proses fine-tuning (Skilled - Manual Log)...")

    # Pipeline: scaling + klasifikasi
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=500))
    ])

    # Hyperparameter grid
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"]
    }
    
    # === 6. MANUAL LOGGING (Parameters) ===
    # Log hyperparameter grid yang kita gunakan
    mlflow.log_params(param_grid)
    mlflow.log_param("cv", 5)
    mlflow.log_param("scoring", "f1")

    # GridSearchCV
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # Train model
    grid.fit(X_train, y_train)

    # Evaluasi hasil tuning
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n=== Hasil Fine-tuning (Skilled) ===")
    print("Best Params :", grid.best_params_)
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    # === 7. MANUAL LOGGING (Metrics) ===
    # Log metrik utama
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Log metrik dari parameter terbaik
    mlflow.log_metric("best_cv_score (f1)", grid.best_score_)
    
    # === 8. MANUAL LOGGING (Artifacts) ===
    # a. Simpan model
    # "best_estimator" adalah nama folder artefak model di MLflow
    mlflow.sklearn.log_model(best_model, "best_estimator") 

    # b. Simpan confusion matrix sebagai gambar
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot()
    plt.title("Confusion Matrix (Test Set)")
    
    # Simpan gambar ke file & log sebagai artifact
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    
    print(f"\n✅ Model dan Confusion Matrix tersimpan di MLflow (Run: LogReg_GridSearch_Manual).")


print("\n✅ Fine-tuning (Skilled) selesai.")
print("👉 Jalankan MLflow UI: mlflow ui --port 5000")