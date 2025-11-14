"""
Automate Preprocessing Script - Mansur
------------------------------------------
File ini menjalankan proses preprocessing secara otomatis berdasarkan hasil eksplorasi
dari Template_Eksperimen_MSML.ipynb.
"""

import pandas as pd
from pathlib import Path

# === 1. Path setup ===
# Karena folder 'diabetes' berada satu level dengan file ini
RAW_PATH = Path("diabetes/diabetes.csv")
OUTPUT_DIR = Path("namadataset_preprocessing")
OUTPUT_FILE = OUTPUT_DIR / "cleaned_dataset.csv"


# === 2. Load raw data ===
def load_data(path: Path) -> pd.DataFrame:
    print(f"📥 Memuat dataset dari {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(f"❌ File tidak ditemukan di lokasi: {path.resolve()}")
    df = pd.read_csv(path)
    print(f"✅ Data berhasil dimuat, jumlah baris: {len(df)}, kolom: {len(df.columns)}")
    return df


# === 3. Preprocessing ===
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Membersihkan data...")

    # Hapus baris kosong & duplikat
    df = df.dropna()
    df = df.drop_duplicates()

    # Contoh pembersihan kolom teks (opsional)
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()

    print("✨ Data berhasil dibersihkan.")
    return df


# === 4. Save preprocessed data ===
def save_data(df: pd.DataFrame, output_path: Path):
    print(f"💾 Menyimpan hasil preprocessing ke {output_path.resolve()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ File preprocessing tersimpan dengan sukses.")


# === 5. Main pipeline ===
def main():
    print("🚀 Memulai proses otomatisasi preprocessing...")
    df = load_data(RAW_PATH)
    df_clean = clean_data(df)
    save_data(df_clean, OUTPUT_FILE)
    print("🎉 Pipeline preprocessing selesai!")


if __name__ == "__main__":
    main()
