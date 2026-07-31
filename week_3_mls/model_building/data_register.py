from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CSV_PATH = DATA_DIR / "machine-failure-prediction.csv"

# Safety check
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {CSV_PATH}")

# Load dataset
df = pd.read_csv(CSV_PATH)

# Basic validation / registration info
print("Dataset loaded successfully.")
print(f"Path: {CSV_PATH}")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# Optional: save it back to ensure consistent formatting
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_PATH, index=False)

print(f"\nDataset registered and saved at: {CSV_PATH}")
