
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "machine-failure-prediction.csv"

# -----------------------------
# Load dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# -----------------------------
# Drop unnecessary columns
# -----------------------------
drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

existing_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_cols)

# -----------------------------
# Target
# -----------------------------
TARGET_COL = "Machine failure"

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------------
# Train/Test Split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Save splits
# -----------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Training samples : {len(Xtrain)}")
print(f"Testing samples  : {len(Xtest)}")
