from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "machine-failure-prediction.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print(df.head())

TARGET_COL = "Machine failure"

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. Available columns: {df.columns.tolist()}"
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print("Saved files: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
