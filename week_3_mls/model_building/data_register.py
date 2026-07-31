from pathlib import Path
import os
import pandas as pd
from github import Github, GithubException

# paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CSV_LOCAL = PROJECT_DIR / "data" / "machine-failure-prediction.csv"

# read dataset
df = pd.read_csv(CSV_LOCAL)

# save/register file
CSV_LOCAL.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_LOCAL, index=False)

print("Dataset registered at:", CSV_LOCAL)
print("Shape:", df.shape)
