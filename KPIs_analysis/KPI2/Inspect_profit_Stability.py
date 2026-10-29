
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
PROJECT = BASE.parent

filename = "scenario_profit_sensitivity_corrected.csv"

matches = list(PROJECT.rglob(filename))

if not matches:
    print(f"\nERROR: {filename} was not found.")
    print("Search folder:", PROJECT)
    print("\nCSV files found in the project:")
    for path in PROJECT.rglob("*.csv"):
        print("-", path.relative_to(PROJECT))
else:
    for file_path in matches:
        print("\n--- FILE FOUND ---")
        print(file_path)

        df = pd.read_csv(file_path)

        print("\n--- DATASET SHAPE ---")
        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\n--- COLUMN NAMES ---")
        for col in df.columns:
            print("-", col)

        print("\n--- FIRST 10 RECORDS ---")
        print(df.head(10).to_string(index=False))

        print("\n--- NUMERIC COLUMN SUMMARY ---")
        print(df.describe(include="number").round(3).to_string())

        print("\n--- MISSING VALUES ---")
        print(df.isna().sum().to_string())
