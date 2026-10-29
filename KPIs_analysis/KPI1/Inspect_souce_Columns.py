
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE.parent / "Nassau Candy Distributor.csv"

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

print("\n--- ALL SOURCE COLUMNS ---")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\n--- FIRST 5 COMPLETE RECORDS ---")
print(df.head().to_string(index=False))

print("\n--- DATE-RELATED COLUMNS ---")
date_cols = [
    col for col in df.columns
    if "date" in col.lower() or "time" in col.lower()
]
print(df[date_cols].head(10).to_string(index=False))

print("\n--- SHIPPING-RELATED COLUMNS ---")
shipping_cols = [
    col for col in df.columns
    if any(word in col.lower() for word in [
        "ship", "delivery", "freight", "factory", "warehouse",
        "region", "city", "state", "postal", "zip", "country"
    ])
]
print(shipping_cols)

for col in shipping_cols:
    print(f"\n{col}:")
    print("Unique values:", df[col].nunique(dropna=True))
    print(df[col].value_counts(dropna=False).head(8).to_string())
