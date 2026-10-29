
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE.parent / "Nassau Candy Distributor.csv"

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

for col in ["Order Date", "Ship Date"]:
    df[col + " (parsed)"] = pd.to_datetime(
        df[col], dayfirst=True, errors="coerce"
    )

print("\n--- ORIGINAL DATE VALUES ---")
print(df[["Order Date", "Ship Date"]].head(15).to_string(index=False))

print("\n--- PARSED DATE RANGES ---")
for col in ["Order Date (parsed)", "Ship Date (parsed)"]:
    print(f"\n{col}")
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Unique dates:", df[col].nunique())

print("\n--- COMMON ORDER DATE VALUES ---")
print(df["Order Date"].value_counts().head(10).to_string())

print("\n--- COMMON SHIP DATE VALUES ---")
print(df["Ship Date"].value_counts().head(10).to_string())

print("\n--- SAMPLE CALCULATED LEAD TIMES ---")
sample = df[["Order Date", "Ship Date",
             "Order Date (parsed)", "Ship Date (parsed)"]].head(15).copy()
sample["Lead Time (days)"] = (
    sample["Ship Date (parsed)"] - sample["Order Date (parsed)"]
).dt.days
print(sample.to_string(index=False))
