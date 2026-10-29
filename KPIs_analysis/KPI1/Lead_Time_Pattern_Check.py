
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE.parent / "Nassau Candy Distributor.csv"

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="%d-%m-%Y", errors="coerce"
)
df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], format="%d-%m-%Y", errors="coerce"
)

df["Lead Time (days)"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

print("\n--- LEAD TIME BY SHIP MODE ---")
print(
    df.groupby("Ship Mode")["Lead Time (days)"]
      .agg(["count", "min", "median", "mean", "max"])
      .round(2)
      .to_string()
)

print("\n--- LEAD TIME BY ORDER YEAR ---")
df["Order Year"] = df["Order Date"].dt.year
print(
    df.groupby("Order Year")["Lead Time (days)"]
      .agg(["count", "min", "median", "mean", "max"])
      .round(2)
      .to_string()
)

print("\n--- ORDER-TO-SHIP DATE GAP DISTRIBUTION ---")
print(
    df["Lead Time (days)"]
      .describe(percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
      .round(2)
      .to_string()
)

print("\n--- DATE PARSING VALIDATION ---")
print("Missing order dates:", df["Order Date"].isna().sum())
print("Missing ship dates:", df["Ship Date"].isna().sum())
print("Ship date before order date:",
      (df["Ship Date"] < df["Order Date"]).sum())
print("Lead times under 365 days:",
      (df["Lead Time (days)"] < 365).sum())
