
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "Nassau Candy Distributor.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Cannot find: {INPUT}")

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

required = ["Order Date", "Ship Date", "Product Name", "Region", "Ship Mode"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates using the same convention as the existing app.
df["Order Date Parsed"] = pd.to_datetime(
    df["Order Date"], dayfirst=True, errors="coerce"
)
df["Ship Date Parsed"] = pd.to_datetime(
    df["Ship Date"], dayfirst=True, errors="coerce"
)

df["Lead Time (days)"] = (
    df["Ship Date Parsed"] - df["Order Date Parsed"]
).dt.days

lead = df["Lead Time (days)"]
valid = lead.dropna()

print("\n--- LEAD-TIME DATA AUDIT ---")
print(f"Total order rows: {len(df):,}")
print(f"Valid calculated lead times: {len(valid):,}")
print(f"Missing/invalid date calculations: {lead.isna().sum():,}")
print(f"Negative lead times: {(lead < 0).sum():,}")
print(f"Lead times over 365 days: {(lead > 365).sum():,}")

if len(valid):
    print("\n--- DISTRIBUTION (DAYS) ---")
    print(valid.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).to_string())

# Summarize by shipping mode to find unusual patterns.
mode_summary = (
    df.groupby("Ship Mode", dropna=False)
      .agg(
          Order_Rows=("Lead Time (days)", "size"),
          Valid_Lead_Times=("Lead Time (days)", "count"),
          Median_Lead_Time=("Lead Time (days)", "median"),
          Mean_Lead_Time=("Lead Time (days)", "mean"),
          Negative_Lead_Times=("Lead Time (days)", lambda s: (s < 0).sum()),
          Over_365_Days=("Lead Time (days)", lambda s: (s > 365).sum()),
      )
      .reset_index()
)

mode_summary.to_csv(BASE / "lead_time_audit_by_ship_mode.csv", index=False)

# Save suspicious records for inspection, without changing the original data.
review = df[
    df["Lead Time (days)"].isna()
    | (df["Lead Time (days)"] < 0)
    | (df["Lead Time (days)"] > 365)
].copy()

review.to_csv(BASE / "lead_time_records_for_review.csv", index=False)

print("\nFiles generated:")
print("- lead_time_audit_by_ship_mode.csv")
print("- lead_time_records_for_review.csv")
print("\nAudit completed. Inspect the flagged records before modelling.")
