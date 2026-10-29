
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
INPUT_FILE = BASE / "KPI4" / "factory_recommendations_ranked.csv"

df = pd.read_csv(INPUT_FILE)

required = [
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Recommendation Status",
    "Estimated Net Profit Impact",
]

missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required].isna().any().any():
    print("WARNING: Missing values exist in required columns.")
    print(df[required].isna().sum().to_string())
else:
    print("Required columns have no missing values.")

print("\n--- DATASET OVERVIEW ---")
print("Recommendation rows:", len(df))
print("Unique products:", df["Product Name"].nunique())
print("Unique current factories:", df["Current Factory"].nunique())
print("Unique scenario factories:", df["Scenario Factory"].nunique())

print("\n--- ALTERNATIVES PER PRODUCT ---")
alternatives = df.groupby("Product Name").agg(
    Alternative_Count=("Scenario Factory", "nunique"),
    Current_Factory_Count=("Current Factory", "nunique"),
    Priority_Recommendations=(
        "Recommendation Status",
        lambda x: (x == "Priority for validation").sum()
    ),
    Review_Recommendations=(
        "Recommendation Status",
        lambda x: (x == "Review before considering").sum()
    ),
    Positive_Impact_Alternatives=(
        "Estimated Net Profit Impact",
        lambda x: (x > 0).sum()
    ),
)

print(alternatives.to_string())

print("\n--- COVERAGE CHECKS ---")
print("Products with at least one alternative:",
      int((alternatives["Alternative_Count"] >= 1).sum()))

print("Products with at least one priority recommendation:",
      int((alternatives["Priority_Recommendations"] >= 1).sum()))

print("Products with no priority recommendation:",
      int((alternatives["Priority_Recommendations"] == 0).sum()))

print("Products with at least one positive estimated impact:",
      int((alternatives["Positive_Impact_Alternatives"] >= 1).sum()))

print("\n--- STATUS COUNTS ---")
print(df["Recommendation Status"].value_counts(dropna=False).to_string())

print("\n--- KPI 4 COVERAGE AUDIT COMPLETE ---")
