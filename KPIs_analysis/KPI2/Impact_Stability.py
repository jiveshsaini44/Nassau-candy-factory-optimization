
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "scenario_profit_sensitivity_corrected.csv"

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip()

required = [
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Assumed Freight Cost per km-unit",
    "Assumed Transfer Cost per Unit",
    "Estimated Net Profit Impact",
    "Positive Under Assumptions",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Estimated Net Profit Impact"] = pd.to_numeric(
    df["Estimated Net Profit Impact"], errors="coerce"
)

# Use the numeric impact itself to determine profitability.
df = df.dropna(subset=["Estimated Net Profit Impact"]).copy()
df["Is Positive"] = df["Estimated Net Profit Impact"] > 0
df["Is Negative"] = df["Estimated Net Profit Impact"] < 0
df["Is Break Even"] = df["Estimated Net Profit Impact"] == 0

print("\n--- DATA VALIDATION ---")
print("Rows evaluated:", len(df))
print("Unique products:", df["Product Name"].nunique())
print("Unique product-factory alternatives:",
      df[["Product Name", "Current Factory", "Scenario Factory"]]
      .drop_duplicates().shape[0])
print("Freight assumptions:",
      sorted(df["Assumed Freight Cost per km-unit"].unique()))
print("Transfer assumptions:",
      sorted(df["Assumed Transfer Cost per Unit"].unique()))

# Overall row-level results across all tested assumptions.
n = len(df)
print("\n--- OVERALL PROFIT IMPACT RESULTS ---")
print(f"Positive rows: {df['Is Positive'].sum()} ({df['Is Positive'].mean()*100:.2f}%)")
print(f"Negative rows: {df['Is Negative'].sum()} ({df['Is Negative'].mean()*100:.2f}%)")
print(f"Break-even rows: {df['Is Break Even'].sum()} ({df['Is Break Even'].mean()*100:.2f}%)")
print(f"Average estimated net impact: {df['Estimated Net Profit Impact'].mean():.2f}")
print(f"Median estimated net impact: {df['Estimated Net Profit Impact'].median():.2f}")
print(f"Minimum estimated net impact: {df['Estimated Net Profit Impact'].min():.2f}")
print(f"Maximum estimated net impact: {df['Estimated Net Profit Impact'].max():.2f}")

# Profitability for each freight/transfer assumption combination.
by_assumption = (
    df.groupby([
        "Assumed Freight Cost per km-unit",
        "Assumed Transfer Cost per Unit"
    ])
    .agg(
        Scenarios=("Estimated Net Profit Impact", "size"),
        Positive=("Is Positive", "sum"),
        Negative=("Is Negative", "sum"),
        Break_Even=("Is Break Even", "sum"),
        Average_Net_Impact=("Estimated Net Profit Impact", "mean"),
        Median_Net_Impact=("Estimated Net Profit Impact", "median"),
    )
    .reset_index()
)

by_assumption["Positive_Rate_Pct"] = (
    by_assumption["Positive"] / by_assumption["Scenarios"] * 100
)

print("\n--- RESULTS BY COST ASSUMPTIONS ---")
print(by_assumption.round(2).to_string(index=False))

# Each alternative should have one result for each tested assumption.
keys = ["Product Name", "Current Factory", "Scenario Factory"]
stability = (
    df.groupby(keys)
    .agg(
        Assumptions_Tested=("Is Positive", "size"),
        Positive_Assumptions=("Is Positive", "sum"),
        Average_Net_Impact=("Estimated Net Profit Impact", "mean"),
        Worst_Net_Impact=("Estimated Net Profit Impact", "min"),
        Best_Net_Impact=("Estimated Net Profit Impact", "max"),
    )
    .reset_index()
)

stability["Stability_Rate_Pct"] = (
    stability["Positive_Assumptions"]
    / stability["Assumptions_Tested"] * 100
)

print("\n--- ALTERNATIVE-LEVEL STABILITY ---")
print("Alternatives profitable under all tested assumptions:",
      (stability["Stability_Rate_Pct"] == 100).sum())
print("Alternatives profitable under at least one assumption:",
      (stability["Positive_Assumptions"] > 0).sum())
print("Alternatives never profitable:",
      (stability["Positive_Assumptions"] == 0).sum())
print("\nTop 10 alternatives by stability rate, then average impact:")
print(
    stability.sort_values(
        ["Stability_Rate_Pct", "Average_Net_Impact"],
        ascending=[False, False]
    ).head(10).round(2).to_string(index=False)
)

# Save detailed results for later dashboard integration.
by_assumption.to_csv(
    BASE / "kpi2_results_by_assumption.csv", index=False
)
stability.to_csv(
    BASE / "kpi2_alternative_stability.csv", index=False
)

print("\nFiles generated:")
print("- kpi2_results_by_assumption.csv")
print("- kpi2_alternative_stability.csv")
