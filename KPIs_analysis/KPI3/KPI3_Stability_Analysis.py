
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

file_path = BASE / "kpi2_alternative_stability.csv"
df = pd.read_csv(file_path)

# Validate the expected number of assumption tests
print("--- ASSUMPTION TEST VALIDATION ---")
print("Number of alternatives:", len(df))
print("Assumption counts:")
print(df["Assumptions_Tested"].value_counts(dropna=False).sort_index())

# Group alternatives by profit stability
def classify_stability(rate):
    if rate == 100:
        return "High: profitable under all assumptions"
    elif rate == 0:
        return "Low: profitable under no assumptions"
    else:
        return "Mixed: profitable under some assumptions"

df["Stability_Group"] = df["Stability_Rate_Pct"].apply(
    classify_stability
)

print("\n--- STABILITY GROUP COUNTS ---")
print(df["Stability_Group"].value_counts().to_string())

print("\n--- STABILITY GROUP SUMMARY ---")
summary = (
    df.groupby("Stability_Group", dropna=False)
    .agg(
        Alternatives=("Product Name", "count"),
        Average_Stability=("Stability_Rate_Pct", "mean"),
        Average_Net_Impact=("Average_Net_Impact", "mean"),
        Average_Worst_Impact=("Worst_Net_Impact", "mean"),
    )
    .round(2)
)
print(summary.to_string())

print("\n--- LOWEST STABILITY ALTERNATIVES ---")
cols = [
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Positive_Assumptions",
    "Assumptions_Tested",
    "Stability_Rate_Pct",
    "Average_Net_Impact",
    "Worst_Net_Impact",
]
print(
    df.sort_values(
        ["Stability_Rate_Pct", "Average_Net_Impact"]
    )[cols].head(10).to_string(index=False)
)

print("\n--- HIGHEST STABILITY ALTERNATIVES ---")
print(
    df.sort_values(
        ["Stability_Rate_Pct", "Worst_Net_Impact"],
        ascending=[False, False]
    )[cols].head(10).to_string(index=False)
)

print("\n--- KPI 3 STABILITY REVIEW COMPLETE ---")
