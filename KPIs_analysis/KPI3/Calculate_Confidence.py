
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

INPUT_FILE = BASE / "kpi2_alternative_stability.csv"
OUTPUT_FILE = BASE / "kpi3_scenario_confidence.csv"
SUMMARY_FILE = BASE / "kpi3_confidence_summary.csv"

df = pd.read_csv(INPUT_FILE)

# 1. Validate required fields
required = [
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Assumptions_Tested",
    "Positive_Assumptions",
    "Stability_Rate_Pct",
    "Average_Net_Impact",
    "Worst_Net_Impact",
    "Best_Net_Impact",
]

missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required].isna().any().any():
    raise ValueError("Required fields contain missing values.")

if not (df["Assumptions_Tested"] == 9).all():
    raise ValueError(
        "Expected exactly 9 tested assumptions for every alternative."
    )

# 2. Calculate the score directly from the tested scenarios
df["Scenario_Confidence_Score"] = (
    df["Positive_Assumptions"] / df["Assumptions_Tested"] * 100
).round(2)

# Verify the provided stability rate matches our calculation
difference = (
    df["Scenario_Confidence_Score"] - df["Stability_Rate_Pct"]
).abs()

if (difference > 0.02).any():
    raise ValueError(
        "Calculated score does not match the existing stability rate."
    )

# 3. Assign descriptive bands
def confidence_band(score):
    if score >= 90:
        return "High scenario robustness"
    elif score >= 50:
        return "Moderate scenario robustness"
    return "Low scenario robustness"

df["Confidence_Category"] = df["Scenario_Confidence_Score"].apply(
    confidence_band
)

# 4. Add interpretation and data limitations
df["Scenario_Evidence_Status"] = (
    "Complete: 9 of 9 cost assumptions tested"
)

df["Operational_Validation_Status"] = (
    "Unverified: operational risk and alternative-factory lead times unavailable"
)

df["Score_Interpretation"] = (
    "Percentage of tested cost assumptions with positive estimated net impact; "
    "not a probability of real-world success"
)

# 5. Sort strongest alternatives first
df = df.sort_values(
    ["Scenario_Confidence_Score", "Worst_Net_Impact", "Average_Net_Impact"],
    ascending=[False, False, False]
)

df.to_csv(OUTPUT_FILE, index=False)

# 6. Create a summary for KPI reporting
summary = (
    df.groupby("Confidence_Category")
    .agg(
        Alternatives=("Product Name", "count"),
        Average_Confidence_Score=("Scenario_Confidence_Score", "mean"),
        Average_Net_Impact=("Average_Net_Impact", "mean"),
        Average_Worst_Case_Impact=("Worst_Net_Impact", "mean"),
    )
    .round(2)
    .reset_index()
)

summary.to_csv(SUMMARY_FILE, index=False)

print("--- KPI 3 SCENARIO CONFIDENCE RESULTS ---")
print("Alternatives evaluated:", len(df))
print("Unique products:", df["Product Name"].nunique())
print("All alternatives tested under 9 assumptions:",
      bool((df["Assumptions_Tested"] == 9).all()))

print("\nConfidence category counts:")
print(df["Confidence_Category"].value_counts().to_string())

print("\nConfidence score summary:")
print(df["Scenario_Confidence_Score"].describe().round(2).to_string())

print("\nTop 10 alternatives:")
print(
    df[
        [
            "Product Name",
            "Current Factory",
            "Scenario Factory",
            "Scenario_Confidence_Score",
            "Confidence_Category",
            "Worst_Net_Impact",
            "Average_Net_Impact",
        ]
    ].head(10).to_string(index=False)
)

print("\nFiles created:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\n--- KPI 3 CALCULATION COMPLETE ---")
