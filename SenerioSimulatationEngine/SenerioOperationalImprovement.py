
import pandas as pd
import numpy as np

# Load existing Step 5A and 5B outputs
distance = pd.read_csv("scenario_factory_comparison.csv")
lead_time = pd.read_csv("scenario_lead_time_predictions.csv")

keys = ["Product Name", "Current Factory", "Scenario Factory"]

# Check that each product-factory scenario appears only once
if distance.duplicated(keys).any():
    raise ValueError("Duplicate scenarios found in distance results.")

if lead_time.duplicated(keys).any():
    raise ValueError("Duplicate scenarios found in lead-time results.")

# Combine the two analyses
df = distance.merge(
    lead_time,
    on=keys,
    how="inner",
    suffixes=("_distance", "_leadtime"),
    validate="one_to_one"
)

# Calculate percentage reduction in predicted lead time
df["Lead Time Reduction (%)"] = np.where(
    df["Current_Predicted_Lead_Time"] > 0,
    (
        df["Estimated Lead Time Reduction (days)"]
        / df["Current_Predicted_Lead_Time"]
    ) * 100,
    np.nan
)

# Identify alternatives that improve both measures
df["Distance Improved"] = (
    df["Potential Distance Reduction (km)"] > 0
)

df["Predicted Lead Time Improved"] = (
    df["Estimated Lead Time Reduction (days)"] > 0
)

df["Improves Both Measures"] = (
    df["Distance Improved"]
    & df["Predicted Lead Time Improved"]
)

# Rank scenarios by lead-time reduction, then distance reduction
alternatives = df[
    df["Scenario Factory"] != df["Current Factory"]
].copy()

alternatives = alternatives.sort_values(
    by=[
        "Improves Both Measures",
        "Estimated Lead Time Reduction (days)",
        "Potential Distance Reduction (km)"
    ],
    ascending=[False, False, False]
)

alternatives["Recommendation Rank"] = range(
    1, len(alternatives) + 1
)

# Save full report and a shortlist
df.to_csv("scenario_operational_improvement.csv", index=False)

alternatives.to_csv(
    "ranked_factory_recommendations.csv",
    index=False
)

print("\n--- STEP 5C: OPERATIONAL IMPROVEMENT ---")
print("Total scenarios:", len(df))
print("Alternative scenarios:", len(alternatives))
print(
    "Alternatives improving both measures:",
    int(alternatives["Improves Both Measures"].sum())
)

print("\n--- TOP 10 CANDIDATE SCENARIOS ---")
columns = [
    "Recommendation Rank",
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Potential Distance Reduction (km)",
    "Estimated Lead Time Reduction (days)",
    "Lead Time Reduction (%)",
    "Improves Both Measures"
]

print(
    alternatives[columns]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nCreated:")
print("- scenario_operational_improvement.csv")
print("- ranked_factory_recommendations.csv")
