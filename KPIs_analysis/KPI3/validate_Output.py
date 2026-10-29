
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

detail_path = BASE / "kpi3_scenario_confidence.csv"
summary_path = BASE / "kpi3_confidence_summary.csv"

df = pd.read_csv(detail_path)
summary = pd.read_csv(summary_path)

print("--- KPI 3 OUTPUT VALIDATION ---")
print("Detail rows:", len(df))
print("Summary rows:", len(summary))
print("Missing confidence scores:",
      df["Scenario_Confidence_Score"].isna().sum())
print("Confidence score outside 0-100:",
      ((df["Scenario_Confidence_Score"] < 0) |
       (df["Scenario_Confidence_Score"] > 100)).sum())

print("\nConfidence categories:")
print(df["Confidence_Category"].value_counts().to_string())

print("\nSummary table:")
print(summary.to_string(index=False))

print("\nDuplicate product-factory alternatives:",
      df.duplicated(
          ["Product Name", "Current Factory", "Scenario Factory"]
      ).sum())

print("\n--- VALIDATION COMPLETE ---")
