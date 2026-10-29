
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent

files = [
    BASE / "KPI4" / "kpi3_scenario_confidence.csv",
    BASE / "KPI4" / "kpi3_confidence_summary.csv",
    BASE / "KPI4" / "factory_recommendations_ranked.csv",
    BASE / "KPI4" / "top_n_factory_recommendations.csv",
]

for path in files:
    print("\n" + "=" * 70)
    print("FILE:", path.name)

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:")
    print(df.columns.tolist())

    if "Product Name" in df.columns:
        print("Unique products:", df["Product Name"].nunique())

    if "Scenario Factory" in df.columns:
        print("Unique scenario factories:",
              df["Scenario Factory"].nunique())

    if "Recommendation Status" in df.columns:
        print("\nRecommendation status counts:")
        print(df["Recommendation Status"].value_counts(
            dropna=False
        ).to_string())

    if "Confidence_Category" in df.columns:
        print("\nConfidence category counts:")
        print(df["Confidence_Category"].value_counts(
            dropna=False
        ).to_string())

print("\n--- KPI 4 INPUT INSPECTION COMPLETE ---")
