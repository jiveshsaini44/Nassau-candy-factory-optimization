
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

files = [
    "factory_recommendations_ranked.csv",
    "top_n_factory_recommendations.csv",
    "kpi2_alternative_stability.csv",
]

for filename in files:
    path = BASE / filename

    print(f"\n{'=' * 65}")
    print("FILE:", filename)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    print("\nMissing values by column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any()
          else "No missing values")

    if "Overall Score" in df.columns:
        print("\nOverall Score summary:")
        print(df["Overall Score"].describe().to_string())

    if "Risk Status" in df.columns:
        print("\nRisk Status counts:")
        print(df["Risk Status"].value_counts(dropna=False).to_string())

    if "Recommendation Status" in df.columns:
        print("\nRecommendation Status counts:")
        print(
            df["Recommendation Status"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "Recommendation Rank" in df.columns:
        print("\nRecommendation Rank summary:")
        print(
            df["Recommendation Rank"]
            .describe()
            .to_string()
        )

    if "Stability_Rate_Pct" in df.columns:
        print("\nStability rate distribution:")
        print(
            df["Stability_Rate_Pct"]
            .describe()
            .to_string()
        )

print("\n--- QUALITY INSPECTION COMPLETE ---")
