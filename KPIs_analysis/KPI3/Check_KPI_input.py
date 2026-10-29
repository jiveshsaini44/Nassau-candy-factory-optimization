
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

files_to_check = [
    "kpi2_alternative_stability.csv",
    "kpi2_results_by_assumption.csv",
    "scenario_profit_sensitivity_corrected.csv",
    "factory_recommendations_ranked.csv",
    "top_n_factory_recommendations.csv",
]

print("\n--- KPI 3 INPUT FILE AUDIT ---")

for filename in files_to_check:
    path = BASE / filename

    if not path.exists():
        print(f"\nNOT FOUND: {filename}")
        continue

    df = pd.read_csv(path)

    print(f"\nFILE: {filename}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Missing values:", int(df.isna().sum().sum()))

print("\n--- AUDIT COMPLETE ---")
