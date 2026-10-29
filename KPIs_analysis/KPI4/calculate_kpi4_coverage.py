
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
INPUT_FILE = (
    BASE.parent / "KPI4" / "factory_recommendations_ranked.csv"
)

DETAIL_FILE = BASE / "kpi4_product_coverage.csv"
SUMMARY_FILE = BASE / "kpi4_coverage_summary.csv"

df = pd.read_csv(INPUT_FILE)

required = [
    "Product Name",
    "Current Factory",
    "Scenario Factory",
    "Recommendation Status",
    "Estimated Net Profit Impact",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Required fields contain missing values.")

# Ensure each product has only one current factory.
current_factory_counts = df.groupby("Product Name")[
    "Current Factory"
].nunique()

if (current_factory_counts != 1).any():
    raise ValueError(
        "Some products map to multiple current factories. "
        "Resolve the mapping before calculating coverage."
    )

# Product-level coverage
product_coverage = (
    df.groupby("Product Name")
    .agg(
        Current_Factory=("Current Factory", "first"),
        Alternative_Count=("Scenario Factory", "nunique"),
        Priority_Recommendations=(
            "Recommendation Status",
            lambda s: (s == "Priority for validation").sum()
        ),
        Review_Recommendations=(
            "Recommendation Status",
            lambda s: (s == "Review before considering").sum()
        ),
        Positive_Impact_Alternatives=(
            "Estimated Net Profit Impact",
            lambda s: (s > 0).sum()
        ),
    )
    .reset_index()
)

product_coverage["Has_Alternative"] = (
    product_coverage["Alternative_Count"] > 0
)
product_coverage["Has_Priority_Recommendation"] = (
    product_coverage["Priority_Recommendations"] > 0
)
product_coverage["Has_Positive_Impact_Alternative"] = (
    product_coverage["Positive_Impact_Alternatives"] > 0
)

product_coverage["Coverage_Status"] = product_coverage.apply(
    lambda row: (
        "Priority option available"
        if row["Has_Priority_Recommendation"]
        else "No priority option - review alternatives"
    ),
    axis=1,
)

# Overall metrics
total_products = df["Product Name"].nunique()
total_alternatives = len(df)

products_with_alternatives = int(
    product_coverage["Has_Alternative"].sum()
)
products_with_priority = int(
    product_coverage["Has_Priority_Recommendation"].sum()
)
products_with_positive_impact = int(
    product_coverage["Has_Positive_Impact_Alternative"].sum()
)
priority_alternatives = int(
    (df["Recommendation Status"] == "Priority for validation").sum()
)

summary = pd.DataFrame([
    {
        "KPI": "Product Recommendation Coverage",
        "Numerator": products_with_alternatives,
        "Denominator": total_products,
        "Coverage_Pct": round(
            products_with_alternatives / total_products * 100, 2
        ),
        "Unit": "Percent of products with at least one alternative",
    },
    {
        "KPI": "Validation-Priority Coverage",
        "Numerator": products_with_priority,
        "Denominator": total_products,
        "Coverage_Pct": round(
            products_with_priority / total_products * 100, 2
        ),
        "Unit": "Percent of products with at least one priority option",
    },
    {
        "KPI": "Alternative Priority Rate",
        "Numerator": priority_alternatives,
        "Denominator": total_alternatives,
        "Coverage_Pct": round(
            priority_alternatives / total_alternatives * 100, 2
        ),
        "Unit": "Percent of alternatives marked priority",
    },
    {
        "KPI": "Positive-Impact Product Coverage",
        "Numerator": products_with_positive_impact,
        "Denominator": total_products,
        "Coverage_Pct": round(
            products_with_positive_impact / total_products * 100, 2
        ),
        "Unit": "Percent of products with a positive estimated impact",
    },
])

product_coverage.to_csv(DETAIL_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

print("--- KPI 4 RECOMMENDATION COVERAGE RESULTS ---")
print("Products:", total_products)
print("Alternatives:", total_alternatives)
print("Products with alternatives:", products_with_alternatives)
print("Products with priority recommendations:", products_with_priority)
print("Products without priority recommendations:",
      total_products - products_with_priority)
print("Priority alternatives:", priority_alternatives)

print("\nCoverage summary:")
print(summary.to_string(index=False))

print("\nProducts requiring review:")
print(
    product_coverage.loc[
        ~product_coverage["Has_Priority_Recommendation"],
        ["Product Name", "Current_Factory", "Alternative_Count",
         "Coverage_Status"]
    ].to_string(index=False)
)

print("\nFiles created:")
print(DETAIL_FILE)
print(SUMMARY_FILE)

print("\n--- KPI 4 CALCULATION COMPLETE ---")
