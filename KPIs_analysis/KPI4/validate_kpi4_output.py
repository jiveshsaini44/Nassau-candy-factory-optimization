
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

DETAIL_FILE = BASE / "kpi4_product_coverage.csv"
SUMMARY_FILE = BASE / "kpi4_coverage_summary.csv"

for path in [DETAIL_FILE, SUMMARY_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

detail = pd.read_csv(DETAIL_FILE)
summary = pd.read_csv(SUMMARY_FILE)

print("--- KPI 4 OUTPUT VALIDATION ---")

# 1. Check file sizes
print("\n1. ROW COUNTS")
print("Product detail rows:", len(detail))
print("Summary metric rows:", len(summary))

# 2. Check missing values
print("\n2. MISSING VALUES")
print("Detail missing values:", int(detail.isna().sum().sum()))
print("Summary missing values:", int(summary.isna().sum().sum()))

# 3. Check product uniqueness
print("\n3. PRODUCT UNIQUENESS")
print("Unique products:", detail["Product Name"].nunique())
print("Duplicate product rows:",
      int(detail["Product Name"].duplicated().sum()))

# 4. Recalculate coverage directly from detail file
total_products = len(detail)

products_with_alternatives = int(
    detail["Has_Alternative"].astype(bool).sum()
)
products_with_priority = int(
    detail["Has_Priority_Recommendation"].astype(bool).sum()
)
products_with_positive_impact = int(
    detail["Has_Positive_Impact_Alternative"].astype(bool).sum()
)

print("\n4. PRODUCT-LEVEL COVERAGE")
print(
    "Products with alternatives:",
    products_with_alternatives,
    "/",
    total_products
)
print(
    "Products with priority recommendations:",
    products_with_priority,
    "/",
    total_products
)
print(
    "Products with positive-impact alternatives:",
    products_with_positive_impact,
    "/",
    total_products
)

# 5. Recalculate alternative priority rate from summary numerator
ranked_input = (
    BASE.parent / "KPI4" / "factory_recommendations_ranked.csv"
)
if not ranked_input.exists():
    raise FileNotFoundError(f"Source file not found: {ranked_input}")

recommendations = pd.read_csv(ranked_input)

priority_count = int(
    (recommendations["Recommendation Status"] ==
     "Priority for validation").sum()
)
total_alternatives = len(recommendations)

print("\n5. ALTERNATIVE-LEVEL COVERAGE")
print("Total alternatives:", total_alternatives)
print("Priority alternatives:", priority_count)

# 6. Verify summary percentages against the source data
expected = {
    "Product Recommendation Coverage": (
        products_with_alternatives, total_products
    ),
    "Validation-Priority Coverage": (
        products_with_priority, total_products
    ),
    "Alternative Priority Rate": (
        priority_count, total_alternatives
    ),
    "Positive-Impact Product Coverage": (
        products_with_positive_impact, total_products
    ),
}

print("\n6. SUMMARY RECONCILIATION")

all_match = True

for kpi, (numerator, denominator) in expected.items():
    expected_pct = round(numerator / denominator * 100, 2)

    row = summary.loc[summary["KPI"] == kpi]

    if len(row) != 1:
        print(f"{kpi}: FAIL - expected exactly one summary row")
        all_match = False
        continue

    stored_numerator = int(row.iloc[0]["Numerator"])
    stored_denominator = int(row.iloc[0]["Denominator"])
    stored_pct = round(float(row.iloc[0]["Coverage_Pct"]), 2)

    passed = (
        stored_numerator == numerator
        and stored_denominator == denominator
        and stored_pct == expected_pct
    )

    print(
        f"{kpi}: {'PASS' if passed else 'FAIL'} | "
        f"Expected {numerator}/{denominator} = {expected_pct}% | "
        f"Stored {stored_numerator}/{stored_denominator} = "
        f"{stored_pct}%"
    )

    if not passed:
        all_match = False

# 7. Check products requiring review
print("\n7. PRODUCTS REQUIRING REVIEW")

review = detail.loc[
    ~detail["Has_Priority_Recommendation"].astype(bool),
    ["Product Name", "Alternative_Count", "Coverage_Status"]
]

if review.empty:
    print("None")
else:
    print(review.to_string(index=False))

# 8. Final status
print("\n--- VALIDATION RESULT ---")

basic_checks = (
    len(detail) == 15
    and detail["Product Name"].nunique() == 15
    and detail["Product Name"].duplicated().sum() == 0
    and detail.isna().sum().sum() == 0
    and summary.isna().sum().sum() == 0
    and len(summary) == 4
)

if basic_checks and all_match:
    print("PASS: KPI 4 output files passed all validation checks.")
else:
    print("REVIEW REQUIRED: One or more checks did not pass.")

print("\n--- KPI 4 VALIDATION COMPLETE ---")
