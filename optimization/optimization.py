
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# STEP 6: OPTIMIZATION & RECOMMENDATION LOGIC
# --------------------------------------------------

INPUT_FILE = "scenario_profit_sensitivity_corrected.csv"

# Select the assumptions to use for the main ranking.
FREIGHT_COST = 0.01
TRANSFER_COST = 0

# Number of alternative factories to recommend per product.
TOP_N = 3

# Ranking weights (must total 1.0).
WEIGHT_DISTANCE = 0.40
WEIGHT_PROFIT = 0.40
WEIGHT_RISK = 0.20

OUTPUT_ALL = "factory_recommendations_ranked.csv"
OUTPUT_TOP = "top_n_factory_recommendations.csv"
OUTPUT_SUMMARY = "step6_recommendation_summary.csv"


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Put this script in the same "
            "folder as your Step 5 CSV."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "Product Name",
        "Current Factory",
        "Scenario Factory",
        "Eligible Order Rows",
        "Eligible Units",
        "Potential Distance Reduction (km)",
        "Estimated Net Profit Impact",
        "Assumed Freight Cost per km-unit",
        "Assumed Transfer Cost per Unit",
        "Positive Under Assumptions",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only one selected cost scenario.
    df = df[
        np.isclose(
            pd.to_numeric(
                df["Assumed Freight Cost per km-unit"],
                errors="coerce"
            ),
            FREIGHT_COST
        )
        &
        np.isclose(
            pd.to_numeric(
                df["Assumed Transfer Cost per Unit"],
                errors="coerce"
            ),
            TRANSFER_COST
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows match the selected freight and transfer costs. "
            "Check the assumptions against the Step 5 CSV."
        )

    # Convert numeric columns safely.
    numeric_cols = [
        "Eligible Order Rows",
        "Eligible Units",
        "Potential Distance Reduction (km)",
        "Estimated Net Profit Impact",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude a scenario if it is not a reassignment.
    df = df[
        df["Scenario Factory"].notna()
        & df["Current Factory"].notna()
        & (
            df["Scenario Factory"].astype(str).str.strip()
            != df["Current Factory"].astype(str).str.strip()
        )
    ].copy()

    if df.empty:
        raise ValueError("No valid alternative-factory scenarios found.")

    # Positive distance reduction means a shorter distance.
    # Negative distance reduction means a longer distance.
    df["Distance Benefit"] = df[
        "Potential Distance Reduction (km)"
    ]

    # Risk cannot be quantified from this CSV alone.
    # Use a neutral ranking component and flag risk as unknown.
    df["Risk Reduction Score"] = np.nan
    df["Risk Status"] = (
        "Unknown - operational risk data unavailable"
    )

    # Normalize distance benefits across scenarios.
    def minmax(series):
        low = series.min()
        high = series.max()

        if pd.isna(low) or pd.isna(high) or high == low:
            return pd.Series(0.5, index=series.index)

        return (series - low) / (high - low)

    df["Distance Score"] = minmax(df["Distance Benefit"])

    # Positive profit impact is better.
    df["Profit Score"] = minmax(
        df["Estimated Net Profit Impact"]
    )

    # Do not pretend unknown risk is measured.
    # The risk component is neutral for all alternatives.
    df["Risk Score Used"] = 0.5

    # Weighted overall ranking score, from 0 to 100.
    df["Overall Score"] = 100 * (
        WEIGHT_DISTANCE * df["Distance Score"]
        + WEIGHT_PROFIT * df["Profit Score"]
        + WEIGHT_RISK * df["Risk Score Used"]
    )

    # Label each scenario.
    df["Profit Assessment"] = np.where(
        df["Estimated Net Profit Impact"] > 0,
        "Positive under selected assumptions",
        np.where(
            df["Estimated Net Profit Impact"] < 0,
            "Negative under selected assumptions",
            "Break-even under selected assumptions"
        )
    )

    df["Distance Assessment"] = np.where(
        df["Distance Benefit"] > 0,
        "Distance reduction",
        np.where(
            df["Distance Benefit"] < 0,
            "Distance increase",
            "No distance change"
        )
    )

    # Rank alternatives within each product.
    df = df.sort_values(
        [
            "Product Name",
            "Overall Score",
            "Estimated Net Profit Impact",
            "Potential Distance Reduction (km)",
        ],
        ascending=[True, False, False, False]
    )

    df["Recommendation Rank"] = (
        df.groupby("Product Name").cumcount() + 1
    )

    df["Recommendation Status"] = np.where(
        (df["Estimated Net Profit Impact"] > 0)
        & (df["Potential Distance Reduction (km)"] > 0),
        "Priority for validation",
        "Review before considering"
    )

    # Export the complete ranking.
    df.to_csv(OUTPUT_ALL, index=False)

    # Export the top N alternatives for each product.
    top = df[df["Recommendation Rank"] <= TOP_N].copy()
    top.to_csv(OUTPUT_TOP, index=False)

    # Summarize the results by product.
    summary = (
        df.groupby("Product Name")
        .agg(
            Current_Factory=("Current Factory", "first"),
            Alternatives_Evaluated=("Scenario Factory", "nunique"),
            Best_Alternative=("Scenario Factory", "first"),
            Best_Overall_Score=("Overall Score", "first"),
            Best_Distance_Reduction_km=(
                "Potential Distance Reduction (km)", "first"
            ),
            Best_Estimated_Profit_Impact=(
                "Estimated Net Profit Impact", "first"
            ),
            Positive_Impact_Alternatives=(
                "Estimated Net Profit Impact",
                lambda x: int((x > 0).sum())
            ),
            Negative_Impact_Alternatives=(
                "Estimated Net Profit Impact",
                lambda x: int((x < 0).sum())
            ),
        )
        .reset_index()
    )

    summary["Risk Assessment"] = (
        "Unknown - validate operational risk separately"
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n--- STEP 6 COMPLETED ---")
    print("Cost assumptions:")
    print("Freight cost per km-unit:", FREIGHT_COST)
    print("Transfer cost per unit:", TRANSFER_COST)
    print("Top N per product:", TOP_N)

    print("\nAlternative scenarios evaluated:", len(df))
    print("Products evaluated:", df["Product Name"].nunique())
    print("Top-N recommendations:", len(top))
    print(
        "Positive-profit scenarios:",
        int((df["Estimated Net Profit Impact"] > 0).sum())
    )
    print(
        "Negative-profit scenarios:",
        int((df["Estimated Net Profit Impact"] < 0).sum())
    )

    print("\nTop recommendations preview:")
    preview_cols = [
        "Product Name",
        "Current Factory",
        "Scenario Factory",
        "Potential Distance Reduction (km)",
        "Estimated Net Profit Impact",
        "Overall Score",
        "Recommendation Rank",
    ]
    print(top[preview_cols].head(15).to_string(index=False))

    print("\nFiles generated:")
    print("-", OUTPUT_ALL)
    print("-", OUTPUT_TOP)
    print("-", OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
