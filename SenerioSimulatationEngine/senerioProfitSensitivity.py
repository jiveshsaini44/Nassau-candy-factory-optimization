"""
import pandas as pd
import numpy as np

# Step 5D: Corrected Profit Sensitivity Analysis

orders = pd.read_csv("Nassau Candy Distributor.csv")
routes = pd.read_csv("route_analysis_results.csv")

product_factory = {
    "Wonka Bar - Nutty Crunch Surprise": "Lots_O_Nuts",
    "Wonka Bar - Fudge Mallows": "Lots_O_Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lots_O_Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked_Choccys",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked_Choccys",
    "Laffy Taffy": "Sugar_Shack",
    "SweeTARTS": "Sugar_Shack",
    "Nerds": "Sugar_Shack",
    "Fun Dip": "Sugar_Shack",
    "Fizzy Lifting Drinks": "Sugar_Shack",
    "Everlasting Gobstopper": "Secret_Factory",
    "Hair Toffee": "The_Other_Factory",
    "Lickable Wallpaper": "Secret_Factory",
    "Wonka Gum": "Secret_Factory",
    "Kazookles": "The_Other_Factory",
}

factories = {
    "Lots_O_Nuts": "Distance_Lots_O_Nuts_km",
    "Wicked_Choccys": "Distance_Wicked_Choccys_km",
    "Sugar_Shack": "Distance_Sugar_Shack_km",
    "Secret_Factory": "Distance_Secret_Factory_km",
    "The_Other_Factory": "Distance_The_Other_Factory_km",
}

keys = [
    "Country/Region",
    "City",
    "State/Province",
    "Postal Code",
    "Region",
]

# 1. Standardize destination join keys
for data in [orders, routes]:
    for col in keys:
        data[col] = data[col].fillna("").astype(str).str.strip()

# Avoid duplicate destination matches
routes = routes.drop_duplicates(subset=keys)

# 2. Join each order to its factory distances
df = orders.merge(
    routes[keys + list(factories.values())],
    on=keys,
    how="left",
    validate="many_to_one"
)

df["Current Factory"] = df["Product Name"].map(product_factory)

for col in ["Units", "Sales", "Cost", "Gross Profit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    df["Current Factory"].notna()
    & df["Units"].notna()
    & (df["Units"] > 0)
    & df["Gross Profit"].notna()
].copy()

# 3. Illustrative assumptions only—not company-confirmed rates
freight_rates = [0.01, 0.05, 0.10]
transfer_costs = [0, 1, 5]

results = []

# 4. Compare every alternative on exactly the same eligible order rows
for product, group in df.groupby("Product Name"):
    current_factory = product_factory[product]
    current_col = factories[current_factory]

    for scenario_factory, scenario_col in factories.items():

        if scenario_factory == current_factory:
            continue

        eligible = group[
            group[current_col].notna()
            & group[scenario_col].notna()
        ].copy()

        if eligible.empty:
            continue

        units = eligible["Units"].sum()

        current_avg_distance = (
            eligible[current_col] * eligible["Units"]
        ).sum() / units

        scenario_avg_distance = (
            eligible[scenario_col] * eligible["Units"]
        ).sum() / units

        # Positive = shorter distance; negative = longer distance
        distance_reduction = (
            current_avg_distance - scenario_avg_distance
        )

        # Signed total distance change weighted by units
        distance_savings_km_units = (
            (eligible[current_col] - eligible[scenario_col])
            * eligible["Units"]
        ).sum()

        # Financial baseline from the SAME eligible orders
        baseline_sales = eligible["Sales"].sum()
        baseline_cost = eligible["Cost"].sum()
        baseline_gp = eligible["Gross Profit"].sum()

        for freight_rate in freight_rates:
            for transfer_cost in transfer_costs:

                estimated_transport_impact = (
                    distance_savings_km_units * freight_rate
                )

                estimated_transfer_cost = units * transfer_cost

                estimated_net_impact = (
                    estimated_transport_impact
                    - estimated_transfer_cost
                )

                results.append({
                    "Product Name": product,
                    "Current Factory": current_factory,
                    "Scenario Factory": scenario_factory,
                    "Eligible Order Rows": len(eligible),
                    "Eligible Units": units,
                    "Baseline Sales on Eligible Orders": baseline_sales,
                    "Baseline Cost on Eligible Orders": baseline_cost,
                    "Baseline Gross Profit on Eligible Orders": baseline_gp,
                    "Current Avg Distance (km)": current_avg_distance,
                    "Scenario Avg Distance (km)": scenario_avg_distance,
                    "Potential Distance Reduction (km)": distance_reduction,
                    "Distance Savings (km-units)": distance_savings_km_units,
                    "Assumed Freight Cost per km-unit": freight_rate,
                    "Assumed Transfer Cost per Unit": transfer_cost,
                    "Estimated Transport Impact": estimated_transport_impact,
                    "Estimated Transfer Cost": estimated_transfer_cost,
                    "Estimated Net Profit Impact": estimated_net_impact,
                    "Illustrative Adjusted Gross Profit": (
                        baseline_gp + estimated_net_impact
                    ),
                    "Positive Under Assumptions": estimated_net_impact > 0,
                })

result = pd.DataFrame(results)

result.to_csv("scenario_profit_sensitivity_corrected.csv", index=False)

print("\n--- CORRECTED STEP 5D ---")
print("Alternative product-factory scenarios:", result[
    ["Product Name", "Scenario Factory"]
].drop_duplicates().shape[0])
print("Sensitivity rows:", len(result))
print("Output: scenario_profit_sensitivity_corrected.csv")

if not result.empty:
    summary = result.groupby(
        [
            "Assumed Freight Cost per km-unit",
            "Assumed Transfer Cost per Unit",
        ],
        as_index=False
    ).agg(
        Scenarios_Evaluated=("Product Name", "size"),
        Positive_Impact_Scenarios=("Positive Under Assumptions", "sum"),
        Average_Estimated_Net_Impact=("Estimated Net Profit Impact", "mean"),
    )

    print("\n--- SENSITIVITY SUMMARY ---")
    print(summary.round(2).to_string(index=False))

    print("\n--- TOP 10 SCENARIOS UNDER EACH ASSUMPTION ---")
    for (rate, transfer), subset in result.groupby(
        ["Assumed Freight Cost per km-unit",
         "Assumed Transfer Cost per Unit"]
    ):
        print(
            f"\nFreight rate={rate}, transfer cost/unit={transfer}"
        )
        print(
            subset.nlargest(3, "Estimated Net Profit Impact")[
                [
                    "Product Name",
                    "Current Factory",
                    "Scenario Factory",
                    "Eligible Order Rows",
                    "Potential Distance Reduction (km)",
                    "Estimated Net Profit Impact",
                ]
            ].round(2).to_string(index=False)
        )
else:
    print("No eligible alternative scenarios found. Check route data.")
"""
"""
import pandas as pd

file = "scenario_profit_sensitivity_corrected.csv"
df = pd.read_csv(file)

print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))"""
