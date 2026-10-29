
import pandas as pd

# Step 5A: Scenario Simulation Engine
orders = pd.read_csv("Nassau Candy Distributor.csv")
routes = pd.read_csv("route_analysis_results.csv")

# Official current product-to-factory assignments
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

# Match orders to destination distances
keys = [
    "Country/Region",
    "City",
    "State/Province",
    "Postal Code",
    "Region",
]

for df in [orders, routes]:
    for col in keys:
        df[col] = df[col].fillna("").astype(str).str.strip()

# Keep one route row per destination key
routes = routes.drop_duplicates(subset=keys)

df = orders.merge(
    routes[keys + list(factories.values())],
    on=keys,
    how="left",
    validate="many_to_one"
)

df["Current Factory"] = df["Product Name"].map(product_factory)
df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

results = []

# Evaluate every factory for every product
for product, group in df.groupby("Product Name"):
    current = product_factory.get(product)

    for factory, distance_col in factories.items():
        valid = group[
            group[distance_col].notna()
            & group["Units"].notna()
            & (group["Units"] > 0)
        ].copy()

        if valid.empty:
            continue

        total_units = valid["Units"].sum()
        weighted_distance = (
            valid[distance_col] * valid["Units"]
        ).sum() / total_units

        current_col = factories[current]
        baseline = valid[
            current_col
        ].notna()

        baseline_units = valid.loc[baseline, "Units"].sum()

        if baseline_units > 0:
            current_distance = (
                valid.loc[baseline, current_col]
                * valid.loc[baseline, "Units"]
            ).sum() / baseline_units
        else:
            current_distance = float("nan")

        results.append({
            "Product Name": product,
            "Current Factory": current,
            "Scenario Factory": factory,
            "Scenario Type": (
                "Current" if factory == current else "Alternative"
            ),
            "Orders Evaluated": len(valid),
            "Units Evaluated": total_units,
            "Current Avg Distance (km)": current_distance,
            "Scenario Avg Distance (km)": weighted_distance,
            "Potential Distance Reduction (km)": (
                current_distance - weighted_distance
                if pd.notna(current_distance) else float("nan")
            ),
            "Estimated Unit-Weighted Distance Savings (km-units)": (
                (current_distance - weighted_distance) * total_units
                if pd.notna(current_distance) else float("nan")
            ),
        })

scenario_df = pd.DataFrame(results)
scenario_df.to_csv("scenario_factory_comparison.csv", index=False)

print("\n--- STEP 5A: SCENARIO SIMULATION ---")
print("Original order rows:", len(orders))
print("Rows matched to destinations:", df[list(factories.values())].notna().any(axis=1).sum())
print("Products evaluated:", df["Product Name"].nunique())
print("Scenario rows created:", len(scenario_df))
print("Output: scenario_factory_comparison.csv")

print("\n--- SAMPLE SCENARIOS ---")
print(scenario_df.head(10).round(2).to_string(index=False))
