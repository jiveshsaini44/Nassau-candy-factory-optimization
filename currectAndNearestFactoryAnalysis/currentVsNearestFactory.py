
import os
import pandas as pd
import numpy as np

# -----------------------------------------
# CURRENT FACTORY VS NEAREST FACTORY
# -----------------------------------------

folder = r"C:\Users\HP\Desktop\project1"

orders_file = os.path.join(
    folder, "Nassau Candy Distributor.csv"
)

routes_file = os.path.join(
    folder, "route_analysis_results.csv"
)

output_file = os.path.join(
    folder, "current_vs_nearest_factory.csv"
)

summary_file = os.path.join(
    folder, "current_vs_nearest_summary.csv"
)

# 1. Official product-to-factory mapping
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

# 2. Load files
orders = pd.read_csv(orders_file)
routes = pd.read_csv(routes_file)

print("\n--- CURRENT VS NEAREST FACTORY ---")
print("Original order rows:", len(orders))
print("Route destination rows:", len(routes))

# 3. Match column names without assuming capitalization
def find_column(df, possible_names):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for name in possible_names:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]

    raise ValueError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns: {list(df.columns)}"
    )

# Original order columns
order_product = find_column(orders, ["Product Name"])
order_country = find_column(orders, ["Country/Region", "Country"])
order_city = find_column(orders, ["City"])
order_state = find_column(orders, ["State/Province", "State"])
order_postal = find_column(orders, ["Postal Code", "zipCode", "PostalCode"])
order_region = find_column(orders, ["Region"])

# Route destination columns
route_country = find_column(routes, ["Country/Region", "Country"])
route_city = find_column(routes, ["City"])
route_state = find_column(routes, ["State/Province", "State"])
route_postal = find_column(routes, ["Postal Code", "zipCode", "PostalCode"])
route_region = find_column(routes, ["Region"])

nearest_col = find_column(routes, ["Nearest_Factory"])
nearest_distance_col = find_column(routes, ["Nearest_Distance_km"])

distance_columns = {
    "Lots_O_Nuts": find_column(routes, ["Distance_Lots_O_Nuts_km"]),
    "Wicked_Choccys": find_column(routes, ["Distance_Wicked_Choccys_km"]),
    "Sugar_Shack": find_column(routes, ["Distance_Sugar_Shack_km"]),
    "Secret_Factory": find_column(routes, ["Distance_Secret_Factory_km"]),
    "The_Other_Factory": find_column(
        routes, ["Distance_The_Other_Factory_km"]
    ),
}

# 4. Create consistent destination matching keys
def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

def clean_postal(series):
    # Normalize postal codes while preserving leading zeros where present
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )

key_pairs = [
    (order_country, route_country, "country"),
    (order_city, route_city, "city"),
    (order_state, route_state, "state"),
    (order_postal, route_postal, "postal"),
    (order_region, route_region, "region"),
]

for order_col, route_col, key_name in key_pairs:
    if key_name == "postal":
        orders["_key_" + key_name] = clean_postal(orders[order_col])
        routes["_key_" + key_name] = clean_postal(routes[route_col])
    else:
        orders["_key_" + key_name] = clean_text(orders[order_col])
        routes["_key_" + key_name] = clean_text(routes[route_col])

keys = ["_key_country", "_key_city", "_key_state",
        "_key_postal", "_key_region"]

# Remove empty-key destination records from the join
routes_valid_keys = routes.copy()
for key in keys:
    routes_valid_keys = routes_valid_keys[
        routes_valid_keys[key] != ""
    ]

# One row per destination key is required for a reliable merge
duplicate_keys = routes_valid_keys.duplicated(keys).sum()

if duplicate_keys:
    raise ValueError(
        f"Route file has {duplicate_keys} duplicate destination keys. "
        "Resolve these before joining to avoid duplicating order rows."
    )

# 5. Prepare route data for merging
route_columns = keys + [nearest_col, nearest_distance_col]
route_columns += list(distance_columns.values())

route_lookup = routes_valid_keys[route_columns].copy()

orders = orders.merge(
    route_lookup,
    on=keys,
    how="left",
    validate="many_to_one",
    indicator=True
)

orders["Route_Match_Status"] = np.where(
    orders["_merge"] == "both",
    "Matched",
    "No destination match"
)

# 6. Map each product to its current factory
orders["Current_Factory"] = orders[order_product].map(product_factory)

# 7. Obtain distance from the current factory
orders["Current_Factory_Distance_km"] = np.nan

for factory_name, distance_col in distance_columns.items():
    mask = orders["Current_Factory"] == factory_name

    orders.loc[mask, "Current_Factory_Distance_km"] = pd.to_numeric(
        orders.loc[mask, distance_col], errors="coerce"
    )

# 8. Compare current factory and nearest factory
orders["Factory_Matches_Nearest"] = np.where(
    orders["Current_Factory"].isna(),
    "Unknown product mapping",
    np.where(
        orders[nearest_col].isna()
        | orders["Current_Factory_Distance_km"].isna(),
        "Distance unavailable",
        np.where(
            orders["Current_Factory"] == orders[nearest_col],
            "Already nearest",
            "Different from nearest"
        )
    )
)

orders["Potential_Distance_Reduction_km"] = (
    orders["Current_Factory_Distance_km"]
    - pd.to_numeric(orders[nearest_distance_col], errors="coerce")
)

orders.loc[
    orders["Factory_Matches_Nearest"] != "Different from nearest",
    "Potential_Distance_Reduction_km"
] = np.nan

# This is a destination-level geographic proxy, not measured freight savings.
# Summarize only rows with valid comparisons.
orders["Comparison_Valid"] = (
    orders["Current_Factory_Distance_km"].notna()
    & pd.to_numeric(orders[nearest_distance_col], errors="coerce").notna()
    & orders["Current_Factory"].notna()
)

# 9. Save order-level comparison
orders.drop(columns=["_merge"], errors="ignore").to_csv(
    output_file, index=False
)

# 10. Create a concise summary
valid = orders[orders["Comparison_Valid"]].copy()

summary_rows = [
    {"Metric": "Total order rows", "Value": len(orders)},
    {
        "Metric": "Rows matched to a destination",
        "Value": int((orders["Route_Match_Status"] == "Matched").sum())
    },
    {
        "Metric": "Rows with valid current-vs-nearest distances",
        "Value": len(valid)
    },
    {
        "Metric": "Rows where current factory is already nearest",
        "Value": int(
            (valid["Current_Factory"] == valid[nearest_col]).sum()
        )
    },
    {
        "Metric": "Rows where nearest factory differs",
        "Value": int(
            (valid["Current_Factory"] != valid[nearest_col]).sum()
        )
    },
]

if len(valid):
    summary_rows.extend([
        {
            "Metric": "Average current-factory distance (km)",
            "Value": round(valid["Current_Factory_Distance_km"].mean(), 2)
        },
        {
            "Metric": "Average nearest-factory distance (km)",
            "Value": round(
                pd.to_numeric(valid[nearest_distance_col]).mean(), 2
            )
        },
        {
            "Metric": "Average potential distance reduction (km)",
            "Value": round(
                (
                    valid["Current_Factory_Distance_km"]
                    - pd.to_numeric(valid[nearest_distance_col])
                ).mean(),
                2
            )
        },
    ])

summary = pd.DataFrame(summary_rows)
summary.to_csv(summary_file, index=False)

print("\n--- COMPARISON SUMMARY ---")
print(summary.to_string(index=False))

print("\nSaved order-level results to:", output_file)
print("Saved summary to:", summary_file)

