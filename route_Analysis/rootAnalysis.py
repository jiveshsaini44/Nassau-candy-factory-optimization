
import os
import numpy as np
import pandas as pd

# -----------------------------------------
# STEP 4B: ROUTE ANALYSIS - DISTANCE CALCULATION
# -----------------------------------------

# Project folder
folder = r"C:\Users\HP\Desktop\project1"

input_file = os.path.join(
    folder, "customer_coordinate_validation.csv"
)

output_file = os.path.join(
    folder, "route_analysis_results.csv"
)

# 1. Load customer destinations and validation flags
df = pd.read_csv(input_file)

print("\n--- STEP 4B: ROUTE ANALYSIS ---")
print("Total customer destinations:", len(df))

# 2. Official factory coordinates
factories = {
    "Lots_O_Nuts": (32.881893, -111.768036),
    "Wicked_Choccys": (32.076176, -81.088371),
    "Sugar_Shack": (48.119140, -96.181150),
    "Secret_Factory": (41.446333, -90.565487),
    "The_Other_Factory": (35.117500, -89.971107),
}

# 3. Ensure coordinates are numeric
df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)
df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

# 4. Only use destinations marked OK with valid coordinates
valid_coordinates = (
    df["coordinate_flag"].eq("OK")
    & df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
    & df["latitude"].notna()
    & df["longitude"].notna()
)

print("Verified destinations:", int(valid_coordinates.sum()))
print("Destinations needing review:", int((~valid_coordinates).sum()))

# 5. Haversine formula: calculate distance in kilometres
def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


# 6. Calculate distance from each destination to every factory
distance_columns = []

for factory_name, (factory_lat, factory_lon) in factories.items():

    column_name = f"Distance_{factory_name}_km"
    distance_columns.append(column_name)

    df[column_name] = np.nan

    df.loc[valid_coordinates, column_name] = haversine_km(
        df.loc[valid_coordinates, "latitude"].to_numpy(),
        df.loc[valid_coordinates, "longitude"].to_numpy(),
        factory_lat,
        factory_lon,
    )

# 7. Find the nearest factory for verified destinations
df["Nearest_Factory"] = "Needs coordinate review"
df["Nearest_Distance_km"] = np.nan

df.loc[valid_coordinates, "Nearest_Factory"] = (
    df.loc[valid_coordinates, distance_columns]
    .idxmin(axis=1)
    .str.replace("Distance_", "", regex=False)
    .str.replace("_km", "", regex=False)
)

df.loc[valid_coordinates, "Nearest_Distance_km"] = (
    df.loc[valid_coordinates, distance_columns]
    .min(axis=1)
)

# 8. Summarize results
print("\n--- NEAREST FACTORY COUNTS ---")
print(
    df.loc[valid_coordinates, "Nearest_Factory"]
    .value_counts()
    .to_string()
)

print("\n--- AVERAGE DISTANCE TO EACH FACTORY (km) ---")
print(
    df.loc[valid_coordinates, distance_columns]
    .mean()
    .round(2)
    .to_string()
)

print("\n--- DISTANCE SUMMARY (km) ---")
print(
    df.loc[valid_coordinates, "Nearest_Distance_km"]
    .describe()
    .round(2)
    .to_string()
)

# 9. Save results
df.to_csv(output_file, index=False)

print("\nResults saved to:", output_file)
print("Total output rows:", len(df))
print("Step 4B distance calculation completed.")

"""

--- STEP 4B: ROUTE ANALYSIS ---
Total customer destinations: 655
Verified destinations: 597
Destinations needing review: 58

--- NEAREST FACTORY COUNTS ---
Nearest_Factory
Lots_O_Nuts          174
Secret_Factory       153
Wicked_Choccys       128
The_Other_Factory    108
Sugar_Shack           34

--- AVERAGE DISTANCE TO EACH FACTORY (km) ---
Distance_Lots_O_Nuts_km          1979.16
Distance_Wicked_Choccys_km       1803.00
Distance_Sugar_Shack_km          1720.49
Distance_Secret_Factory_km       1371.03
Distance_The_Other_Factory_km    1407.95

--- DISTANCE SUMMARY (km) ---
count     597.00
mean      698.80
std       429.67
min         6.50
25%       397.16
50%       619.70
75%       947.40
max      2926.69

Results saved to: C:\Users\HP\Desktop\project1\route_analysis_results.csv
Total output rows: 655
Step 4B distance calculation completed."""