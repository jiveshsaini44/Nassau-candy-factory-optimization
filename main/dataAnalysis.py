

import pandas as pd

df=pd.read_csv('Nassau Candy Distributor.csv')


# Knowing data more in depth
'''
print("Dataset shape:", df.shape)


print("\nColumn names:")
print(df.columns.tolist())


print("\nFirst five rows:")
print(df.head())


print("\nDataset information:")
df.info()


print("\nMissing values:")
print(df.isnull().sum())

'''

#working with dates to fetch some insights

# Convert dates using the format present in your dataset
"""
df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="%d-%m-%Y", errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], format="%d-%m-%Y", errors="coerce"
)

# Calculate shipping lead time in days
df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# Display validation results
print("\n--- DATE VALIDATION ---")
print("Missing order dates:", df["Order Date"].isna().sum())
print("Missing ship dates:", df["Ship Date"].isna().sum())
print("Negative lead times:", (df["Lead Time"] < 0).sum())

print("\n--- LEAD TIME SUMMARY ---")
print(df["Lead Time"].describe())

print("\n--- SAMPLE RECORDS ---")
print(df[["Order Date", "Ship Date", "Lead Time"]].head(10))

"""

#output:
"""--- DATE VALIDATION ---
Missing order dates: 0
Missing ship dates: 0
Negative lead times: 0

--- LEAD TIME SUMMARY ---
count    10194.000000
mean      1320.841868
std        262.444892
min        904.000000
25%       1271.000000
50%       1274.000000
75%       1638.000000
max       1642.000000
Name: Lead Time, dtype: float64

--- SAMPLE RECORDS ---
  Order Date  Ship Date  Lead Time
0 2024-01-03 2026-06-30        909
1 2024-01-04 2026-07-01        909
2 2024-01-04 2026-07-01        909
3 2024-01-04 2026-07-01        909
4 2024-01-05 2026-07-05        912
5 2024-01-06 2026-07-03        909
6 2024-01-06 2026-07-03        909
7 2024-01-06 2026-06-30        906
8 2024-01-06 2026-07-03        909
9 2024-01-06 2026-07-03        909"""



# Recalculate dates and lead time immediately before analysis
"""df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="%d-%m-%Y", errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], format="%d-%m-%Y", errors="coerce"
)

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

print("\n--- MOST COMMON LEAD TIMES ---")
print(df["Lead Time"].value_counts().head(15))

print("\n--- LEAD TIME BY ORDER YEAR ---")
print(
    df.groupby(df["Order Date"].dt.year)["Lead Time"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\n--- LEAD TIME BY SHIPPING MODE ---")
print(
    df.groupby("Ship Mode")["Lead Time"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("median")
)"""

"""--- MOST COMMON LEAD TIMES ---
Lead Time
1273    1326
1274    1035
1639     896
1640     657
1275     612
1271     610
908      591
909      519
1272     456
1637     449
1638     411
1641     397
1276     293
906      289
1269     258
Name: count, dtype: int64

--- LEAD TIME BY ORDER YEAR ---
            count         mean  median   min   max
Order Date                                        
2024         4181  1094.018417  1269.0   904  1277
2025         6013  1478.558290  1637.0  1269  1642

--- LEAD TIME BY SHIPPING MODE ---
                count         mean  median  min   max
Ship Mode                                            
Same Day          547  1333.442413  1269.0  904  1636
First Class      1548  1338.275840  1272.0  905  1638
Second Class     1979  1323.845376  1273.0  906  1640
Standard Class   6120  1314.334641  1274.0  908  1642"""



#PREPARE MODELING DATASET

"""# Map each product to its current factory

df["Order Date"] = pd.to_datetime(
    df["Order Date"], format="%d-%m-%Y", errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], format="%d-%m-%Y", errors="coerce"
)

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month


factory_mapping = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

df["Product Name"] = df["Product Name"].astype(str).str.strip()
df["Current Factory"] = df["Product Name"].map(factory_mapping)

# Check that every product has a factory mapping
print("\n--- FACTORY MAPPING CHECK ---")
print("Unmapped product rows:", df["Current Factory"].isna().sum())

if df["Current Factory"].isna().any():
    print(
        df.loc[
            df["Current Factory"].isna(), "Product Name"
        ].value_counts()
    )

# Keep only the columns required for the first prediction model
model_columns = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode",
    "Units",
    "Lead Time",
    "Order Year",
    "Order Month"
]

model_df = df[model_columns].copy()

# Remove rows only if essential model fields are missing
model_df = model_df.dropna(subset=model_columns)

# Save the prepared dataset
model_df.to_csv("modeling_dataset.csv", index=False)

print("\n--- MODELING DATASET ---")
print("Rows:", len(model_df))
print("Columns:", model_df.columns.tolist())
print("\nMissing values:")
print(model_df.isna().sum())
print("\nCurrent factory distribution:")
print(model_df["Current Factory"].value_counts())
print("\nSaved as modeling_dataset.csv")"""


