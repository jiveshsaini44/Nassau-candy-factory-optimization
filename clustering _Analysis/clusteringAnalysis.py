
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the original dataset
df = pd.read_csv("Nassau Candy Distributor.csv")

# Prepare dates and calculate historical lead time
df["Order Date"] = pd.to_datetime(
    df["Order Date"], dayfirst=True, errors="coerce"
)
df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], dayfirst=True, errors="coerce"
)

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# Calculate historical performance for each product
product_summary = df.groupby("Product Name").agg(
    Average_Units=("Units", "mean"),
    Average_Sales=("Sales", "mean"),
    Average_Gross_Profit=("Gross Profit", "mean"),
    Average_Cost=("Cost", "mean"),
    Average_Lead_Time=("Lead Time", "mean")
).reset_index()

# Select numeric features for clustering
features = [
    "Average_Units",
    "Average_Sales",
    "Average_Gross_Profit",
    "Average_Cost",
    "Average_Lead_Time"
]

# Standardize features so large-scale values do not dominate
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    product_summary[features]
)

# Group products into 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

product_summary["Cluster"] = kmeans.fit_predict(
    scaled_features
)

# Save the results
product_summary.to_csv("product_clusters.csv", index=False)

# Display results
print("\n--- STEP 4A: PRODUCT CLUSTERING ---")
print("Number of products:", len(product_summary))
print("\nProducts in each cluster:")
print(product_summary["Cluster"].value_counts().sort_index())

print("\nProduct cluster assignments:")
print(
    product_summary[
        ["Product Name", "Average_Units",
         "Average_Gross_Profit", "Average_Lead_Time", "Cluster"]
    ].sort_values("Cluster").to_string(index=False)
)

print("\nResults saved to product_clusters.csv")
