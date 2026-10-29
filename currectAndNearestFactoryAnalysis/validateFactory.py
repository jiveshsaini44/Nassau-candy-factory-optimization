import os
import pandas as pd

folder = r"C:\Users\HP\Desktop\project1"
file_path = os.path.join(
    folder, "current_vs_nearest_factory.csv"
)

df = pd.read_csv(file_path)

print("\n--- FACTORY COMPARISON VALIDATION ---")

print("\n1. Product count:")
print("Unique products:", df["Product Name"].nunique())

print("\n2. Product mapping status:")
print(
    df.groupby("Product Name")["Current_Factory"]
    .first()
    .to_string()
)

print("\n3. Comparison status:")
print(df["Factory_Matches_Nearest"].value_counts().to_string())

print("\n4. Results by product:")
summary = df.groupby("Product Name").agg(
    Order_Rows=("Product Name", "size"),
    Current_Factory=("Current_Factory", "first"),
    Valid_Comparisons=("Comparison_Valid", "sum"),
    Average_Current_Distance_km=(
        "Current_Factory_Distance_km", "mean"
    ),
    Average_Potential_Reduction_km=(
        "Potential_Distance_Reduction_km", "mean"
    ),
)

print(summary.round(2).to_string())

print("\n5. Products without a factory mapping:")
unmapped = df.loc[
    df["Current_Factory"].isna(), "Product Name"
].unique()

print(list(unmapped) if len(unmapped) else "None")

print("\nValidation completed.")
