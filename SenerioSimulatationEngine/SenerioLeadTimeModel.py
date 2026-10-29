
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Load existing data ----------
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
    "Country/Region", "City", "State/Province",
    "Postal Code", "Region"
]

for data in [orders, routes]:
    for col in keys:
        data[col] = data[col].fillna("").astype(str).str.strip()

routes = routes.drop_duplicates(subset=keys)

df = orders.merge(
    routes[keys + list(factories.values())],
    on=keys,
    how="left",
    validate="many_to_one"
)

# ---------- 2. Prepare historical lead time ----------
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

df["Lead Time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

df["Current Factory"] = df["Product Name"].map(product_factory)
df["Current Distance"] = np.nan

for factory, distance_col in factories.items():
    mask = df["Current Factory"] == factory
    df.loc[mask, "Current Distance"] = df.loc[mask, distance_col]

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month

feature_columns = [
    "Product Name", "Region", "Ship Mode",
    "Units", "Order Year", "Order Month",
    "Current Distance"
]

categorical = ["Product Name", "Region", "Ship Mode"]
numeric = [
    "Units", "Order Year", "Order Month", "Current Distance"
]

df = df.dropna(subset=["Lead Time"] + feature_columns).copy()

X = df[feature_columns]
y = df["Lead Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

print("\n--- STEP 5B: MODEL EVALUATION ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("MAE (days):", round(mean_absolute_error(y_test, test_predictions), 2))
print("RMSE (days):", round(np.sqrt(mean_squared_error(y_test, test_predictions)), 2))
print("R2:", round(r2_score(y_test, test_predictions), 4))

# ---------- 3. Simulate all five factory scenarios ----------
scenario_results = []

for factory, distance_col in factories.items():
    scenario = df.copy()
    scenario["Scenario Factory"] = factory
    scenario["Scenario Distance"] = scenario[distance_col]

    valid = scenario.dropna(subset=["Scenario Distance"]).copy()

    scenario_X = valid[feature_columns].copy()
    scenario_X["Current Distance"] = valid["Scenario Distance"]

    valid["Predicted Lead Time"] = model.predict(scenario_X)

    summary = valid.groupby(
        ["Product Name", "Current Factory", "Scenario Factory"],
        as_index=False
    ).agg(
        Orders_Evaluated=("Predicted Lead Time", "size"),
        Average_Predicted_Lead_Time=("Predicted Lead Time", "mean"),
        Average_Scenario_Distance_km=("Scenario Distance", "mean")
    )

    scenario_results.append(summary)

results = pd.concat(scenario_results, ignore_index=True)

current_baseline = results[
    results["Scenario Factory"] == results["Current Factory"]
][[
    "Product Name", "Average_Predicted_Lead_Time"
]].rename(columns={
    "Average_Predicted_Lead_Time": "Current_Predicted_Lead_Time"
})

results = results.merge(
    current_baseline,
    on="Product Name",
    how="left",
    validate="many_to_one"
)

results["Estimated Lead Time Reduction (days)"] = (
    results["Current_Predicted_Lead_Time"]
    - results["Average_Predicted_Lead_Time"]
)

results["Scenario Type"] = np.where(
    results["Scenario Factory"] == results["Current Factory"],
    "Current",
    "Alternative"
)

results.to_csv("scenario_lead_time_predictions.csv", index=False)

print("\n--- SCENARIO PREDICTIONS ---")
print("Scenario rows:", len(results))
print("Output: scenario_lead_time_predictions.csv")
print(
    results.head(10).round(2).to_string(index=False)
)

