
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "main" / "modeling_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "main" / "model_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_factory_aware_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"
INFO_PATH = OUTPUT_DIR / "model_info.json"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that main\\modeling_dataset.csv exists."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n--- FACTORY-AWARE MODEL TRAINING ---")
print(f"Dataset: {DATA_PATH}")
print(f"Original rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

required = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode",
    "Units",
    "Lead Time",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------
df = df[required + [
    col for col in ["Order Date", "Order Year", "Order Month"]
    if col in df.columns
]].copy()

df["Lead Time"] = pd.to_numeric(df["Lead Time"], errors="coerce")
df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Lead Time", "Product Name", "Current Factory"])

# Derive date features only if the relevant source exists.
if "Order Date" in df.columns:
    parsed_dates = pd.to_datetime(
        df["Order Date"], errors="coerce", dayfirst=True
    )
    if "Order Year" not in df.columns:
        df["Order Year"] = parsed_dates.dt.year
    if "Order Month" not in df.columns:
        df["Order Month"] = parsed_dates.dt.month

for col in ["Product Name", "Current Factory", "Region", "Ship Mode"]:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

df = df.dropna(subset=["Product Name", "Current Factory"])

# Only use date fields if they contain useful values.
date_features = [
    col for col in ["Order Year", "Order Month"]
    if col in df.columns and df[col].notna().any()
]

categorical_features = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode",
]

numeric_features = ["Units"] + date_features
feature_columns = categorical_features + numeric_features
target_column = "Lead Time"

# Convert numeric features safely.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

X = df[feature_columns].copy()
y = df[target_column].copy()

if len(df) < 10:
    raise ValueError(
        "Fewer than 10 usable rows remain. "
        "There may not be enough data to train and evaluate a model."
    )

if y.nunique() < 2:
    raise ValueError("Lead Time has fewer than two distinct values.")

# --------------------------------------------------
# 4. CHECK FACTORY AND PRODUCT COVERAGE
# --------------------------------------------------
print("\n--- DATA COVERAGE ---")
print(f"Usable rows: {len(df)}")
print(f"Unique products: {df['Product Name'].nunique()}")
print(f"Unique factories: {df['Current Factory'].nunique()}")
print("\nRows per factory:")
print(df["Current Factory"].value_counts().to_string())

product_factory_counts = (
    df.groupby("Product Name")["Current Factory"].nunique()
)

print(
    "\nProducts observed at multiple factories: "
    f"{int((product_factory_counts > 1).sum())} "
    f"of {len(product_factory_counts)}"
)

if (product_factory_counts > 1).sum() == 0:
    print(
        "\nWARNING: Each product appears at only one factory in this "
        "dataset. The model can learn associations with factory, but "
        "counterfactual predictions for moving a product to another "
        "factory are not directly validated by these observations."
    )

# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Dense one-hot encoding works across the model types used here.
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
    ],
    remainder="drop",
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

# --------------------------------------------------
# 6. TRAIN AND COMPARE
# --------------------------------------------------
results = []
trained_models = {}

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        })

        trained_models[name] = pipeline

        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.3f}")

    except Exception as exc:
        print(f"Could not train {name}: {exc}")

if not results:
    raise RuntimeError(
        "All models failed. Review the errors above before continuing."
    )

# Lowest test RMSE selects the model; this is not proof that
# counterfactual factory relocation predictions are reliable.
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(METRICS_PATH, index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
joblib.dump(best_model, MODEL_PATH)

# Save information needed by the future Streamlit application.
model_info = {
    "selected_model": best_name,
    "target": target_column,
    "features": feature_columns,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "dataset_rows_used": int(len(df)),
    "unique_products": int(df["Product Name"].nunique()),
    "unique_factories": int(df["Current Factory"].nunique()),
    "products_observed_at_multiple_factories": int(
        (product_factory_counts > 1).sum()
    ),
    "selection_metric": "lowest holdout RMSE",
    "note": (
        "Holdout metrics measure predictions on the random test split. "
        "They do not by themselves validate predictions for a product "
        "at a factory where it has not been observed."
    ),
}

with open(INFO_PATH, "w", encoding="utf-8") as file:
    json.dump(model_info, file, indent=4)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

print("\n--- SAVED FILES ---")
print(f"Best model: {MODEL_PATH}")
print(f"Metrics:    {METRICS_PATH}")
print(f"Model info: {INFO_PATH}")
print(f"\nSelected model: {best_name}")
print("\nTraining finished. Review the metrics and coverage warning")
print("before using predictions for factory relocation.")
