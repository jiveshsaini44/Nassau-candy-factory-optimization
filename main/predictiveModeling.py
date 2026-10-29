#Train Test Split and Scaling

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the prepared dataset
model_df = pd.read_csv("modeling_dataset.csv")

# Separate input features and prediction target
X = model_df[
    [
        "Product Name",
        "Region",
        "Ship Mode",
        "Units",
        "Order Year",
        "Order Month"
    ]
].copy()

y = model_df["Lead Time"].copy()

# Convert categorical features into numeric columns
X = pd.get_dummies(
    X,
    columns=["Product Name", "Region", "Ship Mode"],
    dtype=int
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale Units using training data only
scaler = StandardScaler()

X_train[["Units"]] = scaler.fit_transform(X_train[["Units"]])
X_test[["Units"]] = scaler.transform(X_test[["Units"]])

# Display results
print("\n--- STEP 1: DATA PREPARATION ---")
print("Original dataset rows:", len(model_df))
print("Encoded feature columns:", X.shape[1])
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Missing values in training features:", X_train.isna().sum().sum())
print("Missing values in testing features:", X_test.isna().sum().sum())
print("Missing values in target:", y.isna().sum())
print("Step 1 preparation completed.")

#PREDICTIVE MODELING

"""from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load the modeling dataset
model_df = pd.read_csv("modeling_dataset.csv")

# Define features and target
X = model_df[
    [
        "Product Name",
        "Region",
        "Ship Mode",
        "Units",
        "Order Year",
        "Order Month"
    ]
].copy()

y = model_df["Lead Time"].copy()
# Encode categorical features
X = pd.get_dummies(
    X,
    columns=["Product Name", "Region", "Ship Mode"],
    dtype=int
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale Units using training data only
scaler = StandardScaler()
X_train[["Units"]] = scaler.fit_transform(X_train[["Units"]])
X_test[["Units"]] = scaler.transform(X_test[["Units"]])

# Define the three models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    )
}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    '''
    print(f"\n{name}")
    print("Training completed.")
    print("Sample predictions:", np.round(predictions[:5], 2))
    '''
    
    # Model Evaluation
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2 Score: {r2:.4f}")
    
    print("STEP 2 completed: Model training and evaluation finished.")
"""

# output:
"""Linear Regression
Training completed.
Sample predictions: [1469.45 1110.95 1481.67 1470.73 1480.46]

Random Forest
Training completed.
Sample predictions: [1341.81 1058.06 1605.11 1529.94 1640.21]

Gradient Boosting
Training completed.
Sample predictions: [1462.73 1092.28 1451.86 1476.63 1472.74]


Linear Regression
MAE: 181.56 days
RMSE: 182.60 days
R2 Score: 0.5285

Random Forest
MAE: 178.95 days
RMSE: 203.16 days
R2 Score: 0.4164

Gradient Boosting
MAE: 180.45 days
RMSE: 182.18 days
R2 Score: 0.5307

    """
    
