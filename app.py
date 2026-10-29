
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# 1. PROJECT PATHS
# =========================================================
ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main"
ARTIFACTS = MAIN / "model_artifacts"

DATA_PATH = MAIN / "modeling_dataset.csv"
MODEL_PATH = ARTIFACTS / "best_factory_aware_model.joblib"
INFO_PATH = ARTIFACTS / "model_info.json"

KPI2_PATH = (
    ROOT / "KPIs_analysis" / "KPI2"
    / "scenario_profit_sensitivity_corrected.csv"
)
KPI3_PATH = (
    ROOT / "KPIs_analysis" / "KPI3"
    / "kpi3_scenario_confidence.csv"
)
KPI4_SUMMARY_PATH = (
    ROOT / "KPIs_analysis" / "KPI4"
    / "kpi4_coverage_summary.csv"
)
KPI4_PRODUCTS_PATH = (
    ROOT / "KPIs_analysis" / "KPI4"
    / "kpi4_product_coverage.csv"
)
RECOMMENDATIONS_PATH = (
    ROOT / "KPIs_analysis" / "KPI3"
    / "factory_recommendations_ranked.csv"
)

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide",
)

# =========================================================
# 2. DATA LOADING
# =========================================================
@st.cache_data
def load_dataset(path, modified_time):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_model_info(path, modified_time):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")
    return None


def find_column(df, candidates):
    """Find a column using case-insensitive exact or partial matching."""
    if df is None or df.empty:
        return None

    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in str(col).strip().lower():
                return col

    return None


def fmt_number(value, decimals=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


# =========================================================
# 3. VALIDATE REQUIRED FILES
# =========================================================
st.title("🏭 Nassau Candy Factory Optimization")
st.caption(
    "Factory prediction simulator • Scenario analysis • "
    "Recommendation coverage"
)

missing_files = [
    p for p in [DATA_PATH, MODEL_PATH, INFO_PATH]
    if not p.exists()
]

if missing_files:
    st.error("Some required model files are missing:")
    for path in missing_files:
        st.write(f"- `{path}`")
    st.stop()

df = load_dataset(
    str(DATA_PATH), DATA_PATH.stat().st_mtime
)
model = load_model(
    str(MODEL_PATH), MODEL_PATH.stat().st_mtime
)
model_info = load_model_info(
    str(INFO_PATH), INFO_PATH.stat().st_mtime
)

required_columns = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode",
    "Units",
    "Lead Time",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()

df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
df["Lead Time"] = pd.to_numeric(df["Lead Time"], errors="coerce")

# Keep only rows with the fields needed for selectors.
selector_df = df.dropna(
    subset=["Product Name", "Current Factory", "Region", "Ship Mode"]
).copy()

factories = sorted(
    selector_df["Current Factory"].astype(str).unique().tolist()
)
products = sorted(
    selector_df["Product Name"].astype(str).unique().tolist()
)

if not products or not factories:
    st.error("No valid products or factories were found in the dataset.")
    st.stop()

# =========================================================
# 4. LOAD EXISTING KPI FILES
# =========================================================
kpi2 = read_csv_if_exists(KPI2_PATH)
kpi3 = read_csv_if_exists(KPI3_PATH)
kpi4_summary = read_csv_if_exists(KPI4_SUMMARY_PATH)
kpi4_products = read_csv_if_exists(KPI4_PRODUCTS_PATH)
recommendations = read_csv_if_exists(RECOMMENDATIONS_PATH)

# =========================================================
# 5. TOP-LEVEL KPI CARDS
# =========================================================
st.subheader("Project overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Products", f"{selector_df['Product Name'].nunique():,}")
c2.metric("Factories", f"{selector_df['Current Factory'].nunique():,}")

if kpi2 is not None and not kpi2.empty:
    impact_col = find_column(
        kpi2,
        ["Average Estimated Net Impact", "Estimated Net Impact",
         "Net Impact", "Net Profit Impact"]
    )
    positive_col = find_column(
        kpi2,
        ["Positive Impact", "Positive Impact Scenario",
         "Net Impact Positive"]
    )

    if impact_col:
        numeric_impact = pd.to_numeric(kpi2[impact_col], errors="coerce")
        c3.metric(
            "Mean modeled net impact",
            fmt_number(numeric_impact.mean())
        )
    else:
        c3.metric("KPI2 scenarios", f"{len(kpi2):,}")
else:
    c3.metric("KPI2 scenarios", "File unavailable")

if kpi3 is not None and not kpi3.empty:
    confidence_col = find_column(
        kpi3, ["Confidence Score", "Scenario Confidence Score",
               "Confidence_Score"]
    )
    if confidence_col:
        scores = pd.to_numeric(kpi3[confidence_col], errors="coerce")
        c4.metric("Mean scenario confidence", f"{scores.mean():.1f}/100"
                  if scores.notna().any() else "N/A")
    else:
        c4.metric("KPI3 alternatives", f"{len(kpi3):,}")
else:
    c4.metric("KPI3 alternatives", "File unavailable")

st.info(
    "Interpretation: KPI2 values are modeled scenario estimates, not "
    "realized savings. KPI3 confidence measures robustness across tested "
    "assumptions, not the probability of real-world success. KPI4 coverage "
    "measures recommendation coverage, not successful relocations."
)

# =========================================================
# 6. MODEL VALIDATION NOTICE
# =========================================================
st.subheader("Model validation status")

st.warning(
    "Experimental relocation predictions: the training dataset contains "
    "no products observed at multiple factories. Comparing a product at "
    "alternative factories is therefore an extrapolation and is not "
    "validated evidence of improved real-world shipping performance."
)

m1, m2, m3 = st.columns(3)
m1.metric("Selected model", model_info.get("selected_model", "Unknown"))
m2.metric("Training rows", f"{model_info.get('training_rows', 0):,}")
m3.metric(
    "Products seen at multiple factories",
    str(model_info.get("products_observed_at_multiple_factories", "Unknown"))
)

# =========================================================
# 7. FACTORY PREDICTION SIMULATOR
# =========================================================
st.divider()
st.header("1. Factory Prediction Simulator")
st.write(
    "Select a product and shipment configuration to compare model-estimated "
    "lead times across factories."
)

left, right = st.columns([1, 1])

with left:
    selected_product = st.selectbox("Product", products)

    product_rows = selector_df[
        selector_df["Product Name"].astype(str) == selected_product
    ]

    # The source data has one observed factory per product.
    observed_factories = sorted(
        product_rows["Current Factory"].astype(str).unique().tolist()
    )
    current_factory = observed_factories[0]

    st.text_input(
        "Observed factory in historical data",
        value=current_factory,
        disabled=True,
    )

    regions = sorted(
        selector_df["Region"].dropna().astype(str).unique().tolist()
    )
    modes = sorted(
        selector_df["Ship Mode"].dropna().astype(str).unique().tolist()
    )

    product_regions = sorted(
        product_rows["Region"].dropna().astype(str).unique().tolist()
    )
    product_modes = sorted(
        product_rows["Ship Mode"].dropna().astype(str).unique().tolist()
    )

    selected_region = st.selectbox(
        "Region",
        product_regions if product_regions else regions,
    )
    selected_mode = st.selectbox(
        "Ship mode",
        product_modes if product_modes else modes,
    )

with right:
    median_units = product_rows["Units"].median()
    if pd.isna(median_units):
        median_units = 1

    selected_units = st.number_input(
        "Units",
        min_value=1,
        max_value=10_000_000,
        value=max(1, int(median_units)),
        step=1,
    )

    years = []
    months = []

    if "Order Year" in selector_df.columns:
        years = sorted(
            pd.to_numeric(
                selector_df["Order Year"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )

    if "Order Month" in selector_df.columns:
        months = sorted(
            pd.to_numeric(
                selector_df["Order Month"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )

    if years:
        selected_year = st.selectbox("Order year", years, index=len(years) - 1)
    else:
        selected_year = None

    if months:
        selected_month = st.selectbox("Order month", months, index=0)
    else:
        selected_month = None

    st.caption(
        "Date selections are model inputs, not a live shipment date forecast."
    )

predict_clicked = st.button(
    "Compare factories",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    features = model_info.get("features", [])
    prediction_rows = []

    for factory in factories:
        row = {
            "Product Name": selected_product,
            "Current Factory": factory,
            "Region": selected_region,
            "Ship Mode": selected_mode,
            "Units": selected_units,
        }

        if selected_year is not None:
            row["Order Year"] = selected_year
        if selected_month is not None:
            row["Order Month"] = selected_month

        # Supply exactly the feature columns used during training.
        input_df = pd.DataFrame([{col: row.get(col, np.nan) for col in features}])

        try:
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            st.error(
                f"Prediction failed for {factory}: {exc}"
            )
            st.stop()

        prediction_rows.append({
            "Factory": factory,
            "Predicted Lead Time": prediction,
            "Observed for this product?": (
                "Yes" if factory in observed_factories else "No"
            ),
        })

    prediction_df = pd.DataFrame(prediction_rows).sort_values(
        "Predicted Lead Time", ascending=True
    ).reset_index(drop=True)

    prediction_df.insert(
        0, "Model rank", range(1, len(prediction_df) + 1)
    )

    st.subheader("Predicted factory comparison")
    st.dataframe(
        prediction_df.style.format(
            {"Predicted Lead Time": "{:,.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    best_row = prediction_df.iloc[0]
    current_prediction = prediction_df.loc[
        prediction_df["Factory"] == current_factory,
        "Predicted Lead Time",
    ].iloc[0]

    difference = current_prediction - best_row["Predicted Lead Time"]

    st.metric(
        "Lowest predicted lead time",
        f"{best_row['Predicted Lead Time']:,.2f}",
        help="Model estimate only; not a validated relocation recommendation.",
    )

    if best_row["Factory"] == current_factory:
        st.write(
            "The lowest estimate is for the product's historically observed factory."
        )
    else:
        st.write(
            f"The model estimates {difference:,.2f} fewer lead-time units "
            f"at {best_row['Factory']} than at {current_factory}. "
            "This comparison is experimental and must not be treated as "
            "a proven improvement."
        )

    st.caption(
        "The model estimates lead time only. This comparison does not "
        "calculate profit, capacity, operational risk, or transfer costs."
    )

    chart_df = prediction_df.set_index("Factory")[
        ["Predicted Lead Time"]
    ]
    st.bar_chart(chart_df)

# =========================================================
# 8. KPI2 - PROFIT SENSITIVITY
# =========================================================
st.divider()
st.header("2. KPI2 — Profit Impact Sensitivity")

if kpi2 is not None and not kpi2.empty:
    st.write(
        f"Loaded {len(kpi2):,} rows from the existing KPI2 scenario analysis."
    )

    impact_col = find_column(
        kpi2,
        ["Average Estimated Net Impact", "Estimated Net Impact",
         "Net Impact", "Net Profit Impact"]
    )

    if impact_col:
        impacts = pd.to_numeric(kpi2[impact_col], errors="coerce")
        positive_rows = int((impacts > 0).sum())
        valid_rows = int(impacts.notna().sum())

        a, b, c = st.columns(3)
        a.metric("Positive impact rows", f"{positive_rows:,}")
        b.metric("Rows with impact values", f"{valid_rows:,}")
        c.metric("Mean estimated net impact", fmt_number(impacts.mean()))

    st.dataframe(kpi2, use_container_width=True, hide_index=True)
else:
    st.warning(f"KPI2 file not found or unreadable: {KPI2_PATH}")

# =========================================================
# 9. KPI3 - SCENARIO CONFIDENCE
# =========================================================
st.divider()
st.header("3. KPI3 — Scenario Confidence")

if kpi3 is not None and not kpi3.empty:
    confidence_col = find_column(
        kpi3, ["Confidence Score", "Scenario Confidence Score",
               "Confidence_Score"]
    )
    category_col = find_column(
        kpi3, ["Confidence Category", "Confidence_Category",
               "Scenario Confidence Category"]
    )

    if category_col:
        st.write("Scenario robustness categories")
        st.bar_chart(
            kpi3[category_col].fillna("Unknown").astype(str)
            .value_counts()
        )

    if confidence_col:
        st.write("Confidence score distribution")
        score_values = pd.to_numeric(
            kpi3[confidence_col], errors="coerce"
        )
        st.write(score_values.describe().to_frame("Confidence score"))

    st.dataframe(kpi3, use_container_width=True, hide_index=True)
else:
    st.warning(f"KPI3 file not found or unreadable: {KPI3_PATH}")

# =========================================================
# 10. KPI4 - RECOMMENDATION COVERAGE
# =========================================================
st.divider()
st.header("4. KPI4 — Recommendation Coverage")

if kpi4_summary is not None and not kpi4_summary.empty:
    st.write("Existing KPI4 coverage summary")
    st.dataframe(
        kpi4_summary,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning(f"KPI4 summary not found or unreadable: {KPI4_SUMMARY_PATH}")

if kpi4_products is not None and not kpi4_products.empty:
    with st.expander("View product-level KPI4 coverage"):
        st.dataframe(
            kpi4_products,
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# 11. RANKED RECOMMENDATIONS
# =========================================================
st.divider()
st.header("5. Existing Ranked Factory Recommendations")

if recommendations is not None and not recommendations.empty:
    st.write(
        "These are existing analytical recommendations, separate from "
        "the experimental model predictions above."
    )
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning(
        "The ranked recommendation file was not found. "
        f"Expected location: {RECOMMENDATIONS_PATH}"
    )

# =========================================================
# 12. RISK AND METHODOLOGY
# =========================================================
st.divider()
st.header("6. Risk & Methodology Notes")

st.warning(
    "Operational risk scores are unavailable in the current analysis. "
    "This app does not invent risk ratings or claim that a factory transfer "
    "is operationally safe."
)

with st.expander("Model evaluation"):
    metrics_path = ARTIFACTS / "model_comparison.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison metrics file is unavailable.")

    st.json(model_info)

with st.expander("Data and interpretation limitations"):
    st.markdown(
        """
        - Predictions are estimates based on historical data.
        - No product was observed at multiple factories in the training data.
        - A lower predicted lead time does not establish that a transfer will
          improve actual delivery performance.
        - KPI2 financial impacts are scenario estimates, not realized savings.
        - KPI3 confidence scores represent robustness across tested
          assumptions, not the probability of success.
        - KPI4 coverage is not proof that recommendations succeeded.
        - Operational risk data and verified alternative-factory lead times
          are not available in this version.
        """
    )

st.caption(
    "Nassau Candy Factory Optimization | Local Streamlit prototype"
)
