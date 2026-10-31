from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("default")

# ---------------------------------------------------------------------
# Nassau Candy Factory Optimization — Streamlit dashboard
# ---------------------------------------------------------------------
# Expected project structure:
# project/
#   app.py
#   main/
#     modeling_dataset.csv
#     model_artifacts/
#       best_factory_aware_model.joblib
#       model_info.json
#   KPIs_analysis/
#     KPI2/scenario_profit_sensitivity_corrected.csv
#     KPI3/kpi3_scenario_confidence.csv
#     KPI3/factory_recommendations_ranked.csv
#     KPI4/kpi4_coverage_summary.csv
#     KPI4/kpi4_product_coverage.csv
#
# The app also searches subfolders for expected CSV filenames.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Nassau Candy | Factory Optimization",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

FACTORY_DISPLAY = {
    "Lot's O' Nuts": "Lot's O' Nuts",
    "Lots_O_Nuts": "Lot's O' Nuts",
    "Wicked Choccy's": "Wicked Choccy's",
    "Wicked_Choccys": "Wicked Choccy's",
    "Sugar Shack": "Sugar Shack",
    "Sugar_Shack": "Sugar Shack",
    "Secret Factory": "Secret Factory",
    "Secret_Factory": "Secret Factory",
    "The Other Factory": "The Other Factory",
    "The_Other_Factory": "The Other Factory",
}

FACTORIES = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory",
]

# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------
def locate_file(relative_candidates: list[str], filename: str | None = None) -> Path | None:
    """Find a file using known project paths, then recursive filename search."""
    for candidate in relative_candidates:
        path = ROOT / candidate
        if path.exists() and path.is_file():
            return path

    target = filename or (Path(relative_candidates[0]).name if relative_candidates else None)
    if target:
        matches = [
            p for p in ROOT.rglob(target)
            if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts
        ]
        if matches:
            # Prefer the expected project output folders over duplicate copies.
            matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
            return matches[0]
    return None


@st.cache_data(show_spinner=False)
def read_csv_path(path_string: str) -> pd.DataFrame:
    df = pd.read_csv(path_string)
    df.columns = df.columns.astype(str).str.strip()
    return df


@st.cache_resource(show_spinner="Loading trained factory-aware model...")
def load_model(path_string: str):
    return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def read_json_path(path_string: str) -> dict:
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(relative_candidates: list[str], filename: str) -> tuple[pd.DataFrame | None, Path | None]:
    path = locate_file(relative_candidates, filename)
    if path is None:
        return None, None
    try:
        return read_csv_path(str(path)), path
    except Exception as exc:
        st.warning(f"Could not read `{filename}`: {exc}")
        return None, path


def find_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = candidate.strip().lower()
        for actual in df.columns:
            if key in str(actual).strip().lower():
                return actual
    return None


def clean_factory(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return FACTORY_DISPLAY.get(value, value)


def number(value, decimals: int = 2) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def show_missing_file(filename: str, expected_path: str) -> None:
    st.info(f"`{filename}` was not found. Expected location: `{expected_path}`")


# ---------------------------------------------------------------------
# Locate project data and artifacts
# ---------------------------------------------------------------------
model_path = locate_file(
    ["main/model_artifacts/best_factory_aware_model.joblib",
     "model_artifacts/best_factory_aware_model.joblib"],
    "best_factory_aware_model.joblib",
)
model_info_path = locate_file(
    ["main/model_artifacts/model_info.json",
     "model_artifacts/model_info.json"],
    "model_info.json",
)
model_data, model_data_path = load_csv(
    ["main/modeling_dataset.csv", "modeling_dataset.csv"],
    "modeling_dataset.csv",
)

kpi2, kpi2_path = load_csv(
    ["KPIs_analysis/KPI2/scenario_profit_sensitivity_corrected.csv",
     "KPIanalysis/KPI2/scenario_profit_sensitivity_corrected.csv"],
    "scenario_profit_sensitivity_corrected.csv",
)
kpi2_summary, _ = load_csv(
    ["KPIs_analysis/KPI2/kpi2_results_by_assumption.csv"],
    "kpi2_results_by_assumption.csv",
)
kpi3, kpi3_path = load_csv(
    ["KPIs_analysis/KPI3/kpi3_scenario_confidence.csv"],
    "kpi3_scenario_confidence.csv",
)
kpi3_summary, _ = load_csv(
    ["KPIs_analysis/KPI3/kpi3_confidence_summary.csv"],
    "kpi3_confidence_summary.csv",
)
recommendations, recommendations_path = load_csv(
    ["KPIs_analysis/KPI3/factory_recommendations_ranked.csv",
     "optimization/factory_recommendations_ranked.csv"],
    "factory_recommendations_ranked.csv",
)
top_recommendations, _ = load_csv(
    ["KPIs_analysis/KPI3/top_n_factory_recommendations.csv"],
    "top_n_factory_recommendations.csv",
)
kpi4_summary, kpi4_summary_path = load_csv(
    ["KPIs_analysis/KPI4/kpi4_coverage_summary.csv"],
    "kpi4_coverage_summary.csv",
)
kpi4_detail, _ = load_csv(
    ["KPIs_analysis/KPI4/kpi4_product_coverage.csv"],
    "kpi4_product_coverage.csv",
)

# ---------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------
st.title("🍬 Nassau Candy Factory Optimization")
st.caption(
    "Factory-allocation decision support combining a trained prediction model "
    "with scenario-based profitability, confidence and recommendation coverage."
)

with st.sidebar:
    st.header("Dashboard controls")
    st.caption("Filters apply to the relevant product-level tables and charts.")

if model_data is None:
    st.error(
        "The trained model dataset `modeling_dataset.csv` could not be found. "
        "Place it in the `main` folder or the project root."
    )
    st.stop()

required_model_cols = [
    "Product Name", "Current Factory", "Region", "Ship Mode",
    "Units", "Lead Time", "Order Year", "Order Month",
]
missing_model_cols = [c for c in required_model_cols if c not in model_data.columns]
if missing_model_cols:
    st.error("The modeling dataset is missing columns: " + ", ".join(missing_model_cols))
    st.stop()

for col in ["Units", "Lead Time", "Order Year", "Order Month"]:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

products = sorted(model_data["Product Name"].dropna().astype(str).unique().tolist())
regions = sorted(model_data["Region"].dropna().astype(str).unique().tolist())
ship_modes = sorted(model_data["Ship Mode"].dropna().astype(str).unique().tolist())

with st.sidebar:
    selected_product = st.selectbox("Product", ["All products"] + products)
    selected_region = st.selectbox("Customer region", ["All regions"] + regions)
    selected_ship_mode = st.selectbox("Ship mode", ["All modes"] + ship_modes)
    priority = st.slider(
        "Decision preference: speed ↔ profit",
        min_value=0,
        max_value=100,
        value=60,
        help="Higher values give more weight to the model's lead-time output when displaying a combined scenario score. This is a preference slider, not a learned freight-cost model.",
    )
    st.caption("100 = speed preference; 0 = profit-impact preference.")

def apply_common_filters(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    product_col = find_column(out, ["Product Name"])
    region_col = find_column(out, ["Region"])
    mode_col = find_column(out, ["Ship Mode"])
    if selected_product != "All products" and product_col:
        out = out[out[product_col].astype(str) == selected_product]
    if selected_region != "All regions" and region_col:
        out = out[out[region_col].astype(str) == selected_region]
    if selected_ship_mode != "All modes" and mode_col:
        out = out[out[mode_col].astype(str) == selected_ship_mode]
    return out


# ---------------------------------------------------------------------
# Model loading and compatibility handling
# ---------------------------------------------------------------------
model = None
model_info = {}
model_error = None

if model_info_path:
    try:
        model_info = read_json_path(str(model_info_path))
    except Exception as exc:
        model_error = f"Could not read model_info.json: {exc}"

if model_path:
    try:
        model = load_model(str(model_path))
    except Exception as exc:
        model_error = (
            "The trained model file exists but could not be loaded. "
            "This commonly happens when the scikit-learn/joblib versions used "
            "to train the model differ from the deployment environment. "
            f"Technical detail: {type(exc).__name__}: {exc}"
        )
else:
    model_error = "The model file best_factory_aware_model.joblib was not found."

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
st.subheader("Project overview")

unique_products = int(model_data["Product Name"].nunique())
unique_factories = int(model_data["Current Factory"].nunique())
training_rows = model_info.get("training_rows", "Not available")
test_rows = model_info.get("test_rows", "Not available")
multiple_factory_products = model_info.get("products_observed_at_multiple_factories", 0)

impact_col = find_column(kpi2, [
    "Estimated Net Profit Impact", "Net Profit Impact", "Estimated Net Impact"
])
confidence_col = find_column(kpi3, [
    "Scenario_Confidence_Score", "Confidence_Score", "Scenario Confidence Score"
])

kpi2_filtered = apply_common_filters(kpi2)
kpi3_filtered = apply_common_filters(kpi3)
rec_filtered = apply_common_filters(recommendations)

mean_impact = (
    pd.to_numeric(kpi2_filtered[impact_col], errors="coerce").mean()
    if kpi2_filtered is not None and impact_col else np.nan
)
mean_confidence = (
    pd.to_numeric(kpi3_filtered[confidence_col], errors="coerce").mean()
    if kpi3_filtered is not None and confidence_col else np.nan
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products in model data", f"{unique_products:,}")
c2.metric("Factories represented", f"{unique_factories:,}")
c3.metric("Mean modeled net impact", number(mean_impact))
c4.metric("Mean scenario confidence", f"{number(mean_confidence, 1)}%")

st.caption(
    "Mean modeled net impact is averaged over the available sensitivity rows and "
    "assumption combinations after the selected filters; it is not realized savings."
)

if model_error:
    st.warning(model_error)
if model is not None:
    st.success("Trained factory-aware model loaded successfully.")
else:
    st.info("The prediction simulator is unavailable until the model artifact loads.")

if model_info:
    with st.expander("Model information"):
        st.write({
            "Selected model": model_info.get("selected_model", "Not specified"),
            "Target": model_info.get("target", "Not specified"),
            "Training rows": model_info.get("training_rows", "Not specified"),
            "Test rows": model_info.get("test_rows", "Not specified"),
            "Features": model_info.get("features", []),
            "Products observed at multiple factories": multiple_factory_products,
        })

st.warning(
    "Validation limitation: the historical modeling dataset contains no products "
    "observed at multiple factories. Therefore, predictions for a product at a "
    "different factory are extrapolations, not validated causal estimates. "
    "The source lead-time dates also require verification before interpreting "
    "the target as real shipping days."
)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_overview, tab_simulator, tab_kpi2, tab_kpi3, tab_kpi4, tab_recs, tab_method = st.tabs([
    "Overview",
    "Live factory simulator",
    "KPI 2 · Profit impact",
    "KPI 3 · Scenario confidence",
    "KPI 4 · Recommendation coverage",
    "Ranked recommendations",
    "Methodology & limitations",
])

with tab_overview:
    st.subheader("Historical modeling dataset")
    st.write(
        "The model was trained on historical product, factory, region, ship-mode, "
        "unit-volume and order-calendar information."
    )
    a, b, c = st.columns(3)
    a.metric("Rows in modeling dataset", f"{len(model_data):,}")
    b.metric("Unique products", f"{model_data['Product Name'].nunique():,}")
    c.metric("Unique current factories", f"{model_data['Current Factory'].nunique():,}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Orders by current factory**")
        st.bar_chart(model_data["Current Factory"].value_counts())
    with right:
        st.markdown("**Orders by ship mode**")
        st.bar_chart(model_data["Ship Mode"].value_counts())

    st.markdown("**Model comparison**")
    comparison, comparison_path = load_csv(
        ["main/model_artifacts/model_comparison.csv", "model_artifacts/model_comparison.csv"],
        "model_comparison.csv",
    )
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("model_comparison.csv was not found.")

    with st.expander("Preview modeling data"):
        st.dataframe(apply_common_filters(model_data), use_container_width=True, hide_index=True)

with tab_simulator:
    st.subheader("Compare factory scenarios for one product")
    st.write(
        "The simulator sends the selected order context to the saved factory-aware "
        "model once for each factory and sorts the resulting model outputs."
    )

    product_options = products
    sim_product = st.selectbox(
        "Product to simulate",
        product_options,
        index=product_options.index(selected_product) if selected_product in product_options else 0,
        key="sim_product",
    )
    product_rows = model_data[model_data["Product Name"].astype(str) == sim_product].copy()
    observed_factories = sorted(product_rows["Current Factory"].dropna().astype(str).unique().tolist())

    if len(observed_factories) == 1:
        current_factory = observed_factories[0]
    elif len(observed_factories) > 1:
        current_factory = st.selectbox("Historical factory assignment", observed_factories)
        st.warning("This product appears at more than one factory in the model data; select the historical assignment to compare.")
    else:
        current_factory = ""

    default_region = (
        selected_region if selected_region != "All regions"
        else (str(product_rows["Region"].mode().iloc[0]) if not product_rows["Region"].dropna().empty else (regions[0] if regions else ""))
    )
    default_mode = (
        selected_ship_mode if selected_ship_mode != "All modes"
        else (str(product_rows["Ship Mode"].mode().iloc[0]) if not product_rows["Ship Mode"].dropna().empty else (ship_modes[0] if ship_modes else ""))
    )
    unit_default = int(max(1, round(product_rows["Units"].median()))) if product_rows["Units"].notna().any() else 1
    year_default = int(product_rows["Order Year"].mode().iloc[0]) if product_rows["Order Year"].notna().any() else 2026
    month_default = int(product_rows["Order Month"].mode().iloc[0]) if product_rows["Order Month"].notna().any() else 1

    x1, x2, x3, x4, x5 = st.columns(5)
    with x1:
        sim_region = st.selectbox("Region", regions, index=regions.index(default_region) if default_region in regions else 0, key="sim_region")
    with x2:
        sim_mode = st.selectbox("Ship mode", ship_modes, index=ship_modes.index(default_mode) if default_mode in ship_modes else 0, key="sim_mode")
    with x3:
        sim_units = st.number_input("Units", min_value=1, value=unit_default, step=1)
    with x4:
        sim_year = st.number_input("Order year", min_value=2000, max_value=2100, value=year_default, step=1)
    with x5:
        sim_month = st.number_input("Order month", min_value=1, max_value=12, value=month_default, step=1)

    st.caption(f"Historical current factory for this product: **{clean_factory(current_factory) or 'Not available'}**")

    if st.button("Compare all five factories", type="primary", disabled=(model is None)):
        required_features = model_info.get("features", [
            "Product Name", "Current Factory", "Region", "Ship Mode",
            "Units", "Order Year", "Order Month",
        ])
        scenario_rows = []
        prediction_error = None

        for factory in FACTORIES:
            row = {
                "Product Name": sim_product,
                "Current Factory": factory,
                "Region": sim_region,
                "Ship Mode": sim_mode,
                "Units": sim_units,
                "Order Year": sim_year,
                "Order Month": sim_month,
            }
            input_frame = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in required_features}])
            try:
                predicted_value = float(model.predict(input_frame)[0])
                scenario_rows.append({
                    "Scenario Factory": factory,
                    "Predicted model output": predicted_value,
                    "Is current factory": factory == current_factory,
                })
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
                break

        if prediction_error:
            st.error("Prediction failed: " + prediction_error)
        elif scenario_rows:
            results = pd.DataFrame(scenario_rows).sort_values("Predicted model output", ascending=True).reset_index(drop=True)
            results.insert(0, "Rank", np.arange(1, len(results) + 1))
            best = results.iloc[0]
            current_rows = results[results["Is current factory"]]
            st.dataframe(
                results.style.format({"Predicted model output": "{:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Lowest model output", number(best["Predicted model output"]))
            m2.metric("Top-ranked scenario", str(best["Scenario Factory"]))
            if not current_rows.empty:
                current_value = float(current_rows.iloc[0]["Predicted model output"])
                m3.metric("Difference vs current", number(current_value - float(best["Predicted model output"])))
            else:
                m3.metric("Current factory comparison", "Unavailable")
            chart = results.set_index("Scenario Factory")[["Predicted model output"]]
            st.bar_chart(chart)
            st.caption(
                "Lower model output is ranked first. This is a model comparison only; "
                "do not interpret it as proven delivery-time savings or approve a "
                "factory move without operational validation."
            )

with tab_kpi2:
    st.subheader("KPI 2 — Estimated profit impact and sensitivity")
    if kpi2_filtered is None:
        show_missing_file("scenario_profit_sensitivity_corrected.csv", "KPIs_analysis/KPI2/")
    else:
        st.metric("Sensitivity rows in current filter", f"{len(kpi2_filtered):,}")
        if impact_col:
            impacts = pd.to_numeric(kpi2_filtered[impact_col], errors="coerce").dropna()
            positive = int((impacts > 0).sum())
            negative = int((impacts < 0).sum())
            break_even = int((impacts == 0).sum())
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Positive rows", f"{positive:,}")
            p2.metric("Negative rows", f"{negative:,}")
            p3.metric("Break-even rows", f"{break_even:,}")
            p4.metric("Average estimated impact", number(impacts.mean()))
            st.caption(
                "These are modeled scenario estimates based on the assumed freight "
                "and transfer-cost values in the source CSV, not realized company profit."
            )
            assumption_cols = [
                c for c in kpi2_filtered.columns
                if "freight" in c.lower() or "transfer" in c.lower()
            ]
            if assumption_cols:
                group_cols = assumption_cols[:2]
                summary = kpi2_filtered.copy()
                summary[impact_col] = pd.to_numeric(summary[impact_col], errors="coerce")
                summary = summary.groupby(group_cols, dropna=False)[impact_col].agg(
                    Scenarios="count", Average_Impact="mean", Median_Impact="median"
                ).reset_index()
                st.markdown("**Impact by cost assumptions**")
                st.dataframe(summary, use_container_width=True, hide_index=True)
            st.markdown("**Sensitivity detail**")
            st.dataframe(kpi2_filtered, use_container_width=True, hide_index=True)
        else:
            st.error("Could not identify the estimated net profit impact column in the KPI2 file.")

with tab_kpi3:
    st.subheader("KPI 3 — Scenario confidence")
    if kpi3_filtered is None:
        show_missing_file("kpi3_scenario_confidence.csv", "KPIs_analysis/KPI3/")
    else:
        if confidence_col:
            scores = pd.to_numeric(kpi3_filtered[confidence_col], errors="coerce").dropna()
            st.metric("Average scenario confidence score", f"{number(scores.mean(), 1)}%")
            category_col = find_column(kpi3_filtered, [
                "Confidence_Category", "Confidence Category", "Scenario Confidence Category"
            ])
            if category_col:
                st.markdown("**Alternatives by confidence category**")
                st.bar_chart(kpi3_filtered[category_col].astype(str).value_counts())
            st.caption(
                "Confidence is the percentage of tested freight/transfer-cost "
                "assumptions producing positive estimated net impact. It is not "
                "a probability that a factory move will succeed."
            )
            st.dataframe(kpi3_filtered, use_container_width=True, hide_index=True)
        else:
            st.error("Could not identify the scenario confidence score column in the KPI3 file.")

with tab_kpi4:
    st.subheader("KPI 4 — Recommendation coverage")
    if kpi4_summary is None:
        show_missing_file("kpi4_coverage_summary.csv", "KPIs_analysis/KPI4/")
    else:
        st.markdown("**Coverage summary**")
        st.dataframe(kpi4_summary, use_container_width=True, hide_index=True)
        st.caption(
            "Coverage indicates whether products have alternatives and/or "
            "validation-priority recommendations. It does not mean every "
            "recommendation is approved for implementation."
        )
    if kpi4_detail is not None:
        st.markdown("**Product-level coverage**")
        st.dataframe(apply_common_filters(kpi4_detail), use_container_width=True, hide_index=True)

with tab_recs:
    st.subheader("Ranked factory recommendations")
    if rec_filtered is None:
        show_missing_file("factory_recommendations_ranked.csv", "KPIs_analysis/KPI3/")
    else:
        status_col = find_column(rec_filtered, ["Recommendation Status", "Priority Status"])
        if status_col:
            st.markdown("**Recommendation status counts**")
            st.bar_chart(rec_filtered[status_col].astype(str).value_counts())
        rank_col = find_column(rec_filtered, ["Recommendation Rank"])
        sort_cols = [c for c in [rank_col] if c]
        display_recs = rec_filtered.sort_values(sort_cols) if sort_cols else rec_filtered
        st.dataframe(display_recs, use_container_width=True, hide_index=True)
        st.caption(
            "The ranking is based on the selected scenario assumptions and the "
            "scoring logic in the recommendation script. Unknown operational risk "
            "is not measured; it must not be interpreted as zero risk."
        )
    if top_recommendations is not None:
        with st.expander("Top-N recommendation output"):
            st.dataframe(apply_common_filters(top_recommendations), use_container_width=True, hide_index=True)

with tab_method:
    st.subheader("How to interpret this dashboard")
    st.markdown(
        """
        **What the project currently provides**
        - A trained model that estimates the historical `Lead Time` target from the saved feature set.
        - A five-factory scenario comparison for the selected product and order context.
        - Profit sensitivity results under nine illustrative freight/transfer-cost combinations.
        - Scenario confidence calculated from the share of tested assumptions with positive estimated net impact.
        - Product and alternative recommendation coverage, plus ranked candidate alternatives.

        **Important limitations**
        - No product was observed at multiple factories in the training data. Alternative-factory model predictions are therefore extrapolations, not validated causal comparisons.
        - The source date audit found implausibly large calculated lead times. Confirm the meaning and quality of `Order Date` and `Ship Date` before calling the model output actual shipping days.
        - Factory distances are straight-line geographical estimates, not road-route distances or carrier invoices.
        - Freight and transfer costs are illustrative assumptions, not confirmed company rates.
        - Operational risk is currently unknown/unmeasured. A neutral score in the ranking code is not an observed risk measurement.
        - The recommendation ranking is a scenario-based decision-support method, not a global mathematical optimizer and not an automatic factory reassignment.
        - Factory capacity, production capability, labor, inventory, service levels and regulatory constraints are not fully represented.
        """
    )
    st.markdown("**Model artifact and data locations**")
    st.write({
        "Model file": str(model_path) if model_path else "Not found",
        "Model metadata": str(model_info_path) if model_info_path else "Not found",
        "Modeling dataset": str(model_data_path) if model_data_path else "Not found",
        "KPI2 input": str(kpi2_path) if kpi2_path else "Not found",
        "KPI3 input": str(kpi3_path) if kpi3_path else "Not found",
        "KPI4 summary": str(kpi4_summary_path) if kpi4_summary_path else "Not found",
        "Recommendation input": str(recommendations_path) if recommendations_path else "Not found",
    })

st.divider()
st.caption(
    "Nassau Candy Factory Optimization · Decision-support prototype · "
    "Validate recommendations with operations, finance and logistics teams before implementation."
)
