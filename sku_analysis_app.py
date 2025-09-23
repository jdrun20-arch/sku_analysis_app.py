import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="SKU Analysis Tool", layout="wide")

# Sidebar for navigation
st.sidebar.title("📑 Navigation")
module = st.sidebar.radio("Select Module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Quality Checks",
    "Insights & Recommendations"
])

# ======================================================
# MODULE 1 - SKU Performance & Shelf Space
# ======================================================
if module == "SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimization")

    # Sidebar Inputs
    st.sidebar.header("⚙️ Module 1 Settings")
    target_retain_pct = st.sidebar.slider("Target to Retain %", 0, 100, 80)
    total_shelf_space = st.sidebar.number_input("Total Shelf Space (cm)", min_value=50, value=200, step=10)
    number_of_layers = st.sidebar.number_input("Number of Layers", min_value=1, value=4, step=1)

    uploaded_file = st.file_uploader("Upload your SKU CSV file", type=["csv"], key="module1_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_columns = ["SKU","Store Code","Sales","Volume","Margins","Width","Facings","Product Type","Variant","Item Size"]
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
        else:
            # Weighted score calculation
            df["Sales %"] = df["Sales"] / df["Sales"].sum()
            df["Volume %"] = df["Volume"] / df["Volume"].sum()
            df["Margin %"] = df["Margins"] / df["Margins"].sum()
            df["Weighted Score"] = (0.3 * df["Sales %"]) + (0.3 * df["Volume %"]) + (0.4 * df["Margin %"])

            total_facings_available = (total_shelf_space * number_of_layers) / df["Width"].mean()
            df["Suggested Facings"] = np.floor(df["Weighted Score"] / df["Weighted Score"].sum() * total_facings_available).astype(int)

            # Recommendations
            threshold = np.percentile(df["Weighted Score"], 100 - target_retain_pct)
            df["Recommendation"] = np.where(df["Weighted Score"] >= threshold, "Retain", "Delist")
            df.loc[df["Suggested Facings"] > df["Facings"], "Recommendation"] = "Expand"

            # Shelf Space Usage
            df["Space Used"] = df["Facings"] * df["Width"]
            total_space_used = df["Space Used"].sum()
            shelf_capacity = total_shelf_space * number_of_layers
            shelf_usage_pct = (total_space_used / shelf_capacity) * 100

            st.subheader("📊 Shelf Space Usage & SKU Fit")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shelf Usage (%)", f"{shelf_usage_pct:.2f}%")
            with col2:
                if total_space_used <= shelf_capacity:
                    st.success("✅ All SKUs fit the shelf space.")
                else:
                    skus_overflow = df[df["Space Used"].cumsum() > shelf_capacity]
                    if not skus_overflow.empty:
                        st.error(f"⚠️ {len(skus_overflow)} SKUs cannot fit in the shelf space.")
                        st.dataframe(skus_overflow[["SKU","Product Type","Variant","Facings","Width","Space Used"]])

            # SKU Distribution Visualization
            st.subheader("📊 SKU Distribution by Product Type & Variant")
            dist = df.groupby(["Product Type","Variant"]).size().reset_index(name="Count")
            st.bar_chart(dist.set_index(["Product Type","Variant"]))

            # Recommendations Table
            st.subheader("📋 SKU Recommendations")
            st.dataframe(df[["SKU","Product Type","Variant","Sales","Volume","Margins","Facings","Suggested Facings","Recommendation"]])

            # Summary by Product Type & Variant
            st.subheader("📊 Summary by Product Type & Variant")
            summary = df.groupby(["Product Type","Variant","Recommendation"]).size().reset_index(name="Count")
            st.dataframe(summary)

            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="SKU Recommendations", index=False)
                summary.to_excel(writer, sheet_name="Summary", index=False)
            st.download_button(
                label="📥 Download Recommendations (Excel)",
                data=output.getvalue(),
                file_name="SKU_Recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("📂 Please upload a CSV file to start analysis.")

# ======================================================
# MODULE 2 - Sales Analysis
# ======================================================
elif module == "Sales Analysis":
    st.title("📈 Sales Analysis")

    uploaded_sales = st.file_uploader("Upload Sales Data CSV", type=["csv"], key="module2_upload")
    if uploaded_sales:
        sales_df = pd.read_csv(uploaded_sales)

        required_cols = ["Store Code","Date","Sales","Baseline"]
        if not set(required_cols).issubset(sales_df.columns):
            st.error(f"CSV must contain: {', '.join(required_cols)}")
        else:
            sales_df["ChangePct"] = ((sales_df["Sales"] - sales_df["Baseline"]) / sales_df["Baseline"]) * 100
            sales_df["Signal"] = np.where(sales_df["ChangePct"] > 10, "📈 Growth",
                                          np.where(sales_df["ChangePct"] < -10, "📉 Decline", "➖ Stable"))
            st.subheader("Sales Insights")
            st.dataframe(sales_df[["Store Code","Date","Sales","Baseline","ChangePct","Signal"]])
    else:
        st.info("📂 Please upload sales data to view analysis.")

# ======================================================
# MODULE 3 - Quality Checks
# ======================================================
elif module == "Quality Checks":
    st.title("✅ Quality Checks")
    st.write("""
    This section checks for:
    - Missing values in critical fields
    - Negative sales/volume/margin values
    - Duplicate SKUs
    """)

    qc_file = st.file_uploader("Upload Data for Quality Check", type=["csv"], key="module3_upload")
    if qc_file:
        qc_df = pd.read_csv(qc_file)
        issues = {}

        # Missing values
        missing = qc_df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            issues["Missing Values"] = missing

        # Negative checks
        negatives = qc_df[(qc_df["Sales"] < 0) | (qc_df["Volume"] < 0) | (qc_df["Margins"] < 0)]
        if not negatives.empty:
            issues["Negative Values"] = negatives

        # Duplicate SKUs
        duplicates = qc_df[qc_df.duplicated(subset=["SKU"], keep=False)]
        if not duplicates.empty:
            issues["Duplicate SKUs"] = duplicates

        if issues:
            st.error("⚠️ Issues found in dataset:")
            for k, v in issues.items():
                st.subheader(k)
                st.dataframe(v)
        else:
            st.success("✅ No issues found!")
    else:
        st.info("📂 Please upload a CSV for quality check.")

# ======================================================
# MODULE 4 - Insights & Recommendations
# ======================================================
elif module == "Insights & Recommendations":
    st.title("💡 Insights & Recommendations")
    st.write("Summarize key findings and propose actions here.")

    insights_text = st.text_area("Write your insights here...")
    if st.button("Save Insights"):
        with open("insights.txt", "w") as f:
            f.write(insights_text)
        st.success("Insights saved successfully.")
