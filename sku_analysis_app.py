import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="Retail Insights App", layout="wide")
st.title("🛒 Retail Insights Dashboard")

# --- SIDEBAR: Module Selector ---
module = st.sidebar.radio(
    "Choose Module:",
    [
        "SKU Performance & Shelf Space",
        "Sales Analysis",
        "Inventory Insights",
        "Promotions Analysis"
    ]
)

# ===============================
# MODULE 1: SKU PERFORMANCE & SHELF SPACE
# ===============================
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")

    uploaded_file = st.sidebar.file_uploader("Upload SKU CSV", type=["csv"])
    shelf_capacity = st.sidebar.number_input("Shelf capacity (total width)", min_value=10, value=1000, step=50)

    if uploaded_file is None:
        st.info("📥 Please upload a CSV file with SKU data to get started.")
    else:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            "SKU", "Store Code", "Sales", "Volume", "Margins", "Width", "Facings",
            "Product Type", "Variant", "Item Size"
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            # --- CALCULATE PERFORMANCE METRICS ---
            df["Sales Contribution"] = df["Sales"] / df["Sales"].sum()
            df["Volume Contribution"] = df["Volume"] / df["Volume"].sum()
            df["Margin Contribution"] = df["Margins"] / df["Margins"].sum()

            df["Performance Score"] = (
                0.5 * df["Sales Contribution"] +
                0.3 * df["Volume Contribution"] +
                0.2 * df["Margin Contribution"]
            )

            # Suggested facings based on relative score
            df["Suggested Facings"] = np.ceil(df["Performance Score"] * 10).astype(int)
            df.loc[df["Suggested Facings"] < 1, "Suggested Facings"] = 1

            # Shelf space calculations
            df["Space Needed"] = df["Suggested Facings"] * df["Width"]
            df["Fits Shelf"] = True  # default; will check later
            total_required_space = df["Space Needed"].sum()

            # Recommendations
            df["Recommendation"] = np.where(
                df["Performance Score"] >= df["Performance Score"].quantile(0.7), "Expand",
                np.where(df["Performance Score"] <= df["Performance Score"].quantile(0.3), "Delist", "Retain")
            )

            # --- SUMMARY ---
            summary = (
                df.groupby(["Product Type", "Variant", "Recommendation"])
                .size()
                .reset_index(name="Count")
            )

            # Sidebar filter
            product_types = summary["Product Type"].unique().tolist()
            selected_types = st.sidebar.multiselect(
                "Filter by Product Type:",
                options=product_types,
                default=product_types
            )

            df_filtered = df[df["Product Type"].isin(selected_types)]
            summary_filtered = summary[summary["Product Type"].isin(selected_types)]

            # --- DISPLAY SECTION ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📏 Shelf Usage")
                st.metric("Total Space Needed", f"{total_required_space:.1f}")
                st.metric("Shelf Capacity", f"{shelf_capacity}")
                st.metric("Usage %", f"{(total_required_space / shelf_capacity) * 100:.1f}%")

            with col2:
                st.subheader("🚫 SKUs That Cannot Fit")
                if total_required_space <= shelf_capacity:
                    st.success("✅ All SKUs can fit in the shelf!")
                else:
                    # Sort by Space Needed and mark overflow
                    df_sorted = df_filtered.sort_values("Space Needed", ascending=False).copy()
                    df_sorted["CumulativeSpace"] = df_sorted["Space Needed"].cumsum()
                    df_sorted["Fits Shelf"] = df_sorted["CumulativeSpace"] <= shelf_capacity
                    overflow_df = df_sorted[df_sorted["Fits Shelf"] == False]

                    if overflow_df.empty:
                        st.success("✅ All selected SKUs fit in shelf!")
                    else:
                        st.warning(f"{len(overflow_df)} SKUs exceed shelf space!")
                        st.dataframe(
                            overflow_df[["SKU", "Product Type", "Variant", "Suggested Facings", "Space Needed"]],
                            use_container_width=True
                        )

            st.subheader("📋 SKU Summary by Product Type & Variant")
            st.dataframe(summary_filtered, use_container_width=True)

            if not summary_filtered.empty:
                summary_filtered["Product-Variant"] = summary_filtered["Product Type"] + " - " + summary_filtered["Variant"]
                fig = px.bar(
                    summary_filtered.sort_values("Count", ascending=True),
                    x="Count",
                    y="Product-Variant",
                    color="Recommendation",
                    orientation="h",
                    text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(height=500, bargap=0.3, xaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📑 SKU Recommendation Table")
            display_cols = [
                "SKU", "Product Type", "Variant", "Item Size", "Sales", "Volume", "Margins",
                "Performance Score", "Recommendation", "Suggested Facings", "Width", "Space Needed"
            ]
            st.dataframe(df_filtered[display_cols], use_container_width=True)

            # --- DOWNLOAD BUTTON ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_filtered[display_cols].to_excel(writer, sheet_name="Recommendations", index=False)

            st.download_button(
                label="📥 Download SKU Recommendations",
                data=output.getvalue(),
                file_name="sku_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ===============================
# MODULE 2: SALES ANALYSIS
# ===============================
elif module == "Sales Analysis":
    st.header("📈 Sales Analysis")
    uploaded_sales = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_sales is not None:
        sales_df = pd.read_csv(uploaded_sales)
        st.dataframe(sales_df.head(), use_container_width=True)

# ===============================
# MODULE 3: INVENTORY INSIGHTS
# ===============================
elif module == "Inventory Insights":
    st.header("📦 Inventory Insights")
    uploaded_inventory = st.sidebar.file_uploader("Upload Inventory CSV", type=["csv"])
    if uploaded_inventory is not None:
        inv_df = pd.read_csv(uploaded_inventory)
        st.dataframe(inv_df.head(), use_container_width=True)

# ===============================
# MODULE 4: PROMOTIONS ANALYSIS
# ===============================
elif module == "Promotions Analysis":
    st.header("🏷 Promotions Analysis")
    uploaded_promos = st.sidebar.file_uploader("Upload Promotions CSV", type=["csv"])
    if uploaded_promos is not None:
        promo_df = pd.read_csv(uploaded_promos)
        st.dataframe(promo_df.head(), use_container_width=True)
