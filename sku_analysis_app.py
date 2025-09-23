import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="SKU Performance & Shelf Space", layout="wide")

st.title("SKU Analysis Dashboard")

# --- SIDEBAR ---
module = st.sidebar.radio(
    "Select Module:",
    ["SKU Performance & Shelf Space", "Sales Analysis", "Inventory Insights", "Promotions Analysis"]
)

# --- MODULE 1: SKU PERFORMANCE & SHELF SPACE ---
if module == "SKU Performance & Shelf Space":
    st.header("Module 1: SKU Performance & Shelf Space")

    uploaded_file = st.file_uploader("Upload SKU Performance CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- DATA PREPARATION ---
        required_columns = [
            "SKU", "Store Code", "Sales", "Volume", "Margins", "Width", "Facings",
            "Product Type", "Variant", "Item Size"
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {', '.join(missing)}")
        else:
            # Calculate contributions
            df["Sales Contribution"] = df["Sales"] / df["Sales"].sum()
            df["Volume Contribution"] = df["Volume"] / df["Volume"].sum()
            df["Margin Contribution"] = df["Margins"] / df["Margins"].sum()

            # Weighted score to rank SKUs
            df["Performance Score"] = (
                0.5 * df["Sales Contribution"] +
                0.3 * df["Volume Contribution"] +
                0.2 * df["Margin Contribution"]
            )

            # --- FACINGS SUGGESTION ---
            df["Suggested Facings"] = np.ceil(df["Performance Score"] * 10).astype(int)
            df.loc[df["Suggested Facings"] < 1, "Suggested Facings"] = 1

            # --- SHELF FIT CHECK ---
            df["Total Width"] = df["Suggested Facings"] * df["Width"]
            shelf_capacity = 1000  # adjustable
            total_required_width = df["Total Width"].sum()
            df["Fits Shelf"] = total_required_width <= shelf_capacity

            # --- RECOMMENDATION LOGIC ---
            df["Recommendation"] = np.where(df["Performance Score"] >= 0.05, "Expand",
                                   np.where(df["Performance Score"] >= 0.02, "Retain", "Delist"))

            # --- SUMMARY TABLE ---
            summary = (
                df.groupby(["Product Type", "Variant", "Recommendation"])
                .size()
                .reset_index(name="Count")
            )

            # --- PRODUCT TYPE FILTER ---
            product_types = summary["Product Type"].unique().tolist()
            selected_types = st.multiselect(
                "Filter by Product Type:",
                options=product_types,
                default=product_types
            )

            filtered_summary = summary[summary["Product Type"].isin(selected_types)]
            filtered_df = df[df["Product Type"].isin(selected_types)]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Shelf Usage")
                st.metric("Total Required Width", f"{total_required_width:.2f}")
                st.metric("Shelf Capacity", f"{shelf_capacity}")
                st.metric("Usage %", f"{(total_required_width/shelf_capacity)*100:.2f}%")

            with col2:
                st.subheader("SKUs that Cannot Fit in Shelf")
                overflow_df = filtered_df[filtered_df["Fits Shelf"] == False]
                if overflow_df.empty:
                    st.success("✅ All SKUs can fit in the shelf!")
                else:
                    st.warning(f"{len(overflow_df)} SKUs exceed shelf space!")
                    st.dataframe(overflow_df[["SKU", "Product Type", "Variant", "Suggested Facings", "Total Width"]])

            # --- SUMMARY TABLE & VISUAL ---
            st.subheader("SKU Summary by Product Type & Variant")
            st.dataframe(filtered_summary, use_container_width=True)

            filtered_summary["Category-Variant"] = (
                filtered_summary["Product Type"] + " - " + filtered_summary["Variant"]
            )

            fig_summary = px.bar(
                filtered_summary.sort_values("Count", ascending=True),
                x="Count",
                y="Category-Variant",
                color="Recommendation",
                orientation="h",
                text="Count",
                category_orders={"Recommendation": ["Expand", "Retain", "Delist"]}
            )
            fig_summary.update_traces(textposition="outside")
            fig_summary.update_layout(
                height=500,
                xaxis_title="Number of SKUs",
                yaxis_title="Product Type - Variant",
                legend_title="Recommendation",
                bargap=0.3
            )
            st.plotly_chart(fig_summary, use_container_width=True)

            # --- FULL TABLE & DOWNLOAD ---
            st.subheader("SKU Recommendation Table")
            st.dataframe(filtered_df, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="Recommendations")
            st.download_button(
                label="📥 Download SKU Recommendations (Excel)",
                data=output.getvalue(),
                file_name="sku_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- MODULE 2: SALES ANALYSIS ---
elif module == "Sales Analysis":
    st.header("Module 2: Sales Analysis")
    uploaded_sales = st.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_sales is not None:
        sales_df = pd.read_csv(uploaded_sales)
        st.dataframe(sales_df.head(), use_container_width=True)

# --- MODULE 3: INVENTORY INSIGHTS ---
elif module == "Inventory Insights":
    st.header("Module 3: Inventory Insights")
    uploaded_inventory = st.file_uploader("Upload Inventory CSV", type=["csv"])
    if uploaded_inventory is not None:
        inv_df = pd.read_csv(uploaded_inventory)
        st.dataframe(inv_df.head(), use_container_width=True)

# --- MODULE 4: PROMOTIONS ANALYSIS ---
elif module == "Promotions Analysis":
    st.header("Module 4: Promotions Analysis")
    uploaded_promos = st.file_uploader("Upload Promotions CSV", type=["csv"])
    if uploaded_promos is not None:
        promo_df = pd.read_csv(uploaded_promos)
        st.dataframe(promo_df.head(), use_container_width=True)
