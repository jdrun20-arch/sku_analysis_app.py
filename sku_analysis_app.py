import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="SKU Analysis App", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload SKU Performance CSV", type=["csv"])

# Thresholds
sales_threshold = st.sidebar.slider("Minimum Sales to Retain SKU", 0, 10000, 500)
volume_threshold = st.sidebar.slider("Minimum Volume to Retain SKU", 0, 10000, 500)
margin_threshold = st.sidebar.slider("Minimum Margin % to Retain SKU", 0.0, 1.0, 0.2)

st.sidebar.markdown("---")
module = st.sidebar.radio("Select Module", 
                          ["SKU Performance & Shelf Space", "Sales Analysis", "Insights", "Custom Analysis"])

# --- MODULE 1: SKU PERFORMANCE & SHELF SPACE ---
if module == "SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimization")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Ensure column names are as expected ---
        required_cols = ['SKU','Store Code','Sales','Volume','Margins','Width','Facings',
                         'Product Type','Variant','Item Size']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            st.stop()

        # --- Compute recommendations ---
        df['Retain'] = (df['Sales'] >= sales_threshold) & (df['Volume'] >= volume_threshold) & (df['Margins'] >= margin_threshold)
        df['Recommendation'] = np.where(df['Retain'], 'Retain', 'Delist')

        # Facings suggestion (weighted score)
        df['SalesNorm'] = df['Sales'] / df['Sales'].max()
        df['VolumeNorm'] = df['Volume'] / df['Volume'].max()
        df['MarginNorm'] = df['Margins'] / df['Margins'].max()

        df['WeightedScore'] = (0.3 * df['SalesNorm']) + (0.3 * df['VolumeNorm']) + (0.4 * df['MarginNorm'])
        df['Suggested Facings'] = (df['WeightedScore'] * 5).round().astype(int).clip(lower=1)

        # Shelf capacity check
        total_shelf_width = 1000  # assume fixed width
        df['TotalWidth'] = df['Width'] * df['Suggested Facings']
        df['Fits Shelf'] = df['TotalWidth'].cumsum() <= total_shelf_width

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📏 Shelf Usage")
            used_width = df.loc[df['Fits Shelf'], 'TotalWidth'].sum()
            st.metric("Shelf Used (cm)", f"{used_width} / {total_shelf_width}")

        with col2:
            st.subheader("🚨 SKUs That Cannot Fit in Shelf")
            overflow = df[~df['Fits Shelf']]
            if overflow.empty:
                st.success("✅ All SKUs fit in the shelf space.")
            else:
                st.dataframe(overflow[['SKU','Product Type','Variant','TotalWidth']])

        # --- Summary by Product Type & Variant ---
        st.subheader("📋 SKU Summary by Product Type & Variant")
        summary = df.groupby(['Product Type','Variant','Recommendation']).size().unstack(fill_value=0)
        st.dataframe(summary)

        # --- Downloadable Excel ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="SKU_Recommendations")
            summary.to_excel(writer, sheet_name="Summary")
        st.download_button("📥 Download Recommendations (Excel)", 
                           data=output.getvalue(),
                           file_name="SKU_Recommendations.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # --- Visualization ---
        st.subheader("📊 SKU Distribution by Product Type & Variant")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.groupby(['Product Type','Variant']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("Count of SKUs")
        ax.set_title("SKU Count by Product Type & Variant")
        ax.legend(title="Variant", bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

    else:
        st.info("⬆️ Please upload a CSV file to begin analysis.")

# --- MODULE 2: SALES ANALYSIS ---
elif module == "Sales Analysis":
    st.title("📈 Sales Analysis")
    st.info("This module analyzes sales performance over time.")
    # Future expansion: timeseries plots, MoM change, etc.

# --- MODULE 3: INSIGHTS ---
elif module == "Insights":
    st.title("💡 Insights")
    st.info("This module provides insights based on SKU performance trends.")
    # Placeholder for AI-driven insights

# --- MODULE 4: CUSTOM ANALYSIS ---
elif module == "Custom Analysis":
    st.title("🛠 Custom Analysis")
    st.info("This module allows custom filters and pivot tables.")
    # Placeholder for custom analysis features
