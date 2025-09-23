import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

st.title("SKU Performance & Shelf Space Optimizer")

uploaded_file = st.file_uploader("Upload your SKU CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Clean Column Names ---
    df.columns = df.columns.str.strip()

    # --- Calculate Metrics ---
    total_shelf_space = df['Width'].sum() * 1.2  # add 20% buffer

    # Compute Contribution %
    df['Sales_Contribution'] = df['Sales'] / df['Sales'].sum()
    df['Volume_Contribution'] = df['Volume'] / df['Volume'].sum()
    df['Margin_Contribution'] = df['Margins'] / df['Margins'].sum()

    # Updated Performance Score with 30-30-40 weighting
    df['Weighted_Contribution'] = (
        df['Sales_Contribution'] * 0.3 +
        df['Volume_Contribution'] * 0.3 +
        df['Margin_Contribution'] * 0.4
    )

    # Recommendation Logic
    def recommend(row):
        if row['Sales_Contribution'] > 0.05 and row['Margin_Contribution'] > 0.05:
            return "Expand"
        elif row['Sales_Contribution'] < 0.01 and row['Margin_Contribution'] < 0.01:
            return "Delist"
        else:
            return "Retain"

    df['Recommendation'] = df.apply(recommend, axis=1)

    # --- Contribution-Based Facings ---
    total_shelf_facings = int(total_shelf_space / df['Width'].mean())
    df['Suggested Facings'] = (df['Weighted_Contribution'] * total_shelf_facings).round()

    expand_facings = 3
    retain_facings = 2
    delist_facings = 0

    df.loc[df['Recommendation'] == "Expand", 'Suggested Facings'] = df['Suggested Facings'].clip(lower=expand_facings)
    df.loc[df['Recommendation'] == "Retain", 'Suggested Facings'] = df['Suggested Facings'].clip(lower=retain_facings)
    df.loc[df['Recommendation'] == "Delist", 'Suggested Facings'] = delist_facings

    df['Space Needed'] = df['Width'] * df['Suggested Facings']

    # --- Shelf Fit Check ---
    total_space_needed = df['Space Needed'].sum()
    df['Fits Shelf'] = total_space_needed <= total_shelf_space

    st.subheader("Shelf Usage & SKUs That Cannot Fit")
    col1, col2 = st.columns(2)

    with col1:
        shelf_usage_pct = (total_space_needed / total_shelf_space) * 100
        st.metric("Shelf Usage", f"{shelf_usage_pct:.1f}%")

    with col2:
        overflow_df = df[df['Fits Shelf'] == False]
        if not overflow_df.empty:
            st.dataframe(overflow_df[['SKU', 'Store Code', 'Recommendation', 'Suggested Facings', 'Space Needed']])
        else:
            st.success("✅ All SKUs fit within the available shelf space.")

    # --- Summary by Product Type & Variant with Counts ---
    st.subheader("SKU Summary by Product Type & Variant")
    summary = df.groupby(['Product Type', 'Variant', 'Recommendation']).size().reset_index(name='Count')
    pivot_summary = summary.pivot_table(index=['Product Type','Variant'], columns='Recommendation', values='Count', fill_value=0).reset_index()

    selected_product_types = st.multiselect("Filter by Product Type", options=pivot_summary['Product Type'].unique(), default=list(pivot_summary['Product Type'].unique()))
    filtered_summary = pivot_summary[pivot_summary['Product Type'].isin(selected_product_types)]

    st.dataframe(filtered_summary)

    chart_df = summary[summary['Product Type'].isin(selected_product_types)]
    chart_df['Product-Variant'] = chart_df['Product Type'] + " - " + chart_df['Variant']
    fig = px.bar(chart_df, x='Count', y='Product-Variant', color='Recommendation', orientation='h', barmode='group', text='Count')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- Downloadable Recommendations ---
    st.subheader("Download SKU Recommendations")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="SKU Recommendations")
        filtered_summary.to_excel(writer, index=False, sheet_name="Summary")

    st.download_button(label="📥 Download Excel", data=output.getvalue(), file_name="sku_recommendations.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file with SKU, Store Code, Sales, Volume, Margins, Width, Facings, Product Type, Variant, Item Size.")
