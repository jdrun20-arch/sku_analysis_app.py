import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

st.set_page_config(layout="wide")

st.title("🛒 Retail Performance & Insights Hub")

# --- Session State for Insights Storage ---
if "insights" not in st.session_state:
    st.session_state.insights = []
if "approved_insights" not in st.session_state:
    st.session_state.approved_insights = []

# --- Sidebar Navigation ---
module = st.sidebar.radio("📌 Select Module", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insights",
    "Approve Insights"
])

# ================================
# 1️⃣ SKU PERFORMANCE & SHELF SPACE
# ================================
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("📂 Upload SKU Performance CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Performance Scoring ---
        df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

        df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)

        cutoff_expand = df['Score'].quantile(0.70)
        cutoff_delist = df['Score'].quantile(0.30)

        def classify(score):
            if score >= cutoff_expand:
                return "Expand"
            elif score <= cutoff_delist:
                return "Delist"
            else:
                return "Retain"

        df['Recommendation'] = df['Score'].apply(classify)

        def justify(row):
            if row['Recommendation'] == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif row['Recommendation'] == "Delist":
                return "Low performance → candidate for phase-out."
            else:
                return "Balanced performance → maintain current space."

        df['Justification'] = df.apply(justify, axis=1)

        # --- SKU Recommendation Summary ---
        st.subheader("📊 SKU Recommendation Summary")
        total_skus = len(df)
        num_expand = len(df[df['Recommendation']=='Expand'])
        num_retain = len(df[df['Recommendation']=='Retain'])
        num_delist = len(df[df['Recommendation']=='Delist'])

        st.write(f"Total SKUs uploaded: {total_skus}")
        st.write(f"✅ Expand SKUs: {num_expand}")
        st.write(f"🔸 Retain SKUs: {num_retain}")
        st.write(f"❌ Delist SKUs: {num_delist}")

        # --- Sidebar Settings ---
        st.sidebar.header("⚙️ Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, 100, 50, 5)

        total_shelf_space = shelf_width * num_layers

        def base_facings(rec):
