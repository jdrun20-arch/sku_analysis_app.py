# sku_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide", page_title="SKU Performance + Shelf Capacity Analyzer")

st.title("📊 SKU Performance + Shelf Capacity Analyzer (all features)")

st.write("""
Upload a CSV with SKU data. The app will auto-detect common column names (Sales, Volume, Margin, Width, Facings).
You can correct mapping in the sidebar if needed.
""")

# ---------------------------
# Utility: possible column name synonyms (for auto-detect)
# ---------------------------
COL_SYNS = {
    'Sales': ['sales', 'revenue', 'net_sales', 'sales_amt', 'sales_value'],
    'Volume': ['volume', 'units', 'qty', 'quantity', 'sales_units'],
    'Margin': ['margin', 'gross_margin', 'gp', 'profit'],
    'Width': ['width', 'size', 'item_width', 'pack_width', 'width_in', 'width_cm', 'w'],
    'Facings': ['facings', 'facing', 'no_facings', 'num_facings', 'faces']
}

def autodetect_columns(df_columns):
    mapped = {}
    cols_lower = {c.lower(): c for c in df_columns}
    for canonical, syns in COL_SYNS.items():
        found = None
        for s in syns:
            if s.lower() in cols_lower:
                found = cols_lower[s.lower()]
                break
        mapped[canonical] = found
    return mapped

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload SKU CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to begin. Example columns: SKU, Sales, Volume, Margin, Width (inches or cm), Facings (optional).")
    st.stop()

# Load CSV
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.sidebar.header("Column mapping (auto-detected)")
auto_map = autodetect_columns(df_raw.columns)
# Allow user to correct mappings using selects (or 'None')
col_map = {}
for key in ['Sales', 'Volume', 'Margin', 'Width', 'Facings']:
    col_map[key] = st.sidebar.selectbox(f"{key} column", options=[None] + list(df_raw.columns), index=(1 + list(df_raw.columns).index(auto_map[key]) if auto_map[key] in df_raw.columns else 0))

# If user leaves Sales/Volume/Margin/Width None -> error
required = ['Sales', 'Volume', 'Margin', 'Width']
missing_required = [r for r in required if not col_map[r]]
if missing_required:
    st.error(f"Please map these required columns in the sidebar: {missing_required}")
    st.stop()

# Copy df and rename mapped columns to canonical names
df = df_raw.copy()
rename_map = {col_map[k]: k for k in col_map if col_map[k]}
df = df.rename(columns=rename_map)

# Units: inches or cm
st.sidebar.
