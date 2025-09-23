import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="Retail Insights App", layout="wide")

# ---------- Config ----------
INSIGHTS_FILE = "insights.csv"
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Date", "Store Code", "Insight", "Status"]).to_csv(INSIGHTS_FILE, index=False)

# ---------- Helpers ----------
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.session_state["_refresh_needed"] = not st.session_state.get("_refresh_needed", False)
        st.success("Change saved. Please refresh the page to see updates.")
        st.stop()

def ensure_insights_df():
    df = pd.read_csv(INSIGHTS_FILE)
    for c in ["Date","Store Code","Insight","Status"]:
        if c not in df.columns:
            df[c] = ""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime("%Y-%m-%d")
    return df

def write_insights_df(df):
    df.to_csv(INSIGHTS_FILE, index=False)

def clean_sales_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    neg_mask = s2.str.match(r'^\(.*\)$')
    s2 = s2.str.replace(r'[\(\)]', '', regex=True)
    s2 = s2.str.replace(r'[^\d\.\-]', '', regex=True)
    out = pd.to_numeric(s2, errors='coerce')
    out[neg_mask] = -out[neg_mask]
    return out.fillna(0.0)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    lowermap = {c.lower(): c for c in df.columns}
    mapping = {}
    for cand in ('date','day','transaction_date'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Date'; break
    for cand in ('store code','store_code','storecode','store'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Store Code'; break
    for cand in ('sales','sale','amount','revenue'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Sales'; break
    for cand in ('sku','sku_code','product','item'):
        if cand in lowermap: mapping[lowermap[cand]] = 'SKU'; break
    for cand in ('width','item width','item width (in)','width_in','item_width'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Width'; break
    return df.rename(columns=mapping)

# ---------- Sidebar Module Selection ----------
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insight",
    "Approve Insights"
])

# ========== MODULE 1: SKU Performance & Shelf Space ==========
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    sku_file = st.file_uploader("Upload SKU CSV (required: SKU, Sales, Volume, Margin). Optional: Width)", type=["csv"])
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        required = ["SKU","Sales","Volume","Margin"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # --- Clean numeric data ---
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)

            # --- Scoring ---
            def norm(series):
                mx = series.replace(0, pd.NA).max()
                if pd.isna(mx) or mx == 0:
                    return pd.Series(0, index=series.index)
                return series / mx
            sku['Sales_Norm'] = norm(sku['Sales'])
            sku['Volume_Norm'] = norm(sku['Volume'])
            sku['Margin_Norm'] = norm(sku['Margin'])
            sku['Score'] = (sku['Sales_Norm']*0.3) + (sku['Volume_Norm']*0.3) + (sku['Margin_Norm']*0.4)
            sku['Rank'] = sku['Score'].rank(method='min', ascending=False).astype(int)

            # --- Recommendations ---
            cutoff_expand = sku['Score'].quantile(0.70)
            cutoff_delist = sku['Score'].quantile(0.30)
            sku['Recommendation'] = sku['Score'].apply(
                lambda s: "Expand" if s>=cutoff_expand else ("Delist" if s<=cutoff_delist else "Retain")
            )
            sku['Justification'] = sku['Recommendation'].map({
                'Expand': "High performance — consider expansion.",
                'Delist': "Low performance — candidate for phase-out.",
                'Retain': "Balanced — maintain."
            })

            # --- Shelf Settings ---
            st.sidebar.header("Shelf settings")
            expand_facings = st.sidebar.slider("Facings for Expand", 1, 10, 3)
            retain_facings = st.sidebar.slider("Facings for Retain", 1, 10, 2)
            delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 1)
            min_facings = st.sidebar.number_input("Minimum facings", 1, 10, 2)
            shelf_width = st.sidebar.number_input("Shelf width per layer", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs", value=False)
            top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, max(5, len(sku))), min(50, max(5, len(sku))))

            total_shelf_space = shelf_width * num_layers

            # --- Calculate Space ---
            def base_fac(rec):
                if rec == "Expand": return max(expand_facings, min_facings)
                if rec == "Retain": return max(retain_facings, min_facings)
                return delist_facings
            sku['Base Facings'] = sku['Recommendation'].apply(base_fac)

            if 'Width' not in sku.columns:
                sku['Width'] = st.sidebar.number_input("Default SKU width", 0.1, 100.0, 5.0)
            else:
                sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(
                    st.sidebar.number_input("Default SKU width (fallback)", 0.1, 100.0, 5.0)
                )

            sku['Suggested Facings'] = sku['Base Facings']
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            df_filtered = sku[sku['Recommendation'] != "Delist"] if hide_delist else sku.copy()
            total_space_used = df_filtered['Space Needed'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

            # --- Display SKU Table ---
            st.subheader("SKU Recommendations")
            def highlight_rec(v):
                if v=="Expand": return "background-color:#d4f7d4"
                if v=="Retain": return "background-color:#fff4cc"
                if v=="Delist": return "background-color:#ffd6d6"
                return ""
            st.dataframe(
                sku[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style
                .applymap(highlight_rec, subset=['Recommendation']),
                use_container_width=True
            )

            # --- Shelf Usage Overview ---
            st.subheader("Shelf usage")
            st.progress(min(space_pct/100, 1.0))
            st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")

            # --- Allocation & Auto-adjusted Facings ---
            df_alloc = df_filtered.sort_values("Score", ascending=False).copy()
            df_alloc['Adjusted Facings'] = df_alloc['Suggested Facings']
            df_alloc['Space Needed Adjusted'] = df_alloc['Width'] * df_alloc['Adjusted Facings']

            remaining_space = total_shelf_space
            for i, row in df_alloc.iterrows():
                space_needed = row['Space Needed Adjusted']
                if space_needed <= remaining_space:
                    remaining_space -= space_needed
                    df_alloc.at[i, 'Fits Shelf'] = True
                else:
                    max_facings = max(1, int(remaining_space / row['Width']))
                    if max_facings > 0:
                        df_alloc.at[i, 'Adjusted Facings'] = max_facings
                        df_alloc.at[i, 'Space Needed Adjusted'] = max_facings * row['Width']
                        df_alloc.at[i, 'Fits Shelf'] = True
                        remaining_space -= df_alloc.at[i, 'Space Needed Adjusted']
                    else:
                        df_alloc.at[i, 'Fits Shelf'] = False
                        df_alloc.at[i, 'Adjusted Facings'] = 0
                        df_alloc.at[i, 'Space Needed Adjusted'] = 0

            skus_that_fit = df_alloc[df_alloc['Fits Shelf']]
            skus_overflow = df_alloc[~df_alloc['Fits Shelf']]

            st.write(f"✅ SKUs that fit: {len(skus_that_fit)} / {len(df_alloc)}")
            st.write(f"❌ SKUs that cannot fit: {len(skus_overflow)} (consider delisting)")

            if not skus_overflow.empty:
                st.subheader("SKUs that cannot fit in shelf")
                st.dataframe(
                    skus_overflow[['SKU','Score','Recommendation','Adjusted Facings','Space Needed Adjusted']],
                    use_container_width=True
                )

            # --- Chart ---
            st.subheader("Top SKUs by Adjusted Space Needed")
            df_chart = skus_that_fit.sort_values('Space Needed Adjusted', ascending=False).head(top_n)
            fig = px.bar(df_chart, x='Space Needed Adjusted', y='SKU', orientation='h', color='Recommendation')
            fig.update_layout(height=30*len(df_chart))
            st.plotly_chart(fig, use_container_width=True)

# ========== MODULE 2: Sales Analysis ==========
# ... Keep your existing Sales Analysis code here ...

# ========== MODULE 3: Submit Insight ==========
# ... Keep your existing Submit Insight code here ...

# ========== MODULE 4: Approve Insights ==========
# ... Keep your existing Approve Insights code here ...
