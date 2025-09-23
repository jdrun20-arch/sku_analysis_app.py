import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
import io

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
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Date'
            break
    for cand in ('store code','store_code','storecode','store'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Store Code'
            break
    for cand in ('sales','sale','amount','revenue'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Sales'
            break
    for cand in ('sku','sku_code','product','item'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'SKU'
            break
    for cand in ('width','item width','item width (in)','width_in','item_width'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Width'
            break
    return df.rename(columns=mapping)

# Recommended export column order
download_cols = [
    "SKU",
    "Product Type",
    "Variant",
    "Item Size",
    "Sales",
    "Volume",
    "Margin",
    "Score",
    "Rank",
    "Recommendation",
    "Justification",
    "Suggested Facings",
    "Width",
    "Space Needed"
]

# ---------- UI ----------
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insight",
    "Approve Insights"
])

import streamlit as st
import pandas as pd
import numpy as np
import io

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

    # --- Summary by Product Type & Variant ---
    st.subheader("SKU Summary by Product Type & Variant")
    summary = df.groupby(['Product Type', 'Variant'])['SKU'].count().reset_index()
    summary.rename(columns={'SKU': 'Count'}, inplace=True)

    selected_product_types = st.multiselect("Filter by Product Type", options=summary['Product Type'].unique(), default=list(summary['Product Type'].unique()))
    filtered_summary = summary[summary['Product Type'].isin(selected_product_types)]

    st.bar_chart(filtered_summary.set_index(['Product Type', 'Variant']))

    # --- Downloadable Recommendations ---
    st.subheader("Download SKU Recommendations")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="SKU Recommendations")
    st.download_button(label="📥 Download Excel", data=output.getvalue(), file_name="sku_recommendations.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file with SKU, Store Code, Sales, Volume, Margins, Width, Facings, Product Type, Variant, Item Size.")

# ================= MODULE 2 =================
elif module == "Sales Analysis":
    st.header("📈 Sales Analysis & Insight Matching")
    sales_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    if sales_file is None:
        st.info("Upload a sales CSV.")
    else:
        sales_raw = pd.read_csv(sales_file)
        sales = normalize_colnames(sales_raw)
        if 'Date' not in sales.columns or 'Sales' not in sales.columns:
            st.error("Missing Date or Sales columns.")
        else:
            sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
            sales = sales.dropna(subset=['Date']).copy()
            sales['Sales'] = clean_sales_series(sales['Sales'])
            if 'Store Code' not in sales.columns:
                sales['Store Code'] = "ALL"

            store_list = sales['Store Code'].unique().tolist()
            selected = st.multiselect("Select store(s)", store_list, default=store_list)
            min_date, max_date = sales['Date'].min().date(), sales['Date'].max().date()
            dr = st.date_input("Date range", [min_date, max_date])
            start_d, end_d = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

            sel = sales[(sales['Store Code'].isin(selected)) & (sales['Date'].between(start_d, end_d))].copy()
            if sel.empty:
                st.info("No data for selection.")
            else:
                # Compute baseline (mean of other days within selection per store)
                sel['Baseline'] = np.nan
                for store, group in sel.groupby('Store Code'):
                    if len(group) == 1:
                        sel.loc[group.index, 'Baseline'] = np.nan
                    else:
                        # baseline for each row is mean of other rows for that store
                        for idx in group.index:
                            sel.loc[idx, 'Baseline'] = group.loc[group.index != idx, 'Sales'].mean()
                sel['ChangePct'] = (sel['Sales'] - sel['Baseline']) / sel['Baseline'] * 100

                pct_thr_up = st.sidebar.slider("Lift threshold (%)", 10, 500, 50, 5)
                pct_thr_down = st.sidebar.slider("Drop threshold (%)", 5, 200, 30, 5)

                insights_df = ensure_insights_df()
                insights_approved = insights_df[insights_df['Status'].str.lower() == 'approved'].copy()
                sel['Date_key'] = sel['Date'].dt.strftime("%Y-%m-%d")
                merged = pd.merge(sel, insights_approved, how='left', left_on=['Store Code','Date_key'], right_on=['Store Code','Date'])
                merged['Matched Insight'] = merged['Insight'].fillna("")

                def classify_row(r):
                    if r['Matched Insight']:
                        if pd.isna(r['Baseline']):
                            store_mean = sales[sales['Store Code'] == r['Store Code']]['Sales'].mean()
                            return "LIFT" if r['Sales'] >= store_mean else "DROP"
                        return "LIFT" if r['Sales'] >= r['Baseline'] else "DROP"
                    if pd.isna(r.get('ChangePct', np.nan)):
                        return "NORMAL"
                    if r['ChangePct'] >= pct_thr_up:
                        return "LIFT"
                    if r['ChangePct'] <= -pct_thr_down:
                        return "DROP"
                    return "NORMAL"

                merged['Signal'] = merged.apply(classify_row, axis=1)
                merged['Qualitative Note'] = merged.apply(
                    lambda r: f"User insight: {r['Matched Insight']}" if r['Matched Insight'] else (
                        f"Sales +{r['ChangePct']:.0f}% vs baseline" if r['Signal'] == "LIFT" else (
                            f"Sales -{abs(r['ChangePct']):.0f}% vs baseline" if r['Signal'] == "DROP" else "Normal"
                        )
                    ),
                    axis=1
                )

                lifts = (merged['Signal'] == "LIFT").sum()
                drops = (merged['Signal'] == "DROP").sum()
                with_insight = (merged['Matched Insight'] != "").sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Lift days", lifts)
                c2.metric("Drop days", drops)
                c3.metric("With insights", with_insight)

                display_cols = [c for c in ['Store Code', 'Date', 'Sales', 'Baseline', 'ChangePct', 'Signal', 'Qualitative Note'] if c in merged.columns]
                def style_sig(v):
                    if v == "LIFT": return "background-color: #d4f7d4"
                    if v == "DROP": return "background-color: #ffd6d6"
                    return ""
                styled = merged[display_cols].style
                if 'Signal' in display_cols:
                    styled = styled.applymap(style_sig, subset=['Signal'])
                if 'Qualitative Note' in display_cols:
                    styled = styled.applymap(lambda x: "font-style: italic;" if isinstance(x, str) and x.startswith("User insight") else "", subset=['Qualitative Note'])
                st.dataframe(styled, use_container_width=True)

# ================= MODULE 3 =================
elif module == "Submit Insight":
    st.header("📝 Submit an Insight")
    insights_df = ensure_insights_df()
    with st.form("insight_form", clear_on_submit=True):
        date = st.date_input("Date", datetime.today())
        store_code = st.text_input("Store Code")
        insight = st.text_area("Insight")
        submitted = st.form_submit_button("Submit Insight")
        if submitted:
            new_row = pd.DataFrame([{
                "Date": date.strftime("%Y-%m-%d"),
                "Store Code": store_code,
                "Insight": insight,
                "Status": "Pending"
            }])
            insights_df = pd.concat([insights_df, new_row], ignore_index=True)
            write_insights_df(insights_df)
            st.success("Insight submitted!")
            safe_rerun()

# ================= MODULE 4 =================
elif module == "Approve Insights":
    st.header("✅ Approve or Reject Insights")
    insights_df = ensure_insights_df()
    pending = insights_df[insights_df['Status'].str.lower() == "pending"]
    if pending.empty:
        st.info("No pending insights.")
    else:
        for i, row in pending.iterrows():
            st.write(f"📅 {row['Date']} | 🏪 {row['Store Code']} | 📝 {row['Insight']}")
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {i}"):
                insights_df.loc[i, 'Status'] = "Approved"
                write_insights_df(insights_df)
                safe_rerun()
            if col2.button(f"Reject {i}"):
                insights_df.loc[i, 'Status'] = "Rejected"
                write_insights_df(insights_df)
                safe_rerun()
