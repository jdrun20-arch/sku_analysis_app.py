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

# ---------- UI ----------
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insight",
    "Approve Insights"
])

# ================= MODULE 1 =================
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    sku_file = st.file_uploader("Upload SKU CSV (required: SKU, Sales, Volume, Margin)", type=["csv"])
    if sku_file is None:
        st.info("Upload a SKU CSV to run this module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        required = ["SKU","Sales","Volume","Margin"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)

            # --- Normalize and Score ---
            def norm(series):
                mx = series.replace(0, pd.NA).max()
                return series / mx if mx and mx != 0 else 0

            sku['Sales_Norm'] = norm(sku['Sales'])
            sku['Volume_Norm'] = norm(sku['Volume'])
            sku['Margin_Norm'] = norm(sku['Margin'])
            sku['Score'] = (sku['Sales_Norm']*0.3) + (sku['Volume_Norm']*0.3) + (sku['Margin_Norm']*0.4)
            sku['Rank'] = sku['Score'].rank(method='min', ascending=False).astype(int)

            # --- Retain % Control ---
            retain_pct = st.sidebar.slider("Target Retain %", 10, 100, 80, 5)
            top_n_to_keep = int(len(sku) * retain_pct / 100)
            sku_sorted = sku.sort_values("Score", ascending=False)
            cutoff_score = sku_sorted.iloc[top_n_to_keep-1]['Score'] if len(sku_sorted)>=top_n_to_keep else 0
            sku['Recommendation'] = sku['Score'].apply(lambda s: "Delist" if s < cutoff_score else "Retain")
            # Top 20% of Retain are Expand
            retain_only = sku[sku['Recommendation']=="Retain"]
            expand_cutoff = retain_only['Score'].quantile(0.80) if not retain_only.empty else 1
            sku.loc[(sku['Recommendation']=="Retain") & (sku['Score']>=expand_cutoff), 'Recommendation'] = "Expand"

            # --- Facings Suggestion based on Score ---
            min_facings = st.sidebar.number_input("Minimum facings per SKU", 1, 10, 1)
            max_facings = st.sidebar.number_input("Maximum facings per SKU", 1, 20, 5)
            sku['Suggested Facings'] = ((sku['Score']/sku['Score'].max())*(max_facings-min_facings)+min_facings).round(0).astype(int)

            # --- Shelf Settings ---
            shelf_width = st.sidebar.number_input("Shelf width per layer", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            total_shelf_space = shelf_width * num_layers
            sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(5.0)
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            total_space_used = sku['Space Needed'].sum()
            space_pct = (total_space_used / total_shelf_space)*100 if total_shelf_space>0 else 0.0

            st.subheader("SKU Distribution by Product Type & Variant")
            if "Product Type" in sku.columns and "Variant" in sku.columns:
                chart_df = sku.groupby(["Product Type","Variant"])['Recommendation'].value_counts().unstack(fill_value=0).reset_index()
                fig = px.bar(chart_df, x="Product Type", y=["Expand","Retain","Delist"],
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             barmode="stack", title="SKU Recommendation Summary")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Shelf Usage")
            st.progress(min(space_pct/100, 1.0))
            st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")

            st.subheader("SKU Recommendations")
            def highlight_rec(v):
                return ("background-color:#d4f7d4" if v=="Expand"
                        else "background-color:#fff4cc" if v=="Retain"
                        else "background-color:#ffd6d6" if v=="Delist" else "")
            st.dataframe(
                sku[['SKU','Score','Rank','Recommendation','Suggested Facings','Space Needed']]
                .style.applymap(highlight_rec, subset=['Recommendation']),
                use_container_width=True
            )

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
                sel['Baseline'] = np.nan
                for store, group in sel.groupby('Store Code'):
                    idxs = group.index.tolist()
                    if len(group) == 1:
                        sel.loc[idxs, 'Baseline'] = np.nan
                    else:
                        for i in idxs:
                            sel.loc[i, 'Baseline'] = group.loc[group.index != i, 'Sales'].mean()
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
                        return "LIFT" if r['Sales'] >= (r['Baseline'] if not pd.isna(r['Baseline']) else r['Sales']) else "DROP"
                    if pd.isna(r['ChangePct']): return "NORMAL"
                    if r['ChangePct'] >= pct_thr_up: return "LIFT"
                    if r['ChangePct'] <= -pct_thr_down: return "DROP"
                    return "NORMAL"

                merged['Signal'] = merged.apply(classify_row, axis=1)

                c1,c2,c3 = st.columns(3)
                c1.metric("Lift days", (merged['Signal']=="LIFT").sum())
                c2.metric("Drop days", (merged['Signal']=="DROP").sum())
                c3.metric("With insights", (merged['Matched Insight'] != "").sum())

                st.dataframe(
                    merged[['Store Code','Date','Sales','Baseline','ChangePct','Signal','Matched Insight']],
                    use_container_width=True
                )

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
    pending = insights_df[insights_df['Status'].str.lower()=="pending"]
    if pending.empty:
        st.info("No pending insights.")
    else:
        for i, row in pending.iterrows():
            st.write(f"📅 {row['Date']} | 🏪 {row['Store Code']} | 📝 {row['Insight']}")
            col1,col2 = st.columns(2)
            if col1.button(f"Approve {i}"):
                insights_df.loc[i,'Status']="Approved"
                write_insights_df(insights_df)
                safe_rerun()
            if col2.button(f"Reject {i}"):
                insights_df.loc[i,'Status']="Rejected"
                write_insights_df(insights_df)
                safe_rerun()
