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

# ========== MODULE 1 ==========
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    sku_file = st.file_uploader("Upload SKU CSV (required: SKU, Sales, Volume, Margins, Width, Facings, Product Type, Variant, Item Size)", type=["csv"])
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        df = pd.read_csv(sku_file)
        df = normalize_colnames(df)

        required = ["SKU","Sales","Volume","Margins","Width","Facings","Product Type","Variant","Item Size"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # Convert numeric columns
            df['Sales'] = clean_sales_series(df['Sales'])
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            df['Margins'] = pd.to_numeric(df['Margins'], errors='coerce').fillna(0)
            df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(5)
            df['Facings'] = pd.to_numeric(df['Facings'], errors='coerce').fillna(1)

            # Calculate performance score
            df["Sales Contribution"] = df["Sales"] / df["Sales"].sum()
            df["Volume Contribution"] = df["Volume"] / df["Volume"].sum()
            df["Margin Contribution"] = df["Margins"] / df["Margins"].sum()
            df["Performance Score"] = 0.3*df["Sales Contribution"] + 0.3*df["Volume Contribution"] + 0.4*df["Margin Contribution"]

            # Recommendation based on quantiles
            cutoff_expand = df["Performance Score"].quantile(0.70)
            cutoff_delist = df["Performance Score"].quantile(0.30)
            df["Recommendation"] = df["Performance Score"].apply(lambda s: "Expand" if s>=cutoff_expand else ("Delist" if s<=cutoff_delist else "Retain"))

            # Suggested facings (dynamic)
            df["Suggested Facings"] = np.ceil(df["Performance Score"] * 10).astype(int).clip(lower=1)
            df["Space Needed"] = df["Width"] * df["Suggested Facings"]

            # Shelf Settings
            st.sidebar.header("Shelf Settings")
            shelf_width = st.sidebar.number_input("Shelf width per layer", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs", value=False)
            top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, max(5, len(df))), min(50, max(5, len(df))))

            total_shelf_space = shelf_width * num_layers
            total_space_used = df["Space Needed"].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space>0 else 0.0

            # Summary by Product Type + Variant
            summary = df.groupby(["Product Type","Variant","Recommendation"]).size().reset_index(name="Count")

            # Filter by product type
            product_types = summary["Product Type"].unique().tolist()
            selected_types = st.multiselect("Filter by Product Type:", product_types, default=product_types)
            df = df[df["Product Type"].isin(selected_types)]
            summary = summary[summary["Product Type"].isin(selected_types)]

            # Display summary + chart
            st.subheader("Summary of Recommendations by Product Type & Variant")
            st.dataframe(summary, use_container_width=True)
            fig_summary = px.bar(summary, x="Count", y="Variant", color="Recommendation", barmode="group", facet_col="Product Type")
            fig_summary.update_layout(height=500, xaxis_title="Count")
            st.plotly_chart(fig_summary, use_container_width=True)

            # Shelf usage + SKUs that cannot fit
            c1,c2 = st.columns(2)
            with c1:
                st.subheader("Shelf Usage")
                st.progress(min(space_pct/100,1.0))
                st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} ({space_pct:.1f}%)")
            with c2:
                st.subheader("SKUs That Cannot Fit")
                df_sorted = df.sort_values("Performance Score", ascending=False)
                df_sorted["Cumulative Space"] = df_sorted["Space Needed"].cumsum()
                cannot_fit = df_sorted[df_sorted["Cumulative Space"]>total_shelf_space]
                if cannot_fit.empty:
                    st.success("✅ All SKUs fit within available shelf space!")
                else:
                    st.dataframe(cannot_fit[["SKU","Product Type","Variant","Space Needed","Suggested Facings","Recommendation"]])

            # Recommendation Table
            st.subheader("SKU Recommendations")
            if hide_delist:
                display_df = df[df["Recommendation"]!="Delist"]
            else:
                display_df = df
            st.dataframe(display_df[["SKU","Product Type","Variant","Sales","Volume","Margins","Performance Score","Recommendation","Suggested Facings","Space Needed"]], use_container_width=True)

            # Downloadable Excel
            output = "sku_recommendations.xlsx"
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                display_df.to_excel(writer, sheet_name="SKU Recommendations", index=False)
                summary.to_excel(writer, sheet_name="Summary", index=False)
            with open(output, "rb") as f:
                st.download_button("📥 Download Recommendations (Excel)", f, file_name=output)

# ========== MODULE 2 ==========
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
                        if pd.isna(r['Baseline']):
                            store_mean = sales[sales['Store Code']==r['Store Code']]['Sales'].mean()
                            return "LIFT" if r['Sales'] >= store_mean else "DROP"
                        return "LIFT" if r['Sales'] >= r['Baseline'] else "DROP"
                    if pd.isna(r['ChangePct']): return "NORMAL"
                    if r['ChangePct'] >= pct_thr_up: return "LIFT"
                    if r['ChangePct'] <= -pct_thr_down: return "DROP"
                    return "NORMAL"

                merged['Signal'] = merged.apply(classify_row, axis=1)
                merged['Qualitative Note'] = merged.apply(lambda r:
                    f"User insight: {r['Matched Insight']}" if r['Matched Insight'] else (
                        f"Sales +{r['ChangePct']:.0f}% vs baseline" if r['Signal']=="LIFT"
                        else f"Sales -{abs(r['ChangePct']):.0f}% vs baseline" if r['Signal']=="DROP" else "Normal"
                    ), axis=1)

                lifts = (merged['Signal']=="LIFT").sum()
                drops = (merged['Signal']=="DROP").sum()
                with_insight = (merged['Matched Insight'] != "").sum()

                c1,c2,c3 = st.columns(3)
                c1.metric("Lift days", lifts)
                c2.metric("Drop days", drops)
                c3.metric("With insights", with_insight)

                def style_sig(v):
                    if v == "LIFT": return "background-color: #d4f7d4"
                    if v == "DROP": return "background-color: #ffd6d6"
                    return ""
                st.dataframe(merged[['Store Code','Date','Sales','Baseline','ChangePct','Signal','Qualitative Note']].style
                             .applymap(style_sig, subset=['Signal'])
                             .applymap(lambda x: "font-style: italic;" if isinstance(x,str) and x.startswith("User insight") else "", subset=['Qualitative Note']),
                             use_container_width=True)

# ========== MODULE 3 ==========
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

# ========== MODULE 4 ==========
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
                insights_df.loc[i, 'Status'] = "Approved"
                write_insights_df(insights_df)
                safe_rerun()
            if col2.button(f"Reject {i}"):
                insights_df.loc[i, 'Status'] = "Rejected"
                write_insights_df(insights_df)
                safe_rerun()
