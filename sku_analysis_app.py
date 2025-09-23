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

import streamlit as st
import pandas as pd
import numpy as np
import io

st.title("📊 SKU Performance & Shelf Space Optimization")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Settings")
target_retain_pct = st.sidebar.slider("Target to Retain %", 0, 100, 80)
total_shelf_space = st.sidebar.number_input("Total Shelf Space (in cm)", min_value=50, value=200, step=10)
number_of_layers = st.sidebar.number_input("Number of Layers", min_value=1, value=4, step=1)

uploaded_file = st.file_uploader("Upload your SKU CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    required_columns = ["SKU","Store Code","Sales","Volume","Margins","Width","Facings","Product Type","Variant","Item Size"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
    else:
        # --- Calculate Weighted Score (Sales 30%, Volume 30%, Margin 40%) ---
        df["Sales %"] = df["Sales"] / df["Sales"].sum()
        df["Volume %"] = df["Volume"] / df["Volume"].sum()
        df["Margin %"] = df["Margins"] / df["Margins"].sum()
        df["Weighted Score"] = (0.3 * df["Sales %"]) + (0.3 * df["Volume %"]) + (0.4 * df["Margin %"])

        # Normalize to suggest facings
        total_facings_available = (total_shelf_space * number_of_layers) / df["Width"].mean()
        df["Suggested Facings"] = np.floor(df["Weighted Score"] / df["Weighted Score"].sum() * total_facings_available).astype(int)

        # --- Determine SKU Status (Retain, Expand, Delist) ---
        threshold = np.percentile(df["Weighted Score"], 100 - target_retain_pct)
        df["Recommendation"] = np.where(df["Weighted Score"] >= threshold, "Retain", "Delist")
        df.loc[df["Suggested Facings"] > df["Facings"], "Recommendation"] = "Expand"

        # --- Shelf Usage ---
        df["Space Used"] = df["Facings"] * df["Width"]
        total_space_used = df["Space Used"].sum()
        shelf_capacity = total_shelf_space * number_of_layers
        shelf_usage_pct = (total_space_used / shelf_capacity) * 100

        st.subheader("📊 Shelf Space Usage & SKU Fit")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shelf Usage (%)", f"{shelf_usage_pct:.2f}%")
        with col2:
            if total_space_used <= shelf_capacity:
                st.success("✅ All SKUs fit the shelf space.")
            else:
                skus_overflow = df[df["Space Used"].cumsum() > shelf_capacity]
                if not skus_overflow.empty:
                    st.error(f"⚠️ {len(skus_overflow)} SKUs cannot fit in the shelf space.")
                    st.dataframe(skus_overflow[["SKU","Product Type","Variant","Facings","Width","Space Used"]])

        # --- SKU Distribution by Product Type & Variant ---
        st.subheader("📊 SKU Distribution by Product Type & Variant")
        dist = df.groupby(["Product Type","Variant"]).size().reset_index(name="Count")
        st.bar_chart(dist.set_index(["Product Type","Variant"]))

        # --- SKU Summary Table ---
        st.subheader("📋 SKU Recommendations")
        st.dataframe(df[["SKU","Product Type","Variant","Sales","Volume","Margins","Facings","Suggested Facings","Recommendation"]])

        # --- Summary Counts ---
        st.subheader("📊 Summary by Product Type & Variant")
        summary = df.groupby(["Product Type","Variant","Recommendation"]).size().reset_index(name="Count")
        st.dataframe(summary)

        # --- Download Recommendations ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="SKU Recommendations", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
        st.download_button(
            label="📥 Download Recommendations (Excel)",
            data=output.getvalue(),
            file_name="SKU_Recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("📂 Please upload a CSV file to start analysis.")

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
