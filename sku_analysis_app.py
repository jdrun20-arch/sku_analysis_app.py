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

# ========== MODULE 1: SKU PERFORMANCE & SHELF SPACE ==========
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    
    sku_file = st.file_uploader(
        "Upload SKU CSV (required: SKU, Sales, Volume, Margin, Width, Facings). Optional: Product Type, Variant, Item Size",
        type=["csv"]
    )
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        # Ensure required columns exist
        required = ["SKU","Sales","Volume","Margin"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # Add optional columns if missing
            for c in ['Width','Facings','Product Type','Variant','Item Size']:
                if c not in sku.columns:
                    sku[c] = ""

            # Clean numeric columns
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)
            sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(5.0)
            sku['Facings'] = pd.to_numeric(sku['Facings'], errors='coerce').fillna(1)

            # ---------------- SCORE & RANK ----------------
            def norm(series):
                mx = series.replace(0, pd.NA).max()
                if pd.isna(mx) or mx==0:
                    return pd.Series(0, index=series.index)
                return series / mx

            sku['Sales_Norm'] = norm(sku['Sales'])
            sku['Volume_Norm'] = norm(sku['Volume'])
            sku['Margin_Norm'] = norm(sku['Margin'])
            sku['Score'] = (sku['Sales_Norm']*0.3) + (sku['Volume_Norm']*0.3) + (sku['Margin_Norm']*0.4)
            sku['Rank'] = sku['Score'].rank(method='min', ascending=False).astype(int)

            # ---------------- RECOMMENDATIONS ----------------
            cutoff_expand = sku['Score'].quantile(0.70)
            cutoff_delist = sku['Score'].quantile(0.30)
            sku['Recommendation'] = sku['Score'].apply(lambda s: "Expand" if s>=cutoff_expand else ("Delist" if s<=cutoff_delist else "Retain"))
            sku['Justification'] = sku['Recommendation'].map({
                'Expand': "High performance — consider expansion.",
                'Delist': "Low performance — candidate for phase-out.",
                'Retain': "Balanced — maintain."
            })

            # ---------------- SIDEBAR SETTINGS ----------------
            st.sidebar.header("Shelf & Variant Settings")
            expand_facings = st.sidebar.slider("Facings for Expand", 1, 10, 3)
            retain_facings = st.sidebar.slider("Facings for Retain", 1, 10, 2)
            delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 1)
            min_facings = st.sidebar.number_input("Minimum facings", 1, 10, 2)
            shelf_width = st.sidebar.number_input("Shelf width per layer", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs", value=False)
            top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, max(5, len(sku))), min(50, max(5, len(sku))))
            max_expand_per_type = st.sidebar.slider("Max Expand SKUs per Product Type", 1, 10, 2)

            total_shelf_space = shelf_width * num_layers

            # ---------------- BASE FACINGS ----------------
            def base_fac(rec):
                if rec=="Expand": return max(expand_facings, min_facings)
                if rec=="Retain": return max(retain_facings, min_facings)
                return delist_facings

            sku['Base Facings'] = sku['Recommendation'].apply(base_fac)
            sku['Suggested Facings'] = sku['Base Facings']
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            # ---------------- VARIANT-AWARE ADJUSTMENT ----------------
            def variant_adjustment(df):
                df = df.sort_values('Score', ascending=False).copy()
                expand_count = 0
                for idx, row in df.iterrows():
                    if row['Recommendation'] == "Expand":
                        if expand_count < max_expand_per_type:
                            expand_count += 1
                        else:
                            df.at[idx, 'Recommendation'] = "Retain"
                            df.at[idx, 'Justification'] = "Top variants limit reached — maintain."
                return df

            sku = sku.groupby('Product Type', group_keys=False).apply(variant_adjustment)

            # ---------------- SHELF ALLOCATION ----------------
            df_filtered = sku[sku['Recommendation'] != "Delist"] if hide_delist else sku.copy()
            df_alloc = df_filtered.sort_values('Score', ascending=False).copy()
            df_alloc['CumulativeSpace'] = df_alloc['Space Needed'].cumsum()
            df_alloc['Fits Shelf'] = df_alloc['CumulativeSpace'] <= total_shelf_space

            # ---------------- DISPLAY ----------------
            st.subheader("Shelf usage & SKUs that cannot fit")
            c1,c2 = st.columns([2,2])

            total_space_used = df_alloc[df_alloc['Fits Shelf']]['Space Needed'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space>0 else 0.0
            with c1:
                st.write("**Shelf Usage**")
                st.progress(min(space_pct/100,1.0))
                st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")
                st.write(f"SKUs allocated: {df_alloc['Fits Shelf'].sum()} / {len(df_alloc)}")

            skus_overflow = df_alloc[~df_alloc['Fits Shelf']]
            with c2:
                if skus_overflow.empty:
                    st.success("✅ All SKUs fit in the shelf.")
                else:
                    st.write("**SKUs that cannot fit**")
                    st.dataframe(skus_overflow[['SKU','Product Type','Variant','Item Size','Space Needed','Recommendation']], use_container_width=True)

            # ---------------- TOP SKUs BY SPACE ----------------
            st.subheader("Top SKUs by Space Needed")
            df_chart = df_alloc[df_alloc['Fits Shelf']].sort_values('Space Needed', ascending=False).head(top_n)
            import plotly.express as px
            fig = px.bar(df_chart, x='Space Needed', y='SKU', orientation='h', color='Recommendation')
            fig.update_layout(height=30*len(df_chart))
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- DOWNLOAD EXCEL ----------------
            st.subheader("Download SKU Recommendations")
            download_cols = ['SKU','Product Type','Variant','Item Size','Sales','Volume','Margin','Score','Rank','Recommendation','Justification','Suggested Facings','Width','Space Needed']
            excel_file = "sku_recommendations.xlsx"
            df_alloc[download_cols].to_excel(excel_file, index=False)
            with open(excel_file,"rb") as f:
                st.download_button("Download Excel", f, file_name=excel_file)
# --- SUMMARY BY PRODUCT TYPE & VARIANT ---
summary = (
    sku.groupby(['Product Type','Variant','Recommendation'])
    .size()
    .reset_index(name='Count')
    .sort_values(['Product Type','Variant','Recommendation'])
)

st.subheader("📋 SKU Summary by Product Type & Variant")
col_summary_table, col_summary_chart = st.columns([1,2])

with col_summary_table:
    st.dataframe(summary, use_container_width=True)

with col_summary_chart:
    import plotly.express as px
    fig_summary = px.bar(
        summary,
        x="Variant",
        y="Count",
        color="Recommendation",
        barmode="group",
        facet_col="Product Type",
        text="Count"
    )
    fig_summary.update_traces(textposition="outside")
    fig_summary.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_summary, use_container_width=True)

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

                # Only display columns that exist
                display_cols = [c for c in ['Store Code','Date','Sales','Baseline','ChangePct','Signal','Qualitative Note'] if c in merged.columns]
                def style_sig(v):
                    if v=="LIFT": return "background-color:#d4f7d4"
                    if v=="DROP": return "background-color:#ffd6d6"
                    return ""
                st.dataframe(
                    merged[display_cols].style
                        .applymap(style_sig, subset=['Signal'] if 'Signal' in display_cols else None)
                        .applymap(lambda x: "font-style: italic;" if isinstance(x,str) and x.startswith("User insight") else "", subset=['Qualitative Note'] if 'Qualitative Note' in display_cols else None),
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
