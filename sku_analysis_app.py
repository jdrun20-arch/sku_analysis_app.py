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

# ========== MODULE 1 ==========
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    sku_file = st.file_uploader(
        "Upload SKU CSV (required: SKU, Store Code, Sales, Volume, Margin, Width, Facings, Product Type, Variant, Item Size)", 
        type=["csv"]
    )

    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        required = ["SKU","Store Code","Sales","Volume","Margin","Width","Facings","Product Type","Variant","Item Size"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # --- CLEANING & SCORING ---
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)
            sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(1)
            sku['Facings'] = pd.to_numeric(sku['Facings'], errors='coerce').fillna(1)

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

            cutoff_expand = sku['Score'].quantile(0.70)
            cutoff_delist = sku['Score'].quantile(0.30)
            sku['Recommendation'] = sku['Score'].apply(lambda s: "Expand" if s>=cutoff_expand else ("Delist" if s<=cutoff_delist else "Retain"))
            sku['Justification'] = sku['Recommendation'].map({
                'Expand': "High performance — consider expansion.",
                'Delist': "Low performance — candidate for phase-out.",
                'Retain': "Balanced — maintain."
            })

            # --- SHELF SETTINGS ---
            st.sidebar.header("Shelf settings")
            expand_facings = st.sidebar.slider("Facings for Expand", 1, 10, 3)
            retain_facings = st.sidebar.slider("Facings for Retain", 1, 10, 2)
            delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 1)
            min_facings = st.sidebar.number_input("Minimum facings", 1, 10, 2)
            shelf_width = st.sidebar.number_input("Shelf width per layer", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs", value=False)

            total_shelf_space = shelf_width * num_layers

            def base_fac(rec):
                if rec == "Expand": return max(expand_facings, min_facings)
                if rec == "Retain": return max(retain_facings, min_facings)
                return delist_facings

            sku['Base Facings'] = sku['Recommendation'].apply(base_fac)
            sku['Suggested Facings'] = sku['Base Facings']
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            # --- FILTER BY PRODUCT TYPE ---
            product_types = sku['Product Type'].dropna().unique().tolist()
            selected_types = st.multiselect(
                "Filter by Product Type:",
                options=product_types,
                default=product_types
            )

            df_filtered = sku[sku['Product Type'].isin(selected_types)].copy()
            if hide_delist:
                df_filtered = df_filtered[df_filtered['Recommendation'] != "Delist"]

            total_space_used = df_filtered['Space Needed'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space>0 else 0.0

            # --- LAYOUT FOR SHELF USAGE + OVERFLOW ---
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Shelf Usage")
                st.progress(min(space_pct/100,1.0))
                st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")
            with c2:
                st.subheader("SKUs that cannot fit in shelf")
                df_sorted = df_filtered.sort_values('Score', ascending=False)
                df_sorted['CumulativeSpace'] = df_sorted['Space Needed'].cumsum()
                df_sorted['Fits Shelf'] = df_sorted['CumulativeSpace'] <= total_shelf_space
                skus_overflow = df_sorted[~df_sorted['Fits Shelf']]
                if skus_overflow.empty:
                    st.success("✅ All SKUs fit within available shelf space!")
                else:
                    st.error(f"🚨 {len(skus_overflow)} SKUs exceed available space.")
                    st.dataframe(skus_overflow[['SKU','Product Type','Variant','Space Needed']], use_container_width=True)

            # --- SKU RECOMMENDATION TABLE ---
            st.subheader("SKU Recommendations")
            def highlight_rec(v):
                if v=="Expand": return "background-color:#d4f7d4"
                if v=="Retain": return "background-color:#fff4cc"
                if v=="Delist": return "background-color:#ffd6d6"
                return ""
            st.dataframe(df_filtered[['SKU','Product Type','Variant','Item Size','Score','Rank','Recommendation','Suggested Facings','Space Needed']].style
                         .applymap(highlight_rec, subset=['Recommendation']),
                         use_container_width=True)

            # --- SUMMARY BY PRODUCT TYPE & VARIANT ---
            st.subheader("Summary by Product Type & Variant")
            summary = (df_filtered.groupby(['Product Type','Variant','Recommendation'])
                       .size().reset_index(name="Count"))

            st.dataframe(summary, use_container_width=True)

            summary['Category-Variant'] = summary['Product Type'] + " - " + summary['Variant']
            fig_summary = px.bar(
                summary.sort_values("Count", ascending=True),
                x="Count",
                y="Category-Variant",
                color="Recommendation",
                orientation="h",
                text="Count",
                category_orders={"Recommendation": ["Expand", "Retain", "Delist"]}
            )
            fig_summary.update_traces(textposition="outside")
            fig_summary.update_layout(height=500, xaxis_title="Number of SKUs", bargap=0.3)
            st.plotly_chart(fig_summary, use_container_width=True)

            # --- DOWNLOADABLE EXCEL ---
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_filtered.to_excel(writer, index=False, sheet_name="SKU Recommendations")
                summary.to_excel(writer, index=False, sheet_name="Summary")
            st.download_button(
                label="📥 Download SKU Recommendations (Excel)",
                data=output.getvalue(),
                file_name="sku_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
