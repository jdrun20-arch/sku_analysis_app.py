# Full app: Modules 1-4 (robust Excel fallback, 30/30/40 weighting, contribution-based facings)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
import importlib.util
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
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime("%Y-%m-%d")
    except Exception:
        df['Date'] = df['Date'].astype(str)
    return df

def write_insights_df(df):
    df.to_csv(INSIGHTS_FILE, index=False)

def clean_sales_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    neg_mask = s2.str.match(r'^\(.*\)$')
    s2 = s2.str.replace(r'[\(\)]', '', regex=True)
    s2 = s2.str.replace(r'[^\d\.\-]', '', regex=True)
    out = pd.to_numeric(s2, errors='coerce')
    if neg_mask.any():
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
    for cand in ('margin','margins','profit'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Margins'; break
    for cand in ('volume','units','unit_sold'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Volume'; break
    for cand in ('sku','sku_code','product','item','item_code'):
        if cand in lowermap: mapping[lowermap[cand]] = 'SKU'; break
    for cand in ('width','item width','width_in','width_cm','size_width'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Width'; break
    for cand in ('facings','facing','num_facings'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Facings'; break
    for cand in ('product type','category','type'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Product Type'; break
    for cand in ('variant','flavor','flavour'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Variant'; break
    for cand in ('item size','size','pack_size'):
        if cand in lowermap: mapping[lowermap[cand]] = 'Item Size'; break
    return df.rename(columns=mapping)

# ---------- UI: module selector in sidebar ----------
st.sidebar.title("Retail Insights")
module = st.sidebar.radio("Choose module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insight",
    "Approve Insights"
])

# ================= MODULE 1 =================
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")

    # Sidebar controls for Module 1
    sku_file = st.sidebar.file_uploader("Upload SKU CSV (required: SKU, Sales, Volume, Margins)", type=["csv"])
    st.sidebar.markdown("**Shelf settings**")
    shelf_width = st.sidebar.number_input("Shelf width per layer (units)", 1.0, 10000.0, 100.0)
    num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
    expand_min_facings = st.sidebar.slider("Minimum facings for Expand", 1, 10, 3)
    retain_min_facings = st.sidebar.slider("Minimum facings for Retain", 1, 10, 2)
    delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 0)
    global_min_facings = st.sidebar.number_input("Global minimum facings", 1, 10, 1)
    max_expand_per_type = st.sidebar.slider("Max Expand SKUs per Product Type", 1, 10, 2)
    hide_delist = st.sidebar.checkbox("Hide Delist SKUs in tables", value=False)
    top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, 50), 10)

    total_shelf_space = float(shelf_width) * int(num_layers)

    if sku_file is None:
        st.info("Upload a SKU CSV to run Module 1.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        # Ensure essential columns; provide safe defaults for optional fields
        essential = ["SKU", "Sales", "Volume", "Margins"]
        missing_ess = [c for c in essential if c not in sku.columns]
        if missing_ess:
            st.error(f"Missing essential columns: {missing_ess}. Please include at least SKU, Sales, Volume, Margins.")
        else:
            # Optional defaults
            optional_defaults = {
                "Store Code": "ALL",
                "Width": 5.0,
                "Facings": 1,
                "Product Type": "Unknown",
                "Variant": "Default",
                "Item Size": ""
            }
            for c, default in optional_defaults.items():
                if c not in sku.columns:
                    sku[c] = default

            # Clean numeric columns
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margins'] = pd.to_numeric(sku['Margins'], errors='coerce').fillna(0)
            sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(5.0)
            sku['Facings'] = pd.to_numeric(sku['Facings'], errors='coerce').fillna(1).astype(int)

            # Normalize text
            sku['Product Type'] = sku['Product Type'].fillna("Unknown").astype(str)
            sku['Variant'] = sku['Variant'].fillna("Default").astype(str)
            sku['Item Size'] = sku['Item Size'].fillna("").astype(str)
            sku['Store Code'] = sku['Store Code'].fillna("ALL").astype(str)

            # ----- Score & recommendation -----
            # Safeguard sums
            n = len(sku)
            sales_sum = sku['Sales'].sum()
            volume_sum = sku['Volume'].sum()
            margins_sum = sku['Margins'].sum()
            if sales_sum <= 0:
                sku['Sales_Contribution'] = 1.0 / n
            else:
                sku['Sales_Contribution'] = sku['Sales'] / sales_sum
            if volume_sum <= 0:
                sku['Volume_Contribution'] = 1.0 / n
            else:
                sku['Volume_Contribution'] = sku['Volume'] / volume_sum
            if margins_sum <= 0:
                sku['Margin_Contribution'] = 1.0 / n
            else:
                sku['Margin_Contribution'] = sku['Margins'] / margins_sum

            # Performance Score with 30/30/40 weights
            sku['Score'] = (
                0.3 * sku['Sales_Contribution'] +
                0.3 * sku['Volume_Contribution'] +
                0.4 * sku['Margin_Contribution']
            )

            # Initial recommendations (quantile-based)
            cutoff_expand = sku['Score'].quantile(0.70)
            cutoff_delist = sku['Score'].quantile(0.30)
            sku['Recommendation'] = sku['Score'].apply(lambda s: "Expand" if s >= cutoff_expand else ("Delist" if s <= cutoff_delist else "Retain"))
            sku['Justification'] = sku['Recommendation'].map({
                'Expand': "High performance — consider expansion.",
                'Delist': "Low performance — candidate for phase-out.",
                'Retain': "Balanced — maintain."
            })

            # ----- Variant-aware limit: allow only top N expands per Product Type -----
            def variant_adjustment(group):
                g = group.sort_values('Score', ascending=False).copy()
                expand_count = 0
                for idx, row in g.iterrows():
                    if row['Recommendation'] == "Expand":
                        if expand_count < max_expand_per_type:
                            expand_count += 1
                        else:
                            g.at[idx, 'Recommendation'] = "Retain"
                            g.at[idx, 'Justification'] = "Top variants limit reached — maintain."
                return g

            sku = sku.groupby('Product Type', group_keys=False).apply(variant_adjustment).reset_index(drop=True)

            # ----- Contribution-based facings (on the current SKU set) -----
            # Recompute contributions on the possibly-updated dataframe
            sales_sum = sku['Sales'].sum() or 1.0
            volume_sum = sku['Volume'].sum() or 1.0
            margins_sum = sku['Margins'].sum() or 1.0
            sku['Sales_Contribution'] = sku['Sales'] / sales_sum
            sku['Volume_Contribution'] = sku['Volume'] / volume_sum
            sku['Margin_Contribution'] = sku['Margins'] / margins_sum

            sku['Weighted_Contribution'] = (
                0.3 * sku['Sales_Contribution'] +
                0.3 * sku['Volume_Contribution'] +
                0.4 * sku['Margin_Contribution']
            )

            # estimate how many facings can fit across shelf using average width
            avg_width = sku['Width'].replace(0, np.nan).dropna().mean()
            if pd.isna(avg_width) or avg_width <= 0:
                avg_width = 5.0
            estimated_total_facings = max(int(total_shelf_space / avg_width), 1)

            # proportional allocation of facings
            sku['Suggested Facings'] = (sku['Weighted_Contribution'] * estimated_total_facings).round().fillna(0).astype(int)

            # apply minimums by recommendation
            sku.loc[sku['Recommendation'] == "Expand", 'Suggested Facings'] = sku.loc[sku['Recommendation'] == "Expand", 'Suggested Facings'].clip(lower=expand_min_facings)
            sku.loc[sku['Recommendation'] == "Retain", 'Suggested Facings']  = sku.loc[sku['Recommendation'] == "Retain", 'Suggested Facings'].clip(lower=retain_min_facings)
            sku.loc[sku['Recommendation'] == "Delist", 'Suggested Facings']  = delist_facings

            # enforce global minimum
            sku['Suggested Facings'] = sku['Suggested Facings'].clip(lower=global_min_facings)

            # final space needed
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            # ----- Product Type filter (sidebar) -----
            product_types = sku['Product Type'].unique().tolist()
            selected_types = st.sidebar.multiselect("Filter Product Type (summary & table)", options=product_types, default=product_types)

            df_filtered = sku[sku['Product Type'].isin(selected_types)].copy()
            if hide_delist:
                df_filtered = df_filtered[df_filtered['Recommendation'] != "Delist"].copy()

            if df_filtered.empty:
                st.warning("No SKUs to display for the selected filters.")
                st.stop()

            # ----- Shelf allocation by width (attempt full suggested facings, allow partial) -----
            df_alloc = df_filtered.sort_values('Score', ascending=False).copy()
            df_alloc['Adjusted Facings'] = df_alloc['Suggested Facings'].astype(int)
            df_alloc['Space Needed Adjusted'] = df_alloc['Width'] * df_alloc['Adjusted Facings']
            df_alloc['Fits Shelf'] = True

            remaining_space = total_shelf_space
            for i, row in df_alloc.iterrows():
                width = row['Width'] if row['Width'] > 0 else avg_width
                desired_space = int(row['Suggested Facings']) * width
                if desired_space <= remaining_space:
                    df_alloc.at[i, 'Adjusted Facings'] = int(row['Suggested Facings'])
                    df_alloc.at[i, 'Space Needed Adjusted'] = df_alloc.at[i, 'Adjusted Facings'] * width
                    df_alloc.at[i, 'Fits Shelf'] = True
                    remaining_space -= df_alloc.at[i, 'Space Needed Adjusted']
                else:
                    max_facings = int(remaining_space // width) if width > 0 else 0
                    if max_facings > 0:
                        df_alloc.at[i, 'Adjusted Facings'] = int(max_facings)
                        df_alloc.at[i, 'Space Needed Adjusted'] = df_alloc.at[i, 'Adjusted Facings'] * width
                        df_alloc.at[i, 'Fits Shelf'] = False
                        remaining_space -= df_alloc.at[i, 'Space Needed Adjusted']
                    else:
                        df_alloc.at[i, 'Adjusted Facings'] = 0
                        df_alloc.at[i, 'Space Needed Adjusted'] = 0
                        df_alloc.at[i, 'Fits Shelf'] = False

            skus_that_fit = df_alloc[df_alloc['Fits Shelf']].copy()
            skus_overflow = df_alloc[~df_alloc['Fits Shelf']].copy()

            # ----- Dashboard: shelf usage and overflow side-by-side -----
            st.subheader("Shelf usage & SKUs that cannot fit")
            col1, col2 = st.columns([2, 2])

            total_space_used = df_alloc['Space Needed Adjusted'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

            with col1:
                st.write("**Shelf Usage**")
                st.progress(min(space_pct/100, 1.0))
                st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")
                st.write(f"SKUs fully allocated: {skus_that_fit.shape[0]} / {df_alloc.shape[0]}")

            with col2:
                st.write("**SKUs that cannot fit (full suggested facings)**")
                if skus_overflow.empty:
                    st.success("✅ All SKUs fit at least partially (or fully) within selected shelf space.")
                else:
                    st.dataframe(
                        skus_overflow[['SKU', 'Product Type', 'Variant', 'Item Size', 'Recommendation', 'Suggested Facings', 'Adjusted Facings', 'Space Needed Adjusted']].rename(
                            columns={'Space Needed Adjusted': 'Space Needed (Adjusted)'}
                        ),
                        use_container_width=True
                    )

            # ----- SKU Recommendation table (styled) -----
            st.subheader("SKU Recommendations")
            display_cols = [
                "SKU", "Store Code", "Product Type", "Variant", "Item Size",
                "Sales", "Volume", "Margins", "Score", "Rank",
                "Recommendation", "Justification",
                "Suggested Facings", "Adjusted Facings", "Width", "Space Needed Adjusted"
            ]
            # keep only existing cols
            display_cols = [c for c in display_cols if c in df_alloc.columns]
            def highlight_rec(v):
                if v == "Expand": return "background-color:#d4f7d4"
                if v == "Retain": return "background-color:#fff4cc"
                if v == "Delist": return "background-color:#ffd6d6"
                return ""
            try:
                st.dataframe(df_alloc[display_cols].style.applymap(highlight_rec, subset=['Recommendation'] if 'Recommendation' in display_cols else None), use_container_width=True)
            except Exception:
                # fallback if style fails
                st.dataframe(df_alloc[display_cols], use_container_width=True)

            # ----- Summary pivot (Product Type x Variant x Recommendation counts) -----
            st.subheader("Summary by Product Type & Variant")
            summary = (
                df_alloc.groupby(['Product Type', 'Variant', 'Recommendation'])
                .size().reset_index(name='Count')
            )
            pivot = summary.pivot_table(index=['Product Type','Variant'], columns='Recommendation', values='Count', fill_value=0).reset_index()
            st.dataframe(pivot, use_container_width=True)

            # Chart: combined label + horizontal bars (no overlap)
            if not summary.empty:
                summary['CategoryVariant'] = summary['Product Type'] + " - " + summary['Variant']
                # sort categories by total count for readability
                totals = summary.groupby('CategoryVariant')['Count'].sum().sort_values()
                order = totals.index.tolist()
                fig_summary = px.bar(
                    summary.sort_values('Count', ascending=True),
                    x='Count',
                    y='CategoryVariant',
                    color='Recommendation',
                    orientation='h',
                    text='Count',
                    category_orders={'CategoryVariant': order}
                )
                fig_summary.update_traces(textposition='outside')
                fig_summary.update_layout(height=400, margin=dict(l=200, r=20, t=30, b=30))
                st.plotly_chart(fig_summary, use_container_width=True)
            else:
                st.info("No summary to plot for selected filters.")

            # ----- Download: try Excel (openpyxl) else fallback to CSV -----
            st.subheader("Download SKU Recommendations")
            excel_buf = io.BytesIO()
            has_openpyxl = importlib.util.find_spec('openpyxl') is not None
            if has_openpyxl:
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    # write df_alloc (rename columns for clarity)
                    excel_df = df_alloc.copy()
                    if 'Space Needed Adjusted' in excel_df.columns:
                        excel_df = excel_df.rename(columns={'Space Needed Adjusted': 'Space Needed'})
                    download_cols = [c for c in [
                        "SKU", "Store Code", "Product Type", "Variant", "Item Size",
                        "Sales", "Volume", "Margins", "Score", "Rank",
                        "Recommendation", "Justification",
                        "Suggested Facings", "Adjusted Facings", "Width", "Space Needed"
                    ] if c in excel_df.columns]
                    excel_df[download_cols].to_excel(writer, index=False, sheet_name="Recommendations")
                    pivot.to_excel(writer, index=False, sheet_name="Summary")
                excel_buf.seek(0)
                st.download_button(
                    label="📥 Download SKU Recommendations (Excel)",
                    data=excel_buf,
                    file_name="sku_recommendations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # fallback to CSV (single file for recommendations)
                csv_buf = io.BytesIO()
                excel_df = df_alloc.copy()
                csv_bytes = excel_df.to_csv(index=False).encode('utf-8')
                st.warning("Excel export requires the 'openpyxl' package. Downloading CSV instead. (To enable Excel, install openpyxl.)")
                st.download_button(
                    label="📥 Download SKU Recommendations (CSV)",
                    data=csv_bytes,
                    file_name="sku_recommendations.csv",
                    mime="text/csv"
                )

# ================= MODULE 2 =================
elif module == "Sales Analysis":
    st.header("📈 Sales Analysis & Insight Matching")
    sales_file = st.sidebar.file_uploader("Upload Sales CSV", type=["csv"])
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
                    if len(group) == 1:
                        sel.loc[group.index, 'Baseline'] = np.nan
                    else:
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
