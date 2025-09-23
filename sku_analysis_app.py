# Full app: Modules 1-4 (robust Excel fallback, contribution-based facings, variant-aware, filters)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
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
    out[neg_mask] = -out[neg_mask]
    return out.fillna(0.0)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    lowermap = {c.lower(): c for c in df.columns}
    mapping = {}
    # common mappings
    for cand in ('date','day','transaction_date'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Date'; break
    for cand in ('store code','store_code','storecode','store'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Store Code'; break
    for cand in ('sales','sale','amount','revenue'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Sales'; break
    for cand in ('margin','margins','profit'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Margin'; break
    for cand in ('volume','units','unit_sold'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Volume'; break
    for cand in ('sku','sku_code','product','item','item_code'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'SKU'; break
    for cand in ('width','item width','width_in','width_cm','size_width'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Width'; break
    for cand in ('facings','facing','num_facings'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Facings'; break
    for cand in ('product type','category','type'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Product Type'; break
    for cand in ('variant','flavor','flavour'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Variant'; break
    for cand in ('item size','size','pack_size'):
        if cand in lowermap:
            mapping[lowermap[cand]] = 'Item Size'; break
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

    sku_file = st.file_uploader(
        "Upload SKU CSV (required: SKU, Sales, Volume, Margin). Optional: Store Code, Width, Facings, Product Type, Variant, Item Size",
        type=["csv"]
    )
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        # Provide safe defaults if columns missing
        defaults = {
            "SKU": None,
            "Store Code": "ALL",
            "Sales": 0.0,
            "Volume": 0.0,
            "Margin": 0.0,
            "Width": 5.0,
            "Facings": 1,
            "Product Type": "Unknown",
            "Variant": "Default",
            "Item Size": ""
        }
        for col, default in defaults.items():
            if col not in sku.columns:
                sku[col] = default

        # Ensure there's at least one non-empty SKU
        if sku['SKU'].isnull().all() or sku['SKU'].astype(str).str.strip().eq('').all():
            st.error("No SKU column found or all SKU values are empty. Please provide a valid SKU column.")
            st.stop()

        # Clean numeric fields
        sku['Sales'] = clean_sales_series(sku['Sales'])
        sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
        sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)
        sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(5.0)
        sku['Facings'] = pd.to_numeric(sku['Facings'], errors='coerce').fillna(1).astype(int)

        # Normalize text columns
        sku['Product Type'] = sku['Product Type'].fillna("Unknown").astype(str)
        sku['Variant'] = sku['Variant'].fillna("Default").astype(str)
        sku['Item Size'] = sku['Item Size'].fillna("").astype(str)
        sku['Store Code'] = sku['Store Code'].fillna("ALL").astype(str)

        # ========== Scoring ==========
        def norm(series):
            mx = series.replace(0, pd.NA).max()
            if pd.isna(mx) or mx == 0:
                return pd.Series(0, index=series.index)
            return series / mx

        sku['Sales_Norm'] = norm(sku['Sales'])
        sku['Volume_Norm'] = norm(sku['Volume'])
        sku['Margin_Norm'] = norm(sku['Margin'])
        sku['Score'] = (sku['Sales_Norm'] * 0.3) + (sku['Volume_Norm'] * 0.3) + (sku['Margin_Norm'] * 0.4)
        sku['Rank'] = sku['Score'].rank(method='min', ascending=False).astype(int)

        # Initial rec (quantile-based)
        cutoff_expand = sku['Score'].quantile(0.70)
        cutoff_delist = sku['Score'].quantile(0.30)
        sku['Recommendation'] = sku['Score'].apply(
            lambda s: "Expand" if s >= cutoff_expand else ("Delist" if s <= cutoff_delist else "Retain")
        )
        sku['Justification'] = sku['Recommendation'].map({
            'Expand': "High performance — consider expansion.",
            'Delist': "Low performance — candidate for phase-out.",
            'Retain': "Balanced — maintain."
        })

        # ---------- Sidebar (controls) ----------
        st.sidebar.header("Shelf & Variant Settings")
        expand_facings = st.sidebar.slider("Minimum facings for Expand", 1, 10, 3)
        retain_facings = st.sidebar.slider("Minimum facings for Retain", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 0)
        min_facings = st.sidebar.number_input("Global minimum facings", 1, 10, 1)
        shelf_width = st.sidebar.number_input("Shelf width per layer (units match Width)", 1.0, 10000.0, 100.0)
        num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from display", value=False)
        top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, max(5, len(sku))), min(50, max(5, len(sku))))
        max_expand_per_type = st.sidebar.slider("Max Expand SKUs per Product Type", 1, 10, 2)

        total_shelf_space = float(shelf_width) * int(num_layers)

        # ---------- Variant-aware adjustment ----------
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

        # ---------- Product Type filter ----------
        product_types = sku['Product Type'].unique().tolist()
        selected_types = st.multiselect("Filter by Product Type (affects summary & allocations)", options=product_types, default=product_types)

        df_filtered = sku[sku['Product Type'].isin(selected_types)].copy()
        if hide_delist:
            df_filtered = df_filtered[df_filtered['Recommendation'] != "Delist"].copy()

        if df_filtered.empty:
            st.warning("No SKUs to display for the selected Product Type(s) / filters.")
            st.stop()

        # ---------- Contribution-based suggested facings (on filtered set) ----------
        sales_sum = df_filtered['Sales'].sum() or 1.0
        volume_sum = df_filtered['Volume'].sum() or 1.0
        margin_sum = df_filtered['Margin'].sum() or 1.0

        df_filtered['Sales_Contribution'] = df_filtered['Sales'] / sales_sum
        df_filtered['Volume_Contribution'] = df_filtered['Volume'] / volume_sum
        df_filtered['Margin_Contribution'] = df_filtered['Margin'] / margin_sum

        df_filtered['Weighted_Contribution'] = (
            df_filtered['Sales_Contribution'] * 0.3 +
            df_filtered['Volume_Contribution'] * 0.3 +
            df_filtered['Margin_Contribution'] * 0.4
        )

        # estimate total facings that could fit across shelf based on average width
        avg_width = df_filtered['Width'].replace(0, np.nan).dropna().mean()
        if pd.isna(avg_width) or avg_width <= 0:
            avg_width = 5.0
        estimated_total_facings = max(int(total_shelf_space / avg_width), 1)

        # proportional allocation of facings
        df_filtered['Suggested Facings'] = (df_filtered['Weighted_Contribution'] * estimated_total_facings).round().fillna(0).astype(int)

        # enforce minimums by recommendation
        df_filtered.loc[df_filtered['Recommendation'] == "Expand", 'Suggested Facings'] = df_filtered.loc[df_filtered['Recommendation'] == "Expand", 'Suggested Facings'].clip(lower=expand_facings)
        df_filtered.loc[df_filtered['Recommendation'] == "Retain", 'Suggested Facings']  = df_filtered.loc[df_filtered['Recommendation'] == "Retain", 'Suggested Facings'].clip(lower=retain_facings)
        df_filtered.loc[df_filtered['Recommendation'] == "Delist", 'Suggested Facings']  = delist_facings

        # enforce global minimum
        df_filtered['Suggested Facings'] = df_filtered['Suggested Facings'].clip(lower=min_facings)

        # final space needed (before allocation)
        df_filtered['Space Needed'] = df_filtered['Width'] * df_filtered['Suggested Facings']

        # ---------- Shelf allocation by width (attempt full suggested facings, allow partial) ----------
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

        # ---------- Dashboard: Shelf usage & Overflow ----------
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

        # ---------- SKU Recommendations table ----------
        st.subheader("SKU Recommendations")
        display_cols = [
            "SKU", "Store Code", "Product Type", "Variant", "Item Size",
            "Sales", "Volume", "Margin", "Score", "Rank",
            "Recommendation", "Justification",
            "Suggested Facings", "Adjusted Facings", "Width", "Space Needed Adjusted"
        ]
        display_cols = [c for c in display_cols if c in df_alloc.columns]
        def highlight_rec(v):
            if v == "Expand": return "background-color:#d4f7d4"
            if v == "Retain": return "background-color:#fff4cc"
            if v == "Delist": return "background-color:#ffd6d6"
            return ""
        st.dataframe(df_alloc[display_cols].style.applymap(highlight_rec, subset=['Recommendation'] if 'Recommendation' in display_cols else None), use_container_width=True)

        # ---------- Summary chart ----------
        st.subheader("Summary by Product Type & Variant")
        summary = (
            df_alloc.groupby(['Product Type', 'Variant', 'Recommendation'])
            .size().reset_index(name='Count')
            .sort_values(['Product Type', 'Variant', 'Recommendation'])
        )
        summary_filtered = summary[summary['Product Type'].isin(selected_types)].copy()
        if not summary_filtered.empty:
            summary_filtered['Category-Variant'] = summary_filtered['Product Type'] + " - " + summary_filtered['Variant']
            fig_summary = px.bar(
                summary_filtered.sort_values("Count", ascending=True),
                x="Count",
                y="Category-Variant",
                color="Recommendation",
                orientation="h",
                text="Count",
                category_orders={"Recommendation": ["Expand", "Retain", "Delist"]},
                height=400
            )
            fig_summary.update_traces(textposition="outside")
            fig_summary.update_layout(margin=dict(l=120, r=20, t=30, b=30))
            st.plotly_chart(fig_summary, use_container_width=True)
        else:
            st.info("No summary rows to display for the selected filters.")

        # ---------- Download (Excel if available; fallback to CSV) ----------
        st.subheader("Download SKU Recommendations")
        excel_alloc = df_alloc.copy()
        if 'Space Needed Adjusted' in excel_alloc.columns:
            excel_alloc = excel_alloc.rename(columns={'Space Needed Adjusted': 'Space Needed'})

        download_cols = [
            "SKU", "Store Code", "Product Type", "Variant", "Item Size",
            "Sales", "Volume", "Margin", "Score", "Rank",
            "Recommendation", "Justification",
            "Suggested Facings", "Adjusted Facings", "Width", "Space Needed"
        ]
        download_cols_actual = [c for c in download_cols if c in excel_alloc.columns]

        buf = io.BytesIO()
        # try Excel with openpyxl; if not available, fallback to CSV
        try:
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                excel_alloc[download_cols_actual].to_excel(writer, index=False, sheet_name="SKU Recommendations")
                summary.to_excel(writer, index=False, sheet_name="Summary")
            buf.seek(0)
            st.download_button(
                label="📥 Download SKU Recommendations (Excel)",
                data=buf,
                file_name="sku_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            # fallback to CSV if Excel writer engine missing
            csv_buf = io.BytesIO()
            excel_alloc[download_cols_actual].to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.warning("Excel export requires 'openpyxl'. Downloading CSV instead. (To enable Excel, install openpyxl in the environment.)")
            st.download_button(
                label="📥 Download SKU Recommendations (CSV)",
                data=csv_buf.getvalue(),
                file_name="sku_recommendations.csv",
                mime="text/csv"
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
