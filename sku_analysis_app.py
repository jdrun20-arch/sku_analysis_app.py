# Complete app (Modules 1-4) — robust Suggested Facings (30/30/40), Retain %, shelf allocation, downloads
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
import importlib.util
from datetime import datetime
#-

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
        out.loc[neg_mask] = -out.loc[neg_mask]
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

    # --- Sidebar inputs (preserve target retain % and distribution chart) ---
    st.sidebar.markdown("### Shelf & Allocation Settings")
    target_retain_pct = st.sidebar.slider("Target to Retain %", 1, 100, 80, step=1)
    shelf_width = st.sidebar.number_input("Shelf width per layer (units)", 1.0, 10000.0, 100.0)
    num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
    min_facings_expand = st.sidebar.slider("Min facings if Expand", 1, 10, 3)
    min_facings_retain = st.sidebar.slider("Min facings if Retain", 1, 10, 2)
    facings_delist = st.sidebar.slider("Facings if Delist", 0, 5, 0)
    min_facings_global = st.sidebar.number_input("Global minimum facings", 1, 10, 1)
    max_facings_per_sku = st.sidebar.number_input("Max facings per SKU", 1, 50, 10)
    default_width = st.sidebar.number_input("Default SKU width (used if missing)", 0.1, 100.0, 5.0)
    hide_delist = st.sidebar.checkbox("Hide Delist SKUs in tables", value=False)

    # File uploader in main area (keeps sidebar clean)
    sku_file = st.file_uploader("Upload SKU CSV (columns: SKU, Sales, Volume, Margins, optional: Width, Facings, Product Type, Variant, Item Size)", type=["csv"])
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        # required
        required = ["SKU","Sales","Volume","Margins"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # defaults for optional columns
            optional_defaults = {
                "Store Code": "ALL",
                "Width": default_width,
                "Facings": 1,
                "Product Type": "Unknown",
                "Variant": "Default",
                "Item Size": ""
            }
            for c, d in optional_defaults.items():
                if c not in sku.columns:
                    sku[c] = d

            # clean numeric columns
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0.0)
            sku['Margins'] = pd.to_numeric(sku['Margins'], errors='coerce').fillna(0.0)
            sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(default_width)
            sku['Facings'] = pd.to_numeric(sku['Facings'], errors='coerce').fillna(1).astype(int)

            # normalize text columns
            sku['Product Type'] = sku['Product Type'].fillna("Unknown").astype(str)
            sku['Variant'] = sku['Variant'].fillna("Default").astype(str)
            sku['Item Size'] = sku['Item Size'].fillna("").astype(str)
            sku['Store Code'] = sku['Store Code'].fillna("ALL").astype(str)

            # --- Contributions & Weighted Score (30/30/40) ---
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

            sku['Score'] = (
                0.3 * sku['Sales_Contribution'] +
                0.3 * sku['Volume_Contribution'] +
                0.4 * sku['Margin_Contribution']
            )
            sku['Rank'] = sku['Score'].rank(method='min', ascending=False).astype(int)

            # --- Retain % decision (top X% by score) ---
            sku_sorted = sku.sort_values('Score', ascending=False).reset_index(drop=True)
            keep_count = max(1, int(np.ceil(len(sku_sorted) * (target_retain_pct / 100.0))))
            sku_sorted['Recommendation'] = 'Delist'
            sku_sorted.loc[:keep_count-1, 'Recommendation'] = 'Retain'

            # mark top of retained as Expand? We'll mark expands where Suggested Facings > current Facings later.
            sku = sku_sorted.copy()

            # --- Compute total facings capacity based on shelf inputs ---
            total_shelf_space = float(shelf_width) * int(num_layers)
            avg_width = sku['Width'].replace(0, np.nan).dropna().mean()
            if pd.isna(avg_width) or avg_width <= 0:
                avg_width = default_width
            total_facings_available = int(np.floor(total_shelf_space / avg_width)) if avg_width > 0 else max(1, len(sku) * min_facings_global)
            total_facings_available = max(total_facings_available, len(sku) * min_facings_global)  # ensure reasonable

            # --- Proportional allocation (robust to zeros) ---
            score_sum = sku['Score'].sum()
            if score_sum <= 0:
                # equal allocation if all scores zero
                raw_alloc = np.array([total_facings_available / len(sku)] * len(sku), dtype=float)
            else:
                raw_alloc = (sku['Score'].values / score_sum) * total_facings_available

            # base allocation (floor) and fractional parts
            base_alloc = np.floor(raw_alloc).astype(int)
            frac = raw_alloc - np.floor(raw_alloc)

            # Minimum required by recommendation
            min_req = np.array([
                min_facings_expand if rec == 'Expand' else (min_facings_retain if rec == 'Retain' else facings_delist)
                for rec in sku['Recommendation'].values
            ], dtype=int)

            # Enforce global minimum
            min_req = np.maximum(min_req, int(min_facings_global))

            # Start allocation as max(base_alloc, min_req)
            alloc = np.maximum(base_alloc, min_req)

            # Ensure not exceeding max_facings_per_sku
            alloc = np.minimum(alloc, int(max_facings_per_sku))

            # If allocation sum differs from available, adjust:
            current_sum = int(alloc.sum())
            if current_sum > total_facings_available:
                # need to reduce; reduce from lowest score SKUs first but not below min_req or 0
                idx_order_reduce = np.argsort(sku['Score'].values)  # low to high
                need_reduce = current_sum - total_facings_available
                for idx in idx_order_reduce:
                    if need_reduce <= 0:
                        break
                    reducible = alloc[idx] - min_req[idx]
                    if reducible <= 0:
                        continue
                    take = min(reducible, need_reduce)
                    alloc[idx] -= take
                    need_reduce -= take
                # If still need_reduce >0, further reduce starting from lowest but allow down to 0
                if need_reduce > 0:
                    for idx in idx_order_reduce:
                        if need_reduce <= 0:
                            break
                        reducible = alloc[idx]
                        if reducible <= 0:
                            continue
                        take = min(reducible, need_reduce)
                        alloc[idx] -= take
                        need_reduce -= take
            elif current_sum < total_facings_available:
                # distribute remaining facings by fractional parts, prioritizing higher fractional and higher score
                remaining = total_facings_available - current_sum
                # create priority: fractional first then score
                priority = np.argsort([-frac[i] + -sku['Score'].values[i]*1e-9 for i in range(len(sku))])  # descending frac, tie-breaker score
                for idx in priority:
                    if remaining <= 0:
                        break
                    if alloc[idx] < max_facings_per_sku:
                        alloc[idx] += 1
                        remaining -= 1
                # if still remain (unlikely), add to highest score until exhausted
                if remaining > 0:
                    for idx in np.argsort(-sku['Score'].values):
                        if remaining <= 0:
                            break
                        addable = max_facings_per_sku - alloc[idx]
                        if addable <= 0:
                            continue
                        add = min(addable, remaining)
                        alloc[idx] += add
                        remaining -= add

            # Final safety: ensure alloc non-negative ints
            alloc = np.clip(alloc.astype(int), 0, int(max_facings_per_sku))

            sku['Suggested Facings'] = alloc

            # Set Recommendation to Expand if Suggested Facings > existing Facings AND currently Retain (or always mark)
            sku.loc[(sku['Suggested Facings'] > sku['Facings']) & (sku['Recommendation'] != 'Delist'), 'Recommendation'] = 'Expand'

            # Compute Space Needed by Suggested Facings
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            # ---- Shelf allocation pass: attempt to allocate suggested facings until shelf capacity exhausted ----
            df_alloc = sku.sort_values('Score', ascending=False).copy()
            df_alloc['Adjusted Facings'] = df_alloc['Suggested Facings'].astype(int)
            df_alloc['Space Needed Adjusted'] = df_alloc['Width'] * df_alloc['Adjusted Facings']
            df_alloc['Fits Shelf'] = True

            remaining_space = total_shelf_space
            for i, row in df_alloc.iterrows():
                width = row['Width'] if row['Width'] > 0 else avg_width
                desired = int(row['Suggested Facings']) * width
                if desired <= remaining_space:
                    df_alloc.at[i, 'Adjusted Facings'] = int(row['Suggested Facings'])
                    df_alloc.at[i, 'Space Needed Adjusted'] = df_alloc.at[i, 'Adjusted Facings'] * width
                    df_alloc.at[i, 'Fits Shelf'] = True
                    remaining_space -= df_alloc.at[i, 'Space Needed Adjusted']
                else:
                    # allocate partial facings that fit (integer)
                    max_facings_fit = int(remaining_space // width) if width > 0 else 0
                    if max_facings_fit > 0:
                        df_alloc.at[i, 'Adjusted Facings'] = max_facings_fit
                        df_alloc.at[i, 'Space Needed Adjusted'] = max_facings_fit * width
                        df_alloc.at[i, 'Fits Shelf'] = False
                        remaining_space -= df_alloc.at[i, 'Space Needed Adjusted']
                    else:
                        df_alloc.at[i, 'Adjusted Facings'] = 0
                        df_alloc.at[i, 'Space Needed Adjusted'] = 0
                        df_alloc.at[i, 'Fits Shelf'] = False

            skus_that_fit = df_alloc[df_alloc['Fits Shelf']].copy()
            skus_overflow = df_alloc[~df_alloc['Fits Shelf']].copy()

            total_space_used = df_alloc['Space Needed Adjusted'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

            # --- KEEP DISTRIBUTION CHART (unchanged feel) ---
            st.subheader("SKU Distribution by Product Type & Variant")
            # build tidy summary for chart
            try:
                chart_df = sku.groupby(['Product Type','Variant','Recommendation']).size().reset_index(name='Count')
                chart_df['CategoryVariant'] = chart_df['Product Type'] + " - " + chart_df['Variant']
                fig = px.bar(chart_df, x='Count', y='CategoryVariant', color='Recommendation', orientation='h', text='Count',
                             category_orders={'Recommendation': ['Expand','Retain','Delist']}, height=400)
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(l=200, r=20, t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Not enough data for distribution chart.")

            # --- Shelf usage & overflow side-by-side ---
            st.subheader("Shelf usage & SKUs that cannot fit")
            col1, col2 = st.columns([2,2])
            with col1:
                st.write("**Shelf Usage**")
                st.progress(min(space_pct/100, 1.0))
                st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")
                st.write(f"SKUs allocated (>=1 facing): { (df_alloc['Adjusted Facings']>0).sum() } / {len(df_alloc)}")
            with col2:
                st.write("**SKUs that cannot fit (full suggested facings)**")
                if skus_overflow.empty:
                    st.success("✅ All SKUs fit at least partially (or fully) within selected shelf space.")
                else:
                    display_overflow_cols = ['SKU','Product Type','Variant','Item Size','Recommendation','Suggested Facings','Adjusted Facings','Space Needed Adjusted']
                    cols_exist = [c for c in display_overflow_cols if c in skus_overflow.columns]
                    st.dataframe(skus_overflow[cols_exist].rename(columns={'Space Needed Adjusted':'Space Needed (Adjusted)'}), use_container_width=True)

            # --- SKU Recommendations table (styled) ---
            st.subheader("SKU Recommendations")
            display_cols = [
                "SKU","Store Code","Product Type","Variant","Item Size",
                "Sales","Volume","Margins","Score","Rank",
                "Recommendation","Suggested Facings","Adjusted Facings","Width","Space Needed Adjusted"
            ]
            display_cols = [c for c in display_cols if c in df_alloc.columns]
            def highlight_rec(v):
                if v == "Expand": return "background-color:#d4f7d4"
                if v == "Retain": return "background-color:#fff4cc"
                if v == "Delist": return "background-color:#ffd6d6"
                return ""
            try:
                st.dataframe(df_alloc[display_cols].style.applymap(highlight_rec, subset=['Recommendation'] if 'Recommendation' in display_cols else None), use_container_width=True)
            except Exception:
                st.dataframe(df_alloc[display_cols], use_container_width=True)

            # --- Summary pivot by product type & variant ---
            st.subheader("Summary by Product Type & Variant")
            summary = (df_alloc.groupby(['Product Type','Variant','Recommendation']).size().reset_index(name='Count'))
            pivot = summary.pivot_table(index=['Product Type','Variant'], columns='Recommendation', values='Count', fill_value=0).reset_index()
            st.dataframe(pivot, use_container_width=True)

            # --- Download recommendations (Excel if openpyxl present else CSV fallback) ---
            st.subheader("Download SKU Recommendations")
            excel_buf = io.BytesIO()
            has_openpyxl = importlib.util.find_spec('openpyxl') is not None
            excel_df = df_alloc.copy()
            if 'Space Needed Adjusted' in excel_df.columns:
                excel_df = excel_df.rename(columns={'Space Needed Adjusted': 'Space Needed'})
            download_cols = [c for c in [
                "SKU","Store Code","Product Type","Variant","Item Size",
                "Sales","Volume","Margins","Score","Rank",
                "Recommendation","Suggested Facings","Adjusted Facings","Width","Space Needed"
            ] if c in excel_df.columns]
            if has_openpyxl:
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    excel_df[download_cols].to_excel(writer, index=False, sheet_name="Recommendations")
                    pivot.to_excel(writer, index=False, sheet_name="Summary")
                excel_buf.seek(0)
                st.download_button(label="📥 Download SKU Recommendations (Excel)", data=excel_buf, file_name="sku_recommendations.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                csv_bytes = excel_df[download_cols].to_csv(index=False).encode('utf-8')
                st.warning("Excel export requires 'openpyxl'. Downloading CSV instead. (To enable Excel, install openpyxl.)")
                st.download_button(label="📥 Download SKU Recommendations (CSV)", data=csv_bytes, file_name="sku_recommendations.csv", mime="text/csv")

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
