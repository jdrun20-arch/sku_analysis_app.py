# app.py
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
    """Try to rerun programmatically; if not available, set flag and stop with a message."""
    try:
        st.rerun()
    except Exception:
        st.session_state["_refresh_needed"] = not st.session_state.get("_refresh_needed", False)
        st.success("Change saved. Please refresh the page to see updates.")
        st.stop()

def ensure_insights_df():
    """Load insights CSV and normalize columns."""
    df = pd.read_csv(INSIGHTS_FILE)
    for c in ["Date","Store Code","Insight","Status"]:
        if c not in df.columns:
            df[c] = ""
    # normalize date to YYYY-MM-DD strings for merging
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime("%Y-%m-%d")
    return df

def write_insights_df(df):
    df.to_csv(INSIGHTS_FILE, index=False)

def clean_sales_series(s: pd.Series) -> pd.Series:
    """Make a pandas Series numeric: remove currency, commas, parentheses, handle negatives."""
    s2 = s.astype(str).str.strip()
    neg_mask = s2.str.match(r'^\(.*\)$')  # (1,234) style
    s2 = s2.str.replace(r'[\(\)]', '', regex=True)
    s2 = s2.str.replace(r'[^\d\.\-]', '', regex=True)  # remove non-digit except . and -
    out = pd.to_numeric(s2, errors='coerce')
    # apply negative where parentheses existed
    out[neg_mask] = -out[neg_mask]
    return out.fillna(0.0)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Map common variants to canonical names: Date, Store Code, Sales, SKU, Width."""
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

# ---------- UI: Module Selector ----------
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

    sku_file = st.file_uploader("Upload SKU CSV (required: SKU, Sales, Volume, Margin). Optional: Width)", type=["csv"])
    if sku_file is None:
        st.info("Upload a SKU CSV to run the SKU module. Minimum columns: SKU, Sales, Volume, Margin.")
    else:
        sku_raw = pd.read_csv(sku_file)
        sku = normalize_colnames(sku_raw)

        required = ["SKU","Sales","Volume","Margin"]
        missing = [c for c in required if c not in sku.columns]
        if missing:
            st.error(f"Missing required columns: {missing}. I can provide a CSV template if needed.")
        else:
            # Clean numbers
            sku['Sales'] = clean_sales_series(sku['Sales'])
            sku['Volume'] = pd.to_numeric(sku['Volume'], errors='coerce').fillna(0)
            sku['Margin'] = pd.to_numeric(sku['Margin'], errors='coerce').fillna(0)

            # Score
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
                'Expand': "High sales/volume/margin — consider more facings/distribution.",
                'Delist': "Low performance — candidate for phase-out.",
                'Retain': "Balanced — keep current allocation."
            })

            # Sidebar settings
            st.sidebar.header("Shelf settings")
            expand_facings = st.sidebar.slider("Facings for Expand", 1, 10, 3)
            retain_facings = st.sidebar.slider("Facings for Retain", 1, 10, 2)
            delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 1)
            min_facings = st.sidebar.number_input("Minimum facings (Expand/Retain)", 1, 10, 2)
            shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0)
            num_layers = st.sidebar.number_input("Number of layers", 1, 10, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & calculations", value=False)
            top_n = st.sidebar.slider("Top SKUs in chart", 5, min(100, max(5, len(sku))), min(50, max(5, len(sku))))

            total_shelf_space = shelf_width * num_layers

            def base_fac(rec):
                if rec == "Expand": return max(expand_facings, min_facings)
                if rec == "Retain": return max(retain_facings, min_facings)
                return delist_facings
            sku['Base Facings'] = sku['Recommendation'].apply(base_fac)

            # Width handling
            if 'Width' not in sku.columns:
                sku['Width'] = st.sidebar.number_input("Default SKU width (inches)", 0.1, 100.0, 5.0, 0.1)
            else:
                sku['Width'] = pd.to_numeric(sku['Width'], errors='coerce').fillna(
                    st.sidebar.number_input("Default SKU width (fallback)", 0.1, 100.0, 5.0, 0.1)
                )

            sku['Suggested Facings'] = sku['Base Facings']
            sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']

            # Optional redistribution when delist facings = 0
            redistribute = st.sidebar.checkbox("Redistribute freed-up space from Delist SKUs (when delist facings = 0)", value=False)
            if delist_facings == 0 and redistribute:
                delist_mask = sku['Recommendation'] == 'Delist'
                freed = (sku.loc[delist_mask, 'Width'] * 1).sum()
                er_mask = sku['Recommendation'].isin(['Expand','Retain'])
                denom = (sku.loc[er_mask, 'Width'] * sku.loc[er_mask, 'Suggested Facings']).sum()
                if denom > 0 and freed > 0:
                    extra = (sku.loc[er_mask, 'Width'] * sku.loc[er_mask, 'Suggested Facings'] / denom * freed) / sku.loc[er_mask, 'Width']
                    sku.loc[er_mask, 'Extra Facings'] = extra.fillna(0)
                    sku['Suggested Facings'] = sku['Suggested Facings'] + sku.get('Extra Facings', 0)
                    sku['Space Needed'] = sku['Width'] * sku['Suggested Facings']
                else:
                    sku['Extra Facings'] = 0
            else:
                sku['Extra Facings'] = 0

            df_filtered = sku[sku['Recommendation'] != "Delist"] if hide_delist else sku.copy()
            total_space_used = df_filtered['Space Needed'].sum()
            space_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space>0 else 0.0

            # Outputs
            st.subheader("SKU Recommendations & Rank")
            def highlight_rec(v):
                if v=="Expand": return "background-color:#d4f7d4"
                if v=="Retain": return "background-color:#fff4cc"
                if v=="Delist": return "background-color:#ffd6d6"
                return ""
            cols_show = ['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']
            st.dataframe(sku[cols_show].style.applymap(highlight_rec, subset=['Recommendation']), use_container_width=True)

            st.subheader("Shelf usage")
            st.progress(min(space_pct/100,1.0))
            st.write(f"Used: {total_space_used:.1f} / {total_shelf_space:.1f} in ({space_pct:.1f}%)")
            if space_pct > 100:
                over = total_space_used - total_shelf_space
                st.warning(f"Over capacity by {over:.1f} in. Consider delisting or reducing facings.")
            else:
                st.success("Within capacity ✅")

            st.subheader("Top SKUs by Space Needed")
            df_chart = df_filtered.sort_values('Space Needed', ascending=False).head(top_n)
            fig = px.bar(df_chart, x='Space Needed', y='SKU', orientation='h', color='Recommendation',
                         hover_data=['Width','Suggested Facings'], color_discrete_map={'Expand':'#2e8b57','Retain':'#ffcc00','Delist':'#ff6666'})
            fig.update_layout(height=30*len(df_chart))
            st.plotly_chart(fig, use_container_width=True)

# ========== MODULE 2: SALES ANALYSIS & INSIGHT MATCHING ==========
elif module == "Sales Analysis":
    st.header("📈 Sales Analysis & Insight Matching")

    sales_file = st.file_uploader("Upload Sales CSV (Date, Store Code, Sales)", type=["csv"])
    if sales_file is None:
        st.info("Upload a sales CSV (Date, Store Code, Sales).")
    else:
        sales_raw = pd.read_csv(sales_file)
        sales = normalize_colnames(sales_raw)

        # Validate
        if 'Date' not in sales.columns or 'Sales' not in sales.columns:
            st.error("Sales CSV must include 'Date' and 'Sales' columns (Store Code recommended).")
        else:
            # Parse & clean
            sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
            sales = sales.dropna(subset=['Date']).copy()
            sales['Date_str'] = sales['Date'].dt.strftime("%Y-%m-%d")
            if 'Store Code' not in sales.columns:
                sales['Store Code'] = "ALL"

            # Clean Sales BEFORE grouping/mean
            sales['Sales'] = clean_sales_series(sales['Sales'])
            # Now it's safe to compute means
            # Filters
            store_list = sales['Store Code'].unique().tolist()
            selected = st.multiselect("Select store(s)", store_list, default=store_list)
            min_date = sales['Date'].min().date()
            max_date = sales['Date'].max().date()
            dr = st.date_input("Date range", [min_date, max_date])
            start_d, end_d = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

            sel = sales[(sales['Store Code'].isin(selected)) & (sales['Date'].between(start_d, end_d))].copy()
            if sel.empty:
                st.info("No sales rows for the selected store(s) and date range.")
            else:
                # Leave-one-out baseline per store
                sel['Baseline'] = np.nan
                for store, group in sel.groupby('Store Code'):
                    idxs = group.index.tolist()
                    if len(group) == 1:
                        sel.loc[idxs, 'Baseline'] = np.nan
                    else:
                        for i in idxs:
                            sel.loc[i, 'Baseline'] = group.loc[group.index != i, 'Sales'].mean()

                sel['ChangePct'] = (sel['Sales'] - sel['Baseline']) / sel['Baseline'] * 100

                # thresholds in sidebar
                pct_thr_up = st.sidebar.slider("Lift threshold (%)", 10, 500, 50, 5)
                pct_thr_down = st.sidebar.slider("Drop threshold (%)", 5, 200, 30, 5)

                # load approved insights
                insights_df = ensure_insights_df()
                insights_approved = insights_df[insights_df['Status'].str.lower() == 'approved'].copy()

                # merge by Store Code + date string
                sel['Date_key'] = sel['Date'].dt.strftime("%Y-%m-%d")
                merged = pd.merge(sel, insights_approved, how='left', left_on=['Store Code','Date_key'], right_on=['Store Code','Date'])
                merged['Matched Insight'] = merged['Insight'].fillna("")

                # classify signal, insights prioritized
                def classify_row(r):
                    if r['Matched Insight']:
                        # if insight present, infer direction vs baseline (or vs store mean if no baseline)
                        if pd.isna(r['Baseline']):
                            store_mean = sales[sales['Store Code']==r['Store Code']]['Sales'].mean()
                            return "LIFT" if r['Sales'] >= store_mean else "DROP"
                        return "LIFT" if r['Sales'] >= r['Baseline'] else "DROP"
                    # no insight: use pct thresholds
                    if pd.isna(r['ChangePct']):
                        return "NORMAL"
                    if r['ChangePct'] >= pct_thr_up:
                        return "LIFT"
                    if r['ChangePct'] <= -pct_thr_down:
                        return "DROP"
                    return "NORMAL"

                merged['Signal'] = merged.apply(classify_row, axis=1)

                def qual_note(r):
                    if r['Matched Insight']:
                        return f"User insight: {r['Matched Insight']}"
                    if r['Signal'] == "LIFT":
                        return f"Sales +{(0 if pd.isna(r['ChangePct']) else r['ChangePct']):.0f}% vs baseline"
                    if r['Signal'] == "DROP":
                        return f"Sales {(0 if pd.isna(r['ChangePct']) else abs(r['ChangePct'])):.0f}% lower vs baseline"
                    return "Normal"
                merged['Qualitative Note'] = merged.apply(qual_note, axis=1)

                # summary
                lifts = (merged['Signal']=="LIFT').sum() if False else (merged['Signal']=="LIFT").sum()
                drops = (merged['Signal']=="DROP").sum()
                with_insight = (merged['Matched Insight'] != "").sum()

                st.subheader("Summary")
                c1,c2,c3 = st.columns(3)
                c1.metric("Lift days", lifts)
                c2.metric("Drop days", drops)
                c3.metric("Days with approved insight", with_insight)

                # styled table
                def style_sig(v):
                    if v == "LIFT": return "background-color: #d4f7d4"
                    if v == "DROP": return "background-color: #ffd6d6"
                    return ""
                disp = ['Store Code','Date','Sales','Baseline','ChangePct','Signal','Qualitative Note']
                st.subheader("Detailed rows")
                st.dataframe(merged[disp].sort_values(['Store Code','Date']).style
                             .applymap(style_sig, subset=['Signal'])
                             .applymap(lambda x: "font-style: italic;" if isinstance(x, str) and x.startswith("User insight") else "", subset=['Qualitative Note']),
                             use_container_width=True)

                # narrative generation
                st.subheader("Automated Narrative")
                narrative = []
                for store, grp in merged.groupby('Store Code'):
                    lifts_store = grp[grp['Signal']=="LIFT"]
                    drops_store = grp[grp['Signal']=="DROP"]
                    if not lifts_store.empty or not drops_store.empty:
                        narrative.append(f"Store {store}: {len(lifts_store)} lift(s), {len(drops_store)} drop(s).")
                        for _, r in lifts_store.iterrows():
                            line = f"📈 {r['Date'].date()}: ₱{r['Sales']:,.0f}"
                            if r['Matched Insight']:
                                line += f" — likely due to: {r['Matched Insight']}"
                            else:
                                pct = 0 if pd.isna(r['ChangePct']) else r['ChangePct']
                                line += f" — +{pct:.0f}% vs baseline (no approved insight)"
                            narrative.append(line)
                        for _, r in drops_store.iterrows():
                            line = f"📉 {r['Date'].date()}: ₱{r['Sales']:,.0f}"
                            if r['Matched Insight']:
                                line += f" — possible cause: {r['Matched Insight']}"
                            else:
                                pct = 0 if pd.isna(r['ChangePct']) else abs(r['ChangePct'])
                                line += f" — -{pct:.0f}% vs baseline (no approved insight)"
                            narrative.append(line)
                if narrative:
                    st.markdown("\n\n".join(narrative))
                else:
                    st.info("No unusual patterns in the selected range.")

                # chart with insight markers
                st.subheader("Sales chart (with insight markers)")
                fig = px.line(merged, x='Date', y='Sales', color='Store Code', markers=True, title="Sales trend")
                for store, grp in merged.groupby('Store Code'):
                    marks = grp[grp['Matched Insight'] != ""]
                    if not marks.empty:
                        fig.add_scatter(x=marks['Date'], y=marks['Sales'], mode='markers', marker_symbol='star', marker_size=12,
                                        name=f"Insight - {store}", hovertext=marks['Matched Insight'])
                fig.update_xaxes(range=[start_d, end_d])
                st.plotly_chart(fig, use_container_width=True)

# ========== MODULE 3: SUBMIT INSIGHT ==========
elif module == "Submit Insight":
    st.header("📝 Submit Insight")
    with st.form("insight_form"):
        date_input = st.date_input("Date")
        store_code = st.text_input("Store Code")
        insight_text = st.text_area("Insight (short description)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            ins_df = ensure_insights_df()
            new_row = {"Date": pd.to_datetime(date_input).strftime("%Y-%m-%d"), "Store Code": store_code, "Insight": insight_text, "Status": "Pending"}
            ins_df = pd.concat([ins_df, pd.DataFrame([new_row])], ignore_index=True)
            write_insights_df(ins_df)
            st.success("Submitted — pending approval.")
    st.subheader("All insights (latest first)")
    st.dataframe(ensure_insights_df().sort_values('Date', ascending=False), use_container_width=True)

# ========== MODULE 4: APPROVE INSIGHTS ==========
elif module == "Approve Insights":
    st.header("✅ Approve / Reject Insights")
    ins = ensure_insights_df()
    pending = ins[ins['Status'].str.lower() == 'pending'].copy()
    if pending.empty:
        st.info("No pending insights.")
    else:
        pending = pending.reset_index()
        for _, row in pending.iterrows():
            orig_idx = int(row['index'])
            st.markdown("---")
            st.write(f"📅 {row['Date']}  |  🏬 {row['Store Code']}")
            st.write(row['Insight'])
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve", key=f"approve_{orig_idx}"):
                ins.loc[ins.index == orig_idx, 'Status'] = "Approved"
                write_insights_df(ins)
                safe_rerun()
            if col2.button("❌ Reject", key=f"reject_{orig_idx}"):
                ins.loc[ins.index == orig_idx, 'Status'] = "Rejected"
                write_insights_df(ins)
                safe_rerun()
