# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import numpy as np

st.set_page_config(layout="wide", page_title="Retail Assortment + Sales Insights (Fixed)")

INSIGHTS_CSV = "insights.csv"
if not os.path.exists(INSIGHTS_CSV):
    pd.DataFrame(columns=["Timestamp", "Date", "Store Code", "Note", "Status", "Submitted By"]).to_csv(INSIGHTS_CSV, index=False)

# -------------------------
# Helpers: load/save insights
# -------------------------
def load_insights():
    df = pd.read_csv(INSIGHTS_CSV)
    # normalize column name variants to 'Date' and 'Store Code'
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' in cols_lower and cols_lower['date'] != 'Date':
        df.rename(columns={cols_lower['date']: 'Date'}, inplace=True)
    for cand in ('store code','store_code','storecode','store'):
        if cand in cols_lower and cols_lower[cand] != 'Store Code':
            df.rename(columns={cols_lower[cand]: 'Store Code'}, inplace=True)
            break
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    return df

def save_insights(df):
    df.to_csv(INSIGHTS_CSV, index=False)

# -------------------------
# Helper: normalize sales columns (accept common variants)
# -------------------------
def normalize_sales_columns(df):
    cols = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in cols}
    colmap = {}
    # Date
    if 'date' in lower_map:
        colmap[lower_map['date']] = 'Date'
    elif 'day' in lower_map:
        colmap[lower_map['day']] = 'Date'
    # Store Code
    for v in ('store code','store_code','storecode','store'):
        if v in lower_map:
            colmap[lower_map[v]] = 'Store Code'
            break
    # Sales
    for v in ('sales','sale','amount','revenue'):
        if v in lower_map:
            colmap[lower_map[v]] = 'Sales'
            break
    # SKU optional
    for v in ('sku','sku_code','product'):
        if v in lower_map:
            colmap[lower_map[v]] = 'SKU'
            break
    if colmap:
        df = df.rename(columns=colmap)
    return df

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("Modules")
module = st.sidebar.radio("Choose module:", [
    "1️⃣ SKU Performance & Shelf Space",
    "2️⃣ Sales Analysis",
    "3️⃣ Submit Insight",
    "4️⃣ Approve Insights"
])

# -------------------------
# MODULE 1: SKU Performance & Shelf Space (restored)
# -------------------------
if module == "1️⃣ SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("Upload SKU CSV (columns: SKU, Sales, Volume, Margin, optional Width, Store Code)", type=["csv"], key="sku1")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validate
        required_cols = ["SKU","Sales","Volume","Margin"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)} (required: SKU, Sales, Volume, Margin).")
            st.stop()

        # numeric safety
        for c in ["Sales","Volume","Margin"]:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # scoring
        df['Sales_Norm'] = df['Sales'] / df['Sales'].replace(0, pd.NA).max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].replace(0, pd.NA).max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].replace(0, pd.NA).max()
        df[['Sales_Norm','Volume_Norm','Margin_Norm']] = df[['Sales_Norm','Volume_Norm','Margin_Norm']].fillna(0)
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)
        df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)

        cutoff_expand = df['Score'].quantile(0.70)
        cutoff_delist = df['Score'].quantile(0.30)
        def classify(s):
            if s >= cutoff_expand: return "Expand"
            if s <= cutoff_delist: return "Delist"
            return "Retain"
        df['Recommendation'] = df['Score'].apply(classify)

        def justify(rec):
            return {
                "Expand":"High sales, volume, or margin → increase facings or distribution.",
                "Delist":"Low performance → candidate for phase-out.",
                "Retain":"Balanced performance → maintain current space."
            }.get(rec, "")
        df['Justification'] = df['Recommendation'].apply(justify)

        # UI settings
        st.sidebar.header("Settings (SKU)")
        expand_facings = st.sidebar.slider("Facings for Expand", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & calc", value=False)
        top_n = st.sidebar.slider("Top N SKUs in chart", 5, max(5, len(df)), 20)

        total_shelf_space = shelf_width * num_layers

        def base_fac(rec):
            if rec=="Expand": return max(expand_facings, min_facings)
            if rec=="Retain": return max(retain_facings, min_facings)
            return delist_facings
        df['Base Facings'] = df['Recommendation'].apply(base_fac)

        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU width (inches)", 0.1, 100.0, 5.0, 0.1)
            df['Width'] = default_width
        else:
            df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(st.sidebar.number_input("Default SKU width (fallback)", 0.1, 100.0, 5.0, 0.1))

        df['Space Needed'] = df['Width'] * df['Base Facings']

        # redistribute if delist facings == 0
        if delist_facings == 0:
            delist_mask = df['Recommendation'] == 'Delist'
            freed = (df.loc[delist_mask, 'Width'] * base_fac('Delist')).sum()
            mask_er = df['Recommendation'].isin(['Expand','Retain'])
            denom = (df.loc[mask_er, 'Width'] * df.loc[mask_er, 'Base Facings']).sum()
            if denom > 0 and freed > 0:
                extra = (df.loc[mask_er, 'Width'] * df.loc[mask_er, 'Base Facings'] / denom * freed) / df.loc[mask_er, 'Width']
                df.loc[mask_er, 'Extra Facings'] = extra.fillna(0)
            else:
                df['Extra Facings'] = 0
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        df_filtered = df[df['Recommendation'] != "Delist"] if hide_delist else df.copy()
        total_used = df_filtered['Space Needed'].sum()
        pct_used = (total_used / total_shelf_space) * 100 if total_shelf_space>0 else 0.0

        st.subheader("SKU Summary")
        st.write(f"Total SKUs: {len(df)} | Expand: {(df['Recommendation']=='Expand').sum()} | Retain: {(df['Recommendation']=='Retain').sum()} | Delist: {(df['Recommendation']=='Delist').sum()}")

        # display table
        def style_rec(v):
            if v=="Expand": return "background-color:#d4f7d4"
            if v=="Retain": return "background-color:#fff4cc"
            if v=="Delist": return "background-color:#ffd6d6"
            return ""
        show_cols = ['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']
        if 'Store Code' in df.columns:
            show_cols.insert(1,'Store Code')
        st.dataframe(df[show_cols].style.applymap(style_rec, subset=['Recommendation']), use_container_width=True)

        st.subheader("Shelf Space")
        st.progress(min(pct_used/100, 1.0))
        st.write(f"Used: {total_used:.1f}/{total_shelf_space:.1f} in ({pct_used:.1f}%)")
        if pct_used>100:
            over = total_used - total_shelf_space
            df_sorted = df_filtered.sort_values(['Space Needed','Score'], ascending=[False, True])
            cum = 0.0; to_remove=[]
            for _,r in df_sorted.iterrows():
                cum += float(r['Space Needed'])
                to_remove.append(r['SKU'])
                if cum >= over:
                    break
            st.warning(f"Overcapacity by {over:.1f} in. Suggested remove {len(to_remove)} SKU(s): {to_remove[:10]}")
        else:
            st.success("Within capacity ✅")

        # chart
        st.subheader("Top SKUs by Space Needed")
        df_chart = df_filtered.sort_values('Space Needed', ascending=False).head(top_n)
        fig = px.bar(df_chart, x='Space Needed', y='SKU', orientation='h', color='Recommendation',
                     hover_data=['Width','Suggested Facings','Justification'],
                     color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=30*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

        # download
        st.download_button("Download SKU recommendations", df.to_csv(index=False).encode('utf-8'), "sku_recs.csv", "text/csv")

# -------------------------
# MODULE 2: Sales Analysis (robust + qualitative + highlighting)
# -------------------------
elif module == "2️⃣ Sales Analysis":
    st.title("📈 Sales Analysis with Qualitative Insights")
    st.markdown("Upload Sales CSV with Date (YYYY-MM-DD), Store Code, Sales. Optional: SKU.")

    sales_file = st.file_uploader("Upload Sales CSV", type=["csv"], key="sales2")
    if sales_file:
        sales_df = pd.read_csv(sales_file)
        sales_df = normalize_sales_columns(sales_df)

        # validate presence of required columns after normalization
        req_sales = ['Date','Store Code','Sales']
        missing = [c for c in req_sales if c not in sales_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Accepted variants: StoreCode/Store_Code/Store Code, Sales/Sale/Amount, Date.")
            st.stop()

        # parse date and ensure numeric sales
        sales_df['Date_parsed'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        if sales_df['Date_parsed'].isna().any():
            st.error("Some dates couldn't be parsed. Use YYYY-MM-DD.")
            st.stop()
        sales_df['Date'] = sales_df['Date_parsed'].dt.strftime("%Y-%m-%d")
        sales_df['Sales'] = pd.to_numeric(sales_df['Sales'], errors='coerce').fillna(0)

        # settings
        st.sidebar.header("Sales Analysis Settings")
        pct_thr = st.sidebar.slider("Lift/Drop threshold (%)", 5, 200, 30, 5)
        match_window = st.sidebar.slider("Match approved insights within ±N days", 0, 7, 0, 1)

        # baseline per store
        baseline = sales_df.groupby('Store Code', as_index=False)['Sales'].mean().rename(columns={'Sales':'BaselineSales'})
        merged = pd.merge(sales_df, baseline, on='Store Code', how='left')

        # compute pct change vs baseline (safe)
        def safe_pct(s, b):
            if pd.isna(b) or b==0:
                return np.nan
            return (s - b) / b * 100.0
        merged['ChangePct'] = merged.apply(lambda r: safe_pct(r['Sales'], r['BaselineSales']), axis=1)

        def classify_pct(pct):
            if pd.isna(pct):
                return 'Normal'
            if pct >= pct_thr:
                return 'Lift'
            if pct <= -pct_thr:
                return 'Drop'
            return 'Normal'

        merged['Signal'] = merged.apply(lambda r: ('Lift' if (r['BaselineSales']==0 and r['Sales']>0) else classify_pct(r['ChangePct'])), axis=1)

        # load approved insights and normalize
        insights_df = load_insights()
        if 'Status' in insights_df.columns:
            approved = insights_df[insights_df['Status'].str.lower()=='approved'].copy()
        else:
            approved = pd.DataFrame(columns=insights_df.columns)
        if not approved.empty and 'Date' in approved.columns:
            approved['Date_parsed'] = pd.to_datetime(approved['Date'], errors='coerce')

        # match insights within match_window days
        def find_matches(r):
            if approved.empty:
                return ""
            try:
                row_date = pd.to_datetime(r['Date'])
            except Exception:
                return ""
            store = str(r['Store Code'])
            if match_window == 0:
                mask = (approved['Store Code'].astype(str) == store) & (approved['Date'] == r['Date'])
            else:
                low = row_date - pd.Timedelta(days=match_window)
                high = row_date + pd.Timedelta(days=match_window)
                mask = (approved['Store Code'].astype(str) == store) & (approved['Date_parsed'] >= low) & (approved['Date_parsed'] <= high)
            matched = approved[mask]
            if matched.empty:
                return ""
            return " ; ".join(matched['Note'].astype(str).tolist())
        merged['MatchedInsight'] = merged.apply(find_matches, axis=1)

        # system + qualitative narratives
        def system_text(r):
            pct = r['ChangePct']; sig = r['Signal']; ins = r['MatchedInsight']
            if sig=='Lift':
                pct_label = ("a big increase" if pd.isna(pct) else f"{pct:.0f}% higher")
                return f"Sales {pct_label} vs baseline — likely related to: {ins}." if ins else f"Sales {pct_label} vs baseline — no matched insight; possible foot traffic, promo, or event."
            if sig=='Drop':
                pct_label = ("a big drop" if pd.isna(pct) else f"{abs(pct):.0f}% lower")
                return f"Sales {pct_label} vs baseline — possibly due to: {ins}." if ins else f"Sales {pct_label} vs baseline — no matched insight; consider weather/stockouts/ops."
            return "Sales within normal range."

        def qualitative_text(r):
            sig = r['Signal']; pct = 0 if pd.isna(r['ChangePct']) else abs(r['ChangePct']); ins = r['MatchedInsight']
            if sig=='Lift':
                degree = "an extreme surge" if pct>=100 else ("a strong uplift" if pct>=50 else "a moderate lift")
                if ins:
                    return f"{degree} ({int(r['ChangePct'])}%) likely due to: {ins}. Recommend stocking fast-movers and increasing staffing."
                else:
                    return f"{degree} ({int(r['ChangePct'])}%) with no matched insight. Check promos, local events, competitor activity."
            if sig=='Drop':
                degree = "a severe decline" if pct>=100 else ("a significant drop" if pct>=50 else "a moderate dip")
                if ins:
                    return f"{degree} ({int(abs(r['ChangePct']))}%) possibly caused by: {ins}. Investigate ops and supply."
                else:
                    return f"{degree} ({int(abs(r['ChangePct']))}%) with no matched insight. Investigate traffic, stockouts, weather."
            return "No significant change vs baseline."

        merged['System Analysis'] = merged.apply(system_text, axis=1)
        merged['Qualitative Analysis'] = merged.apply(qualitative_text, axis=1)

        # counts
        st.subheader("Summary")
        st.write(f"Lift days: {(merged['Signal']=='Lift').sum()}  |  Drop days: {(merged['Signal']=='Drop').sum()}")

        # top matched reasons
        reasons = merged[merged['MatchedInsight'] != ""]['MatchedInsight'].str.split(" ; ").explode().value_counts()
        if not reasons.empty:
            st.write("Top matched reasons from approved insights:")
            for r,cnt in reasons.head(5).items():
                st.write(f"- {r} ({cnt})")

        # prepare display columns safely
        display_cols = [c for c in ['Date','Store Code','SKU','Sales','BaselineSales','ChangePct','Signal','MatchedInsight','System Analysis','Qualitative Analysis'] if c in merged.columns]

        # styling: Signal color; Qualitative Analysis italic
        def color_signal(v):
            if v == 'Lift': return 'background-color: #d4f7d4;'
            if v == 'Drop': return 'background-color: #ffd6d6;'
            return ''
        def italic_text(v):
            return 'font-style: italic;' if isinstance(v, str) and v else ''

        # sort safely
        sort_by = []
        if 'Store Code' in merged.columns:
            sort_by.append('Store Code')
        if 'Date_parsed' in merged.columns:
            sort_by.append('Date_parsed')
        elif 'Date' in merged.columns:
            sort_by.append('Date')
        merged_sorted = merged.sort_values(sort_by) if sort_by else merged

        st.subheader("Detailed Sales Analysis")
        styler = merged_sorted[display_cols].style
        if 'Signal' in merged_sorted.columns:
            styler = styler.applymap(color_signal, subset=['Signal'])
        if 'Qualitative Analysis' in merged_sorted.columns:
            styler = styler.applymap(italic_text, subset=['Qualitative Analysis'])
        st.dataframe(styler, use_container_width=True)

        # trend chart for selected store
        st.subheader("Sales trend (select store)")
        stores = merged['Store Code'].unique().tolist()
        if stores:
            sel = st.selectbox("Store", stores)
            df_store = merged_sorted[merged_sorted['Store Code']==sel]
            if not df_store.empty:
                xcol = 'Date_parsed' if 'Date_parsed' in df_store.columns else 'Date'
                fig = px.line(df_store, x=xcol, y='Sales', markers=True, title=f"Sales trend - {sel}")
                if 'BaselineSales' in df_store.columns:
                    fig.add_scatter(x=df_store[xcol], y=df_store['BaselineSales'], mode='lines', name='Baseline')
                # annotate matched insights
                ann = df_store[df_store['MatchedInsight'] != ""]
                if not ann.empty:
                    fig.add_scatter(x=ann[xcol], y=ann['Sales'], mode='markers', marker_symbol='x', marker_size=12, name='Matched Insight')
                st.plotly_chart(fig, use_container_width=True)

        # download
        st.download_button("Download analysis CSV", merged.to_csv(index=False).encode('utf-8'), "sales_analysis.csv", "text/csv")

# -------------------------
# MODULE 3: Submit Insight (persist)
# -------------------------
elif module == "3️⃣ Submit Insight":
    st.title("📝 Submit Insight (persists to insights.csv)")
    with st.form("insight_form"):
        date_input = st.date_input("Date of event")
        store_code = st.text_input("Store Code")
        note = st.text_area("Note / Insight")
        submitted_by = st.text_input("Submitted by")
        submit = st.form_submit_button("Submit")
        if submit:
            if not store_code or not note or not submitted_by:
                st.error("Please complete Store Code, Note, Submitted by.")
            else:
                ins_df = load_insights()
                new = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Date": date_input.strftime("%Y-%m-%d"),
                    "Store Code": store_code,
                    "Note": note,
                    "Status": "Pending",
                    "Submitted By": submitted_by
                }
                ins_df = pd.concat([ins_df, pd.DataFrame([new])], ignore_index=True)
                save_insights(ins_df)
                st.success("Submitted and pending approval.")

    st.subheader("All insights (latest first)")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)

# -------------------------
# MODULE 4: Approve Insights (persist)
# -------------------------
elif module == "4️⃣ Approve Insights":
    st.title("✅ Approve / Reject Insights")
    ins_df = load_insights()
    ins_df['Status'] = ins_df['Status'].astype(str)
    pending_idx = ins_df[ins_df['Status'].str.lower()=='pending'].index.tolist()
    if not pending_idx:
        st.info("No pending insights.")
    else:
        for idx in pending_idx:
            row = ins_df.loc[idx]
            st.markdown("---")
            st.write(f"Index: {idx}")
            st.write(f"Date: {row['Date']}  |  Store Code: {row['Store Code']}")
            st.write(row['Note'])
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {idx}", key=f"appr_{idx}"):
                ins_df.at[idx,'Status'] = "Approved"
                save_insights(ins_df)
                st.success("Approved")
                st.rerun()
            if col2.button(f"Reject {idx}", key=f"rej_{idx}"):
                ins_df.at[idx,'Status'] = "Rejected"
                save_insights(ins_df)
                st.warning("Rejected")
                st.rerun()
    st.subheader("All insights")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)
