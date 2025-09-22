# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="Assortment + Sales Insights App")
INSIGHTS_FILE = "insights.csv"

# Ensure insights CSV exists
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Timestamp", "Date", "Store Code", "Insight Type", "Details", "Status", "Submitted By"]).to_csv(INSIGHTS_FILE, index=False)

# ----------------------------
# SIDEBAR - Module selection
# ----------------------------
st.sidebar.header("📌 Modules")
module = st.sidebar.radio("Choose module:", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insight",
    "Approve Insights"
])

# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

def compute_score(df, sales_col="Sales", volume_col="Volume", margin_col="Margin"):
    df = df.copy()
    for c in [sales_col, volume_col, margin_col]:
        if c not in df.columns:
            df[c] = 0
    # Avoid division by zero
    df['Sales_Norm'] = df[sales_col] / df[sales_col].replace(0, np.nan).max()
    df['Volume_Norm'] = df[volume_col] / df[volume_col].replace(0, np.nan).max()
    df['Margin_Norm'] = df[margin_col] / df[margin_col].replace(0, np.nan).max()
    df[['Sales_Norm','Volume_Norm','Margin_Norm']] = df[['Sales_Norm','Volume_Norm','Margin_Norm']].fillna(0)
    df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)
    return df

# ----------------------------
# MODULE: SKU Performance & Shelf Space
# ----------------------------
if module == "SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space")

    st.markdown("Upload SKU file with columns: `SKU`, `Store Code`, `Sales`, `Volume`, `Margin`, `Width` (width per facing in inches). `Width` is optional.")

    uploaded = st.file_uploader("Upload SKU CSV (for shelf/assortment)", type=["csv"])
    if uploaded is not None:
        df = safe_read_csv(uploaded)
        if df is None:
            st.stop()

        # Sidebar settings (module-specific)
        st.sidebar.header("⚙️ Shelf Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 12, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 12, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 6, 1)
        min_facings = st.sidebar.number_input("Minimum facings for Expand/Retain", min_value=1, value=2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", min_value=1.0, value=100.0, step=1.0)
        num_layers = st.sidebar.number_input("Number of vertical layers (shelves)", min_value=1, value=1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from space calc", value=False)
        top_n = st.sidebar.slider("Top N SKUs to show on chart", 10, 200, 50)

        total_shelf_space = shelf_width * num_layers

        # Validate minimal columns
        required = ["SKU", "Store Code", "Sales", "Volume", "Margin"]
        missing = ensure_columns(df, required)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Use the template or add these columns.")
            st.stop()

        # Compute performance score and rank
        df = compute_score(df, "Sales", "Volume", "Margin")
        df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)

        # Classify
        cutoff_expand = df['Score'].quantile(0.70)
        cutoff_delist = df['Score'].quantile(0.30)

        def classify(s):
            if s >= cutoff_expand: return "Expand"
            if s <= cutoff_delist: return "Delist"
            return "Retain"

        df['Recommendation'] = df['Score'].apply(classify)

        # Justification
        def justification(r):
            if r == "Expand": return "High performance — consider more facings/distribution."
            if r == "Delist": return "Low performance — candidate for phase-out."
            return "Average performance — maintain current space."
        df['Justification'] = df['Recommendation'].apply(justification)

        # Base facings
        def base_fac(rec):
            if rec == "Expand": return max(expand_facings, min_facings)
            if rec == "Retain": return max(retain_facings, min_facings)
            return delist_facings
        df['Base Facings'] = df['Recommendation'].apply(base_fac)

        # Width handling
        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU width (inches, used if 'Width' missing)", min_value=0.1, value=5.0, step=0.1)
            df['Width'] = default_width
        else:
            df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(st.sidebar.number_input("Default SKU width (inches) (fallback)", min_value=0.1, value=5.0, step=0.1))

        # Initial space needed
        df['Space_Needed_Base'] = df['Width'] * df['Base Facings']

        # Redistribute freed space if Delist facings = 0
        if delist_facings == 0:
            delist_mask = df['Recommendation'] == 'Delist'
            freed_total = (df.loc[delist_mask, 'Width'] * base_fac('Delist')).sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand','Retain'])
            denom = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
            if denom > 0 and freed_total > 0:
                df.loc[expand_retain_mask, 'Extra Facings'] = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / denom * freed_total) / df.loc[expand_retain_mask, 'Width']
                df['Extra Facings'] = df['Extra Facings'].fillna(0)
            else:
                df['Extra Facings'] = 0
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = (df['Base Facings'] + df['Extra Facings']).round(2)
        df['Space_Needed'] = df['Width'] * df['Suggested Facings']

        # Filter for display/space calc
        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"].copy()
        else:
            df_filtered = df.copy()

        # Total space used and percent
        total_used = df_filtered['Space_Needed'].sum()
        pct_used = (total_used / total_shelf_space) * 100 if total_shelf_space > 0 else np.nan

        # Summary counts
        total_skus = len(df)
        cnt_expand = (df['Recommendation'] == 'Expand').sum()
        cnt_retain = (df['Recommendation'] == 'Retain').sum()
        cnt_delist = (df['Recommendation'] == 'Delist').sum()

        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total SKUs", total_skus)
        col2.metric("Expand", cnt_expand)
        col3.metric("Retain", cnt_retain)
        col4.metric("Delist", cnt_delist)

        st.subheader("Shelf Space")
        st.write(f"Total shelf capacity: **{total_shelf_space:.1f} inches** ({shelf_width:.1f} in × {num_layers} layers)")
        st.progress(min(pct_used/100, 1.0))
        st.write(f"Used: **{total_used:.1f}/{total_shelf_space:.1f} in ({pct_used:.1f}%)**")

        # Actionable suggestion: find minimum SKUs to remove (largest space) to fit
        if pct_used > 100:
            over_inch = total_used - total_shelf_space
            df_sorted = df_filtered.sort_values(by='Space_Needed', ascending=False)
            cum = 0.0
            to_remove = []
            for _, r in df_sorted.iterrows():
                cum += r['Space_Needed']
                to_remove.append(str(r['SKU']))
                if cum >= over_inch:
                    break
            st.warning(
                f"⚠️ Shelf is over capacity.\nYou need to free **{over_inch:.1f} inches**.\n"
                f"Suggested minimum SKUs to remove ({len(to_remove)}):\n- " + "\n- ".join(to_remove)
            )
        else:
            st.success("✅ Shelf plan fits within available space.")

        # Detailed table
        st.subheader("Detailed SKU Recommendations")
        display_cols = ['SKU','Store Code','Score','Rank','Recommendation','Justification','Base Facings','Extra Facings','Suggested Facings','Width','Space_Needed']
        st.dataframe(df[display_cols].sort_values(by='Space_Needed', ascending=False).reset_index(drop=True), use_container_width=True)

        # Chart: top N skus by Space_Needed
        st.subheader(f"Top {min(top_n, len(df_filtered))} SKUs by Space Needed")
        df_chart = df_filtered.sort_values(by='Space_Needed', ascending=False).head(top_n)
        fig = px.bar(
            df_chart,
            y='SKU',
            x='Space_Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space_Needed':':.1f','Width':':.1f','Suggested Facings':True,'Justification':True},
            color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=30*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MODULE: Sales Analysis
# ----------------------------
elif module == "Sales Analysis":
    st.title("📈 Sales Analysis (with Insights matching)")
    st.markdown("Upload a **Sales CSV** with columns: `Date` (YYYY-MM-DD), `SKU`, `Store Code`, `Sales`.")

    sales_file = st.file_uploader("Upload Sales CSV (date+sku+store+sales)", type=["csv"])
    if sales_file is not None:
        sales_df = safe_read_csv(sales_file)
        if sales_df is None:
            st.stop()

        # Validate
        missing = ensure_columns(sales_df, ["Date","SKU","Store Code","Sales"])
        if missing:
            st.error(f"Missing columns in sales file: {', '.join(missing)}")
            st.stop()

        # parse date
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        if sales_df['Date'].isna().any():
            st.error("Some Date values couldn't be parsed. Use YYYY-MM-DD format.")
            st.stop()

        # Basic settings
        st.sidebar.header("Sales Analysis Settings")
        window_days = st.sidebar.slider("Spike detection lookback window (days)", 3, 30, 7)
        spike_multiplier = st.sidebar.slider("Spike multiplier vs average (e.g., 2.0 means 2x)", 1.1, 10.0, 2.0)

        # Aggregate daily sales by SKU+Store
        daily = sales_df.groupby(['Date','Store Code','SKU'], as_index=False)['Sales'].sum()

        # Detect spikes: For each SKU+Store, compute rolling mean and compare
        daily = daily.sort_values(['Store Code','SKU','Date'])
        daily['RollingMean'] = daily.groupby(['Store Code','SKU'])['Sales'].transform(lambda s: s.shift(1).rolling(window=window_days, min_periods=1).mean().fillna(0))
        daily['RollingMean'] = daily['RollingMean'].replace(0, 1e-9)
        daily['Spike'] = daily['Sales'] >= (daily['RollingMean'] * spike_multiplier)
        daily['PctChange'] = ((daily['Sales'] - daily['RollingMean']) / daily['RollingMean']) * 100

        # Load approved insights
        insights_df = pd.read_csv(INSIGHTS_FILE)
        insights_df['Date'] = pd.to_datetime(insights_df['Date'], errors='coerce')
        approved = insights_df[insights_df['Status'] == 'Approved'][['Date','Store Code','Insight Type','Details']]

        # Match spikes to approved insights by same Date and Store Code
        daily['Matched Insight'] = None
        for idx, r in daily[daily['Spike']].iterrows():
            same = approved[(approved['Date'] == r['Date']) & (approved['Store Code'].astype(str) == str(r['Store Code']))]
            if not same.empty:
                daily.at[idx, 'Matched Insight'] = "; ".join((same['Insight Type'].astype(str) + ": " + same['Details'].astype(str)).tolist())

        # Display spikes
        spikes = daily[daily['Spike']].sort_values(['PctChange','Date'], ascending=[False, False])

        st.subheader("Detected Sales Spikes")
        st.write(f"Spike rule: Sales >= {spike_multiplier} × rolling mean (previous {window_days} days).")
        if spikes.empty:
            st.info("No spikes detected using current settings.")
        else:
            display_cols = ['Date','Store Code','SKU','Sales','RollingMean','PctChange','Matched Insight']
            st.dataframe(spikes[display_cols].sort_values('PctChange', ascending=False).reset_index(drop=True), use_container_width=True)

            st.subheader("Inspect Spike")
            spike_indices = spikes.index.tolist()
            sel_idx = st.selectbox("Select spike to inspect (by index)", spike_indices)
            r = spikes.loc[sel_idx]
            st.write(f"Date: {r['Date'].date()}, Store: {r['Store Code']}, SKU: {r['SKU']}")
            st.write(f"Sales: {r['Sales']:.1f}, Rolling mean: {r['RollingMean']:.1f}, Change: {r['PctChange']:.1f}%")
            if pd.notna(r['Matched Insight']):
                st.success(f"Matched approved insight: {r['Matched Insight']}")
            else:
                st.info("No approved insight matched this spike. Consider checking insights module or expanding date matching window.")

        # Quick chart: aggregate daily sales for a selected SKU+Store
        st.subheader("Sales Trend for a SKU (choose to view)")
        # Build options for selectbox
        unique_opts = daily.apply(lambda row: f"{row['Store Code']} | {row['SKU']}", axis=1).unique().tolist()
        if unique_opts:
            sku_opt = st.selectbox("Choose a SKU (Store | SKU)", unique_opts)
            store_code, sku_sel = [s.strip() for s in sku_opt.split("|")]
            sel_df = daily[(daily['Store Code'].astype(str) == store_code) & (daily['SKU'] == sku_sel)].sort_values('Date')
            if not sel_df.empty:
                fig = px.line(sel_df, x='Date', y='Sales', title=f"Sales trend - Store {store_code}, SKU {sku_sel}")
                fig.add_scatter(x=sel_df['Date'], y=sel_df['RollingMean'], mode='lines', name='Rolling Mean')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for selected SKU+Store.")

# ----------------------------
# MODULE: Submit Insight
# ----------------------------
elif module == "Submit Insight":
    st.title("📝 Submit Insight")
    st.markdown("Field users can submit observations/events here (e.g., Rally, Class suspended, Promo). These go to the manager for approval.")

    with st.form("insight_form"):
        date_input = st.date_input("Date of event")
        store_code = st.text_input("Store Code")
        insight_type = st.selectbox("Insight Type", ["External Event", "Class Suspended", "Promo", "Competitor Activity", "Other"])
        details = st.text_area("Details / Description")
        submitted_by = st.text_input("Submitted By (name)")
        submitted = st.form_submit_button("Submit Insight")

        if submitted:
            if not store_code or not details or not submitted_by:
                st.error("Please fill Store Code, Details, and Submitted By.")
            else:
                df_ins = pd.read_csv(INSIGHTS_FILE)
                new = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Date": date_input.strftime("%Y-%m-%d"),
                    "Store Code": str(store_code),
                    "Insight Type": insight_type,
                    "Details": details,
                    "Status": "Pending Approval",
                    "Submitted By": submitted_by
                }
                df_ins = pd.concat([df_ins, pd.DataFrame([new])], ignore_index=True)
