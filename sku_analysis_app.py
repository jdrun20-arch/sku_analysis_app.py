# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import numpy as np

# ----------------------------
# Config
# ----------------------------
st.set_page_config(layout="wide", page_title="Retail Assortment + Sales Insights")

INSIGHTS_CSV = "insights.csv"
# Ensure the insights CSV file exists
if not os.path.exists(INSIGHTS_CSV):
    pd.DataFrame(columns=["Timestamp", "Date", "Store Code", "Note", "Status", "Submitted By"]).to_csv(INSIGHTS_CSV, index=False)

# ----------------------------
# Helpers for insights persistence
# ----------------------------
def load_insights():
    df = pd.read_csv(INSIGHTS_CSV)
    # normalize columns
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    return df

def save_insights(df):
    df.to_csv(INSIGHTS_CSV, index=False)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("📌 App Modules")
module = st.sidebar.radio("Choose module:", [
    "1️⃣ SKU Performance & Shelf Space",
    "2️⃣ Sales Analysis",
    "3️⃣ Submit Insight",
    "4️⃣ Approve Insights"
])

# ----------------------------
# MODULE 1: SKU Performance & Shelf Space (restored)
# ----------------------------
if module == "1️⃣ SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimizer")

    uploaded_file = st.file_uploader("📂 Upload your SKU CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ["SKU", "Sales", "Volume", "Margin"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Required: SKU, Sales, Volume, Margin. Optional: Width, Store Code.")
            st.stop()

        # --- Performance Scoring (safe numeric conversion) ---
        for c in ["Sales", "Volume", "Margin"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df['Sales_Norm'] = df['Sales'] / df['Sales'].replace(0, pd.NA).max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].replace(0, pd.NA).max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].replace(0, pd.NA).max()
        df[['Sales_Norm','Volume_Norm','Margin_Norm']] = df[['Sales_Norm','Volume_Norm','Margin_Norm']].fillna(0)
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

        # --- Ranking & Classification ---
        df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)
        cutoff_expand = df['Score'].quantile(0.70)
        cutoff_delist = df['Score'].quantile(0.30)

        def classify(score):
            if score >= cutoff_expand:
                return "Expand"
            elif score <= cutoff_delist:
                return "Delist"
            else:
                return "Retain"

        df['Recommendation'] = df['Score'].apply(classify)

        def justify(rec):
            if rec == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif rec == "Delist":
                return "Low performance → candidate for phase-out."
            else:
                return "Balanced performance → maintain current space."

        df['Justification'] = df['Recommendation'].apply(justify)

        # --- Summary counts ---
        total_skus = len(df)
        num_expand = int((df['Recommendation']=='Expand').sum())
        num_retain = int((df['Recommendation']=='Retain').sum())
        num_delist = int((df['Recommendation']=='Delist').sum())

        st.subheader("📊 SKU Recommendation Summary")
        st.write(f"Total SKUs uploaded: {total_skus}")
        st.write(f"Expand SKUs: {num_expand}  |  Retain SKUs: {num_retain}  |  Delist SKUs: {num_delist}")

        # --- Sidebar settings for module 1 ---
        st.sidebar.header("⚙️ Settings (Module 1)")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)

        max_show = max(1, total_skus)
        default_show = min(50, max_show)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 1, max_show, default_show)

        total_shelf_space = shelf_width * num_layers

        # --- Suggested base facings (function with body fixed) ---
        def base_facings(rec):
            if rec == "Expand":
                return max(expand_facings, min_facings)
            elif rec == "Retain":
                return max(retain_facings, min_facings)
            else:
                return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        # --- Width handling ---
        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches) (used if Width missing)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width
        else:
            df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(st.sidebar.number_input("Default SKU Width (inches) (fallback)", 1.0, 100.0, 5.0, 0.1))

        # --- Initial space needed ---
        df['Space Needed'] = df['Width'] * df['Base Facings']

        # --- Redistribute freed-up space if delist facings == 0 ---
        if delist_facings == 0:
            delist_mask = df['Recommendation'] == 'Delist'
            freed_space = (df.loc[delist_mask, 'Width'] * base_facings('Delist')).sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand', 'Retain'])
            denom = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
            if denom > 0 and freed_space > 0:
                extra_facings = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / denom * freed_space) / df.loc[expand_retain_mask, 'Width']
                df.loc[expand_retain_mask, 'Extra Facings'] = extra_facings.fillna(0)
            else:
                df['Extra Facings'] = 0
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        # Filter for calculations & chart
        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"].copy()
        else:
            df_filtered = df.copy()

        # --- Recalculate total space used ---
        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

        # --- Show detailed table ---
        st.subheader("📋 SKU Recommendations & Performance Rank")
        display_cols = ['SKU', 'Score', 'Rank', 'Recommendation', 'Justification', 'Suggested Facings', 'Space Needed']
        if 'Store Code' in df.columns:
            display_cols.insert(1, 'Store Code')
        st.dataframe(df[display_cols].style.applymap(lambda v: 'background-color: #c6efce' if v=="Expand" else ('background-color: #ffc7ce' if v=="Delist" else ('background-color: #ffeb9c' if v=="Retain" else '')), subset=['Recommendation']), use_container_width=True)

        # --- Shelf Space Usage ---
        st.subheader("📊 Shelf Space Usage")
        st.write("**Explanation:** Total shelf space needed versus available across all layers. Over 100% indicates overcapacity.")
        st.progress(min(space_usage_pct/100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space:.1f} in ({space_usage_pct:.1f}%)")

        if space_usage_pct > 100:
            over_inch = total_space_used - total_shelf_space
            df_sorted = df_filtered.sort_values(by=['Space Needed','Score'], ascending=[False, True])
            cum_space = 0.0
            num_skus_to_remove = 0
            skus_to_remove = []
            for _, row in df_sorted.iterrows():
                cum_space += float(row['Space Needed'])
                num_skus_to_remove += 1
                skus_to_remove.append(str(row['SKU']))
                if cum_space >= over_inch:
                    break
            st.text(
                f"⚠️ Shelf space is full!\n"
                f"You may need to remove {num_skus_to_remove} SKU(s) or reduce facings.\n"
                f"Suggested SKUs to remove based on space and performance:\n- " + "\n- ".join(skus_to_remove)
            )
        else:
            st.success("✅ Your shelf plan fits within the available space.")

        # --- Top SKUs by Space Needed chart ---
        st.subheader("📊 Top SKUs by Space Needed")
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus).copy()
        df_chart['SKU_Label'] = df_chart['SKU']
        fig = px.bar(
            df_chart,
            y='SKU_Label',
            x='Space Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space Needed':':.1f','Width':':.1f','Suggested Facings':True,'Justification':True},
            color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

        # --- Download results ---
        st.subheader("⬇️ Download Results")
        out_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", out_csv, "SKU_Recommendations.csv", "text/csv")


# ----------------------------
# MODULE 2: Sales Analysis (qualitative + insights matching)
# ----------------------------
elif module == "2️⃣ Sales Analysis":
    st.title("📈 Sales Analysis with Qualitative Insights")
    st.markdown("Upload a Sales CSV with columns: `Date` (YYYY-MM-DD), `Store Code`, `Sales`. Optional: `SKU`.")

    sales_file = st.file_uploader("📂 Upload Sales CSV", type=["csv"], key="sales_upload")
    if sales_file:
        sales_df = pd.read_csv(sales_file)

        # Validate columns
        required_sales_cols = ["Date", "Store Code", "Sales"]
        missing = [c for c in required_sales_cols if c not in sales_df.columns]
        if missing:
            st.error(f"Missing required columns in Sales CSV: {', '.join(missing)}")
            st.stop()

        # Parse Date
        sales_df['Date_parsed'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        if sales_df['Date_parsed'].isna().any():
            st.error("Some Date values could not be parsed. Use YYYY-MM-DD format.")
            st.stop()
        sales_df['Date'] = sales_df['Date_parsed'].dt.strftime("%Y-%m-%d")
        sales_df['Sales'] = pd.to_numeric(sales_df['Sales'], errors='coerce').fillna(0)

        # Sidebar controls for sales analysis
        st.sidebar.header("⚙️ Sales Analysis Settings")
        pct_threshold = st.sidebar.slider("Lift/Drop threshold (%)", 5, 200, 30, 5)
        match_window = st.sidebar.slider("Insight match window (days ±N)", 0, 7, 0, 1)

        # Baseline: average sales per store across uploaded period
        baseline = sales_df.groupby('Store Code', as_index=False)['Sales'].mean().rename(columns={'Sales':'BaselineSales'})
        merged = pd.merge(sales_df, baseline, on='Store Code', how='left')

        # Compute change percent (handle zero baseline)
        def safe_pct_change(sales, base):
            if pd.isna(base) or base == 0:
                return np.nan
            return (sales - base) / base * 100.0

        merged['ChangePct'] = merged.apply(lambda r: safe_pct_change(r['Sales'], r['BaselineSales']), axis=1)

        def classify_signal(pct):
            if pd.isna(pct):
                return 'Lift' if pct is np.nan and (pct is np.nan) else 'Normal'  # keep as Normal for baseline 0 handled below
            if pct >= pct_threshold:
                return 'Lift'
            if pct <= -pct_threshold:
                return 'Drop'
            return 'Normal'

        # If baseline is zero and sales > 0 => Lift
        merged['Signal'] = merged.apply(lambda r: ('Lift' if (r['BaselineSales'] == 0 and r['Sales'] > 0) else classify_signal(r['ChangePct'])), axis=1)

        # Load approved insights
        insights_df = load_insights()
        if 'Status' in insights_df.columns:
            approved = insights_df[insights_df['Status'].str.lower() == 'approved'].copy()
        else:
            approved = pd.DataFrame(columns=insights_df.columns)

        # Normalize approved dates
        if not approved.empty and 'Date' in approved.columns:
            approved['Date_parsed'] = pd.to_datetime(approved['Date'], errors='coerce')

        # Match insights within ±match_window days
        def find_matching_insights(row):
            if approved.empty:
                return ""
            row_date = pd.to_datetime(row['Date'])
            store = str(row['Store Code'])
            # find approved rows with matching store and date within window
            mask = (approved['Store Code'].astype(str) == store)
            if match_window == 0:
                mask = mask & (approved['Date'] == row['Date'])
            else:
                low = row_date - pd.Timedelta(days=match_window)
                high = row_date + pd.Timedelta(days=match_window)
                mask = mask & (approved['Date_parsed'] >= low) & (approved['Date_parsed'] <= high)
            matched = approved[mask]
            if matched.empty:
                return ""
            # join notes
            return " ; ".join(matched['Note'].astype(str).tolist())

        merged['MatchedInsight'] = merged.apply(find_matching_insights, axis=1)

        # System Analysis & Qualitative Analysis templates
        def system_analysis_text(row):
            pct = row['ChangePct']
            signal = row['Signal']
            insight = row['MatchedInsight']
            if signal == 'Lift':
                if pd.isna(pct):
                    pct_label = "a big increase"
                else:
                    pct_label = f"{pct:.0f}% higher"
                if insight:
                    return f"Sales {pct_label} vs baseline — likely related to approved insight(s): {insight}."
                else:
                    return f"Sales {pct_label} vs baseline — no approved insight found; possible increased foot traffic, promo, or external event."
            elif signal == 'Drop':
                if pd.isna(pct):
                    pct_label = "a big drop"
                else:
                    pct_label = f"{abs(pct):.0f}% lower"
                if insight:
                    return f"Sales {pct_label} vs baseline — possibly due to approved insight(s): {insight}."
                else:
                    return f"Sales {pct_label} vs baseline — no approved insight found; possible low foot traffic (weather, disruption) or stockouts."
            else:
                return "Sales within normal range."

        def qualitative_text(row):
            signal = row['Signal']
            pct = row['ChangePct']
            insight = row['MatchedInsight']
            # More nuanced wording based on magnitude
            mag = 0 if pd.isna(pct) else abs(pct)
            if signal == 'Lift':
                if mag >= 100:
                    degree = "an extreme surge"
                elif mag >= 50:
                    degree = "a strong uplift"
                else:
                    degree = "a moderate lift"
                if insight:
                    return f"{degree} ({pct:.0f}%) likely caused by: {insight}. Recommend stocking fast movers and increasing staffing during similar events."
                else:
                    return f"{degree} ({pct:.0f}%) with no matched insight. Consider checking promos, local events, or competitor activity. Review inventory for hot SKUs."
            elif signal == 'Drop':
                if mag >= 100:
                    degree = "a severe decline"
                elif mag >= 50:
                    degree = "a significant drop"
                else:
                    degree = "a moderate dip"
                if insight:
                    return f"{degree} ({abs(pct):.0f}%) possibly caused by: {insight}. Recommend investigating operations, supply, and local conditions."
                else:
                    return f"{degree} ({abs(pct):.0f}%) with no matched insight. Investigate foot traffic, stockouts, or adverse weather."
            else:
                return "No significant change vs baseline."

        merged['System Analysis'] = merged.apply(system_analysis_text, axis=1)
        merged['Qualitative Analysis'] = merged.apply(qualitative_text, axis=1)

        # Summary counts and top reasons
        num_lifts = int((merged['Signal']=='Lift').sum())
        num_drops = int((merged['Signal']=='Drop').sum())
        st.subheader("Summary")
        st.write(f"Lift days: {num_lifts}  |  Drop days: {num_drops}")

        # Add a quick 'top reasons' by scraping matched insights
        reason_series = merged[merged['MatchedInsight'] != ""]['MatchedInsight'].str.split(" ; ").explode().value_counts()
        top_reasons = reason_series.head(5).to_dict()
        if top_reasons:
            st.write("Top matched reasons (from approved insights):")
            for r, cnt in top_reasons.items():
                st.write(f"- {r} ({cnt} match(es))")

        # Display analysis table
        display_cols = ['Date','Store Code']
        if 'SKU' in merged.columns:
            display_cols.append('SKU')
        display_cols += ['Sales','BaselineSales','ChangePct','Signal','MatchedInsight','System Analysis','Qualitative Analysis']
        st.subheader("Sales Analysis (detailed)")
        st.dataframe(merged[display_cols].sort_values(['Store Code','Date_parsed'], ascending=[True, True]), use_container_width=True)

        # Store-level summary
        st.subheader("Store-level summary")
        for store, g in merged.groupby('Store Code'):
            lifts = (g['Signal']=='Lift').sum()
            drops = (g['Signal']=='Drop').sum()
            st.write(f"• Store {store}: {lifts} lift day(s), {drops} drop day(s).")
            if lifts > 0:
                st.write("  - Observations: Events or promos likely drive spikes; consider planning inventory/staff.")
            if drops > 0:
                st.write("  - Observations: Investigate local conditions, stockouts, or operational issues.")

        # Chart: trend for selected store
        st.subheader("Sales trend (select store)")
        stores = merged['Store Code'].unique().tolist()
        if stores:
            sel_store = st.selectbox("Select store", stores)
            store_df = merged[merged['Store Code'] == sel_store].sort_values('Date_parsed')
            if not store_df.empty:
                fig = px.line(store_df, x='Date_parsed', y='Sales', markers=True, title=f"Sales trend - Store {sel_store}")
                fig.add_scatter(x=store_df['Date_parsed'], y=store_df['BaselineSales'], mode='lines', name='Baseline (avg)')
                # mark matched insights
                annot = store_df[store_df['MatchedInsight'] != ""]
                if not annot.empty:
                    fig.add_scatter(x=annot['Date_parsed'], y=annot['Sales'], mode='markers', marker_symbol='x', marker_size=12, name='Matched Insight')
                st.plotly_chart(fig, use_container_width=True)

        # Download results
        st.subheader("⬇️ Download analysis results")
        out_csv = merged.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sales Analysis CSV", out_csv, "sales_analysis_with_qualitative.csv", "text/csv")


# ----------------------------
# MODULE 3: Submit Insight (persist to CSV)
# ----------------------------
elif module == "3️⃣ Submit Insight":
    st.title("📝 Submit Insight")
    st.markdown("Submit an insight (Date: YYYY-MM-DD). These will be stored as Pending until a manager approves.")

    with st.form("insight_form"):
        date_input = st.date_input("Date of event")
        store_code = st.text_input("Store Code")
        note = st.text_area("Insight / Note (e.g., 'Rally in front of store', 'Class suspended')")
        submitted_by = st.text_input("Submitted by (name)")
        submitted = st.form_submit_button("Submit Insight")

        if submitted:
            if not store_code or not note or not submitted_by:
                st.error("Please fill Store Code, Note, and Submitted by.")
            else:
                df_ins = load_insights()
                new = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Date": date_input.strftime("%Y-%m-%d"),
                    "Store Code": str(store_code),
                    "Note": note,
                    "Status": "Pending",
                    "Submitted By": submitted_by
                }
                df_ins = pd.concat([df_ins, pd.DataFrame([new])], ignore_index=True)
                save_insights(df_ins)
                st.success("Insight submitted and pending approval.")

    st.subheader("All Insights (latest first)")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)


# ----------------------------
# MODULE 4: Approve Insights (persist approvals)
# ----------------------------
elif module == "4️⃣ Approve Insights":
    st.title("✅ Approve / Reject Insights (Manager)")
    insights_df = load_insights()
    # Normalize Status values
    insights_df['Status'] = insights_df['Status'].astype(str)
    pending_idx = insights_df[insights_df['Status'].str.lower() == 'pending'].index.tolist()

    if not pending_idx:
        st.info("No pending insights for approval.")
    else:
        st.write("Pending insights:")
        for orig_idx in pending_idx:
            row = insights_df.loc[orig_idx]
            st.markdown("---")
            st.write(f"**Index:** {orig_idx}")
            st.write(f"**Date:** {row['Date']}")
            st.write(f"**Store Code:** {row['Store Code']}")
            st.write(f"**Note:** {row['Note']}")
            st.write(f"**Submitted By:** {row.get('Submitted By','-')} at {row.get('Timestamp','-')}")
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {orig_idx}", key=f"approve_{orig_idx}"):
                insights_df.at[orig_idx, 'Status'] = "Approved"
                save_insights(insights_df)
                st.success(f"Approved insight {orig_idx}")
                st.rerun()
            if col2.button(f"Reject {orig_idx}", key=f"reject_{orig_idx}"):
                insights_df.at[orig_idx, 'Status'] = "Rejected"
                save_insights(insights_df)
                st.warning(f"Rejected insight {orig_idx}")
                st.rerun()

    st.subheader("All insights (for reference)")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)
