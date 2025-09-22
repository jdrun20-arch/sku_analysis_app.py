# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# ----------------------------
# Config
# ----------------------------
st.set_page_config(layout="wide", page_title="Retail Assortment + Sales Insights")
INSIGHTS_CSV = "insights.csv"

# ensure insights csv exists and load it
if not os.path.exists(INSIGHTS_CSV):
    pd.DataFrame(columns=["Timestamp", "Date", "Store Code", "Note", "Status", "Submitted By"]).to_csv(INSIGHTS_CSV, index=False)

def load_insights():
    df = pd.read_csv(INSIGHTS_CSV)
    # Ensure Date is string (we'll parse when needed)
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    return df

def save_insights(df):
    df.to_csv(INSIGHTS_CSV, index=False)

# load insights into session state for convenience
if "insights_df" not in st.session_state:
    st.session_state.insights_df = load_insights()

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

        # --- Performance Scoring ---
        # safe numeric conversions
        for c in ["Sales", "Volume", "Margin"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df['Sales_Norm'] = df['Sales'] / df['Sales'].replace(0, pd.NA).max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].replace(0, pd.NA).max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].replace(0, pd.NA).max()
        df[['Sales_Norm','Volume_Norm','Margin_Norm']] = df[['Sales_Norm','Volume_Norm','Margin_Norm']].fillna(0)
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

        # --- SKU Performance Ranking ---
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

        # --- Justification Column ---
        def justify(rec):
            if rec == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif rec == "Delist":
                return "Low performance → candidate for phase-out."
            else:
                return "Balanced performance → maintain current space."
        df['Justification'] = df['Recommendation'].apply(justify)

        # --- SKU Recommendation Summary ---
        st.subheader("📊 SKU Recommendation Summary")
        total_skus = len(df)
        num_expand = int((df['Recommendation']=='Expand').sum())
        num_retain = int((df['Recommendation']=='Retain').sum())
        num_delist = int((df['Recommendation']=='Delist').sum())
        st.write(f"Total SKUs uploaded: {total_skus}")
        st.write(f"Expand SKUs: {num_expand}")
        st.write(f"Retain SKUs: {num_retain}")
        st.write(f"Delist SKUs: {num_delist}")

        # --- Sidebar Settings (module-specific) ---
        st.sidebar.header("⚙️ Settings (Module 1)")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, max(10, total_skus), min(50, max(10, total_skus)), 5)

        total_shelf_space = shelf_width * num_layers

        # --- Suggested Facings (base) ---
        def base_facings(rec):
            if rec == "Expand":
                return max(expand_facings, min_facings)
            elif rec == "Retain":
                return max(retain_facings, min_facings)
            else:
                return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        # --- Handle Width (use Width column if present, otherwise prompt default) ---
        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches) (used if Width column missing)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width
        else:
            df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(st.sidebar.number_input("Default SKU Width (inches) (fallback)", 1.0, 100.0, 5.0, 0.1))

        # --- Initial space needed by base facings ---
        df['Space Needed'] = df['Width'] * df['Base Facings']

        # --- Redistribute freed-up space from Delist SKUs if delist_facings == 0 ---
        if delist_facings == 0:
            delist_mask = df['Recommendation'] == 'Delist'
            # space freed (in inches) if delisted facings set to 0
            freed_space = (df.loc[delist_mask, 'Width'] * base_facings('Delist')).sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand', 'Retain'])
            denom = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
            if denom > 0 and freed_space > 0:
                # distribute freed inches proportionally by current width*basefacings, then convert back to facings for each SKU
                extra_facings = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / denom * freed_space) / df.loc[expand_retain_mask, 'Width']
                df.loc[expand_retain_mask, 'Extra Facings'] = extra_facings.fillna(0)
            else:
                df['Extra Facings'] = 0
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        # optionally hide delist SKUs from calculations and charts
        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"].copy()
        else:
            df_filtered = df.copy()

        # --- Recalculate total space used ---
        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

        # --- Detailed Results display ---
        st.subheader("📋 SKU Recommendations & Performance Rank")
        st.write("**Explanation:** Each SKU's performance, recommended action, justification, suggested facings, and shelf space needed.")
        def color_table(val):
            if val == "Expand": return "background-color: #c6efce"
            elif val == "Delist": return "background-color: #ffc7ce"
            elif val == "Retain": return "background-color: #ffeb9c"
            return ""
        display_cols = ['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']
        # if Store Code present include it for clarity
        if 'Store Code' in df.columns:
            display_cols.insert(1, 'Store Code')
        st.dataframe(df[display_cols].style.applymap(color_table, subset=['Recommendation']), use_container_width=True)

        # --- Shelf Space Usage ---
        st.subheader("📊 Shelf Space Usage")
        st.write("**Explanation:** Total shelf space needed versus available across all layers. Over 100% indicates overcapacity.")
        st.progress(min(space_usage_pct/100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space:.1f} in ({space_usage_pct:.1f}%)")

        # --- Actionable Message: how many SKUs to remove to fit ---
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

        # --- Top SKUs by Space Needed chart (with hover details) ---
        st.subheader("📊 Top SKUs by Space Needed")
        st.write("**Explanation:** This chart shows which SKUs take up the most shelf space. Hover to see item width and suggested facings.")
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
        # create SKU_Label for display
        df_chart['SKU_Label'] = df_chart['SKU']
        fig = px.bar(
            df_chart,
            y='SKU_Label',
            x='Space Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space Needed': ':.1f', 'Width': ':.1f', 'Suggested Facings': True, 'Justification': True},
            color_discrete_map={'Expand':'#4CAF50', 'Retain':'#FFC107', 'Delist':'#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

        # --- Download results as CSV ---
        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "SKU_Recommendations.csv", "text/csv")

# ----------------------------
# MODULE 2: Sales Analysis
# ----------------------------
elif module == "2️⃣ Sales Analysis":
    st.title("📈 Sales Analysis")
    st.markdown("Upload a Sales CSV with columns: `Date` (YYYY-MM-DD), `Store Code`, `Sales` (numeric).")
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
            st.error("Some Date values could not be parsed. Ensure Date column uses YYYY-MM-DD format.")
            st.stop()

        sales_df['Date'] = sales_df['Date_parsed'].dt.strftime("%Y-%m-%d")
        sales_df['Sales'] = pd.to_numeric(sales_df['Sales'], errors='coerce').fillna(0)

        st.subheader("Uploaded Sales (sample)")
        st.dataframe(sales_df.head(), use_container_width=True)

        # Load approved insights from disk
        insights_df = load_insights()
        approved = insights_df[insights_df['Status'] == "Approved"].copy()
        # ensure Date column consistent
        if 'Date' in approved.columns:
            approved['Date'] = approved['Date'].astype(str)

        # Merge sales with approved insights on Date + Store Code
        merged = pd.merge(sales_df, approved[['Date','Store Code','Note']], on=['Date','Store Code'], how='left')
        merged['Note'] = merged['Note'].fillna("-")

        st.subheader("Sales merged with approved insights (if any)")
        st.dataframe(merged, use_container_width=True)

        # quick chart: aggregated by date for chosen store
        st.subheader("Sales Trend by Store")
        stores = merged['Store Code'].unique().tolist()
        store_sel = st.selectbox("Select Store to view", options=stores)
        df_store = merged[merged['Store Code'] == store_sel].sort_values('Date_parsed')
        if not df_store.empty:
            fig = px.line(df_store, x='Date_parsed', y='Sales', title=f"Sales trend - Store {store_sel}")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# MODULE 3: Submit Insight
# ----------------------------
elif module == "3️⃣ Submit Insight":
    st.title("📝 Submit Insight")
    st.markdown("Submit an insight (Date format: YYYY-MM-DD). These go into the queue for manager approval.")

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
                # append to CSV and to session state
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
                st.session_state.insights_df = df_ins
                st.success("Insight submitted and pending approval.")

    st.subheader("All Insights (latest first)")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)

# ----------------------------
# MODULE 4: Approve Insights
# ----------------------------
elif module == "4️⃣ Approve Insights":
    st.title("✅ Approve / Reject Insights (Manager)")
    insights_df = load_insights()
    pending = insights_df[insights_df['Status'].str.lower() == "pending"]

    if pending.empty:
        st.info("No pending insights for approval.")
    else:
        st.write("Pending insights:")
        # display each pending row with Approve/Reject buttons (use unique keys)
        for idx, row in pending.reset_index().iterrows():
            original_index = row['index']  # original index in insights_df
            st.markdown("---")
            st.write(f"**Index:** {original_index}")
            st.write(f"**Date:** {row['Date']}")
            st.write(f"**Store Code:** {row['Store Code']}")
            st.write(f"**Note:** {row['Note']}")
            st.write(f"**Submitted By:** {row.get('Submitted By','-')} at {row.get('Timestamp','-')}")
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {original_index}", key=f"approve_{original_index}"):
                insights_df.at[original_index, 'Status'] = "Approved"
                save_insights(insights_df)
                st.success(f"Approved insight {original_index}")
                st.rerun()
            if col2.button(f"Reject {original_index}", key=f"reject_{original_index}"):
                # mark rejected or drop — we'll mark as Rejected to keep record
                insights_df.at[original_index, 'Status'] = "Rejected"
                save_insights(insights_df)
                st.warning(f"Rejected insight {original_index}")
                st.rerun()

    st.subheader("All insights (for reference)")
    st.dataframe(load_insights().sort_values("Timestamp", ascending=False), use_container_width=True)
