import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Retail Optimization App")

# --- Persistent Insights Storage ---
INSIGHTS_FILE = "approved_insights.csv"
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Date", "Store Code", "Insight", "Status"]).to_csv(INSIGHTS_FILE, index=False)

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Select Module", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insights",
    "Approve Insights"
])

# =========================
# 1️⃣ SKU PERFORMANCE MODULE
# =========================
if page == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("📂 Upload your SKU Performance CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Performance Scoring ---
        df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

        # --- Ranking & Recommendations ---
        df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)
        cutoff_expand = df['Score'].quantile(0.70)
        cutoff_delist = df['Score'].quantile(0.30)

        def classify(score):
            if score >= cutoff_expand: return "Expand"
            elif score <= cutoff_delist: return "Delist"
            return "Retain"

        df['Recommendation'] = df['Score'].apply(classify)

        def justify(row):
            if row['Recommendation'] == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif row['Recommendation'] == "Delist":
                return "Low performance → candidate for phase-out."
            return "Balanced performance → maintain current space."

        df['Justification'] = df.apply(justify, axis=1)

        # --- Summary ---
        st.subheader("📊 SKU Recommendation Summary")
        st.write(f"Total SKUs: {len(df)} | Expand: {len(df[df['Recommendation']=='Expand'])} | "
                 f"Retain: {len(df[df['Recommendation']=='Retain'])} | "
                 f"Delist: {len(df[df['Recommendation']=='Delist'])}")

        # --- Sidebar Settings ---
        st.sidebar.header("⚙️ Shelf Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts", value=False)
        top_n_skus = st.sidebar.slider("Top SKUs to show", 10, 100, 50, 5)

        total_shelf_space = shelf_width * num_layers

        # --- Facings Logic ---
        def base_facings(rec):
            if rec == "Expand": return max(expand_facings, min_facings)
            elif rec == "Retain": return max(retain_facings, min_facings)
            return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width

        df['Space Needed'] = df['Width'] * df['Base Facings']

        if delist_facings == 0:
            freed_space = df.loc[df['Recommendation']=="Delist", 'Space Needed'].sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand','Retain'])
            total_expand_retain_width = df.loc[expand_retain_mask, 'Space Needed'].sum()
            df.loc[expand_retain_mask, 'Extra Facings'] = (
                df.loc[expand_retain_mask, 'Space Needed'] / total_expand_retain_width * freed_space / df.loc[expand_retain_mask, 'Width']
            ).fillna(0)
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        if hide_delist:
            df_filtered = df[df['Recommendation']!="Delist"]
        else:
            df_filtered = df.copy()

        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100

        # --- Results Table ---
        st.subheader("📋 SKU Recommendations & Performance Rank")
        st.dataframe(df[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style.applymap(
            lambda x: "background-color: #c6efce" if x=="Expand" else 
                      "background-color: #ffc7ce" if x=="Delist" else 
                      "background-color: #ffeb9c" if x=="Retain" else "",
            subset=['Recommendation']
        ), use_container_width=True)

        # --- Shelf Space Usage ---
        st.subheader("📊 Shelf Space Usage")
        st.progress(min(space_usage_pct/100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

        # --- Top SKUs Chart ---
        st.subheader("📊 Top SKUs by Space Needed")
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
        fig = px.bar(df_chart, y='SKU', x='Space Needed', color='Recommendation', orientation='h',
                     hover_data={'Space Needed':':.1f','Width':':.1f','Suggested Facings':True,'Justification':True},
                     color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 2️⃣ SALES ANALYSIS MODULE
# =========================
elif page == "Sales Analysis":
    st.header("📈 Sales Analysis & Insight Matching")
    sales_file = st.file_uploader("📂 Upload your Sales CSV", type=["csv"])
    if sales_file is not None:
        sales_df = pd.read_csv(sales_file)
        sales_df['Date_parsed'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        sales_df = sales_df.sort_values(['Store Code','Date_parsed'])

        # --- Filters ---
        store_codes = sales_df['Store Code'].unique().tolist()
        selected_stores = st.multiselect("🏬 Select Store Code(s)", store_codes, default=store_codes)
        min_date, max_date = sales_df['Date_parsed'].min(), sales_df['Date_parsed'].max()
        date_range = st.date_input("📅 Select Date Range", [min_date, max_date])
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

        filtered_df = sales_df[(sales_df['Store Code'].isin(selected_stores)) &
                               (sales_df['Date_parsed'].between(start_date, end_date))]

        # --- Signal Detection ---
        avg_sales = filtered_df['Sales'].mean()
        filtered_df['Signal'] = filtered_df['Sales'].apply(
            lambda x: "⬆️ High Spike" if x > avg_sales*1.5 else
                      ("⬇️ Significant Drop" if x < avg_sales*0.5 else "Normal")
        )

        # --- Merge with Approved Insights ---
        approved_insights = pd.read_csv(INSIGHTS_FILE)
        approved_insights = approved_insights[approved_insights['Status']=="Approved"]
        approved_insights['Date'] = pd.to_datetime(approved_insights['Date'], errors='coerce')
        merged = pd.merge(filtered_df, approved_insights, how='left',
                          left_on=['Store Code','Date_parsed'],
                          right_on=['Store Code','Date'])
        merged['Qualitative Note'] = merged.apply(
            lambda row: f"*{row['Insight']}*" if pd.notnull(row['Insight']) else "", axis=1
        )

        # --- Styled Table ---
        def highlight_signal(val):
            if "High" in str(val): return "background-color: #c6efce"
            elif "Drop" in str(val): return "background-color: #ffc7ce"
            return ""

        display_cols = ['Store Code','Date_parsed','Sales','Signal','Qualitative Note']
        st.dataframe(merged[display_cols].style.applymap(highlight_signal, subset=['Signal']), use_container_width=True)

        # --- Narrative Generation ---
        st.subheader("📝 Automated Narrative")
        spike_days = merged[merged['Signal']=="⬆️ High Spike"]
        drop_days = merged[merged['Signal']=="⬇️ Significant Drop"]

        if not spike_days.empty or not drop_days.empty:
            narrative = []
            for _, row in spike_days.iterrows():
                note = f"📈 **{row['Date_parsed'].date()}**: Sales spiked to ₱{row['Sales']:,.0f}."
                if row['Qualitative Note']:
                    note += f" Likely reason: {row['Qualitative Note']}"
                narrative.append(note)
            for _, row in drop_days.iterrows():
                note = f"📉 **{row['Date_parsed'].date()}**: Sales dropped to ₱{row['Sales']:,.0f}."
                if row['Qualitative Note']:
                    note += f" Possible cause: {row['Qualitative Note']}"
                narrative.append(note)
            st.markdown("\n".join(narrative))
        else:
            st.info("No unusual sales patterns detected in the selected period.")

        # --- Chart ---
        fig = px.line(merged, x='Date_parsed', y='Sales', color='Store Code',
                      title="📈 Sales Trend with Insights")
        fig.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 3️⃣ SUBMIT INSIGHTS MODULE
# =========================
elif page == "Submit Insights":
    st.header("📝 Submit New Insights")
    with st.form("submit_form"):
        date = st.date_input("📅 Date")
        store_code = st.text_input("🏬 Store Code")
        insight = st.text_area("💡 Insight (e.g., Rally near store, class suspension, typhoon)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            df = pd.read_csv(INSIGHTS_FILE)
            new_row = pd.DataFrame([{"Date": date, "Store Code": store_code, "Insight": insight, "Status": "Pending"}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(INSIGHTS_FILE, index=False)
            st.success("✅ Insight submitted for approval!")

# =========================
# 4️⃣ APPROVE INSIGHTS MODULE
# =========================
elif page == "Approve Insights":
    st.header("✅ Approve or Reject Insights")
    df = pd.read_csv(INSIGHTS_FILE)
    pending = df[df['Status']=="Pending"]
    if not pending.empty:
        for idx, row in pending.iterrows():
            st.write(f"📅 {row['Date']} | 🏬 {row['Store Code']} | 💡 {row['Insight']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve {idx}"):
                    df.loc[idx,'Status']="Approved"
                    df.to_csv(INSIGHTS_FILE, index=False)
                    st.experimental_rerun()
            with col2:
                if st.button(f"Reject {idx}"):
                    df.loc[idx,'Status']="Rejected"
                    df.to_csv(INSIGHTS_FILE, index=False)
                    st.experimental_rerun()
    else:
        st.info("No pending insights for approval.")
