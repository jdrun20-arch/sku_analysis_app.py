import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Retail Intelligence App")

# File to store approved insights
INSIGHTS_FILE = "approved_insights.csv"
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Date", "Store Code", "Insight", "Status"]).to_csv(INSIGHTS_FILE, index=False)

# --- MODULE NAVIGATION ---
module = st.sidebar.radio("📌 Select Module", ["SKU Performance & Shelf Space", "Sales Analysis & Insight Matching", "Submit Insight", "Approve Insights"])

# ==========================
# 1️⃣ SKU PERFORMANCE & SHELF SPACE
# ==========================
if module == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"], key="sku_csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Performance Scoring ---
        df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
        df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

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

        def justify(row):
            if row['Recommendation'] == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif row['Recommendation'] == "Delist":
                return "Low performance → candidate for phase-out."
            else:
                return "Balanced performance → maintain current space."

        df['Justification'] = df.apply(justify, axis=1)

        # --- SKU Recommendation Summary ---
        st.subheader("📊 SKU Recommendation Summary")
        st.write(f"Total SKUs uploaded: {len(df)}")
        st.write(f"Expand SKUs: {len(df[df['Recommendation']=='Expand'])}")
        st.write(f"Retain SKUs: {len(df[df['Recommendation']=='Retain'])}")
        st.write(f"Delist SKUs: {len(df[df['Recommendation']=='Delist'])}")

        # --- Sidebar Settings ---
        st.sidebar.header("⚙️ Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, 100, 50, 5)

        total_shelf_space = shelf_width * num_layers

        def base_facings(rec):
            if rec == "Expand":
                return max(expand_facings, min_facings)
            elif rec == "Retain":
                return max(retain_facings, min_facings)
            else:
                return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width

        df['Space Needed'] = df['Width'] * df['Base Facings']

        if delist_facings == 0:
            delist_space = (df['Recommendation'] == 'Delist') * df['Width'] * base_facings('Delist')
            freed_space = delist_space.sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand', 'Retain'])
            total_expand_retain_width = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
            df.loc[expand_retain_mask, 'Extra Facings'] = (
                df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / total_expand_retain_width * freed_space / df.loc[expand_retain_mask, 'Width']
            ).fillna(0)
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"]
        else:
            df_filtered = df.copy()

        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100

        st.subheader("📋 SKU Recommendations & Performance Rank")
        def color_table(val):
            if val == "Expand": return "background-color: #c6efce"
            elif val == "Delist": return "background-color: #ffc7ce"
            elif val == "Retain": return "background-color: #ffeb9c"
            return ""

        st.dataframe(df[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style.applymap(color_table, subset=['Recommendation']), use_container_width=True)

        st.subheader("📊 Shelf Space Usage")
        st.progress(min(space_usage_pct/100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

        if space_usage_pct > 100:
            st.warning("⚠️ Shelf space exceeds 100%! Consider delisting SKUs or reducing facings.")
        else:
            st.success("✅ Your shelf plan fits within the available space.")

        st.subheader("📊 Top SKUs by Space Needed")
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
        fig = px.bar(
            df_chart,
            y='SKU',
            x='Space Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space Needed': ':.1f','Width': ':.1f','Suggested Facings': True,'Justification': True},
            color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

# ==========================
# 2️⃣ SALES ANALYSIS & INSIGHT MATCHING
# ==========================
elif module == "Sales Analysis & Insight Matching":
    st.header("📈 Sales Analysis & Insight Matching")
    sales_file = st.file_uploader("📂 Upload Sales CSV", type=["csv"], key="sales_csv")

    if sales_file is not None:
        sales_df = pd.read_csv(sales_file)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        sales_df = sales_df.dropna(subset=['Date'])
        sales_df = sales_df.sort_values(by=['Store Code','Date'])

        avg_by_store = sales_df.groupby('Store Code')['Sales'].mean(numeric_only=True).to_dict()
        sales_df['Signal'] = sales_df.apply(
            lambda row: "🔺 High" if row['Sales'] > avg_by_store[row['Store Code']] * 1.5 else
                        ("🔻 Low" if row['Sales'] < avg_by_store[row['Store Code']] * 0.5 else "Normal"),
            axis=1
        )

        approved_insights = pd.read_csv(INSIGHTS_FILE)
        approved_insights = approved_insights[approved_insights['Status']=="Approved"]
        approved_insights['Date'] = pd.to_datetime(approved_insights['Date'], errors='coerce')

        merged = pd.merge(sales_df, approved_insights, how="left", on=["Date","Store Code"])
        merged['Qualitative Note'] = merged.apply(
            lambda row: f"*{row['Insight']}*" if pd.notna(row['Insight']) else "", axis=1
        )

        def color_signal(val):
            if "High" in val: return "background-color: #c6efce"
            elif "Low" in val: return "background-color: #ffc7ce"
            return ""

        st.dataframe(
            merged[['Store Code','Date','Sales','Signal','Qualitative Note']].style
            .applymap(color_signal, subset=['Signal'])
            .applymap(lambda x: "font-style: italic" if isinstance(x, str) and x.startswith("*") else "", subset=['Qualitative Note']),
            use_container_width=True
        )

# ==========================
# 3️⃣ SUBMIT INSIGHT
# ==========================
elif module == "Submit Insight":
    st.header("📝 Submit New Insight")
    date = st.date_input("📅 Select Date")
    store_code = st.text_input("🏬 Store Code")
    insight = st.text_area("💡 Insight (e.g., 'Rally near store increased traffic')")
    if st.button("📤 Submit Insight"):
        new = pd.DataFrame([[date, store_code, insight, "Pending"]], columns=["Date","Store Code","Insight","Status"])
        old = pd.read_csv(INSIGHTS_FILE)
        combined = pd.concat([old,new], ignore_index=True)
        combined.to_csv(INSIGHTS_FILE, index=False)
        st.success("✅ Insight submitted for approval.")

# ==========================
# 4️⃣ APPROVE INSIGHTS
# ==========================
elif module == "Approve Insights":
    st.header("✅ Approve or Reject Submitted Insights")
    insights_df = pd.read_csv(INSIGHTS_FILE)
    pending = insights_df[insights_df['Status']=="Pending"]

    if pending.empty:
        st.info("No pending insights.")
    else:
        for i, row in pending.iterrows():
            st.write(f"📅 {row['Date']} | 🏬 {row['Store Code']} | 💡 {row['Insight']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{i}"):
                    insights_df.loc[i, 'Status'] = "Approved"
                    insights_df.to_csv(INSIGHTS_FILE, index=False)
                    st.success("Insight approved.")
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{i}"):
                    insights_df.loc[i, 'Status'] = "Rejected"
                    insights_df.to_csv(INSIGHTS_FILE, index=False)
                    st.warning("Insight rejected.")
                    st.rerun()
