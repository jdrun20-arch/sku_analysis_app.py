import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📊 App Navigation")
module = st.sidebar.radio(
    "Choose Module:",
    ["1️⃣ SKU Performance & Shelf Space", "2️⃣ Sales Analysis", "3️⃣ Submit Insights", "4️⃣ Approve Insights"]
)

# --- Session State for Insights ---
if "insights" not in st.session_state:
    st.session_state.insights = []  # store submitted insights as list of dicts
if "approved_insights" not in st.session_state:
    st.session_state.approved_insights = []  # store approved insights


# ============================================================
# MODULE 1: SKU PERFORMANCE & SHELF SPACE (RESTORED VERSION)
# ============================================================
if module == "1️⃣ SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimizer")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Performance Scoring ---
        df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
        df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
        df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
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
        total_skus = len(df)
        num_expand = len(df[df['Recommendation']=='Expand'])
        num_retain = len(df[df['Recommendation']=='Retain'])
        num_delist = len(df[df['Recommendation']=='Delist'])

        st.write(f"Total SKUs uploaded: {total_skus}")
        st.write(f"Expand SKUs: {num_expand}")
        st.write(f"Retain SKUs: {num_retain}")
        st.write(f"Delist SKUs: {num_delist}")

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

        # --- Suggested Facings ---
        def base_facings(rec):
            if rec == "Expand":
                return max(expand_facings, min_facings)
            elif rec == "Retain":
                return max(retain_facings, min_facings)
            else:
                return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        # --- Handle Width ---
        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width

        df['Space Needed'] = df['Width'] * df['Base Facings']

        # --- Redistribute freed-up space from Delist SKUs ---
        if delist_facings == 0:
            delist_space = (df['Recommendation'] == 'Delist') * df['Width'] * base_facings('Delist')
            freed_space = delist_space.sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand', 'Retain'])
            total_expand_retain_width = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
            df.loc[expand_retain_mask, 'Extra Facings'] = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / total_expand_retain_width * freed_space / df.loc[expand_retain_mask, 'Width']).fillna(0)
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"]
        else:
            df_filtered = df.copy()

        # --- Recalculate total space used ---
        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100

        # --- Detailed Results ---
        st.subheader("📋 SKU Recommendations & Performance Rank")
        st.write("**Explanation:** Each SKU's performance, recommended action, justification, suggested facings, and shelf space needed.")

        def color_table(val):
            if val == "Expand": return "background-color: #c6efce"
            elif val == "Delist": return "background-color: #ffc7ce"
            elif val == "Retain": return "background-color: #ffeb9c"
            return ""

        st.dataframe(df[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style.applymap(color_table, subset=['Recommendation']), use_container_width=True)

        # --- Shelf Space Usage ---
        st.subheader("📊 Shelf Space Usage")
        st.progress(min(space_usage_pct/100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

        if space_usage_pct > 100:
            over_inch = total_space_used - total_shelf_space
            df_sorted = df_filtered.sort_values(by=['Space Needed','Score'], ascending=[False, True])
            cum_space = 0
            num_skus_to_remove = 0
            skus_to_remove = []
            for _, row in df_sorted.iterrows():
                cum_space += row['Space Needed']
                num_skus_to_remove += 1
                skus_to_remove.append(row['SKU'])
                if cum_space >= over_inch:
                    break
            st.text(
                f"⚠️ Shelf space is full!\n"
                f"You may need to remove {num_skus_to_remove} SKU(s) or reduce facings.\n"
                f"Suggested SKUs to remove based on space and performance:\n- " + "\n- ".join(skus_to_remove)
            )
        else:
            st.success("✅ Your shelf plan fits within the available space.")

        # --- Top SKUs by Space Needed ---
        st.subheader("📊 Top SKUs by Space Needed")
        df_filtered['SKU_Label'] = df_filtered['SKU']
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
        fig = px.bar(
            df_chart,
            y='SKU_Label',
            x='Space Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space Needed': ':.1f','Width': ':.1f','Suggested Facings': True,'Justification': True},
            color_discrete_map={'Expand':'#4CAF50', 'Retain':'#FFC107', 'Delist':'#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODULE 2: SALES ANALYSIS
# ============================================================
elif module == "2️⃣ Sales Analysis":
    st.title("📈 Sales Analysis")
    uploaded_sales = st.file_uploader("📂 Upload Sales CSV (Date, Store_Code, Sales)", type=["csv"])
    if uploaded_sales:
        sales_df = pd.read_csv(uploaded_sales, parse_dates=['Date'])
        st.write("Uploaded Sales Data", sales_df.head())
        # Merge with approved insights
        if st.session_state.approved_insights:
            insights_df = pd.DataFrame(st.session_state.approved_insights)
            merged = pd.merge(sales_df, insights_df, on=['Store_Code','Date'], how='left')
            st.subheader("📊 Sales + Insights Analysis")
            st.write("Merged data (insights appear when available):")
            st.dataframe(merged)

# ============================================================
# MODULE 3: SUBMIT INSIGHTS
# ============================================================
elif module == "3️⃣ Submit Insights":
    st.title("📝 Submit Store Insights")
    store_code = st.text_input("Store Code")
    date = st.date_input("Date", value=datetime.today())
    note = st.text_area("Insight (e.g., 'Class suspension', 'Rally near store')")
    if st.button("Submit Insight"):
        st.session_state.insights.append({"Store_Code": store_code, "Date": pd.to_datetime(date), "Note": note, "Status": "Pending"})
        st.success("✅ Insight submitted for approval.")

    if st.session_state.insights:
        st.subheader("Pending Insights")
        st.dataframe(pd.DataFrame(st.session_state.insights))

# ============================================================
# MODULE 4: APPROVE INSIGHTS
# ============================================================
elif module == "4️⃣ Approve Insights":
    st.title("✅ Approve or Reject Insights")
    if st.session_state.insights:
        for i, insight in enumerate(st.session_state.insights):
            st.write(f"**Store:** {insight['Store_Code']} | **Date:** {insight['Date'].date()} | **Note:** {insight['Note']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve {i}"):
                    insight['Status'] = "Approved"
                    st.session_state.approved_insights.append(insight)
                    st.session_state.insights.pop(i)
                    st.experimental_rerun()
            with col2:
                if st.button(f"Reject {i}"):
                    st.session_state.insights.pop(i)
                    st.experimental_rerun()
    else:
        st.info("No pending insights for approval.")
