import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

st.set_page_config(layout="wide")
st.title("📊 Retail Performance Dashboard")

# ---- Setup Tabs ----
tabs = st.tabs([
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submission of Insights",
    "Approval of Insights"
])

# --- Persistent Data ---
if "insights" not in st.session_state:
    st.session_state["insights"] = pd.DataFrame(columns=["Date", "Store Code", "Insight", "Status"])

# ============================
# TAB 1: SKU Performance
# ============================
with tabs[0]:
    st.header("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"], key="sku_upload")

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

        # Justification
        def justify(row):
            if row['Recommendation'] == "Expand":
                return "High sales, volume, or margin → increase facings or distribution."
            elif row['Recommendation'] == "Delist":
                return "Low performance → candidate for phase-out."
            else:
                return "Balanced performance → maintain current space."

        df['Justification'] = df.apply(justify, axis=1)

        # --- Sidebar for SKU Space Settings ---
        st.sidebar.header("⚙️ SKU Space Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts", value=False)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, 100, 50, 5)

        total_shelf_space = shelf_width * num_layers

        # Suggested Facings
        def base_facings(rec):
            if rec == "Expand":
                return max(expand_facings, min_facings)
            elif rec == "Retain":
                return max(retain_facings, min_facings)
            else:
                return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        if 'Width' not in df.columns:
            df['Width'] = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)

        df['Space Needed'] = df['Width'] * df['Base Facings']

        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"]
        else:
            df_filtered = df.copy()

        # Shelf Space Calculation
        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used / total_shelf_space) * 100

        def color_table(val):
            if val == "Expand": return "background-color: #c6efce"
            elif val == "Delist": return "background-color: #ffc7ce"
            elif val == "Retain": return "background-color: #ffeb9c"
            return ""

        st.subheader("📋 SKU Recommendations")
        st.dataframe(
            df[['SKU','Score','Rank','Recommendation','Justification','Base Facings','Space Needed']]
            .style.applymap(color_table, subset=['Recommendation']),
            use_container_width=True
        )

        st.subheader("📊 Shelf Space Usage")
        st.progress(min(space_usage_pct / 100, 1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

        st.subheader("📊 Top SKUs by Space Needed")
        df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
        fig = px.bar(
            df_chart,
            y='SKU',
            x='Space Needed',
            color='Recommendation',
            orientation='h',
            hover_data={'Space Needed': ':.1f', 'Width': ':.1f'},
            color_discrete_map={'Expand': '#4CAF50', 'Retain': '#FFC107', 'Delist': '#F44336'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=25 * len(df_chart))
        st.plotly_chart(fig, use_container_width=True)

# ============================
# TAB 2: SALES ANALYSIS
# ============================
with tabs[1]:
    st.header("📈 Sales Analysis & Signals")
    uploaded_sales = st.file_uploader("📂 Upload Sales Data CSV", type=["csv"], key="sales_upload")
    if uploaded_sales:
        sales_df = pd.read_csv(uploaded_sales)
        if "Date" in sales_df.columns:
            sales_df['Date_parsed'] = pd.to_datetime(sales_df['Date'], errors='coerce')
            sales_df = sales_df.dropna(subset=['Date_parsed'])
            sales_df['Signal'] = sales_df['Sales'].pct_change().apply(
                lambda x: "LIFT" if x > 0.1 else ("DROP" if x < -0.1 else "STABLE")
            )
            sales_df['Qualitative Note'] = sales_df['Signal'].apply(
                lambda x: "Possible rally/holiday → high foot traffic" if x == "LIFT"
                else ("Possible storm or external factor → low foot traffic" if x == "DROP" else "")
            )

            def color_signal(val):
                if val == "LIFT": return "background-color: #c6efce"
                elif val == "DROP": return "background-color: #ffc7ce"
                return ""

            def italicize(val):
                return "font-style: italic;" if isinstance(val, str) and val else ""

            styled_df = sales_df.style.applymap(color_signal, subset=['Signal']).applymap(
                italicize, subset=['Qualitative Note']
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No 'Date' column found — cannot run analysis.")

# ============================
# TAB 3: SUBMISSION OF INSIGHTS
# ============================
with tabs[2]:
    st.header("📝 Submit New Insight")
    date_input = st.date_input("Date of Event", value=datetime.date.today())
    store_code = st.text_input("Store Code")
    insight_text = st.text_area("Insight / Observation")
    if st.button("Submit Insight"):
        if insight_text.strip():
            new_row = pd.DataFrame([{"Date": date_input, "Store Code": store_code, "Insight": insight_text, "Status": "Pending"}])
            st.session_state["insights"] = pd.concat([st.session_state["insights"], new_row], ignore_index=True)
            st.success("Insight submitted for approval.")
        else:
            st.error("Please enter some text before submitting.")

# ============================
# TAB 4: APPROVAL OF INSIGHTS
# ============================
with tabs[3]:
    st.header("✅ Approve Insights")
    pending = st.session_state["insights"][st.session_state["insights"]["Status"] == "Pending"]
    if not pending.empty:
        for i, row in pending.iterrows():
            st.write(f"📅 {row['Date']} | 🏬 {row['Store Code']} | ✍️ {row['Insight']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve {i}"):
                    st.session_state["insights"].loc[i, "Status"] = "Approved"
                    st.success("Insight approved.")
            with col2:
                if st.button(f"Reject {i}"):
                    st.session_state["insights"].loc[i, "Status"] = "Rejected"
                    st.warning("Insight rejected.")
    else:
        st.info("No pending insights for approval.")
