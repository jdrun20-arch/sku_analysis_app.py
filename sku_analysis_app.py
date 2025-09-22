import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Retail Analytics App", layout="wide")

st.title("🛒 Retail Analytics App")

# --- SIDEBAR MODULE SELECTOR ---
module = st.sidebar.radio("Select Module", [
    "1️⃣ SKU Performance & Shelf Space",
    "2️⃣ Sales Analysis",
    "3️⃣ Submit Insights",
    "4️⃣ Approve Insights"
])

# --- MODULE 1: SKU PERFORMANCE & SHELF SPACE ---
if module == "1️⃣ SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space")
    st.write("Upload a CSV file with columns: SKU, Sales, Volume, Margin, Width")

    uploaded_file = st.file_uploader("Upload SKU CSV", type="csv", key="sku")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        expected_cols = {"SKU", "Sales", "Volume", "Margin", "Width"}
        if not expected_cols.issubset(df.columns):
            st.error(f"Your file must contain the following columns: {', '.join(expected_cols)}")
        else:
            # Normalize
            df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
            df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
            df['Margin_Norm'] = df['Margin'] / df['Margin'].max()

            df['Score'] = (df['Sales_Norm']*0.3)+(df['Volume_Norm']*0.3)+(df['Margin_Norm']*0.4)
            cutoff_expand = df['Score'].quantile(0.7)
            cutoff_delist = df['Score'].quantile(0.3)

            def classify(score):
                if score >= cutoff_expand:
                    return "Expand"
                elif score <= cutoff_delist:
                    return "Delist"
                else:
                    return "Retain"

            df['Recommendation'] = df['Score'].apply(classify)
            df['Suggested Facings'] = df['Recommendation'].map({"Expand": 3, "Retain": 2, "Delist": 0})
            df['Space Needed'] = df['Width'] * df['Suggested Facings']

            st.subheader("Detailed Results")
            st.dataframe(df)

            st.subheader("Top SKUs by Space Needed")
            df_sorted = df.sort_values("Space Needed", ascending=False).head(20)
            fig = px.bar(df_sorted, x="Space Needed", y="SKU", orientation="h",
                         hover_data=["Width", "Suggested Facings"], title="Top SKUs by Space Needed")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "SKU_Recommendations.csv", "text/csv")

# --- MODULE 2: SALES ANALYSIS ---
elif module == "2️⃣ Sales Analysis":
    st.header("📈 Sales Analysis")
    st.write("Upload a CSV file with columns: Date, Store Code, Sales")

    sales_file = st.file_uploader("Upload Sales CSV", type="csv", key="sales")

    if sales_file is not None:
        df_sales = pd.read_csv(sales_file, parse_dates=["Date"])

        expected_cols = {"Date", "Store Code", "Sales"}
        if not expected_cols.issubset(df_sales.columns):
            st.error(f"Your file must contain the following columns: {', '.join(expected_cols)}")
        else:
            st.subheader("Sales Over Time")
            fig = px.line(df_sales, x="Date", y="Sales", color="Store Code", title="Sales Trend by Store")
            st.plotly_chart(fig, use_container_width=True)

# --- MODULE 3: SUBMIT INSIGHTS ---
elif module == "3️⃣ Submit Insights":
    st.header("📝 Submit Store Insights")
    with st.form("insight_form"):
        store_code = st.text_input("Store Code")
        date = st.date_input("Date")
        insight = st.text_area("Insight (e.g., rally, weather, class suspension)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("✅ Insight submitted and pending manager approval.")

# --- MODULE 4: APPROVE INSIGHTS ---
elif module == "4️⃣ Approve Insights":
    st.header("✅ Approve Insights")
    st.info("This section would list pending insights for approval (needs DB integration)")
    st.write("For prototype purposes, manager can review submissions manually.")
