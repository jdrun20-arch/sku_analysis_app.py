import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Retail Insights App", layout="wide")

# --- Helper Functions ---
def load_csv(file):
    return pd.read_csv(file)

def parse_date_column(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df

# --- MODULE 1: SKU PERFORMANCE & SHELF SPACE ---
st.header("📊 SKU Performance & Shelf Space")
sku_file = st.file_uploader("Upload SKU Performance CSV", type=["csv"], key="sku")
if sku_file:
    sku_df = load_csv(sku_file)
    sku_df = parse_date_column(sku_df, 'Date')

    # Calculate recommended space (inches) based on facings * item size
    if {'Facings','Item Size (in)'} <= set(sku_df.columns):
        sku_df['Needed Space (in)'] = sku_df['Facings'] * sku_df['Item Size (in)']

    # Recommendation Logic
    avg_sales = sku_df.groupby('SKU')['Sales'].mean().to_dict()
    sku_df['Recommendation'] = sku_df['SKU'].apply(
        lambda x: 'Expand' if avg_sales[x] > sku_df['Sales'].mean()*1.2 else (
                  'Delist' if avg_sales[x] < sku_df['Sales'].mean()*0.5 else 'Retain'))

    st.dataframe(sku_df, use_container_width=True)

    # Summary Counts
    counts = sku_df['Recommendation'].value_counts()
    st.metric("Total SKUs Uploaded", len(sku_df))
    st.metric("To Retain", counts.get('Retain',0))
    st.metric("To Expand", counts.get('Expand',0))
    st.metric("To Delist", counts.get('Delist',0))

    fig = px.bar(sku_df, x='SKU', y='Needed Space (in)', color='Recommendation', hover_data=['Item Size (in)','Facings'])
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: SALES ANALYSIS & INSIGHT MATCHING ---
st.header("📈 Sales Analysis & Insight Matching")
sales_file = st.file_uploader("Upload Sales CSV", type=["csv"], key="sales")
if sales_file:
    sales_df = load_csv(sales_file)
    sales_df = parse_date_column(sales_df, 'Date')

    # Clean and ensure numeric sales values
    if 'Sales' in sales_df.columns:
        sales_df['Sales'] = (sales_df['Sales']
                             .astype(str)
                             .str.replace('[₱,]', '', regex=True)
                             .astype(float))

        # Calculate average per store
        avg_by_store = sales_df.groupby('Store Code')['Sales'].mean().to_dict()

        # Signal Detection
        sales_df['Signal'] = sales_df.apply(lambda row: (
            'High' if row['Sales'] > avg_by_store[row['Store Code']] * 1.5 else
            ('Low' if row['Sales'] < avg_by_store[row['Store Code']] * 0.5 else 'Normal')
        ), axis=1)

        # Load Approved Insights if exists
        try:
            insights_df = pd.read_csv('approved_insights.csv')
            insights_df = parse_date_column(insights_df, 'Date')

            merged = pd.merge(sales_df, insights_df, on=['Store Code','Date'], how='left')
            merged['Qualitative Note'] = merged['Insight'].fillna('').apply(lambda x: f"*{x}*" if x else '')
        except FileNotFoundError:
            merged = sales_df.copy()
            merged['Qualitative Note'] = ''

        # Display with highlighting
        def color_signal(val):
            if val == 'High':
                return 'background-color: lightgreen'
            elif val == 'Low':
                return 'background-color: salmon'
            return ''

        st.dataframe(merged.style.applymap(color_signal, subset=['Signal']), use_container_width=True)
        fig2 = px.line(merged, x='Date', y='Sales', color='Store Code', title='Sales Trend with Signals')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Sales column missing from file.")

# --- MODULE 3: SUBMIT INSIGHTS ---
st.header("📝 Submit Insight")
with st.form("submit_insight"):
    store_code = st.text_input("Store Code")
    date_input = st.date_input("Date")
    insight_text = st.text_area("Insight")
    submitted = st.form_submit_button("Submit")
    if submitted:
        new_data = pd.DataFrame([[store_code, date_input, insight_text]], columns=['Store Code','Date','Insight'])
        try:
            existing = pd.read_csv('pending_insights.csv')
            pending = pd.concat([existing, new_data], ignore_index=True)
        except FileNotFoundError:
            pending = new_data
        pending.to_csv('pending_insights.csv', index=False)
        st.success("Insight submitted for approval!")

# --- MODULE 4: APPROVE INSIGHTS ---
st.header("✅ Approve/Reject Insights")
try:
    pending = pd.read_csv('pending_insights.csv')
    st.dataframe(pending, use_container_width=True)
    for idx, row in pending.iterrows():
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Approve {idx}"):
                try:
                    approved = pd.read_csv('approved_insights.csv')
                    approved = pd.concat([approved, pd.DataFrame([row])], ignore_index=True)
                except FileNotFoundError:
                    approved = pd.DataFrame([row])
                approved.to_csv('approved_insights.csv', index=False)
                pending.drop(idx, inplace=True)
                pending.to_csv('pending_insights.csv', index=False)
                st.experimental_rerun()
        with c2:
            if st.button(f"Reject {idx}"):
                pending.drop(idx, inplace=True)
                pending.to_csv('pending_insights.csv', index=False)
                st.experimental_rerun()
except FileNotFoundError:
    st.info("No pending insights.")
