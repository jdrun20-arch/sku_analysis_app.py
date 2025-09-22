import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")

st.title("📊 SKU Performance & Shelf Space Optimizer")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
    df['Margin_Norm'] = df['Margin'] / df['Margin'].max()

    df['Score'] = (df['Sales_Norm'] * 0.30) + (df['Volume_Norm'] * 0.30) + (df['Margin_Norm'] * 0.40)

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

    st.sidebar.header("⚙️ Settings")
    expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
    retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
    delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
    min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", min_value=1, value=2, step=1)
    total_shelf_space = st.sidebar.number_input("Total Shelf Space (inches)", min_value=1.0, value=100.0, step=1.0)
    hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)

    def suggested_facings(rec):
        if rec == "Expand":
            return max(expand_facings, min_facings)
        elif rec == "Retain":
            return max(retain_facings, min_facings)
        else:
            return delist_facings

    df['Suggested Facings'] = df['Recommendation'].apply(suggested_facings)
    df['Space Needed'] = df['Width'] * df['Suggested Facings']

    if hide_delist:
        df_filtered = df[df['Recommendation'] != "Delist"]
    else:
        df_filtered = df.copy()

    total_space_used = df_filtered['Space Needed'].sum()
    space_usage_pct = (total_space_used / total_shelf_space) * 100

    st.subheader("📋 Detailed Results")
    def color_table(val):
        if val == "Expand":
            return "background-color: #c6efce"
        elif val == "Delist":
            return "background-color: #ffc7ce"
        elif val == "Retain":
            return "background-color: #ffeb9c"
        return ""

    st.dataframe(df.style.applymap(color_table, subset=["Recommendation"]), use_container_width=True)

    st.subheader("📊 Shelf Space Usage")
    fig1, ax1 = plt.subplots(figsize=(6,1))
    color = 'green' if total_space_used <= total_shelf_space else 'red'
    ax1.barh(["Shelf"], [total_space_used], color=color)
    ax1.set_xlim(0, total_shelf_space)
    ax1.set_xlabel(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")
    st.pyplot(fig1)

    st.subheader("📊 Per-SKU Space Allocation")
    if 'SKU' in df_filtered.columns:
        df_sorted = df_filtered.sort_values(by="Space Needed", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.barh(df_sorted['SKU'], df_sorted['Space Needed'], color="skyblue")
        ax2.set_xlabel("Space Needed (inches)")
        st.pyplot(fig2)
    else:
        st.warning("No 'SKU' column found for per-SKU visualization.")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, "SKU_Recommendations.csv", "text/csv")
