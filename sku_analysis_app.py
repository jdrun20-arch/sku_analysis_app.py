import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📊 SKU Performance & Shelf Space Optimizer")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Performance Scoring ---
    df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
    df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
    df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)

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

    # --- Sidebar Settings ---
    st.sidebar.header("⚙️ Settings")
    expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
    retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
    delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
    min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
    total_shelf_space = st.sidebar.number_input("Total Shelf Space (inches)", 1.0, 10000.0, 100.0, 1.0)
    hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)
    top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, 100, 50, 5)

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

    total_space_used = df_filtered['Space Needed'].sum()
    space_usage_pct = (total_space_used / total_shelf_space) * 100

    # --- Detailed Results ---
    st.subheader("📋 SKU Recommendations")
    st.write("**What this table tells you:** Each SKU, its recommendation, how many facings it should have, and how much shelf space it needs.")
    def color_table(val):
        if val == "Expand": return "background-color: #c6efce"
        elif val == "Delist": return "background-color: #ffc7ce"
        elif val == "Retain": return "background-color: #ffeb9c"
        return ""

    st.dataframe(df.style.applymap(color_table, subset=["Recommendation"]), use_container_width=True)

    # --- Shelf Space Usage ---
    st.subheader("📊 Shelf Space Usage")
    st.write("**Simple explanation:** How much of your shelf is being used. If over 100%, you need to reduce facings or remove SKUs.")
    bar_color = 'green' if space_usage_pct <= 100 else 'red'
    st.progress(min(space_usage_pct/100, 1.0))
    st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

    # --- Interactive Per-SKU Space Allocation using Plotly ---
    st.subheader("📊 Top SKUs by Space Needed")
    st.write("**Simple explanation:** This chart shows which SKUs take up the most shelf space. Freed-up space from Delist SKUs is distributed to Expand/Retain SKUs.")

    if 'SKU' not in df_filtered.columns:
        text_cols = df_filtered.select_dtypes(include='object').columns.tolist()
        if text_cols:
            df_filtered['SKU_Label'] = df_filtered[text_cols[0]]
        else:
            df_filtered['SKU_Label'] = [f"SKU_{i+1}" for i in range(len(df_filtered))]
    else:
        df_filtered['SKU_Label'] = df_filtered['SKU']

    df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
    fig = px.bar(df_chart, y='SKU_Label', x='Space Needed', color='Recommendation', orientation='h',
                 color_discrete_map={'Expand':'#4CAF50', 'Retain':'#FFC107', 'Delist':'#F44336'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
    st.plotly_chart(fig, use_container_width=True)

    # --- Download CSV ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv, "SKU_Recommendations.csv", "text/csv")
