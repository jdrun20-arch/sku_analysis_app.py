import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📊 SKU Performance & Shelf Space Optimizer + Sales Analysis")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Performance Scoring ---
    df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
    df['Margin_Norm'] = df['Margin'] / df['Margin'].max()
    df['Score'] = (df['Sales_Norm'] * 0.3) + (df['Volume_Norm'] * 0.3) + (df['Margin_Norm'] * 0.4)

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
    num_expand = len(df[df['Recommendation'] == 'Expand'])
    num_retain = len(df[df['Recommendation'] == 'Retain'])
    num_delist = len(df[df['Recommendation'] == 'Delist'])

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
        total_expand_retain_width = (
            df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']
        ).sum()
        df.loc[expand_retain_mask, 'Extra Facings'] = (
            df.loc[expand_retain_mask, 'Width']
            * df.loc[expand_retain_mask, 'Base Facings']
            / total_expand_retain_width
            * freed_space
            / df.loc[expand_retain_mask, 'Width']
        ).fillna(0)
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
        if val == "Expand":
            return "background-color: #c6efce"
        elif val == "Delist":
            return "background-color: #ffc7ce"
        elif val == "Retain":
            return "background-color: #ffeb9c"
        return ""

    st.dataframe(
        df[['SKU', 'Score', 'Rank', 'Recommendation', 'Justification', 'Suggested Facings', 'Space Needed']]
        .style.applymap(color_table, subset=['Recommendation']),
        use_container_width=True
    )

    # --- Shelf Space Usage ---
    st.subheader("📊 Shelf Space Usage")
    st.progress(min(space_usage_pct / 100, 1.0))
    st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

    if space_usage_pct > 100:
        over_inch = total_space_used - total_shelf_space
        df_sorted = df_filtered.sort_values(by=['Space Needed', 'Score'], ascending=[False, True])
        cum_space = 0
        skus_to_remove = []
        for _, row in df_sorted.iterrows():
            cum_space += row['Space Needed']
            skus_to_remove.append(row['SKU'])
            if cum_space >= over_inch:
                break
        st.text(
            f"⚠️ Shelf space is full!\n"
            f"You may need to remove {len(skus_to_remove)} SKU(s) or reduce facings.\n"
            f"Suggested SKUs to remove:\n- " + "\n- ".join(skus_to_remove)
        )
    else:
        st.success("✅ Your shelf plan fits within the available space.")

    # --- Top SKUs by Space Needed Chart ---
    st.subheader("📊 Top SKUs by Space Needed")
    df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
    fig = px.bar(
        df_chart,
        y='SKU',
        x='Space Needed',
        color='Recommendation',
        orientation='h',
        hover_data={'Space Needed': ':.1f', 'Width': ':.1f', 'Suggested Facings': True, 'Justification': True},
        color_discrete_map={'Expand': '#4CAF50', 'Retain': '#FFC107', 'Delist': '#F44336'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=25 * len(df_chart))
    st.plotly_chart(fig, use_container_width=True)

    # --- Sales Signal Analysis ---
    st.subheader("📈 Sales Trend & Signal Analysis")
    if 'Date' in df.columns:
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date_parsed'])
        df['Signal'] = df['Sales'].pct_change().apply(
            lambda x: "LIFT" if x > 0.1 else ("DROP" if x < -0.1 else "STABLE")
        )
        df['Qualitative Note'] = df['Signal'].apply(
            lambda x: "Possible rally/holiday → high foot traffic" if x == "LIFT"
            else ("Possible storm or external factor → low foot traffic" if x == "DROP" else "")
        )

        display_cols = [col for col in ["Store Code", "Date_parsed", "Sales", "Volume", "Signal", "Qualitative Note"] if col in df.columns]

        def color_signal(val):
            if val == "LIFT":
                return "background-color: #c6efce; color: black;"
            elif val == "DROP":
                return "background-color: #ffc7ce; color: black;"
            return ""

        def italicize(val):
            return "font-style: italic;" if isinstance(val, str) and val else ""

        styled_df = df[display_cols].sort_values(['Store Code', 'Date_parsed']).style.applymap(
            color_signal, subset=['Signal']
        ).applymap(
            italicize, subset=['Qualitative Note']
        )

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No 'Date' column found — cannot perform sales signal analysis.")
