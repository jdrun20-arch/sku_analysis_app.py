import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("📊 SKU Performance Analysis")
st.write("Upload your SKU file to get recommendations (Expand, Retain, Delist) with explanations.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Normalize
    df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
    df['Margin_Norm'] = df['Margin'] / df['Margin'].max()

    # Weighted score
    df['Score'] = (df['Sales_Norm'] * 0.30) + (df['Volume_Norm'] * 0.30) + (df['Margin_Norm'] * 0.40)

    # Cutoffs
    cutoff_expand = df['Score'].quantile(0.70)
    cutoff_delist = df['Score'].quantile(0.30)

    # Classification logic
    def classify(score):
        if score >= cutoff_expand:
            return "Expand"
        elif score <= cutoff_delist:
            return "Delist"
        else:
            return "Retain"

    df['Recommendation'] = df['Score'].apply(classify)

    # Explanations
    def explain(row):
        if row['Recommendation'] == "Expand":
            return "High sales, volume, or margin → Increase facings or distribution."
        elif row['Recommendation'] == "Delist":
            return "Low performance → Candidate for phase-out."
        else:
            return "Balanced performance → Maintain current space."
    
    def move_out_plan(row):
        if row['Recommendation'] == "Delist":
            return "Consider promo bundling, discounting, or supplier return."
        else:
            return "-"
    
    df['Explanation'] = df.apply(explain, axis=1)
    df['Move-Out Plan'] = df.apply(move_out_plan, axis=1)

    # Section: Results
    st.subheader("📋 Detailed Results")

    # Color table
    def color_table(val):
        if val == "Expand":
            return "background-color: #c6efce"  # light green
        elif val == "Delist":
            return "background-color: #ffc7ce"  # light red
        elif val == "Retain":
            return "background-color: #ffeb9c"  # light yellow
        return ""

    st.dataframe(df.style.applymap(color_table, subset=["Recommendation"]))

    # Section: Summary Chart
    st.subheader("📊 Summary of Recommendations")

    summary = df['Recommendation'].value_counts()

    fig, ax = plt.subplots()
    summary.plot(kind='bar', color=["#c6efce", "#ffeb9c", "#ffc7ce"], ax=ax)
    ax.set_title("Recommendation Breakdown")
    ax.set_ylabel("Number of SKUs")
    st.pyplot(fig)

    # Download results
    st.subheader("⬇️ Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="SKU_Recommendations.csv",
        mime="text/csv"
    )


