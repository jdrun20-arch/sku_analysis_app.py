import pandas as pd
import streamlit as st
import io
import openpyxl
from openpyxl.styles import PatternFill

st.title("📊 SKU Performance Analyzer")
st.write("Upload your SKU list to get Expand/Retain/Delist recommendations with explanations and action plans.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Normalize
    df['Sales_Norm'] = df['Sales'] / df['Sales'].max()
    df['Volume_Norm'] = df['Volume'] / df['Volume'].max()
    df['Margin_Norm'] = df['Margin'] / df['Margin'].max()

    # Weighted Score
    df['Score'] = (df['Sales_Norm'] * 0.30) + (df['Volume_Norm'] * 0.30) + (df['Margin_Norm'] * 0.40)

    # Classification
    cutoff_expand = df['Score'].quantile(0.70)
    cutoff_delist = df['Score'].quantile(0.30)

    def classify_with_details(score, sales, volume, margin):
        if score >= cutoff_expand:
            return (
                "Expand",
                "High sales, strong margins, or high volume – a growth driver for the category.",
                "Increase shelf space, add more facings, consider promotion support or multi-store rollout."
            )
        elif score <= cutoff_delist:
            return (
                "Delist",
                "Low sales, weak margins, or slow movement – low contribution to overall performance.",
                "Plan clearance: run promo bundling, offer buy-1-take-1, or markdowns to clear stock."
            )
        else:
            return (
                "Retain",
                "Moderate performance – contributes steadily without underperforming.",
                "Maintain current shelf space, monitor performance quarterly."
            )

    df[['Recommendation', 'Explanation', 'Action Plan']] = df.apply(
        lambda row: pd.Series(classify_with_details(row['Score'], row['Sales'], row['Volume'], row['Margin'])),
        axis=1
    )

    # Show table in app
    st.dataframe(df[['SKU Name', 'Sales', 'Volume', 'Margin', 'Score', 'Recommendation', 'Explanation', 'Action Plan']])

    # Export to Excel with color formatting
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    # Open Excel with openpyxl and apply colors
    wb = openpyxl.load_workbook(output)
    ws = wb.active

    header = [cell.value for cell in ws[1]]
    recommendation_col = header.index("Recommendation") + 1

    for row in range(2, ws.max_row + 1):
        cell = ws.cell(row=row, column=recommendation_col)
        if cell.value == "Expand":
            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        elif cell.value == "Delist":
            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        else:
            cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")

    # Save the formatted file to memory again
    formatted_output = io.BytesIO()
    wb.save(formatted_output)
    formatted_output.seek(0)

    # Download Button
    st.download_button(
        label="📥 Download Excel with Recommendations",
        data=formatted_output,
        file_name="SKU_Recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
