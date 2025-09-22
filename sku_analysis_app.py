import streamlit as st
import pandas as pd
from datetime import datetime
import os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(layout="wide")
INSIGHTS_FILE = "insights.csv"

# Create insights CSV if not exists
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Timestamp","Store Code","Insight","Status","Submitted By"]).to_csv(INSIGHTS_FILE,index=False)

# ----------------------------
# SIDEBAR: Module Selection
# ----------------------------
st.sidebar.header("📌 Select Module")
module = st.sidebar.radio("Choose Module:", 
                          ["SKU Performance & Shelf Space", 
                           "Submit Insight", 
                           "Approve Insight"])

# ----------------------------
# MODULE 1: SKU PERFORMANCE & SHELF SPACE
# ----------------------------
if module == "SKU Performance & Shelf Space":
    st.title("📊 SKU Performance & Shelf Space Optimizer")

    uploaded_file = st.file_uploader("Upload SKU CSV (must include Store Code)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Sidebar Settings for SKU module ---
        st.sidebar.header("⚙️ SKU Settings")
        expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
        retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
        delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
        min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
        shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
        num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
        hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts & space calc", value=False)
        top_n_skus = st.sidebar.slider("Number of SKUs to show in chart", 10, 100, 50, 5)

        total_shelf_space = shelf_width * num_layers

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

        # --- Suggested Facings ---
        def base_facings(rec):
            if rec == "Expand": return max(expand_facings, min_facings)
            elif rec == "Retain": return max(retain_facings, min_facings)
            else: return delist_facings

        df['Base Facings'] = df['Recommendation'].apply(base_facings)

        if 'Width' not in df.columns:
            default_width = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)
            df['Width'] = default_width

        df['Space Needed'] = df['Width'] * df['Base Facings']

        # Redistribute freed space from Delist SKUs
        if delist_facings == 0:
            delist_space = (df['Recommendation'] == 'Delist') * df['Width'] * base_facings('Delist')
            freed_space = delist_space.sum()
            expand_retain_mask = df['Recommendation'].isin(['Expand','Retain'])
            total_expand_retain_width = (df.loc[expand_retain_mask,'Width']*df.loc[expand_retain_mask,'Base Facings']).sum()
            df.loc[expand_retain_mask,'Extra Facings'] = (df.loc[expand_retain_mask,'Width']*df.loc[expand_retain_mask,'Base Facings']/total_expand_retain_width * freed_space / df.loc[expand_retain_mask,'Width']).fillna(0)
        else:
            df['Extra Facings'] = 0

        df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
        df['Space Needed'] = df['Width'] * df['Suggested Facings']

        if hide_delist:
            df_filtered = df[df['Recommendation'] != "Delist"]
        else:
            df_filtered = df.copy()

        # --- Shelf Space Usage ---
        total_space_used = df_filtered['Space Needed'].sum()
        space_usage_pct = (total_space_used/total_shelf_space)*100

        st.subheader("📋 SKU Recommendations & Performance Rank")
        st.write("**Explanation:** Each SKU's performance, recommended action, justification, suggested facings, and shelf space needed. Rank helps decide which SKUs to delist if space is limited.")

        def color_table(val):
            if val == "Expand": return "background-color: #c6efce"
            elif val == "Delist": return "background-color: #ffc7ce"
            elif val == "Retain": return "background-color: #ffeb9c"
            return ""

        st.dataframe(df_filtered[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style.applymap(color_table, subset=['Recommendation']), use_container_width=True)

        st.subheader("📊 Shelf Space Usage")
        st.write("**Explanation:** How much of your shelf is being used across all layers. Over 100% means you need to remove SKUs or reduce facings.")
        st.progress(min(space_usage_pct/100,1.0))
        st.write(f"Used: {total_space_used:.1f}/{total_shelf_space} in ({space_usage_pct:.1f}%)")

        if space_usage_pct>100:
            over_inch = total_space_used - total_shelf_space
            df_sorted = df_filtered.sort_values(by=['Space Needed','Score'], ascending=[False, True])
            cum_space = 0
            num_skus_to_remove = 0
            skus_to_remove = []
            for _, row in df_sorted.iterrows():
                cum_space += row['Space Needed']
                num_skus_to_remove += 1
                skus_to_remove.append(row['SKU'])
                if cum_space>=over_inch: break
            st.text(f"⚠️ Shelf space is full!\nYou may need to remove {num_skus_to_remove} SKU(s) or reduce facings.\nSuggested SKUs to remove based on space and performance:\n- " + "\n- ".join(skus_to_remove))
        else:
            st.success("✅ Your shelf plan fits within the available space.")

# ----------------------------
# MODULE 2: Submit Insight
# ----------------------------
elif module=="Submit Insight":
    st.header("📝 Submit New Insight")
    store_code = st.text_input("Store Code")
    insight_text = st.text_area("Insight Description")
    submitted_by = st.text_input("Submitted By (Name)")
    if st.button("Submit Insight"):
        if store_code and insight_text and submitted_by:
            df_insights = pd.read_csv(INSIGHTS_FILE)
            df_insights = pd.concat([df_insights, pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Store Code": store_code,
                "Insight": insight_text,
                "Status": "Pending Approval",
                "Submitted By": submitted_by
            }])], ignore_index=True)
            df_insights.to_csv(INSIGHTS_FILE,index=False)
            st.success("✅ Insight submitted successfully and pending approval.")
        else:
            st.error("Please fill all fields.")

# ----------------------------
# MODULE 3: Approve Insight
# ----------------------------
elif module=="Approve Insight":
    st.header("✅ Approve Pending Insights")
    df_insights = pd.read_csv(INSIGHTS_FILE)
    pending_df = df_insights[df_insights["Status"]=="Pending Approval"]
    if pending_df.empty:
        st.info("No pending insights for approval.")
    else:
        for idx, row in pending_df.iterrows():
            st.write(f"**Store:** {row['Store Code']}, **Submitted By:** {row['Submitted By']}, **Time:** {row['Timestamp']}")
            st.write(f"**Insight:** {row['Insight']}")
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {idx}"):
                df_insights.at[idx,"Status"]="Approved"
                df_insights.to_csv(INSIGHTS_FILE,index=False)
                st.success(f"Insight {idx} approved.")
            if col2.button(f"Reject {idx}"):
                df_insights.at[idx,"Status"]="Rejected"
                df_insights.to_csv(INSIGHTS_FILE,index=False)
                st.warning(f"Insight {idx} rejected.")
