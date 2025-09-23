# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Retail Optimization App (patched)")

# --- Persistent Insights Storage ---
INSIGHTS_FILE = "approved_insights.csv"
if not os.path.exists(INSIGHTS_FILE):
    pd.DataFrame(columns=["Date", "Store Code", "Insight", "Status"]).to_csv(INSIGHTS_FILE, index=False)

def safe_rerun():
    """
    Try to programmatically rerun the Streamlit script. If that isn't available
    in this runtime (AttributeError or other), set a small session-state toggle
    and stop the script so the user can refresh safely.
    """
    try:
        # preferred approach
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            raise AttributeError("experimental_rerun not available")
    except Exception:
        # fallback: write a small flag and stop
        st.session_state["_needs_manual_refresh"] = not st.session_state.get("_needs_manual_refresh", False)
        st.success("✅ Change saved. Please refresh the page to see the updated state.")
        st.stop()

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Select Module", [
    "SKU Performance & Shelf Space",
    "Sales Analysis",
    "Submit Insights",
    "Approve Insights"
])

# =========================
# 1️⃣ SKU PERFORMANCE MODULE
# =========================
if page == "SKU Performance & Shelf Space":
    st.header("📊 SKU Performance & Shelf Space Optimizer")
    uploaded_file = st.file_uploader("📂 Upload your SKU Performance CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Basic validation: required columns
        required = ["SKU", "Sales", "Volume", "Margin"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Expecting at least: {', '.join(required)}")
        else:
            # --- Performance Scoring ---
            for c in ["Sales", "Volume", "Margin"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            df['Sales_Norm'] = df['Sales'] / df['Sales'].replace(0, pd.NA).max()
            df['Volume_Norm'] = df['Volume'] / df['Volume'].replace(0, pd.NA).max()
            df['Margin_Norm'] = df['Margin'] / df['Margin'].replace(0, pd.NA).max()
            df[['Sales_Norm','Volume_Norm','Margin_Norm']] = df[['Sales_Norm','Volume_Norm','Margin_Norm']].fillna(0)
            df['Score'] = (df['Sales_Norm']*0.3) + (df['Volume_Norm']*0.3) + (df['Margin_Norm']*0.4)
            df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)

            cutoff_expand = df['Score'].quantile(0.70)
            cutoff_delist = df['Score'].quantile(0.30)

            def classify(score):
                if score >= cutoff_expand: return "Expand"
                if score <= cutoff_delist: return "Delist"
                return "Retain"
            df['Recommendation'] = df['Score'].apply(classify)

            def justify(rec):
                return {
                    "Expand":"High sales, volume, or margin → increase facings or distribution.",
                    "Delist":"Low performance → candidate for phase-out.",
                    "Retain":"Balanced performance → maintain current space."
                }.get(rec, "")
            df['Justification'] = df['Recommendation'].apply(justify)

            # --- Sidebar Settings ---
            st.sidebar.header("⚙️ Shelf Settings")
            expand_facings = st.sidebar.slider("Facings for Expand SKUs", 1, 10, 3)
            retain_facings = st.sidebar.slider("Facings for Retain SKUs", 1, 10, 2)
            delist_facings = st.sidebar.slider("Facings for Delist SKUs", 0, 5, 1)
            min_facings = st.sidebar.number_input("Minimum Facings (Expand/Retain)", 1, 10, 2)
            shelf_width = st.sidebar.number_input("Shelf width per layer (inches)", 1.0, 10000.0, 100.0, 1.0)
            num_layers = st.sidebar.number_input("Number of layers in gondola", 1, 20, 1)
            hide_delist = st.sidebar.checkbox("Hide Delist SKUs from charts", value=False)
            top_n_skus = st.sidebar.slider("Top SKUs to show", 5, max(5, len(df)), min(50, max(5, len(df))), 5)

            total_shelf_space = shelf_width * num_layers

            def base_facings(rec):
                if rec == "Expand": return max(expand_facings, min_facings)
                if rec == "Retain": return max(retain_facings, min_facings)
                return delist_facings

            df['Base Facings'] = df['Recommendation'].apply(base_facings)

            # Width handling
            if 'Width' not in df.columns:
                default_width = st.sidebar.number_input("Default SKU Width (inches)", 1.0, 100.0, 5.0, 0.1)
                df['Width'] = default_width
            else:
                df['Width'] = pd.to_numeric(df['Width'], errors='coerce').fillna(
                    st.sidebar.number_input("Default SKU Width (fallback)", 1.0, 100.0, 5.0, 0.1)
                )

            df['Space Needed'] = df['Width'] * df['Base Facings']

            # Redistribute extra facings if delist facings zero
            if delist_facings == 0:
                delist_mask = df['Recommendation'] == 'Delist'
                freed_space = (df.loc[delist_mask, 'Width'] * base_facings('Delist')).sum()
                expand_retain_mask = df['Recommendation'].isin(['Expand','Retain'])
                denom = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings']).sum()
                if denom > 0 and freed_space > 0:
                    extra_facings = (df.loc[expand_retain_mask, 'Width'] * df.loc[expand_retain_mask, 'Base Facings'] / denom * freed_space) / df.loc[expand_retain_mask, 'Width']
                    df.loc[expand_retain_mask, 'Extra Facings'] = extra_facings.fillna(0)
                else:
                    df['Extra Facings'] = 0
            else:
                df['Extra Facings'] = 0

            df['Suggested Facings'] = df['Base Facings'] + df['Extra Facings']
            df['Space Needed'] = df['Width'] * df['Suggested Facings']

            if hide_delist:
                df_filtered = df[df['Recommendation'] != "Delist"].copy()
            else:
                df_filtered = df.copy()

            total_space_used = df_filtered['Space Needed'].sum()
            space_usage_pct = (total_space_used / total_shelf_space) * 100 if total_shelf_space > 0 else 0.0

            # --- Outputs ---
            st.subheader("📋 SKU Recommendations & Performance Rank")
            st.dataframe(df[['SKU','Score','Rank','Recommendation','Justification','Suggested Facings','Space Needed']].style.applymap(
                lambda x: "background-color: #c6efce" if x=="Expand" else ("background-color: #ffc7ce" if x=="Delist" else ("background-color: #ffeb9c" if x=="Retain" else "")),
                subset=['Recommendation']
            ), use_container_width=True)

            st.subheader("📊 Shelf Space Usage")
            st.progress(min(space_usage_pct/100, 1.0))
            st.write(f"Used: {total_space_used:.1f}/{total_shelf_space:.1f} in ({space_usage_pct:.1f}%)")

            if space_usage_pct > 100:
                over_inch = total_space_used - total_shelf_space
                df_sorted = df_filtered.sort_values(by=['Space Needed','Score'], ascending=[False, True])
                cum_space = 0.0
                skus_to_remove = []
                for _, row in df_sorted.iterrows():
                    cum_space += float(row['Space Needed'])
                    skus_to_remove.append(str(row['SKU']))
                    if cum_space >= over_inch:
                        break
                st.warning(f"⚠️ Shelf space overcapacity by {over_inch:.1f} in. Suggested remove {len(skus_to_remove)} SKU(s): {skus_to_remove[:20]}")
            else:
                st.success("✅ Your shelf plan fits within the available space.")

            st.subheader("📊 Top SKUs by Space Needed")
            df_chart = df_filtered.sort_values(by='Space Needed', ascending=False).head(top_n_skus)
            fig = px.bar(df_chart, y='SKU', x='Space Needed', color='Recommendation', orientation='h',
                         hover_data={'Space Needed':':.1f','Width':':.1f','Suggested Facings':True,'Justification':True},
                         color_discrete_map={'Expand':'#4CAF50','Retain':'#FFC107','Delist':'#F44336'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=25*len(df_chart))
            st.plotly_chart(fig, use_container_width=True)

# =========================
# 2️⃣ SALES ANALYSIS MODULE
# =========================
elif page == "Sales Analysis":
    st.header("📈 Sales Analysis & Insight Matching")
    sales_file = st.file_uploader("📂 Upload your Sales CSV", type=["csv"])
    if sales_file is not None:
        sales_df = pd.read_csv(sales_file)

        # normalize and parse dates
        if 'Date' not in sales_df.columns:
            st.error("Sales file must include a 'Date' column (YYYY-MM-DD).")
        else:
            sales_df['Date_parsed'] = pd.to_datetime(sales_df['Date'], errors='coerce')
            sales_df = sales_df.dropna(subset=['Date_parsed'])
            sales_df = sales_df.sort_values(['Store Code','Date_parsed'])

            # --- Filters ---
            store_codes = sales_df['Store Code'].unique().tolist()
            selected_stores = st.multiselect("🏬 Select Store Code(s)", store_codes, default=store_codes)
            min_date, max_date = sales_df['Date_parsed'].min(), sales_df['Date_parsed'].max()
            date_range = st.date_input("📅 Select Date Range", [min_date, max_date])
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

            filtered_df = sales_df[(sales_df['Store Code'].isin(selected_stores)) &
                                   (sales_df['Date_parsed'].between(start_date, end_date))].copy()

            if filtered_df.empty:
                st.info("No sales rows in the selected store(s) and date range.")
            else:
                # --- Signal detection per store relative to filtered period average ---
                avg_by_store = filtered_df.groupby('Store Code')['Sales'].mean().to_dict()
                def detect_signal(row):
                    avg = avg_by_store.get(row['Store Code'], 0)
                    if pd.isna(avg) or avg == 0:
                        return "Normal"
                    if row['Sales'] > avg * 1.5:
                        return "⬆️ High Spike"
                    if row['Sales'] < avg * 0.5:
                        return "⬇️ Significant Drop"
                    return "Normal"
                filtered_df['Signal'] = filtered_df.apply(detect_signal, axis=1)

                # --- Load and merge approved insights ---
                approved_insights = pd.read_csv(INSIGHTS_FILE)
                approved_insights = approved_insights[approved_insights['Status']=="Approved"].copy()
                approved_insights['Date'] = pd.to_datetime(approved_insights['Date'], errors='coerce')

                merged = pd.merge(filtered_df, approved_insights, how='left',
                                  left_on=['Store Code','Date_parsed'],
                                  right_on=['Store Code','Date'])
                merged['Qualitative Note'] = merged['Insight'].fillna("")

                # --- Summary counts & metrics ---
                lifts = (merged['Signal'] == "⬆️ High Spike").sum()
                drops = (merged['Signal'] == "⬇️ Significant Drop").sum()
                with_insight = merged['Qualitative Note'].astype(bool).sum()

                st.subheader("📊 Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("📈 Lift Days", lifts)
                c2.metric("📉 Drop Days", drops)
                c3.metric("📝 Days w/ Insights", with_insight)

                # --- Styled table ---
                def highlight_signal(v):
                    if "High" in str(v): return "background-color: #c6efce"
                    if "Drop" in str(v): return "background-color: #ffc7ce"
                    return ""
                display_cols = ['Store Code','Date_parsed','Sales','Signal','Qualitative Note']
                st.dataframe(merged[display_cols].sort_values(['Store Code','Date_parsed']).style.applymap(
                    highlight_signal, subset=['Signal']
                ).applymap(lambda v: "font-style: italic;" if isinstance(v, str) and v else "", subset=['Qualitative Note']),
                use_container_width=True)

                # --- Narrative Generation ---
                st.subheader("📝 Automated Narrative")
                narrative_lines = []
                spike_days = merged[merged['Signal']=="⬆️ High Spike"]
                drop_days = merged[merged['Signal']=="⬇️ Significant Drop"]

                for _, r in spike_days.iterrows():
                    note = f"📈 **{r['Date_parsed'].date()}** (Store {r['Store Code']}): sales = ₱{r['Sales']:,.0f}"
                    if r['Qualitative Note']:
                        note += f"  — likely: {r['Qualitative Note']}"
                    else:
                        note += "  — no approved insight found (investigate promo/event)."
                    narrative_lines.append(note)

                for _, r in drop_days.iterrows():
                    note = f"📉 **{r['Date_parsed'].date()}** (Store {r['Store Code']}): sales = ₱{r['Sales']:,.0f}"
                    if r['Qualitative Note']:
                        note += f"  — possible cause: {r['Qualitative Note']}"
                    else:
                        note += "  — no approved insight found (check weather/ops/stock)."
                    narrative_lines.append(note)

                if narrative_lines:
                    st.markdown("\n\n".join(narrative_lines))
                else:
                    st.info("No unusual sales patterns detected in the selected period.")

                # --- Chart (auto-zoom to selected date range) ---
                fig = px.line(merged, x='Date_parsed', y='Sales', color='Store Code', title="📈 Sales Trend with Insights")
                fig.update_xaxes(range=[start_date, end_date])
                # add markers for insights
                for store, grp in merged.groupby('Store Code'):
                    insights_grp = grp[grp['Qualitative Note'] != ""]
                    if not insights_grp.empty:
                        fig.add_scatter(x=insights_grp['Date_parsed'], y=insights_grp['Sales'], mode='markers', marker_symbol='star', marker_size=10, name=f"Insight - {store}", hovertext=insights_grp['Qualitative Note'])
                st.plotly_chart(fig, use_container_width=True)

# =========================
# 3️⃣ SUBMIT INSIGHTS MODULE
# =========================
elif page == "Submit Insights":
    st.header("📝 Submit New Insights")
    with st.form("submit_form"):
        date = st.date_input("📅 Date")
        store_code = st.text_input("🏬 Store Code")
        insight = st.text_area("💡 Insight (e.g., Rally near store, class suspension, typhoon)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            # append to CSV
            df_ins = pd.read_csv(INSIGHTS_FILE)
            new_row = pd.DataFrame([{
                "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "Store Code": store_code,
                "Insight": insight,
                "Status": "Pending"
            }])
            df_ins = pd.concat([df_ins, new_row], ignore_index=True)
            df_ins.to_csv(INSIGHTS_FILE, index=False)
            st.success("✅ Insight submitted for approval!")

# =========================
# 4️⃣ APPROVE INSIGHTS MODULE
# =========================
elif page == "Approve Insights":
    st.header("✅ Approve or Reject Insights")
    df = pd.read_csv(INSIGHTS_FILE)
    pending = df[df['Status'] == "Pending"].copy()
    if pending.empty:
        st.info("No pending insights for approval.")
    else:
        # iterate pending rows and provide buttons; use unique keys
        for idx, row in pending.iterrows():
            st.markdown("---")
            st.write(f"📅 {row['Date']}  |  🏬 {row['Store Code']}")
            st.write(row['Insight'])
            col1, col2 = st.columns(2)
            # use explicit keys so Streamlit can distinguish buttons reliably
            if col1.button("Approve", key=f"approve_{idx}"):
                df.loc[idx, 'Status'] = "Approved"
                df.to_csv(INSIGHTS_FILE, index=False)
                safe_rerun()
            if col2.button("Reject", key=f"reject_{idx}"):
                df.loc[idx, 'Status'] = "Rejected"
                df.to_csv(INSIGHTS_FILE, index=False)
                safe_rerun()
