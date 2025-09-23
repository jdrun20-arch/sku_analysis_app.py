# ========== MODULE 2: SALES ANALYSIS ==========
elif module == "Sales Analysis":
    st.header("📊 Sales Analysis")
    sales_file = st.file_uploader("📂 Upload Sales CSV", type=["csv"])
    if sales_file:
        sales_df = pd.read_csv(sales_file)
        sales_df['Date'] = pd.to_datetime(sales_df['Date'], errors='coerce')
        sales_df = sales_df.dropna(subset=['Date'])
        sales_df['Date_parsed'] = sales_df['Date']

        # Leave-one-out baseline per store
        baseline_list = []
        for store, group in sales_df.groupby('Store Code'):
            for idx, row in group.iterrows():
                baseline = group.loc[group.index != idx, 'Sales'].mean()
                baseline_list.append((idx, baseline))
        baseline_df = pd.DataFrame(baseline_list, columns=['idx','BaselineSales']).set_index('idx')
        sales_df = sales_df.join(baseline_df)
        sales_df['ChangePct'] = ((sales_df['Sales'] - sales_df['BaselineSales']) / sales_df['BaselineSales']) * 100

        approved_insights = st.session_state.insights[st.session_state.insights['Approved']==True]
        merged = pd.merge(sales_df, approved_insights, how='left',
                          left_on=['Date_parsed','Store Code'],
                          right_on=['Date','Store Code'])
        merged['MatchedInsight'] = merged['Insight'].fillna("")

        def classify_pct(pct):
            if pct>50: return "LIFT"
            elif pct<-30: return "DROP"
            else: return "STABLE"

        def classify_signal(row):
            if row['MatchedInsight']:
                return "LIFT" if row['Sales']>=row['BaselineSales'] else "DROP"
            return classify_pct(row['ChangePct'])

        merged['Signal'] = merged.apply(classify_signal, axis=1)

        def qualitative(row):
            if row['MatchedInsight']:
                return f"*User Insight:* {row['MatchedInsight']} (System detected {row['Signal']} of {row['ChangePct']:.0f}%)."
            if row['Signal']=="LIFT":
                return f"Sales lifted {row['ChangePct']:.0f}% vs baseline."
            elif row['Signal']=="DROP":
                return f"Sales dropped {row['ChangePct']:.0f}% vs baseline."
            else:
                return "Sales within normal range."
        merged['Qualitative Note'] = merged.apply(qualitative, axis=1)

        # ✅ NEW: Summary Counts
        lifts = (merged['Signal']=="LIFT").sum()
        drops = (merged['Signal']=="DROP").sum()
        with_insight = (merged['MatchedInsight']!="").sum()

        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Lift Days", lifts)
        col2.metric("📉 Drop Days", drops)
        col3.metric("📝 Days w/ Insights", with_insight)

        # Display with coloring + italics
        display_cols = ["Store Code","Date_parsed","Sales","BaselineSales","ChangePct","Signal","Qualitative Note"]
        def color_signal(val):
            if val=="LIFT": return "background-color:#c6efce"
            elif val=="DROP": return "background-color:#ffc7ce"
            return ""
        def italicize(val): return "font-style: italic"

        st.dataframe(
            merged[display_cols].sort_values(['Store Code','Date_parsed'],ascending=[True,True])
                .style.applymap(color_signal, subset=['Signal'])
                .applymap(italicize, subset=['Qualitative Note']),
            use_container_width=True
        )
