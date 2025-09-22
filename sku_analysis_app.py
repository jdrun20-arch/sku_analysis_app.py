# sku_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide", page_title="SKU Performance + Shelf Capacity Analyzer")

st.title("📊 SKU Performance + Shelf Capacity Analyzer (all features)")

st.write("""
Upload a CSV with SKU data. The app will auto-detect common column names (Sales, Volume, Margin, Width, Facings).
You can correct mapping in the sidebar if needed.
""")

# ---------------------------
# Utility: possible column name synonyms (for auto-detect)
# ---------------------------
COL_SYNS = {
    'Sales': ['sales', 'revenue', 'net_sales', 'sales_amt', 'sales_value'],
    'Volume': ['volume', 'units', 'qty', 'quantity', 'sales_units'],
    'Margin': ['margin', 'gross_margin', 'gp', 'profit'],
    'Width': ['width', 'size', 'item_width', 'pack_width', 'width_in', 'width_cm', 'w'],
    'Facings': ['facings', 'facing', 'no_facings', 'num_facings', 'faces']
}

def autodetect_columns(df_columns):
    mapped = {}
    cols_lower = {c.lower(): c for c in df_columns}
    for canonical, syns in COL_SYNS.items():
        found = None
        for s in syns:
            if s.lower() in cols_lower:
                found = cols_lower[s.lower()]
                break
        mapped[canonical] = found
    return mapped

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload SKU CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to begin. Example columns: SKU, Sales, Volume, Margin, Width (inches or cm), Facings (optional).")
    st.stop()

# Load CSV
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.sidebar.header("Column mapping (auto-detected)")
auto_map = autodetect_columns(df_raw.columns)
# Allow user to correct mappings using selects (or 'None')
col_map = {}
for key in ['Sales', 'Volume', 'Margin', 'Width', 'Facings']:
    col_map[key] = st.sidebar.selectbox(f"{key} column", options=[None] + list(df_raw.columns), index=(1 + list(df_raw.columns).index(auto_map[key]) if auto_map[key] in df_raw.columns else 0))

# If user leaves Sales/Volume/Margin/Width None -> error
required = ['Sales', 'Volume', 'Margin', 'Width']
missing_required = [r for r in required if not col_map[r]]
if missing_required:
    st.error(f"Please map these required columns in the sidebar: {missing_required}")
    st.stop()

# Copy df and rename mapped columns to canonical names
df = df_raw.copy()
rename_map = {col_map[k]: k for k in col_map if col_map[k]}
df = df.rename(columns=rename_map)

# Units: inches or cm
st.sidebar.header("Width units")
width_unit = st.sidebar.selectbox("Units used in the Width column", ("inches", "cm"))
if width_unit == "cm":
    st.sidebar.caption("Width in cm will be converted to inches (1 inch = 2.54 cm).")

# Convert numeric columns
numeric_cols = ['Sales', 'Volume', 'Margin', 'Width']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
# Drop rows missing key numerics
df = df.dropna(subset=['Sales', 'Volume', 'Margin', 'Width']).reset_index(drop=True)

# Convert width to inches if needed
if width_unit == 'cm':
    df['Width_in'] = df['Width'] / 2.54
else:
    df['Width_in'] = df['Width']
# We'll use Width_in for computations
if 'Facings' in df.columns:
    df['Facings'] = pd.to_numeric(df['Facings'], errors='coerce').fillna(1).astype(int)
else:
    df['Facings'] = 1  # default

# ---------------------------
# Sidebar: scoring & facing params
# ---------------------------
st.sidebar.header("Scoring weights")
w_sales = st.sidebar.number_input("Sales weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
w_volume = st.sidebar.number_input("Volume weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
w_margin = st.sidebar.number_input("Margin weight", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
# normalize weights if necessary
total_w = w_sales + w_volume + w_margin
if total_w == 0:
    st.sidebar.error("At least one weight must be > 0.")
    st.stop()
w_sales, w_volume, w_margin = w_sales/total_w, w_volume/total_w, w_margin/total_w

st.sidebar.header("Facings proportional controls")
# baseline facings assumption: use existing 'Facings' column as base
min_multiplier = st.sidebar.number_input("Minimum multiplier (floor)", min_value=0.0, value=0.25, step=0.05)
max_multiplier = st.sidebar.number_input("Maximum multiplier (ceiling)", min_value=0.1, value=3.0, step=0.1)
# sensitivity controls how strongly score affects multiplier (higher -> stronger)
sensitivity = st.sidebar.slider("Facing sensitivity (how strongly score changes facings)", 0.1, 5.0, 1.0)

st.sidebar.markdown("Recommendation cutoffs (quantiles)")
q_expand = st.sidebar.slider("Expand quantile (upper)", 0.5, 0.95, 0.70, 0.05)
q_delist = st.sidebar.slider("Delist quantile (lower)", 0.01, 0.5, 0.30, 0.05)

# Shelf space
st.sidebar.header("Shelf & trade-off")
shelf_space = st.sidebar.number_input("Total shelf space available (in inches)", min_value=0.0, value=48.0, step=1.0)
reduction_priority = st.sidebar.radio("When space is insufficient, reduce facings in order:", 
                                      ("Delist → Retain → Expand", "Retain → Delist → Expand", "Custom (choose below)"))
if reduction_priority.startswith("Custom"):
    st.sidebar.markdown("Choose custom priority (first reduced -> last reduced)")
    p1 = st.sidebar.selectbox("1st to reduce", ("Delist","Retain","Expand"), index=0)
    p2 = st.sidebar.selectbox("2nd to reduce", ("Delist","Retain","Expand"), index=1)
    p3 = st.sidebar.selectbox("3rd to reduce", ("Delist","Retain","Expand"), index=2)
    priority_order = [p1, p2, p3]
else:
    priority_order = reduction_priority.split(" → ")

# Safety: ensure priority_order contains all three types
if set(priority_order) != {"Delist", "Retain", "Expand"}:
    priority_order = ["Delist","Retain","Expand"]

# ---------------------------
# Scoring
# ---------------------------
def normalize(series):
    mx = series.max()
    mn = series.min()
    if pd.isna(mx) or mx == mn:
        # return zeroed series if constant
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)

df['Sales_N'] = normalize(df['Sales'])
df['Volume_N'] = normalize(df['Volume'])
df['Margin_N'] = normalize(df['Margin'])
df['Score'] = df['Sales_N'] * w_sales + df['Volume_N'] * w_volume + df['Margin_N'] * w_margin

# Cutoffs
cutoff_expand = df['Score'].quantile(q_expand)
cutoff_delist = df['Score'].quantile(q_delist)

def classify(score):
    if score >= cutoff_expand:
        return "Expand"
    elif score <= cutoff_delist:
        return "Delist"
    else:
        return "Retain"

df['Recommendation'] = df['Score'].apply(classify)

# Explanation
def explain(rec):
    if rec == "Expand":
        return "High score — increase facings or distribution."
    if rec == "Delist":
        return "Low score — candidate for reduction or phase-out."
    return "Moderate score — maintain."
df['Explanation'] = df['Recommendation'].apply(explain)

# ---------------------------
# Continuous proportional facings logic
# multiplier design:
#   multiplier = clamp(min_multiplier, max_multiplier, base + (score_norm - 0.5) * sensitivity)
# but better: map score [0,1] -> multiplier range with sensitivity controlling curvature
# ---------------------------
def score_to_multiplier(score, min_mult, max_mult, sensitivity):
    # score in [0,1]
    # apply a nonlinear transform to allow sensitivity; using tanh for soft control
    centered = (score - 0.5) * 2  # [-1,1]
    factor = np.tanh(centered * sensitivity)  # [-tanh(s), tanh(s)]
    # normalize factor to [0,1]
    f_norm = (factor + np.tanh(sensitivity)) / (2 * np.tanh(sensitivity))
    multiplier = min_mult + f_norm * (max_mult - min_mult)
    return multiplier

df['Multiplier'] = df['Score'].apply(lambda x: score_to_multiplier(x, min_multiplier, max_multiplier, sensitivity))

# For recommendation-specific nudges: allow Expand to bias multiplier upward slightly, Delist downward
df.loc[df['Recommendation'] == 'Expand', 'Multiplier'] = df.loc[df['Recommendation'] == 'Expand', 'Multiplier'] * 1.05
df.loc[df['Recommendation'] == 'Delist', 'Multiplier'] = df.loc[df['Recommendation'] == 'Delist', 'Multiplier'] * 0.95

# Suggested facings (floating) -> round to int but allow zero
df['Suggested_Facings'] = (df['Facings'] * df['Multiplier']).round().astype(int)
df['Suggested_Facings'] = df['Suggested_Facings'].clip(lower=0)  # ensure non-negative

# Space calculations
df['Space_Current'] = df['Width_in'] * df['Facings']
df['Space_Suggested'] = df['Width_in'] * df['Suggested_Facings']

total_current_space = df['Space_Current'].sum()
total_suggested_space = df['Space_Suggested'].sum()

# ---------------------------
# Display key metrics
# ---------------------------
st.markdown("## 📋 Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Shelf space available (in)", f"{shelf_space:.2f}")
c2.metric("Total required (current facings) (in)", f"{total_current_space:.2f}")
c3.metric("Total required (suggested facings) (in)", f"{total_suggested_space:.2f}")

if total_suggested_space <= shelf_space:
    st.success("✅ Suggested plan fits within shelf space.")
else:
    st.error("⚠️ Suggested plan exceeds shelf space and will be adjusted according to reduction priority.")

# ---------------------------
# If over capacity: perform iterative reduction according to priority_order
# ---------------------------
df['Suggested_Facings_Adjusted'] = df['Suggested_Facings'].copy()
df['Space_Suggested_Adjusted'] = df['Width_in'] * df['Suggested_Facings_Adjusted']

def reduce_until_fit(df_in, shelf_space, priority_order, max_iters=100000):
    df_adj = df_in.copy()
    current_needed = df_adj['Space_Suggested_Adjusted'].sum()
    iters = 0
    # Build index groups by recommendation in priority order
    rec_groups = []
    for rec in priority_order:
        rec_idx = df_adj[df_adj['Recommendation'] == rec].sort_values(by='Score', ascending=True).index.tolist()
        rec_groups.append(rec_idx)
    # Flattened list by priority (we'll loop over groups repeatedly)
    flat_order = [idx for group in rec_groups for idx in group]
    # If there are no reducible facings, stop
    while current_needed > shelf_space and iters < max_iters:
        changed = False
        # Iterate through priority buckets
        for rec in priority_order:
            candidates = df_adj[df_adj['Recommendation'] == rec].sort_values(by='Score', ascending=True)
            # attempt reduce on lowest score SKU in this rec group
            for idx in candidates.index:
                if df_adj.at[idx, 'Suggested_Facings_Adjusted'] > 0:
                    # Reduce by 1
                    df_adj.at[idx, 'Suggested_Facings_Adjusted'] -= 1
                    df_adj.at[idx, 'Space_Suggested_Adjusted'] = df_adj.at[idx, 'Width_in'] * df_adj.at[idx, 'Suggested_Facings_Adjusted']
                    current_needed = df_adj['Space_Suggested_Adjusted'].sum()
                    changed = True
                    break  # move to top of priority list (start again)
            if current_needed <= shelf_space:
                break
        if not changed:
            break
        iters += 1
    return df_adj

if total_suggested_space > shelf_space:
    df_adjusted = reduce_until_fit(df.copy(), shelf_space, priority_order)
    df['Suggested_Facings_Adjusted'] = df_adjusted['Suggested_Facings_Adjusted']
    df['Space_Suggested_Adjusted'] = df['Suggested_Facings_Adjusted'] * df['Width_in']
    final_suggested_space = df['Space_Suggested_Adjusted'].sum()
    st.write(f"Adjusted suggested total space: **{final_suggested_space:.2f} in** (target: {shelf_space:.2f} in)")
    if final_suggested_space <= shelf_space:
        st.success("✅ Adjusted plan now fits the shelf space.")
    else:
        st.warning("⚠️ Even after adjustments, plan does not fit. Consider further manual edits or increasing shelf space.")
else:
    df['Suggested_Facings_Adjusted'] = df['Suggested_Facings']
    df['Space_Suggested_Adjusted'] = df['Space_Suggested']

# ---------------------------
# Display large, styled table
# ---------------------------
st.subheader("📋 Detailed Results (large view)")
display_cols = ['Recommendation', 'Score', 'Sales', 'Volume', 'Margin', 'Explanation',
                'Facings', 'Suggested_Facings', 'Suggested_Facings_Adjusted',
                'Width_in', 'Space_Current', 'Space_Suggested', 'Space_Suggested_Adjusted']
display_cols = [c for c in display_cols if c in df.columns]
df_display = df[display_cols].copy()
# Round numeric columns for display
for c in ['Score', 'Sales', 'Volume', 'Margin', 'Width_in', 'Space_Current', 'Space_Suggested', 'Space_Suggested_Adjusted']:
    if c in df_display.columns:
        df_display[c] = df_display[c].apply(lambda x: round(float(x), 3))

def rec_color(val):
    if val == "Expand":
        return "background-color: #c6efce"
    elif val == "Delist":
        return "background-color: #ffc7ce"
    elif val == "Retain":
        return "background-color: #ffeb9c"
    return ""

styled = df_display.style.applymap(rec_color, subset=['Recommendation'])
st.dataframe(styled, height=600)

# ---------------------------
# Summary chart
# ---------------------------
st.subheader("📊 Recommendation Breakdown")
summary = df['Recommendation'].value_counts().reindex(['Expand','Retain','Delist']).fillna(0)
fig, ax = plt.subplots()
bars = ax.bar(summary.index.astype(str), summary.values)
# color bars
for bar, rec in zip(bars, summary.index):
    if rec == "Expand":
        bar.set_color("#c6efce")
    elif rec == "Retain":
        bar.set_color("#ffeb9c")
    elif rec == "Delist":
        bar.set_color("#ffc7ce")
ax.set_ylabel("Number of SKUs")
ax.set_title("Recommendation Breakdown")
st.pyplot(fig)

# ---------------------------
# Downloads: CSV + Excel with color
# ---------------------------
st.subheader("⬇️ Download Results")

csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv_bytes, file_name="SKU_Recommendations.csv", mime="text/csv")

def to_excel_bytes(df_export):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='SKU_Recommendations')
        workbook = writer.book
        worksheet = writer.sheets['SKU_Recommendations']
        # Formats
        fmt_expand = workbook.add_format({'bg_color': '#c6efce'})
        fmt_retain = workbook.add_format({'bg_color': '#ffeb9c'})
        fmt_delist = workbook.add_format({'bg_color': '#ffc7ce'})
        # Find recommendation column
        headers = df_export.columns.tolist()
        if 'Recommendation' in headers:
            rec_col = headers.index('Recommendation')
            for row_idx, rec_val in enumerate(df_export['Recommendation'], start=1):
                if rec_val == 'Expand':
                    worksheet.write(row_idx, rec_col, rec_val, fmt_expand)
                elif rec_val == 'Retain':
                    worksheet.write(row_idx, rec_col, rec_val, fmt_retain)
                elif rec_val == 'Delist':
                    worksheet.write(row_idx, rec_col, rec_val, fmt_delist)
                else:
                    worksheet.write(row_idx, rec_col, rec_val)
        # Autosize
        for i, col in enumerate(headers):
            max_len = max(df_export[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
        writer.close()
        output.seek(0)
        return output.getvalue()

# Order export columns
preferred_order = ['Recommendation', 'Score', 'Sales', 'Volume', 'Margin', 'Facings',
                   'Suggested_Facings', 'Suggested_Facings_Adjusted', 'Width_in',
                   'Space_Current', 'Space_Suggested', 'Space_Suggested_Adjusted', 'Explanation']
export_cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
df_export = df[export_cols].copy()
excel_data = to_excel_bytes(df_export)
st.download_button("Download color-coded Excel (.xlsx)", data=excel_data, file_name="SKU_Recommendations.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# Tips & optional extra features
# ---------------------------
with st.expander("💡 Tips, notes & optional extras"):
    st.markdown("""
    - The app auto-detects column names using common synonyms — correct them in the sidebar if detection isn't right.
    - Width used for calculations is converted to **inches** and available in the `Width_in` column.
    - Suggested facings are computed continuously from the normalized Score and then rounded to integers. You can tune `min`/`max` multipliers and `sensitivity` in the sidebar.
    - Reduction priority controls which recommendation groups lose facings first when space is short.
    - If you'd like these improvements next:  
      • A visual layout preview (mock shelf strip) showing facings as boxes,  
      • A solver that optimizes revenue subject to width constraints (ILP optimization),  
      • Bulk mapping presets for common retailer templates.  
      Tell me which one to add and I’ll extend the app.
    """)

st.success("App ready — you can tune parameters in the sidebar and re-upload different CSVs for testing.")

