from heapq import nlargest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  STEP 1 – LOAD DATA
# ─────────────────────────────────────────────
print("=" * 58)
print("  LOADING DATASET")
print("=" * 58)


df = pd.read_csv(
    r"C:\Users\jagri\Downloads\archive\Playstore_final.csv",
    on_bad_lines='skip'
)
print(f"  Raw shape : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"  Columns   : {list(df.columns)}\n")

# ─────────────────────────────────────────────
#  STEP 2 – DATA CLEANING
# ─────────────────────────────────────────────
# Cleaning
df_clean = df.dropna()
print(df_clean.columns.tolist())

print("=" * 58)
print("  DATA CLEANING")
print("=" * 58)

# --- Drop the notorious bad row (row 10472 has shifted columns) ---
df = df[df['Category'] != '1.9']

# --- Rating: numeric, drop NaN ---
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5) | df['Rating'].isna()]

# --- Reviews: remove commas → int ---
df['Reviews'] = pd.to_numeric(df['Reviews'].astype(str).str.replace(',', ''), errors='coerce')

# --- Installs: remove '+', ',' → int ---
df['Installs'] = (df['Installs']
                  .astype(str)
                  .str.replace('[+,]', '', regex=True)
                  .str.strip())
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# --- Price: remove '$' → float ---
df['Price'] = (df['Price']
               .astype(str)
               .str.replace('$', '', regex=False)
               .str.strip())
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

# --- Size: convert to MB (float) ---
def parse_size(s):
    s = str(s).strip()
    if s.endswith('M'):
        try: return float(s[:-1])
        except: return np.nan
    elif s.endswith('k'):
        try: return float(s[:-1]) / 1024
        except: return np.nan
    return np.nan

df['Size_MB'] = df['Size'].apply(parse_size)

# --- Last Updated: to datetime ---

# --- Drop rows with missing Rating or Installs ---
before = len(df)
df_clean = df.dropna(subset=['Rating', 'Installs']).copy()
after = len(df_clean)
print(df.columns)

print(f"  Rows before cleaning : {before:,}")
print(f"  Rows after  cleaning : {after:,}")
print(f"  Dropped              : {before - after:,} rows\n")

# ─────────────────────────────────────────────
#  STEP 3 – DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("=" * 58)
print("  DESCRIPTIVE STATISTICS")
print("=" * 58)

desc = df_clean[['Rating', 'Installs', 'Reviews', 'Size_MB']].describe().round(2)
print(desc.to_string())

# print(f"\n  Free apps   : {(df_clean['Type'] == 'Free').sum():,}")
# print(f"  Paid apps   : {(df_clean['Type'] == 'Paid').sum():,}")
print(f"  Categories  : {df_clean['Category'].nunique()}")
print(f"  Most popular category: {df_clean['Category'].value_counts().idxmax()}")
print(f"  Avg Rating  : {df_clean['Rating'].mean():.2f}")
print(f"  Avg Installs: {df_clean['Installs'].mean():,.0f}\n")

# ─────────────────────────────────────────────
#  STEP 4 – INSTALL BUCKET ANALYSIS
# ─────────────────────────────────────────────
print("=" * 58)
print("  INSTALL BUCKET DISTRIBUTION")
print("=" * 58)

bins   = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, float('inf')]
labels = ['<1K', '1K–10K', '10K–100K', '100K–1M', '1M–10M', '10M+']
df_clean['Install_Bucket'] = pd.cut(df_clean['Installs'], bins=bins, labels=labels)

bucket = df_clean['Install_Bucket'].value_counts().sort_index()
for b, cnt in bucket.items():
    pct = cnt / len(df_clean) * 100
    bar = '█' * int(pct / 1.5)
    print(f"  {b:>12} | {bar:<35} {cnt:>5} ({pct:.1f}%)")

# ─────────────────────────────────────────────
#  STEP 5 – CATEGORY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  TOP 10 CATEGORIES BY TOTAL INSTALLS")
print("=" * 58)


cat_stats = df_clean.groupby('Category').agg(
    Total_Installs=('Installs', 'sum'),
    Avg_Rating=('Rating', 'mean'),
    Avg_Reviews=('Reviews', 'mean')
).round(2)

print(cat_stats.head(10).to_string())

# ─────────────────────────────────────────────
#  STEP 6 – FREE vs PAID ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FREE vs PAID COMPARISON")
print("=" * 58)

# ─────────────────────────────────────────────
#  STEP 7 – CORRELATION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  CORRELATION ANALYSIS")
print("=" * 58)

corr_data = df_clean[['Rating', 'Installs', 'Reviews', 'Size_MB']].dropna()
corr_matrix = corr_data.corr(method='pearson').round(3)
print("  Pearson Correlation Matrix:")
print(corr_matrix.to_string())

r, p = stats.pearsonr(corr_data['Rating'], np.log1p(corr_data['Installs']))
print(f"\n  Rating vs log(Installs) → r = {r:.4f}, p = {p:.4e}")

# ─────────────────────────────────────────────
#  STEP 8 – TOP APPS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  TOP 10 APPS BY INSTALLS")
print("=" * 58)

df_clean['Type'] = np.where(df_clean['Price'] > 0, 'Paid', 'Free')

top10 = df_clean.nlargest(10, 'Installs')[
    ['App Name', 'Category', 'Rating', 'Installs', 'Type']
]

print("\n" + "=" * 58)
print("  TOP 10 HIGHEST RATED (min 10,000 reviews)")
print("=" * 58)
top_rated = (
    df_clean[df_clean['Reviews'] >= 10000]
    .nlargest(10, 'Rating')[['App Name', 'Category', 'Rating', 'Reviews']]
)
top_rated = (
    df_clean[df_clean['Reviews'] >= 10000]
    .nlargest(10, 'Rating')[['App Name', 'Category', 'Rating', 'Reviews']]
)

print(top_rated)
print(top_rated.to_string(index=False))

# ─────────────────────────────────────────────
#  STEP 9 – CONTENT RATING ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  CONTENT RATING BREAKDOWN")
print("=" * 58)

cr = df_clean.groupby('Content Rating').agg(
    Avg_Rating=('Rating', 'mean'),
    Avg_Installs=('Installs', 'mean')
).round(2).sort_values('Avg_Installs', ascending=False)

print(cr)
# ─────────────────────────────────────────────
#  STEP 10 – VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n⏳ Generating 9-chart dashboard...")

# ── Dark theme setup ──
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor':   '#161B22',
    'axes.edgecolor':   '#30363D',
    'text.color':       '#C9D1D9',
    'axes.labelcolor':  '#8B949E',
    'xtick.color':      '#8B949E',
    'ytick.color':      '#8B949E',
    'grid.color':       '#21262D',
    'grid.alpha':       0.7,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

C1, C2, C3, C4 = '#58A6FF', '#F78166', '#56D364', '#FFA657'

fig = plt.figure(figsize=(22, 17))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Google Play Store  ·  E-Commerce App Analysis',
             fontsize=24, fontweight='bold', color='white',
             y=0.99, x=0.5)

# ── 1. Rating Distribution ──
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(df_clean['Rating'], bins=35, color=C1, edgecolor='#0D1117', linewidth=0.4, alpha=0.9)
ax1.axvline(df_clean['Rating'].mean(), color=C2, lw=1.8, ls='--',
            label=f"Mean {df_clean['Rating'].mean():.2f}")
ax1.axvline(df_clean['Rating'].median(), color=C3, lw=1.8, ls=':',
            label=f"Median {df_clean['Rating'].median():.2f}")
ax1.set_title('Rating Distribution', color='white', fontweight='bold', pad=8)
ax1.set_xlabel('Rating'); ax1.set_ylabel('Count')
ax1.legend(fontsize=8); ax1.grid(True, axis='y')

# ── 2. Log Installs Distribution ──
ax2 = fig.add_subplot(3, 3, 2)
log_inst = np.log10(df_clean['Installs'].replace(0, np.nan).dropna())
ax2.hist(log_inst, bins=30, color=C2, edgecolor='#0D1117', linewidth=0.4, alpha=0.9)
ax2.set_title('Installs Distribution (log₁₀)', color='white', fontweight='bold', pad=8)
ax2.set_xlabel('log₁₀(Installs)'); ax2.set_ylabel('Count')
ax2.grid(True, axis='y')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'10^{int(x)}' if x == int(x) else ''))

# ── 3. Install Bucket Pie ──
ax3 = fig.add_subplot(3, 3, 3)
bucket_data = df_clean['Install_Bucket'].value_counts().sort_index()
colors_pie = [C1, C2, C3, C4, '#BC8CFF', '#FF7EB6']
wedges, texts, autotexts = ax3.pie(
    bucket_data, labels=bucket_data.index,
    autopct='%1.1f%%', startangle=140,
    colors=colors_pie[:len(bucket_data)],
    pctdistance=0.82,
    wedgeprops=dict(linewidth=1.5, edgecolor='#0D1117')
)
for t in texts + autotexts:
    t.set_color('white'); t.set_fontsize(8)
ax3.set_title('Install Bucket Share', color='white', fontweight='bold', pad=8)

# ── 4. Top 10 Categories by Total Installs ──
ax4 = fig.add_subplot(3, 3, 4)
top_cats = cat_stats.head(10)[::-1]
bars = ax4.barh(top_cats.index, top_cats['Total_Installs'] / 1e9,
                color=C1, edgecolor='#0D1117', alpha=0.88)
ax4.set_title('Top 10 Categories: Total Installs', color='white', fontweight='bold', pad=8)
ax4.set_xlabel('Total Installs (Billions)'); ax4.grid(True, axis='x')
for bar in bars:
    ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1f}B', va='center', fontsize=7, color='white')

# ── 5. Avg Rating per Top 10 Category ──
ax5 = fig.add_subplot(3, 3, 5)
cat_rated = cat_stats.sort_values('Avg_Rating', ascending=False).head(10)
bars = ax5.bar(range(len(cat_rated)), cat_rated['Avg_Rating'],
               color=C3, edgecolor='#0D1117', alpha=0.88)
ax5.set_xticks(range(len(cat_rated)))
ax5.set_xticklabels(cat_rated.index, rotation=38, ha='right', fontsize=7)
ax5.set_ylim(3.5, 5.0)
ax5.set_title('Top 10 Categories: Avg Rating', color='white', fontweight='bold', pad=8)
ax5.set_ylabel('Average Rating'); ax5.grid(True, axis='y')
for bar in bars:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, color='white')

# ── 6. Rating vs log(Installs) Scatter ──
ax6 = fig.add_subplot(3, 3, 6)
sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)
mask = sample['Installs'] > 0
sc = ax6.scatter(sample.loc[mask, 'Rating'],
                 np.log10(sample.loc[mask, 'Installs']),
                 alpha=0.25, s=8,
                 c=sample.loc[mask, 'Rating'], cmap='cool',
                 linewidths=0)
# Trend line
x = sample.loc[mask, 'Rating']
y = np.log10(sample.loc[mask, 'Installs'])
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax6.plot(xs, m*xs+b, color=C2, lw=2, ls='--', label=f'Trend (r={r:.3f})')
ax6.set_title('Rating vs log₁₀(Installs)', color='white', fontweight='bold', pad=8)
ax6.set_xlabel('Rating'); ax6.set_ylabel('log₁₀(Installs)')
ax6.legend(fontsize=8); ax6.grid(True)
plt.colorbar(sc, ax=ax6, label='Rating')

# ── 7. Free vs Paid Avg Rating (grouped bar) ──
ax7 = fig.add_subplot(3, 3, 7)
fvp_cat = (df_clean[df_clean['Type'].isin(['Free', 'Paid'])]
           .groupby(['Category', 'Type'])['Rating']
           .mean().unstack().dropna()
           .sort_values('Free', ascending=False).head(10))
x_pos = np.arange(len(fvp_cat))
width = 0.38
ax7.bar(x_pos - width/2, fvp_cat['Free'], width, label='Free', color=C1, alpha=0.88)
ax7.bar(x_pos + width/2, fvp_cat['Paid'], width, label='Paid', color=C4, alpha=0.88)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(fvp_cat.index, rotation=38, ha='right', fontsize=6.5)
ax7.set_ylim(3.0, 5.0)
ax7.set_title('Free vs Paid Rating by Category', color='white', fontweight='bold', pad=8)
ax7.set_ylabel('Avg Rating'); ax7.legend(fontsize=9); ax7.grid(True, axis='y')

# ── 8. Content Rating Distribution ──
ax8 = fig.add_subplot(3, 3, 8)
cr_counts = df_clean['Content Rating'].value_counts()
wedges2, texts2, autotexts2 = ax8.pie(
    cr_counts, labels=cr_counts.index,
    autopct='%1.1f%%', startangle=90,
    colors=[C1, C3, C2, C4, '#BC8CFF', '#FF7EB6'][:len(cr_counts)],
    pctdistance=0.80,
    wedgeprops=dict(linewidth=1.5, edgecolor='#0D1117')
)
for t in texts2 + autotexts2:
    t.set_color('white'); t.set_fontsize(8)
ax8.set_title('Content Rating Breakdown', color='white', fontweight='bold', pad=8)

# ── 9. Correlation Heatmap ──
ax9 = fig.add_subplot(3, 3, 9)
heat_data = df_clean[['Rating', 'Installs', 'Reviews', 'Size_MB']].dropna()
heat_data['log_Installs'] = np.log1p(heat_data['Installs'])
heat_data['log_Reviews']  = np.log1p(heat_data['Reviews'])
corr_sub = heat_data[['Rating', 'log_Installs', 'log_Reviews', 'Size_MB']].corr()
corr_sub.index   = ['Rating', 'log(Installs)', 'log(Reviews)', 'Size (MB)']
corr_sub.columns = ['Rating', 'log(Installs)', 'log(Reviews)', 'Size (MB)']
mask_upper = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)
sns.heatmap(corr_sub, ax=ax9, mask=mask_upper,
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.3f', annot_kws={'size': 9, 'color': 'white'},
            linewidths=1, linecolor='#0D1117',
            cbar_kws={'label': 'Correlation'})
ax9.set_title('Correlation Heatmap', color='white', fontweight='bold', pad=8)
ax9.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('google_playstore_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("✅ Chart saved → google_playstore_analysis.png\n")
print("=" * 58)
print("  ANALYSIS COMPLETE")
print("=" * 58)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loadingg dataset
# df = pd.read_csv(
#     r"C:\Users\jagri\Downloads\archive\Playstore_final.csv",
#     on_bad_lines='skip',
#     low_memory=False
# )
df = pd.read_csv(
    r"C:\Users\jagri\Downloads\archive\Playstore_final.csv",
    on_bad_lines='skip'
)

st.sidebar.header("🔍 Filter Apps")

df_clean = df.dropna()
category_list = sorted(
    df_clean['Category'].dropna().astype(str).unique()
)

selected_category = st.sidebar.selectbox(
    "📂 Select Category",
    ["All"] + list(category_list)
)

# Rating dropdown
selected_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    0.0,
    5.0,
    3.0
)

# Reviews dropdown
selected_reviews = st.sidebar.selectbox(
    "📝 Minimum Reviews",
    [0, 1000, 5000, 10000, 50000]
)

# Sort dropdown
sort_option = st.sidebar.selectbox(
    "📊 Sort Apps By",
    ["Installs", "Rating", "Reviews"]
)

# Number of apps dropdown
top_n = st.sidebar.selectbox(
    "🔝 Number of Apps to Show",
    [5, 10, 15, 20]
)
# -------------------------------------------------
# FILTER DATA
# -------------------------------------------------

filtered_df = df_clean.copy()

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df['Category'] == selected_category
    ]

filtered_df = filtered_df[
    filtered_df['Rating'] >= selected_rating
]

filtered_df = filtered_df[
    filtered_df['Reviews'] >= selected_reviews
]

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📊 E-Commerce Product Analysis Dashboard")

st.markdown("### Analyze App Ratings, Reviews and Installs")

# -------------------------------------------------
# METRICS
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Apps",
    len(filtered_df)
)

col2.metric(
    "Average Rating",
    round(filtered_df['Rating'].mean(), 2)
)

temp_installs = (
    filtered_df['Installs']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
)

temp_installs = pd.to_numeric(
    temp_installs,
    errors='coerce'
)

col3.metric(
    "Total Installs",
    f"{temp_installs.sum():,.0f}"
)

# Clean Installs column properly

df_clean = df.dropna()
df_clean['Installs'] = (
    df_clean['Installs']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
)

df_clean['Installs'] = pd.to_numeric(
    df_clean['Installs'],
    errors='coerce'
)

# Remove invalid rows

df_clean = df_clean.dropna(subset=['Installs'])

# Convert to integer
df_clean['Installs'] = df_clean['Installs'].astype(int)

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------

st.subheader("📁 Dataset Preview")

st.dataframe(filtered_df.head(top_n))

# -------------------------------------------------
# BAR CHART
# -------------------------------------------------

st.subheader("🏆 Top Categories")

cat_rating = (
    filtered_df.groupby('Category')['Rating']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(cat_rating)

# -------------------------------------------------
# HISTOGRAM
# -------------------------------------------------

st.subheader("⭐ Ratings Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(filtered_df['Rating'], bins=20, ax=ax)

st.pyplot(fig)

# -------------------------------------------------
# SCATTER PLOT
# -------------------------------------------------

st.subheader("📈 Reviews vs Ratings")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x='Reviews',
    y='Rating',
    data=filtered_df,
    ax=ax2
)

st.pyplot(fig2)

# -------------------------------------------------
# HEATMAP
# -------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

# Create a temporary clean dataframe
heatmap_df = filtered_df.copy()

# Clean numeric columns properly
heatmap_df['Installs'] = (
    heatmap_df['Installs']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
)

heatmap_df['Installs'] = pd.to_numeric(
    heatmap_df['Installs'],
    errors='coerce'
)

heatmap_df['Reviews'] = pd.to_numeric(
    heatmap_df['Reviews'],
    errors='coerce'
)

heatmap_df['Rating'] = pd.to_numeric(
    heatmap_df['Rating'],
    errors='coerce'
)

# Keep only valid rows
heatmap_df = heatmap_df[
    ['Rating', 'Reviews', 'Installs']
].dropna()

# Calculate correlation
corr = heatmap_df.corr()

# Plot
fig3, ax3 = plt.subplots(figsize=(7,5))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    linewidths=1,
    ax=ax3
)

st.pyplot(fig3)