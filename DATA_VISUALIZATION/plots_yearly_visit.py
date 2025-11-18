#!/usr/bin/env python3

import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# ---------- CONFIG ----------
input_path = "/mnt/d/DETECT/OUTPUT/yearly_visit_processing/yearly_recoded.csv"
output_dir = "/mnt/d/DETECT/OUTPUT/yearly_visit_processing/demographic_plots_recoded"
os.makedirs(output_dir, exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv(input_path, parse_dates=['visit_date'])
df['subid'] = df['subid'].astype(str)
df['year'] = df['visit_date'].dt.year

# ---------- Demographic Features ----------
DEMO_FEATURES = [
    'birthyr', 'sex', 'hispanic', 'race_group', 'educ_group', 'livsitua_recoded', 'independ',
    'residenc', 'alzdis', 'maristat', 'cogstat',
    'primlang', 'moca_category', 'age_bucket', 'birthyr', 'alzdis', 'maristat_recoded',
]

cat_features = [col for col in DEMO_FEATURES if col in df.columns and
                (not is_numeric_dtype(df[col]) or df[col].nunique() <= 50)]

# ---------- Output Directories ----------
freq_dir = os.path.join(output_dir, 'frequency_tables')
os.makedirs(freq_dir, exist_ok=True)
plot_dirs = {
    'stacked_counts': os.path.join(output_dir, 'barplots_counts_stacked'),
    'stacked_props': os.path.join(output_dir, 'barplots_props_stacked'),
    'side_counts': os.path.join(output_dir, 'barplots_counts_side'),
    'side_props': os.path.join(output_dir, 'barplots_props_side')
}
for d in plot_dirs.values():
    os.makedirs(d, exist_ok=True)

# ---------- Plotting ----------
def plot_stacked(df_plot, feature, kind='count'):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                label = f'{height:.1f}%' if kind == 'prop' else f'{int(height)}'
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_y() + height),
                            ha='center', va='bottom', fontsize=8)

    ax.set_title(f"{feature.capitalize()} Distribution Over Years (Stacked)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion" if kind == 'prop' else "Count")
    ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_side_by_side(df, feature, kind='count'):
    plot_df = df.groupby(['year', feature]).size().reset_index(name='count')
    total_per_year = plot_df.groupby('year')['count'].transform('sum')
    if kind == 'prop':
        plot_df['count'] = 100 * plot_df['count'] / total_per_year
    
    # --- Order by overall mean count ---
    order = plot_df.groupby(feature)['count'].mean().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x='year', y='count', hue=feature, hue_order=order, ax=ax)

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                label = f'{height:.1f}%' if kind == 'prop' else f'{int(height)}'
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, height),
                            ha='center', va='bottom', fontsize=8)

    ax.set_title(f"{feature.capitalize()} Over Years (Side-by-Side)")
    ax.set_ylabel("Proportion" if kind == 'prop' else "Count")
    ax.set_xlabel("Year")
    ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# ---------- Loop through features ----------
for feature in cat_features:
    if feature not in df.columns:
        continue

    freq_table = df.groupby('year')[feature].value_counts(dropna=False).unstack().fillna(0).astype(int)
    freq_table.to_csv(os.path.join(freq_dir, f"{feature}_freq_by_year.csv"))

    # ---- Stacked Count ----
    fig1 = plot_stacked(freq_table, feature, kind='count')
    fig1.savefig(os.path.join(plot_dirs['stacked_counts'], f"{feature}_stacked_counts.png"))
    plt.close(fig1)

    # ---- Stacked Proportion ----
    freq_prop = freq_table.div(freq_table.sum(axis=1), axis=0) * 100
    fig2 = plot_stacked(freq_prop, feature, kind='prop')
    fig2.savefig(os.path.join(plot_dirs['stacked_props'], f"{feature}_stacked_props.png"))
    plt.close(fig2)

    # ---- Side-by-Side Count ----
    fig3 = plot_side_by_side(df, feature, kind='count')
    fig3.savefig(os.path.join(plot_dirs['side_counts'], f"{feature}_side_counts.png"))
    plt.close(fig3)

    # ---- Side-by-Side Proportion ----
    fig4 = plot_side_by_side(df, feature, kind='prop')
    fig4.savefig(os.path.join(plot_dirs['side_props'], f"{feature}_side_props.png"))
    plt.close(fig4)
