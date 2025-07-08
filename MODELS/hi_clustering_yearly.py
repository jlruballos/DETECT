import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

base_path = '/mnt/d/DETECT'

# Create output directory if it does not exist
output_path = os.path.join(base_path, 'OUTPUT', 'Hierarchical_Clustering_Yearly')
os.makedirs(output_path, exist_ok=True)
plot_path = os.path.join(output_path, "plots")
os.makedirs(plot_path, exist_ok=True)
box_plot_path = os.path.join(output_path, "boxplots")
os.makedirs(box_plot_path, exist_ok=True)

df = pd.read_csv("/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv", parse_dates=["date"])


demo_cols = ['birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat','moca_avg', 'cogstat', 'amyloid']


df_clean = df.copy()

#intial amyloid participants
initial_amyloid_pos = set(df_clean[df_clean['amyloid'] == 1]['subid'].unique())

# Example preprocessing
demo_df = df_clean[demo_cols].copy()

# Drop amyloid for unsupervised clustering
demo_df = demo_df.drop(columns=['amyloid'])

#create a new column for age from birth year
df_clean['age'] = df_clean['date'].dt.year - df_clean['birthyr']

#encode sex as numeric
df_clean['sex_encoded'] = df_clean['sex'].map({1: 1, 2: 0})

#encode cognitive status as numeric
cogstat_map = {
    1: 0,  # Better than normal
    2: 1,  # Normal
    3: 2,  # One or two abnormal scores
    4: 3,  # Three or more abnormal scores
    0: -1  # Unable to render opinion
}

df_clean['cogstat_encoded'] = df_clean['cogstat'].map(cogstat_map)

df_clean['hispanic_encoded'] = df_clean['hispanic'].fillna(0).astype(int)
df_clean['race_encoded'] = df_clean['race'].fillna(-1).astype(int)
df_clean['educ_encoded'] = df_clean['educ'].fillna(-1).astype(int)
df_clean['independ_encoded'] = df_clean['independ'].fillna(-1).astype(int)
df_clean['residenc_encoded'] = df_clean['residenc'].fillna(-1).astype(int)
df_clean['livsitua_encoded'] = df_clean['livsitua'].fillna(-1).astype(int)
df_clean['maristat_encoded'] = df_clean['maristat'].fillna(-1).astype(int)
df_clean['moca_avg'] = df_clean['moca_avg'].fillna(df_clean['moca_avg'].mean())

#encode alzheimers disease status
df_clean['alzdis_encoded'] = df_clean['alzdis'].fillna(0).astype(int)  # Fill NaN with 0 and convert to int

# Define feature and demographic columns
feature_cols = [
    'steps', 'awakenings', 'bedexitcount', 'end_sleep_time', 'gait_speed',
     'durationinsleep', 'inbed_time', 'outbed_time', 'durationawake', 'sleepscore',
    'waso', 'hrvscore', 'start_sleep_time', #'time_to_sleep',
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', 'time_in_bed_after_sleep',
    #include demographic features
    'age', 'sex_encoded', 'cogstat_encoded', 'alzdis_encoded', 'hispanic_encoded',
    'race_encoded', 'educ_encoded', 'independ_encoded', 'residenc_encoded',
    'livsitua_encoded', 'maristat_encoded', 'moca_avg'
]

#drop rows with NaN in feature columns or date
#df_clean = df_clean.dropna(subset=feature_cols + ['date']).copy()

#fill NaN values in feature columns with the mean of each column
df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].mean())

#remove rows with negative values in features
df_clean = df_clean[(df_clean[feature_cols] >= 0).all(axis=1)].copy()


#prepare for clustering
df_clean['amyloid'] = df_clean['amyloid'].astype(int) # #0 for negative, 1 for positive

df_clean['year'] = df_clean['date'].dt.year  # Add year column

#cluster per year
yearly_clusters = []
metrics = []

for year, group in df_clean.groupby('year'):
    subgroup = group.dropna(subset=feature_cols).copy()
    if subgroup.empty:
        print(f"Skipping year {year} - no data available.")
        continue
    print(f"Processing year: {year}")

    # Standardize
    X = subgroup[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    link_matrix = linkage(X_scaled, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(
		link_matrix,
		truncate_mode='lastp',  # show only last p merged clusters
		p=25,                   # number of leaf nodes to show
		leaf_rotation=45.,
		leaf_font_size=10.
	)
    plt.title(f"Dendrogram - {year}")
    plt.xlabel("Sample index or (cluster size)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"dendrogram_{year}.png"), dpi=300)
    plt.close()
    
    #PCA reduction before clustering
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"Year {year}: Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} PCA components.")

    # Hierarchical Clustering and Silhouette Score
    sil_scores = []
    k_range = range(2, 11)
    best_k, best_sil_score = 2, -1
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        
        print(f"Year {year} - k={k}| Silhouette: {sil:.3f}")

        #PCA Hierarchical calculation
        model_pca = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_pca = model_pca.fit_predict(X_pca)
        sil_pca = silhouette_score(X_pca, labels_pca)
        print(f"Year {year} - k={k} | Silhouette (PCA): {sil_pca:.3f}")

        sil_scores.append(sil)
        
        if sil > best_sil_score:
            best_k, best_sil_score = k, sil
    
    print(f"Best k for year {year} based on silhouette: {best_k}")

    # Final clustering
    model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    subgroup['cluster'] = model.fit_predict(X_scaled)
    
    composition = subgroup.groupby('cluster')['amyloid'].value_counts(normalize=True).unstack().fillna(0)
    print(f"Amyloid composition in clusters for {year}:\n{composition}")
    composition_file = os.path.join(output_path, f"amyloid_composition_year_{year}.csv")
    composition.to_csv(composition_file)
    print(f"Saved amyloid composition to {composition_file}")

    # Filter out clusters with <2 participants
    participant_counts = subgroup.groupby('cluster')['subid'].nunique()
    valid_clusters = participant_counts[participant_counts >= 2].index
    subgroup = subgroup[subgroup['cluster'].isin(valid_clusters)].copy()

    if subgroup.empty:
        print(f"⚠️ All clusters in year {year} had <2 participants — skipping.")
        continue

    # Refit + PCA
    X_scaled_filtered = scaler.fit_transform(subgroup[feature_cols])
    subgroup['cluster'] = AgglomerativeClustering(n_clusters=len(valid_clusters), linkage='ward').fit_predict(X_scaled_filtered)
    pca = PCA(n_components=2).fit_transform(X_scaled_filtered)
    subgroup['pca1'], subgroup['pca2'] = pca[:, 0], pca[:, 1]

    # Save metrics
    sil_score = silhouette_score(X_scaled_filtered, subgroup['cluster']) if len(valid_clusters) > 1 else None
    ch_score = calinski_harabasz_score(X_scaled_filtered, subgroup['cluster']) if len(valid_clusters) > 1 else None
    db_score = davies_bouldin_score(X_scaled_filtered, subgroup['cluster']) if len(valid_clusters) > 1 else None
    metrics.append({
        'year': str(year),
        'n_clusters': len(valid_clusters),
        'silhouette_score': round(sil_score, 4) if sil_score else None,
        'calinski_harabasz_score': round(ch_score, 4) if ch_score else None,
		'davies_bouldin_score': round(db_score, 4) if db_score else None,
		'n_participants': subgroup['subid'].nunique(),
        'min_participants_in_cluster': participant_counts.min(),
        'max_participants_in_cluster': participant_counts.max(),
        'total_days': len(subgroup),
        'unique_participants': subgroup['subid'].nunique(),
    })
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
		data=subgroup,
		x='pca1',
		y='pca2',
		hue='cluster',
		palette='Set2',
		s=60,
		legend='full'
	)
    plt.title(f"PCA Clusters - Year: {year}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

	# Save plot
    plot_filename = f"pca_clusters_year_{str(year)}.png"
    plot_file_path = os.path.join(plot_path, plot_filename)
    plt.savefig(plot_file_path, dpi=300)
    plt.close()
    
    print(f"Saved PCA plot with subid labels to {plot_file_path}")
    
    subgroup['amyloid_status'] = subgroup['amyloid']

	#save filtered participants with clusters
    participant_clusters = (
		subgroup
		.groupby(['subid', 'year', 'amyloid_status', 'cluster'])
		.size()
		.reset_index(name='days')
	)
	# Save to file
    filename = f'participants_year_{year}.csv'
    participant_clusters.to_csv(os.path.join(output_path, filename), index=False)
    print(f"Saved participant info to {filename}")

	#append to yearly clusters
    yearly_clusters.append(subgroup)

# Combine and save
final_df = pd.concat(yearly_clusters, ignore_index=True)
final_df.to_csv(os.path.join(output_path, 'yearly_hierarchical_clusters_all.csv'), index=False)
pd.DataFrame(metrics).to_csv(os.path.join(output_path, 'clustering_metrics_all.csv'), index=False)

composition.to_csv(os.path.join(output_path, f"amyloid_composition_year_{year}.csv"))

#feature averages by cluster and year
cluster_summary = final_df.groupby(['year', 'amyloid_status','cluster'])[feature_cols].mean().round(2)
print(cluster_summary)

#save cluster summary to CSV
summary_file = os.path.join(output_path, 'yearly_cluster_summary_amyloid.csv')
cluster_summary.to_csv(summary_file)
print(f"Cluster summary saved to {summary_file}")

#feature averages by cluster and year
cluster_summary = final_df.groupby(['year', 'cluster'])[feature_cols].mean().round(2)

general_summary_file = os.path.join(output_path, 'yearly_cluster_summary.csv')
cluster_summary.to_csv(general_summary_file)
print(f"General cluster summary saved to {general_summary_file}")

# Boxplots
for feature in feature_cols:
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=final_df,
        x='cluster',
        y=feature,
        palette='Set2',
        hue='cluster',
        showmeans=False,  # Set to False since we're annotating medians manually
        legend=False
    )

    # Get unique clusters in order
    clusters = sorted(final_df['cluster'].unique())

    # Annotate number of participants and median values per cluster
    for i, cluster in enumerate(clusters):
        sub_data = final_df[final_df['cluster'] == cluster]
        count = sub_data['subid'].nunique()
        median_val = sub_data[feature].median()
        
        # Place text slightly above top whisker
        y_max = sub_data[feature].max()
        ax.text(i, y_max * 1.02, f"n={count}", ha='center', fontsize=9, color='blue')
        ax.text(i, y_max * 0.95, f"med={median_val:.1f}", ha='center', fontsize=9, color='black')

    plt.title(f"{feature} by Cluster (All Participants)")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(box_plot_path, f"boxplot_{feature}_all.png"), dpi=300)
    plt.close()

# Plot cluster counts by year
plt.figure(figsize=(12, 6))
sns.countplot(data=final_df, x='year', hue='cluster', palette='Set2')
plt.title("Cluster Counts by Year")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "cluster_counts_by_year.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close after saving