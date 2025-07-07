import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

base_path = '/mnt/d/DETECT'

# Create output directory if it does not exist
output_path = os.path.join(base_path, 'OUTPUT', 'KMeans_Clustering_Yearly')
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

#create a new column for age from birth year
df_clean['age'] = df_clean['date'].dt.year - df_clean['birthyr']

#encode sex as numeric
df_clean['sex_encoded'] = df_clean['sex'].map({1: 1, 2: 0})

#encode cognitive status as numeric
df_clean['cogstat_encoded'] = df_clean['cogstat']

# Define feature and demographic columns
feature_cols = [
    'steps', 'awakenings', 'bedexitcount', 'end_sleep_time', 'gait_speed',
     'durationinsleep', 'inbed_time', 'outbed_time', 'durationawake', 'sleepscore',
    'waso', 'hrvscore', 'start_sleep_time', #'time_to_sleep',
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', 'time_in_bed_after_sleep',
    #include demographic features
    'age', 'sex_encoded', 'cogstat_encoded'
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
    
    # Tune K using both Elbow (Inertia) and Silhouette Score
    inertias = []
    sil_scores = []
    k_range = range(2, 11)
    best_k, best_sil_score = 2, -1
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)
        
        inertias.append(inertia)
        sil_scores.append(sil)
        
        print(f"Year {year} - k={k} | Inertia: {inertia:.0f} | Silhouette: {sil:.3f}")
        if sil > best_sil_score:
            best_k, best_sil_score = k, sil

	# Plot Elbow and Silhouette side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

	# Elbow plot
    axs[0].plot(k_range, inertias, marker='o')
    axs[0].set_title(f'Elbow Method (Inertia) - {year}')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('Inertia')

	# Silhouette plot
    axs[1].plot(k_range, sil_scores, marker='o', color='green')
    axs[1].set_title(f'Silhouette Scores - {year}')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"elbow_silhouette_{year}.png"), dpi=300)
    plt.close()
    
    print(f"Best k for year {year} based on silhouette: {best_k}")

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    subgroup['cluster'] = kmeans.fit_predict(X_scaled)
    
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
    subgroup['cluster'] = KMeans(n_clusters=len(valid_clusters), random_state=42).fit_predict(X_scaled_filtered)
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
final_df.to_csv(os.path.join(output_path, 'yearly_kmeans_clusters_all.csv'), index=False)
pd.DataFrame(metrics).to_csv(os.path.join(output_path, 'clustering_metrics_all.csv'), index=False)

composition.to_csv(os.path.join(output_path, f"amyloid_composition_year_{year}.csv"))

#feature averages by cluster and year
cluster_summary = final_df.groupby(['year', 'amyloid_status','cluster'])[feature_cols].mean().round(2)
print(cluster_summary)

#save cluster summary to CSV
summary_file = os.path.join(output_path, 'yearly_cluster_summary_by_amyloid.csv')
cluster_summary.to_csv(summary_file)
print(f"Cluster summary saved to {summary_file}")

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