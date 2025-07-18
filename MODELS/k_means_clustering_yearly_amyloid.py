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
output_path = os.path.join(base_path, 'OUTPUT', 'KMeans_Clustering_Yearly_Amyloid')
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

# Encode cognitive status using mapping
cogstat_map = {
    1: 0,  # Better than normal
    2: 1,  # Normal
    3: 2,  # One or two abnormal scores
    4: 3,  # Three or more abnormal scores
    0: -1  # Unable to render opinion
}
df_clean['cogstat_encoded'] = df_clean['cogstat'].map(cogstat_map)

# Encode and fill other demographics
df_clean['hispanic_encoded'] = df_clean['hispanic'].fillna(0).astype(int)
df_clean['race_encoded'] = df_clean['race'].fillna(-1).astype(int)
df_clean['educ_encoded'] = df_clean['educ'].fillna(-1).astype(int)
df_clean['independ_encoded'] = df_clean['independ'].fillna(-1).astype(int)
df_clean['residenc_encoded'] = df_clean['residenc'].fillna(-1).astype(int)
df_clean['livsitua_encoded'] = df_clean['livsitua'].fillna(-1).astype(int)
df_clean['maristat_encoded'] = df_clean['maristat'].fillna(-1).astype(int)
df_clean['moca_avg'] = df_clean['moca_avg'].fillna(df_clean['moca_avg'].mean())
df_clean['alzdis_encoded'] = df_clean['alzdis'].fillna(0).astype(int)


# Define feature and demographic columns
feature_cols = [
    'steps', 'awakenings', 'bedexitcount', 'end_sleep_time', 'gait_speed',
     'durationinsleep', 'inbed_time', 'outbed_time', 'durationawake', 'sleepscore',
    'waso', 'hrvscore', 'start_sleep_time', 'time_to_sleep',
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', 'time_in_bed_after_sleep', 'Night_Bathroom_Visits', 'Night_Kitchen_Visits',
     'label_fall', 'label_hospital', 'label_accident',
    #include demographic features
    # 'age', 'sex_encoded', 'cogstat_encoded', 'alzdis_encoded', 'hispanic_encoded',
    # 'race_encoded', 'educ_encoded', 'independ_encoded', 'residenc_encoded',
    # 'livsitua_encoded', 'maristat_encoded', 'moca_avg'
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

#print initial amyloid participants
print(f"Initial amyloid-positive participants: {len(initial_amyloid_pos)}")

for year, group in df_clean.groupby('year'):
    for status in [0,1]: #0 = negative, 1 = positive
        #subgroup = group[group['amyloid'] == status].dropna(subset=feature_cols).copy()
        subgroup = group[group['amyloid'] == status].copy()
        #print inital amyloid participants
        if status == 1:
            print(f"Year {year} - Initial amyloid-positive participants: {len(subgroup['subid'].unique())}")
   
        if subgroup.empty:
            print(f"Skipping year {year} for amyloid status {status} - no data available.")
            continue
        print(f"Processing year: {year}, Amyloid: {status}")

        #standardize the feature values
        X = subgroup[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Year {year}: Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} PCA components.")
        
        best_k = 2  # Reset best_k for each month
        best_sil_score = -1  # Reset best silhouette score for each month
        for k in range(3, 20):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            labels_pca = kmeans.fit_predict(X_pca)
            sil_score = silhouette_score(X_scaled, labels)
            sil_score_pca = silhouette_score(X_pca, labels_pca)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch_score_pca = calinski_harabasz_score(X_pca, labels_pca)
            db_pca = davies_bouldin_score(X_pca, labels_pca)
            print(f"Year {year} - k={k:<2} | Silhouette Score: {sil_score:.3f} | Silhouette (PCA): {sil_score_pca:.3f} | Calinski-Harabasz: {ch_score:.3f} | Calinski-Harabasz (PCA): {ch_score_pca:.3f} | Davies-Bouldin: {db:.3f} | Davies-Bouldin (PCA): {db_pca:.3f}")
            if sil_score_pca > best_sil_score:
                best_k = k
                best_sil_score = sil_score_pca

        print(f"Best k for year {year}: {best_k} with Silhouette Score: {best_sil_score:.3f}")
		
		#run final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        subgroup['cluster'] = kmeans.fit_predict(X_scaled)
        subgroup['n_clusters'] = best_k  # Store the number of clusters used
        subgroup['silhouette_score'] = best_sil_score
        
        unique_labels = subgroup['cluster'].nunique()

        if unique_labels > 1:
           sil_score = silhouette_score(X_scaled, subgroup['cluster'])
           ch_score = calinski_harabasz_score(X_scaled, subgroup['cluster'])
           db = davies_bouldin_score(X_scaled, subgroup['cluster'])
        else:
           sil_score = ch_score = db = None
           print(f"⚠️ Skipping metrics for year {year}, amyloid {status} — only 1 cluster remains.")
           
        # Run PCA
        pca_plot = PCA(n_components=2)
        X_pca_plot = pca_plot.fit_transform(X_scaled)
        subgroup['pca1'] = X_pca_plot[:, 0]
        subgroup['pca2'] = X_pca_plot[:, 1]

        subgroup['amyloid_status'] = status
        
        #unique participants in each cluster
        participant_counts = subgroup.groupby('cluster')['subid'].nunique()

        metrics.append({
        'year': str(year),
        'amyloid_status': status,
        'n_clusters_initial': best_k,
        'silhouette': round(best_sil_score, 4),
        'silhouette_final': round(sil_score, 4) if sil_score is not None else None,
		'calinski_harabasz': round(ch_score, 4) if ch_score is not None else None,
		'davies_bouldin': round(db, 4) if db is not None else None,
        'min_participants_in_cluster': participant_counts.min(),
        'max_participants_in_cluster': participant_counts.max(),
        'cluster_sizes': participant_counts.to_dict(),
        'n_clusters_final': len(subgroup['cluster'].unique()),
        'total_days': len(subgroup),
        'unique_participants': subgroup['subid'].nunique()

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

		# # Add subid labels
        # for _, row in subgroup.iterrows():
        #     plt.text(
		# 		row['pca1'] + 0.05,  # small horizontal offset
		# 		row['pca2'] + 0.05,  # small vertical offset
		# 		str(int(row['subid'])),  # cast to int then string
		# 		fontsize=7,
		# 		alpha=0.6
		# 	)
        plt.title(f"PCA Clusters - Year: {year}, Amyloid: {status}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

		# Save plot
        plot_filename = f"pca_clusters_year_{str(year)}_amyloid_{status}.png"
        plot_file_path = os.path.join(plot_path, plot_filename)
        plt.savefig(plot_file_path, dpi=300)
        plt.close()
        
        print(f"Saved PCA plot with subid labels to {plot_file_path}")
		#save filtered participants with clusters
        participant_clusters = (
			subgroup
            .groupby(['subid', 'year', 'amyloid_status', 'cluster'])
            .size()
            .reset_index(name='days')
		)
        # Save to file
        filename = f'participants_year_{year}_amyloid_{status}.csv'
        participant_clusters.to_csv(os.path.join(output_path, filename), index=False)
        print(f"Saved participant info to {filename}")

		#append to yearly clusters
        yearly_clusters.append(subgroup)

# Combine all yearly clusters into a single DataFrame
df_yearly_clusters = pd.concat(yearly_clusters, ignore_index=True)

# Compare how many amyloid-positive participants were retained
final_amyloid_pos = set(df_yearly_clusters[df_yearly_clusters['amyloid'] == 1]['subid'].unique())
lost_participants = initial_amyloid_pos - final_amyloid_pos

print(f"Initial amyloid-positive participants: {len(initial_amyloid_pos)}")
print(f"Final clustered amyloid-positive participants: {len(final_amyloid_pos)}")
print("Participants dropped from clustering:", lost_participants)

pd.DataFrame({'lost_amyloid_positive_subids': list(lost_participants)}).to_csv(
    os.path.join(output_path, "lost_amyloid_positive_participants.csv"), index=False
)

dropped_df = df[df['subid'].isin(lost_participants)]
dropped_df[['subid',  'amyloid']].drop_duplicates().sort_values(by='subid')

#save dropped participants to CSV
dropped_file = os.path.join(output_path, 'dropped_participants.csv')
dropped_df.to_csv(dropped_file, index=False)
print(f"Saved dropped participants to {dropped_file}")

df_metrics = pd.DataFrame(metrics)
metrics_file = os.path.join(output_path, "clustering_metrics.csv")
df_metrics.to_csv(metrics_file, index=False)
print(f"Saved clustering metrics to {metrics_file}")

# Convert year to string and sort
df_yearly_clusters['year'] = df_yearly_clusters['year'].astype(str)
df_yearly_clusters = df_yearly_clusters.sort_values(by='year')

# Save the yearly clustering results
output_file = os.path.join(output_path, 'yearly_kmeans_clustersby_amyloid.csv')
df_yearly_clusters.to_csv(output_file, index=False)
print(f"Yearly clustering results saved to {output_file}")

#feature averages by cluster and year
cluster_summary = df_yearly_clusters.groupby(['year', 'amyloid_status','cluster'])[feature_cols].mean().round(2)
print(cluster_summary)

#save cluster summary to CSV
summary_file = os.path.join(output_path, 'yearly_cluster_summary_by_amyloid.csv')
cluster_summary.to_csv(summary_file)
print(f"Cluster summary saved to {summary_file}")

sizes = df_yearly_clusters.groupby(['year', 'amyloid_status', 'cluster']).size().unstack(fill_value=0)
print(sizes)

# Plot cluster counts by year
plt.figure(figsize=(12, 6))
sns.countplot(data=df_yearly_clusters, x='year', hue='cluster', palette='Set2')
plt.title("Cluster Counts by Year")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "cluster_counts_by_year.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close after saving

# Plot participants per year by amyloid status
plt.figure(figsize=(12, 6))
sns.countplot(data=df_yearly_clusters, x='year', hue='amyloid_status', palette='Set1')
plt.title("Participants per Year by Amyloid Status")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "amyloid_by_year.png"), dpi=300, bbox_inches='tight')
plt.close()

#genearate blox plots for each year, cluster and feature
subset_amyloid_pos = df_yearly_clusters[
	(df_yearly_clusters['amyloid_status'] == 1) &
    (df_yearly_clusters['year'] == '2024')  # Change to the year you want to plot
]

subset_amyloid_neg = df_yearly_clusters[
	(df_yearly_clusters['amyloid_status'] == 0) &
    (df_yearly_clusters['year'] == '2024')  # Change to the year you want to plot
]

for feature in feature_cols:
	# Positive group boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=subset_amyloid_pos, x='cluster', y=feature, hue='cluster', palette='Set2', legend=False)

    # Add subid count annotations
    counts = subset_amyloid_pos.groupby('cluster')['subid'].nunique()
    for i, cluster in enumerate(sorted(subset_amyloid_pos['cluster'].unique())):
        count = counts[cluster]
        ax.text(i, ax.get_ylim()[1]*0.95, f'n={count}', ha='center', va='top', fontsize=9)

    plt.title(f"Amyloid Positive - {feature} by Cluster (Year: {year})")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(box_plot_path, f"amyloid_pos_{feature}_by_cluster_{year}.png"), dpi=300, bbox_inches='tight')
    plt.close()

	# Negative group boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=subset_amyloid_neg, x='cluster', y=feature, hue='cluster', palette='Set2', legend=False)

	# Add subid count annotations
    counts = subset_amyloid_neg.groupby('cluster')['subid'].nunique()
    for i, cluster in enumerate(sorted(subset_amyloid_neg['cluster'].unique())):
        count = counts[cluster]
        ax.text(i, ax.get_ylim()[1]*0.95, f'n={count}', ha='center', va='top', fontsize=9)
    
    plt.title(f"Amyloid Negative - {feature} by Cluster (Year: {year})")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(box_plot_path, f"amyloid_neg_{feature}_by_cluster_{year}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
#plot both negative and positive in one figure per feature
subset = df_yearly_clusters[df_yearly_clusters['year'] == '2024'].copy()  # Change to the year you want to plot


subset.loc[:, 'group'] = subset.apply(
    lambda row: f"Amyloid {row['amyloid_status']} - Cluster {row['cluster']}", axis=1
)

for feature in feature_cols:
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=subset, x='group', y=feature, hue='group', palette='Set3', legend=False)

    # Annotate with subid count per group
    counts = subset.groupby('group')['subid'].nunique()
    for i, group_label in enumerate(sorted(subset['group'].unique())):
        count = counts[group_label]
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={count}", ha='center', fontsize=8)

    plt.title(f"{feature} by Amyloid Status and Cluster (Year {year})")
    plt.xlabel("Amyloid Group & Cluster")
    plt.ylabel(feature)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(box_plot_path, f"grouped_boxplot_{feature}_{year}.png"), dpi=300)
    plt.close()
