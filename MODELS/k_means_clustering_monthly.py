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
output_path = os.path.join(base_path, 'OUTPUT', 'KMeans_Clustering_Monthly')
os.makedirs(output_path, exist_ok=True)
plot_path = os.path.join(output_path, "plots")
os.makedirs(plot_path, exist_ok=True)

df = pd.read_csv("/mnt/d/DETECT/OUTPUT/sequence_feature/labeled_daily_data_mean.csv", parse_dates=["date"])


demo_cols = ['birthyr', 'sex', 'hispanic', 'race', 'educ', 'livsitua', 'independ', 'residenc', 'alzdis', 'maristat','moca_avg', 'cogstat', 'amyloid']


df_clean = df.copy()

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
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', #'end_sleep_time',
     'durationinsleep', 'inbed_time', 'outbed_time', #'durationawake', 'sleepscore',
    'waso', 'hrvscore', #'start_sleep_time', 'time_to_sleep',
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', #'time_in_bed_after_sleep',
    #include demographic features
    'age', 'sex_encoded', 'cogstat_encoded', 'alzdis_encoded', 'hispanic_encoded',
    'race_encoded', 'educ_encoded', 'independ_encoded', 'residenc_encoded',
    'livsitua_encoded', 'maristat_encoded', 'moca_avg'
]

#fill NaN values in feature columns with the mean of each column
df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].mean())

#remove rows with negative values in features
df_clean = df_clean[(df_clean[feature_cols] >= 0).all(axis=1)].copy()

#prepare for clustering
df_clean['amyloid'] = df_clean['amyloid'].astype(int) # #0 for negative, 1 for positive

df_clean['month'] = df_clean['date'].dt.to_period('M')  # Add month column

#cluster per month
monthly_clusters = []
metrics = []

for month, group in df_clean.groupby('month'):
    for status in [0,1]: #0 = negative, 1 = positive
        subgroup = group[group['amyloid'] == status].dropna(subset=feature_cols).copy()
        if subgroup.empty:
            print(f"Skipping month {month} for amyloid status {status} - no data available.")
            continue
        print(f"Processing month: {month}, Amyloid: {status}")

        #standardize the feature values
        X = subgroup[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        #PCA reduction before clustering
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_pca = pca.fit_transform(X_scaled)
        print(f"Month {month}: Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} PCA components.")

        best_k = 2  # Reset best_k for each month
        best_sil_score = -1  # Reset best silhouette score for each month
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            labels_pca = kmeans.fit_predict(X_pca)
            sil_score = silhouette_score(X_scaled, labels)
            sil_score_pca = silhouette_score(X_pca, labels_pca)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            print(f"Month {month} - k={k:<2} | Silhouette Score: {sil_score:.3f} | Silhouette (PCA): {sil_score_pca:.3f}")
            if sil_score > best_sil_score:
                best_k = k
                best_sil_score = sil_score
        
        print(f"Best k for month {month}: {best_k} with Silhouette Score: {best_sil_score:.3f}")
		
		#run final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        subgroup['cluster'] = kmeans.fit_predict(X_scaled)
        subgroup['n_clusters'] = best_k  # Store the number of clusters used
        subgroup['silhouette_score'] = best_sil_score

        subgroup['amyloid_status'] = status

        metrics.append({
        'month': str(month),
        'amyloid_status': status,
        'n_clusters': best_k,
        'silhouette': round(best_sil_score, 4),
		'calinski_harabasz': round(ch_score, 4),
		'davies_bouldin': round(db, 4)
         })
        
        # Run PCA and attach components before plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        subgroup['pca1'] = X_pca[:, 0]
        subgroup['pca2'] = X_pca[:, 1]

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
        plt.title(f"PCA Clusters - Month: {month}, Amyloid: {status}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

		# Save plot
        plot_filename = f"pca_clusters_month_{str(month)}_amyloid_{status}.png"
        plot_file_path = os.path.join(plot_path, plot_filename)
        plt.savefig(plot_file_path, dpi=300)
        plt.close()
        
        print(f"Saved PCA plot with subid labels to {plot_file_path}")

		#append to monthly clusters
        monthly_clusters.append(subgroup)
    
# Combine all monthly clusters into a single DataFrame
df_monthly_clusters = pd.concat(monthly_clusters, ignore_index=True)

df_metrics = pd.DataFrame(metrics)
metrics_file = os.path.join(output_path, "clustering_metrics.csv")
df_metrics.to_csv(metrics_file, index=False)
print(f"Saved clustering metrics to {metrics_file}")

# Convert month to string and sort
df_monthly_clusters['month'] = df_monthly_clusters['month'].astype(str)
df_monthly_clusters = df_monthly_clusters.sort_values(by='month')

# Save the monthly clustering results
output_file = os.path.join(output_path, 'monthly_kmeans_clustersby_amyloid.csv')
df_monthly_clusters.to_csv(output_file, index=False)
print(f"Monthly clustering results saved to {output_file}")

#feature averages by cluster and month
cluster_summary = df_monthly_clusters.groupby(['month', 'amyloid_status','cluster'])[feature_cols].mean().round(2)
print(cluster_summary)

#save cluster summary to CSV
summary_file = os.path.join(output_path, 'monthly_cluster_summary_by_amyloid.csv')
cluster_summary.to_csv(summary_file)
print(f"Cluster summary saved to {summary_file}")

sizes = df_monthly_clusters.groupby(['month', 'amyloid_status', 'cluster']).size().unstack(fill_value=0)
print(sizes)

#feature averages by cluster and month
agg_df = df_monthly_clusters.groupby(['month', 'cluster'])[feature_cols].mean().round(2)
print(agg_df)
# Save aggregated DataFrame to CSV
agg_file = os.path.join(output_path, 'monthly_aggregated_clusters.csv')
agg_df.to_csv(agg_file)
print(f"Aggregated cluster data saved to {agg_file}")

# Plot cluster counts by month
plt.figure(figsize=(12, 6))
sns.countplot(data=df_monthly_clusters, x='month', hue='cluster', palette='Set2')
plt.title("Cluster Counts by Month")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "cluster_counts_by_month.png"), dpi=300, bbox_inches='tight')
plt.close()  # Close after saving

# Plot participants per month by amyloid status
plt.figure(figsize=(12, 6))
sns.countplot(data=df_monthly_clusters, x='month', hue='amyloid_status', palette='Set1')
plt.title("Participants per Month by Amyloid Status")
plt.ylabel("Number of Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "amyloid_by_cluster.png"), dpi=300, bbox_inches='tight')
plt.close()


