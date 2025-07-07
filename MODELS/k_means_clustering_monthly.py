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

# Define feature and demographic columns
feature_cols = [
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', #'end_sleep_time',
     'durationinsleep', 'inbed_time', 'outbed_time', #'durationawake', 'sleepscore',
    'waso', 'hrvscore', #'start_sleep_time', 'time_to_sleep',
     'tossnturncount', 'maxhr', 'avghr', 'avgrr', #'time_in_bed_after_sleep',
    #include demographic features
    'age', 'sex_encoded'
]

#drop rows with NaN in feature columns or date
df_clean = df_clean.dropna(subset=feature_cols + ['date']).copy()

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
        
        best_k = 2  # Reset best_k for each month
        best_sil_score = -1  # Reset best silhouette score for each month
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            print(f"Month {month} - k={k:<2} | Silhouette Score: {sil_score:.3f}")
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


