import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

base_path = '/mnt/d/DETECT'
best_k = 2  # Initialize best_k for silhouette score
best_sil_score = -1  # Initialize best silhouette score

# Create output directory if it does not exist
output_path = os.path.join(base_path, 'OUTPUT', 'KMeans_Clustering_Participants')
plot_path = os.path.join(output_path, "plots")
os.makedirs(output_path, exist_ok=True)
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
    'steps', 'gait_speed', 'awakenings', 'bedexitcount', 'end_sleep_time',
    'inbed_time', 'outbed_time', 'sleepscore', 'durationinsleep', 'durationawake',
    'waso', 'hrvscore', 'start_sleep_time', 'time_to_sleep',
    'time_in_bed_after_sleep', 'tossnturncount', 'maxhr', 'avghr', 'avgrr',
    #include demographic features
    'age', 'sex_encoded', 'amyloid'
]

#extract just the feature values for clustering
X = df_clean[feature_cols]

#aggregate features by subject and date
agg_df = df_clean.groupby(['subid'])[feature_cols].agg(['mean', 'std']).reset_index()

# Drop any rows with NaNs after aggregation
agg_df.fillna(agg_df.mean(), inplace=True)

# Flatten the multi-level columns
agg_df.columns =['subid'] + [f"{feat}_{stat}" for feat in feature_cols for stat in ['mean', 'std']]

#store seperately to join later
X = agg_df.drop(columns=['subid'])

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Finding Best k ---")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    print(f"k={k:<2} | Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_k = k
        best_sil_score = sil_score
print(f"\nBest k found: {best_k} with Silhouette Score: {best_sil_score:.3f}")

# Perform KMeans clustering
k = best_k  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
agg_df['cluster'] = kmeans.fit_predict(X_scaled)

#sue the input matrix to compute cluster metrics
X_input = X_scaled
labels = agg_df['cluster']

# Calculate clustering metrics
sil_score = silhouette_score(X_input, labels) #silhouette score
ch_score = calinski_harabasz_score(X_input, labels) # Calinski-Harabasz score
db_score = davies_bouldin_score(X_input, labels) # Davies-Bouldin score


# Print results
print("\n--- Cluster Evaluation Metrics ---")
print(f"Silhouette Score:          {sil_score:.3f}  (higher is better)")
print(f"Calinski-Harabasz Index:   {ch_score:.2f}   (higher is better)")
print(f"Davies-Bouldin Index:      {db_score:.3f}  (lower is better)")

# print cluster sizes
print("Cluster sizes:")
print(agg_df['cluster'].value_counts())

#look at average feature values per cluster
print("Average feature values per cluster:")
cluster_summary = agg_df.groupby('cluster').mean().round(2)
print(cluster_summary)

#plot cluster distribution
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

agg_df['pca1'] = X_pca[:, 0]
agg_df['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=agg_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100)
plt.title("KMeans Clustering of Participants (PCA-reduced features)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")

# Save plot
plt.savefig(os.path.join(plot_path, "pca_clusters.png"), dpi=300, bbox_inches='tight')
plt.close()  # closes the figure to free memory

# Proportion of Amyloid Positives
plt.figure(figsize=(6, 4))
sns.barplot(data=agg_df, x='cluster', y='amyloid_mean', ci='sd')
plt.title('Proportion of Amyloid Positive by Cluster')
plt.ylabel('Amyloid Positivity Rate')
plt.xlabel('Cluster')
plt.ylim(0, 1)

plt.savefig(os.path.join(plot_path, "amyloid_by_cluster.png"), dpi=300, bbox_inches='tight')
plt.close()

# Merge back demographics
demo_info = df_clean.groupby('subid')[demo_cols].first().reset_index()
agg_df = agg_df.merge(demo_info, on='subid', how='left')

# Demographic comparisons
print("Sex distribution by cluster:")
print(agg_df.groupby('cluster')['sex'].value_counts(normalize=True).rename("proportion").reset_index())

print("\nAmyloid status distribution by cluster:")
print(agg_df.groupby('cluster')['amyloid'].value_counts(normalize=True).rename("proportion").reset_index())


agg_df.to_csv(os.path.join(output_path, "clustered_participants.csv"), index=False)
cluster_summary.to_csv(os.path.join(output_path, "cluster_feature_summary.csv"))



