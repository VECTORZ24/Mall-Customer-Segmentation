"""
Task-02: K-Means Clustering on Mall Customers dataset
Group customers based on purchase history (Annual Income & Spending Score).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------------------------------
# 1. Load & inspect the data
# -------------------------------------------------
raw_data = pd.read_csv("Mall_Customers.csv")
print(raw_data.head())
print(raw_data.isnull().sum())
print(raw_data.describe())

# -------------------------------------------------
# 2. Select features
# -------------------------------------------------
# Purchase-history-relevant features: Annual Income & Spending Score
data = raw_data[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

# -------------------------------------------------
# 3. Scale the features
# -------------------------------------------------
# KMeans is distance-based, so features should be on comparable scales
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------------------------
# 4. Find the optimal number of clusters (Elbow Method)
# -------------------------------------------------
wcss = []  # within-cluster sum of squares
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.close()

# -------------------------------------------------
# 5. Confirm with Silhouette Score
# -------------------------------------------------
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    sil_scores.append(silhouette_score(scaled_data, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='o', linestyle='--', color='orange')
plt.title('Silhouette Score by Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=150)
plt.close()

best_k = 5  # confirmed by elbow (bend at 5) and silhouette score
print(f"\nChosen number of clusters: {best_k}")

# -------------------------------------------------
# 6. Fit final K-Means model
# -------------------------------------------------
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
raw_data['Cluster'] = kmeans_final.fit_predict(scaled_data)

# -------------------------------------------------
# 7. Visualize the clusters
# -------------------------------------------------
plt.figure(figsize=(9, 6))
palette = sns.color_palette('tab10', n_colors=best_k)

for cluster_id in range(best_k):
    cluster_points = raw_data[raw_data['Cluster'] == cluster_id]
    plt.scatter(
        cluster_points['Annual Income (k$)'],
        cluster_points['Spending Score (1-100)'],
        s=60, color=palette[cluster_id], label=f'Cluster {cluster_id}'
    )

# Plot centroids (convert back to original scale)
centroids_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    s=250, c='black', marker='X', label='Centroids'
)

plt.title('Customer Segments by Annual Income & Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.savefig('customer_clusters.png', dpi=150)
plt.close()

# -------------------------------------------------
# 8. Cluster profile summary
# -------------------------------------------------
cluster_summary = raw_data.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean().round(1)
cluster_summary['Count'] = raw_data['Cluster'].value_counts().sort_index()
print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv('cluster_summary.csv')
raw_data.to_csv('mall_customers_with_clusters.csv', index=False)

print("\nDone. Saved: elbow_method.png, silhouette_scores.png, customer_clusters.png,")
print("cluster_summary.csv, mall_customers_with_clusters.csv")
