# Customer Segmentation using K-Means Clustering

## 📌 Objective
Create a K-Means clustering algorithm to group customers of a retail store based on their purchase history, using the **Mall Customers** dataset.

## 📂 Dataset
`Mall_Customers.csv` — 200 records, 5 columns:

| Column | Description |
|---|---|
| CustomerID | Unique identifier for each customer |
| Gender | Male / Female |
| Age | Customer's age |
| Annual Income (k$) | Customer's annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall based on customer behavior and spending nature |

No missing values were found in the dataset.

## 🛠️ Approach
1. **Feature Selection** — `Annual Income (k$)` and `Spending Score (1-100)` were chosen as they best represent purchasing behavior.
2. **Feature Scaling** — Applied `StandardScaler` since K-Means relies on Euclidean distance and features must be on comparable scales.
3. **Choosing k (number of clusters)**
   - **Elbow Method**: Plotted WCSS (inertia) for k = 1–10 and looked for the "elbow" bend.
   - **Silhouette Score**: Computed for k = 2–10 to confirm cluster separation quality.
   - Both methods pointed to **k = 5** as optimal.
4. **Model Fitting** — Trained `KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)` on the scaled data.
5. **Visualization** — Plotted the 5 clusters in Income vs. Spending Score space, with centroids marked.
6. **Cluster Profiling** — Grouped customers by cluster and computed average Age, Income, and Spending Score per segment.

## 📊 Results

| Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Count | Segment |
|---|---|---|---|---|---|
| 0 | 42.7 | 55.3 | 49.5 | 81 | Average customers |
| 1 | 32.7 | 86.5 | 82.1 | 39 | High income, high spending — **prime target** |
| 2 | 25.3 | 25.7 | 79.4 | 22 | Low income, high spending — young/impulsive |
| 3 | 41.1 | 88.2 | 17.1 | 35 | High income, low spending — frugal |
| 4 | 45.2 | 26.3 | 20.9 | 23 | Low income, low spending |

**Business takeaway:** Cluster 1 (high income + high spending) is the ideal group for premium marketing campaigns. Cluster 3 (high income but low spending) represents an untapped opportunity that could respond well to loyalty programs or targeted promotions.

## 📁 Project Files
```
├── Mall_Customers.csv                     # Raw dataset
├── kmeans_mall_customers.py               # Full clustering pipeline script
├── elbow_method.png                       # WCSS vs. k plot
├── silhouette_scores.png                  # Silhouette score vs. k plot
├── customer_clusters.png                  # Final cluster visualization with centroids
├── cluster_summary.csv                    # Per-cluster averages and counts
├── mall_customers_with_clusters.csv       # Original data with assigned cluster labels
└── README.md                              # This file
```

## ▶️ How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn
python kmeans_mall_customers.py
```
Make sure `Mall_Customers.csv` is in the same directory as the script.

## 🧰 Tech Stack
- Python
- pandas
- matplotlib / seaborn
- scikit-learn (`KMeans`, `StandardScaler`, `silhouette_score`)
