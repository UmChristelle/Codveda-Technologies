# Level 2 Task 3: Clustering Analysis (K-Means)

## Objective

Use K-Means clustering to group similar housing records based on numerical feature patterns, determine an appropriate number of clusters, and visualize the clusters in two dimensions.

## Dataset

- Source dataset: `4) house Prediction Data Set.csv`
- Local file used: `data/house_prediction_dataset.csv`
- Records: 506
- Variables: 14 numerical columns

## Workflow

- Loaded and labeled the dataset columns
- Selected six informative numerical features for clustering
- Standardized the selected features using z-score scaling
- Computed inertia values for multiple `k` values using the elbow method
- Chose `k = 3` as the clustering baseline
- Fit a manual K-Means model
- Exported cluster assignments and cluster profile summaries
- Visualized clusters using the `RM` and `LSTAT` feature space

## Selected Features

- `RM`
- `LSTAT`
- `MEDV`
- `CRIM`
- `NOX`
- `DIS`

## Key Findings

- The elbow curve shows a clear improvement up to around 3 clusters.
- The resulting clusters separate neighborhoods with different room counts, lower-status population percentages, and home values.
- The `RM` and `LSTAT` feature pair provides a clear two-dimensional view of cluster separation.

## Environment Note

- `scikit-learn` was not available in this environment, so K-Means and standardization were implemented directly with NumPy and pandas.
- The workflow still follows the same analytical principles required for the task: scaling, elbow analysis, clustering, and visualization.

## Files

- `task3_clustering_analysis.py`
- `task3_clustering_analysis.ipynb`
- `outputs/elbow_method.csv`
- `outputs/cluster_assignments.csv`
- `outputs/cluster_profiles.csv`
- `outputs/01_elbow_curve.png`
- `outputs/02_cluster_scatter_rm_lstat.png`
