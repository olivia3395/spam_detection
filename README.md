
# Spam Detection with K-Means and Isolation Forest

## Project Overview
This project demonstrates two approaches to spam detection using:
1. **K-Means Clustering**: An unsupervised learning technique to group SMS messages into clusters based on similarity.
2. **Isolation Forest**: An anomaly detection algorithm that identifies outliers (potential spam messages) by modeling deviations from the majority of the data.

Both methods are evaluated on a labeled SMS dataset, and their results are visualized using dimensionality reduction techniques like PCA.



## Features
1. **Dataset**: The SMS Spam Collection dataset, which contains messages labeled as either "spam" or "ham".
2. **Text Preprocessing**: Messages are cleaned (removing special characters, converting to lowercase, etc.) and converted into numerical representations using TF-IDF vectorization.
3. **Unsupervised Learning**:
   - K-Means groups messages into clusters (e.g., spam and ham).
   - Isolation Forest detects anomalies that might correspond to spam messages.
4. **Evaluation**: Models are evaluated using true labels, and performance is assessed with metrics like accuracy, precision, and recall.
5. **Visualization**: PCA is used to reduce the feature space for 2D visualization of clusters and outliers.


## Visualization

### K-Means Clustering
- **Description**: K-Means divides the data into two clusters. One cluster (majority) contains ham messages, and the other (minority) contains spam messages.
- **Observation**: Due to the imbalanced nature of the dataset, most points belong to the majority cluster, with some overlap between clusters in the PCA-reduced space.

### Isolation Forest
- **Description**: Isolation Forest flags outliers in the dataset, which correspond to potential spam messages.
- **Observation**: The model identifies outliers as blue points, while inliers are shown as red points. Overlap between the two indicates possible false positives.


