# High-Dimensional Data Clustering Framework

## Overview
A scalable framework for clustering high-dimensional datasets in cloud environments. This project leverages deep learning for dimensionality reduction (Autoencoders) followed by optimized clustering algorithms to handle big data efficiently.

## Features
-   **Dimensionality Reduction**: Deep Autoencoders compress features while preserving latent structure.
-   **Scalable Clustering**: K-Means / DBSCAN optimized for distributed cloud execution.
-   **Cloud Native**: Designed to run on Spark or similar distributed computing clusters.
-   **Visualization**: t-SNE / PCA plots for interpreting cluster results.

## Technology Stack
-   **DL Framework**: TensorFlow / Keras.
-   **Big Data**: Apache Spark (PySpark).
-   **Language**: Python.

## Usage Flow
1.  **Ingest**: Load massive high-dimensional dataset (e.g., genomics, financial logs).
2.  **Compress**: Train Autoencoder to reduce feature space.
3.  **Cluster**: Apply clustering on the reduced embeddings.
4.  **Output**: Return cluster assignments and centroids.

## Quick Start
```bash
# Clone the repository
git clone "https://github.com/Nytrynox/High-Dimensional-Clustering.git"

# Install requirements
pip install -r requirements.txt

# Run pipeline
python main.py --input data.csv
```

## License
MIT License

## Author
**Karthik Idikuda**
