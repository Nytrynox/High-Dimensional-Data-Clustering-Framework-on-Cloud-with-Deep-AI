# High-Dimensional Data Clustering on Cloud

Scalable deep learning framework for clustering high-dimensional datasets on cloud infrastructure using autoencoders and distributed computing.

## Features

- Autoencoder-based dimensionality reduction
- K-means and DBSCAN clustering on latent representations
- Distributed processing with Apache Spark
- Cluster quality evaluation (silhouette score, NMI)
- Visualization of cluster boundaries in reduced space

## Tech Stack

Python, TensorFlow, PySpark, Scikit-learn, Matplotlib

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/karthik-idikuda/High-Dimensional-Data-Clustering-Framework-on-Cloud-with-Deep-AI.git
cd High-Dimensional-Data-Clustering-Framework-on-Cloud-with-Deep-AI
pip install -r requirements.txt
```

### Usage

```bash
python cluster.py --data input.csv
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
