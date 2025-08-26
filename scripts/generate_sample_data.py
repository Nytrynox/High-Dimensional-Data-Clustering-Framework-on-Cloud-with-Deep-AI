"""
Sample high-dimensional dataset generator for testing clustering algorithms
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def generate_sample_datasets():
    """Generate various sample datasets for testing"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Dataset 1: High-dimensional blobs
    print("Generating high-dimensional blobs dataset...")
    X1, y1 = make_blobs(
        n_samples=1000,
        centers=5,
        n_features=100,
        random_state=42,
        cluster_std=1.5
    )
    
    # Add noise features
    noise_features = np.random.randn(X1.shape[0], 50)
    X1_with_noise = np.column_stack([X1, noise_features])
    
    df1 = pd.DataFrame(X1_with_noise, columns=[f"feature_{i}" for i in range(X1_with_noise.shape[1])])
    df1["true_label"] = y1
    df1.to_csv(data_dir / "high_dim_blobs.csv", index=False)
    
    # Dataset 2: Classification dataset
    print("Generating classification dataset...")
    X2, y2 = make_classification(
        n_samples=2000,
        n_features=200,
        n_informative=50,
        n_redundant=20,
        n_clusters_per_class=1,
        n_classes=8,
        random_state=42
    )
    
    df2 = pd.DataFrame(X2, columns=[f"feature_{i}" for i in range(X2.shape[1])])
    df2["true_label"] = y2
    df2.to_csv(data_dir / "classification_dataset.csv", index=False)
    
    # Dataset 3: Mixed data types (for preprocessing testing)
    print("Generating mixed dataset...")
    np.random.seed(42)
    n_samples = 1500
    
    # Numerical features
    numerical_features = np.random.randn(n_samples, 80)
    
    # Categorical features (encoded as numbers)
    categorical_features = np.random.choice([0, 1, 2, 3, 4], size=(n_samples, 10))
    
    # Binary features
    binary_features = np.random.choice([0, 1], size=(n_samples, 20))
    
    # Combine all features
    X3 = np.column_stack([numerical_features, categorical_features, binary_features])
    
    # Generate clusters
    scaler = StandardScaler()
    X3_scaled = scaler.fit_transform(X3)
    
    # Create clusters based on first few dimensions
    y3 = np.zeros(n_samples)
    for i in range(n_samples):
        if X3_scaled[i, 0] > 1:
            y3[i] = 0
        elif X3_scaled[i, 1] > 0.5:
            y3[i] = 1
        elif X3_scaled[i, 2] < -0.5:
            y3[i] = 2
        else:
            y3[i] = 3
    
    # Add some missing values
    missing_mask = np.random.random((n_samples, X3.shape[1])) < 0.05
    X3_with_missing = X3.copy().astype(float)
    X3_with_missing[missing_mask] = np.nan
    
    df3 = pd.DataFrame(X3_with_missing, columns=[f"feature_{i}" for i in range(X3_with_missing.shape[1])])
    df3["true_label"] = y3
    df3.to_csv(data_dir / "mixed_dataset.csv", index=False)
    
    # Dataset 4: Time series features
    print("Generating time series dataset...")
    n_samples = 800
    n_time_steps = 50
    
    # Generate time series data
    time_series_data = []
    labels = []
    
    for i in range(n_samples):
        # Randomly choose pattern type
        pattern_type = np.random.choice([0, 1, 2, 3])
        
        if pattern_type == 0:  # Linear trend
            ts = np.linspace(0, 10, n_time_steps) + np.random.randn(n_time_steps) * 0.5
        elif pattern_type == 1:  # Sine wave
            ts = np.sin(np.linspace(0, 4*np.pi, n_time_steps)) + np.random.randn(n_time_steps) * 0.3
        elif pattern_type == 2:  # Exponential decay
            ts = np.exp(-np.linspace(0, 3, n_time_steps)) + np.random.randn(n_time_steps) * 0.2
        else:  # Random walk
            ts = np.cumsum(np.random.randn(n_time_steps) * 0.1)
        
        time_series_data.append(ts)
        labels.append(pattern_type)
    
    # Convert to DataFrame
    ts_array = np.array(time_series_data)
    df4 = pd.DataFrame(ts_array, columns=[f"t_{i}" for i in range(n_time_steps)])
    
    # Add statistical features
    df4["mean"] = ts_array.mean(axis=1)
    df4["std"] = ts_array.std(axis=1)
    df4["min"] = ts_array.min(axis=1)
    df4["max"] = ts_array.max(axis=1)
    df4["median"] = np.median(ts_array, axis=1)
    
    df4["true_label"] = labels
    df4.to_csv(data_dir / "timeseries_dataset.csv", index=False)
    
    # Dataset 5: Very high dimensional (for dimensionality reduction testing)
    print("Generating very high dimensional dataset...")
    X5, y5 = make_blobs(
        n_samples=500,
        centers=4,
        n_features=1000,
        random_state=42,
        cluster_std=2.0
    )
    
    df5 = pd.DataFrame(X5, columns=[f"dim_{i}" for i in range(X5.shape[1])])
    df5["true_label"] = y5
    df5.to_csv(data_dir / "very_high_dim.csv", index=False)
    
    print("Sample datasets generated successfully!")
    print(f"Datasets saved to: {data_dir.absolute()}")
    
    # Generate summary
    summary = {
        "high_dim_blobs.csv": {
            "samples": 1000,
            "features": 150,
            "clusters": 5,
            "description": "High-dimensional blob clusters with noise features"
        },
        "classification_dataset.csv": {
            "samples": 2000,
            "features": 200,
            "clusters": 8,
            "description": "Multi-class classification dataset"
        },
        "mixed_dataset.csv": {
            "samples": 1500,
            "features": 110,
            "clusters": 4,
            "description": "Mixed data types with missing values"
        },
        "timeseries_dataset.csv": {
            "samples": 800,
            "features": 55,
            "clusters": 4,
            "description": "Time series data with statistical features"
        },
        "very_high_dim.csv": {
            "samples": 500,
            "features": 1000,
            "clusters": 4,
            "description": "Very high dimensional dataset for dimensionality reduction testing"
        }
    }
    
    # Save summary
    import json
    with open(data_dir / "datasets_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    generate_sample_datasets()
