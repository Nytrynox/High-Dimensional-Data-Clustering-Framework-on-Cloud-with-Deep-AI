# 🆓 FREE Setup Guide for High-Dimensional Data Clustering Framework

This guide shows you how to run the entire clustering framework **completely free** using local development and free cloud services.

## 🎯 **Option 1: Completely Local (100% Free)**

### Step 1: Setup Python Environment
```bash
# Create virtual environment (free)
python -m venv clustering-env
source clustering-env/bin/activate  # On Windows: clustering-env\Scripts\activate

# Install free dependencies
pip install -r requirements.txt
```

### Step 2: Run Locally
```bash
# Start the API server (free)
python src/main.py serve

# Open Jupyter notebook for analysis (free)
jupyter notebook notebooks/clustering_analysis.ipynb

# Generate sample data (free)
python scripts/generate_sample_data.py
```

### Step 3: Use Free Tools
- **VS Code**: Free IDE with excellent Python support
- **Docker Desktop**: Free containerization
- **Git**: Free version control
- **MLflow**: Free ML experiment tracking

## 🌤️ **Option 2: Free Cloud Deployment**

### Google Colab (Completely Free)
```python
# Upload your code to Google Drive
# Open in Colab and run:
!pip install -r requirements.txt
!python src/main.py cluster --algorithm=kmeans --n-clusters=5
```

### GitHub Codespaces (60 hours free/month)
1. Push code to GitHub (free)
2. Open in Codespaces
3. Run: `pip install -r requirements.txt`
4. Develop in cloud environment

### Render.com (Free Web App Hosting)
```yaml
# render.yaml (free tier)
services:
  - type: web
    name: clustering-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

## 🔧 **Free Azure Alternative**

### Use Azure Free Account
- **$200 credit** for 30 days
- **Always-free services** after credit expires
- Perfect for testing and small workloads

### Deploy Script (Uses Free Tier)
```bash
# Uses only free/low-cost services
./scripts/deploy_azure.sh -g "rg-clustering-free" -e "dev"
```

## 📊 **Free Data Analysis Workflow**

### 1. Generate Sample Data (Free)
```bash
python scripts/generate_sample_data.py
```

### 2. Run Clustering Analysis (Free)
```bash
# K-Means clustering
python src/main.py cluster --algorithm=kmeans --data=data/sample_data.csv

# DBSCAN clustering  
python src/main.py cluster --algorithm=dbscan --data=data/sample_data.csv

# Deep embedding clustering
python src/main.py train --model=deep_embedding
```

### 3. Visualize Results (Free)
```bash
# Start Jupyter notebook
jupyter notebook notebooks/clustering_analysis.ipynb

# Or use the plotting utilities
python -c "from src.utils.visualization import plot_clusters; plot_clusters('results/clusters.csv')"
```

## 🎮 **Free Development Stack**

### Core Components (All Free)
- **Python + Libraries**: Open source
- **FastAPI**: Free web framework
- **PyTorch/TensorFlow**: Free ML frameworks
- **Jupyter**: Free data analysis
- **Docker**: Free containerization
- **Git + GitHub**: Free version control

### Free Monitoring & Logging
```python
# Simple free logging
import logging
logging.basicConfig(level=logging.INFO)

# Use MLflow (free) instead of paid services
import mlflow
mlflow.start_run()
mlflow.log_metric("silhouette_score", score)
```

## 💡 **Cost-Free Best Practices**

### 1. Local-First Development
- Test everything locally before cloud deployment
- Use sample data for initial development
- Optimize algorithms on small datasets first

### 2. Resource Management
```python
# Auto-cleanup for free resources
import atexit
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
```

### 3. Free Alternatives
- **Instead of Azure Cosmos DB**: Use SQLite (free)
- **Instead of Azure Storage**: Use local files or GitHub
- **Instead of paid monitoring**: Use Python logging + MLflow

## 🚀 **Quick Start Commands**

```bash
# 1. Setup (free)
python -m venv clustering-env
source clustering-env/bin/activate
pip install -r requirements.txt

# 2. Generate data (free)
python scripts/generate_sample_data.py

# 3. Run clustering (free)
python src/main.py cluster --algorithm=kmeans

# 4. Analyze results (free)
jupyter notebook notebooks/clustering_analysis.ipynb

# 5. Start API server (free)
python src/main.py serve
```

## 📈 **Scaling Up (When Ready)**

When you're ready to move beyond free tiers:

1. **Azure Free Account**: Start with $200 credit
2. **GitHub Pro**: For private repos and more Actions minutes
3. **Cloud GPUs**: When you need serious compute power

## 🛠️ **Troubleshooting Free Setup**

### Common Issues
```bash
# If pip install fails
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# If Jupyter doesn't start
pip install jupyter notebook
jupyter notebook --ip=0.0.0.0

# If Docker issues on M1 Mac
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Memory Management (For Free Tiers)
```python
# Optimize for limited memory
import gc
import psutil

def cleanup_memory():
    gc.collect()
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

---

**🎉 You now have everything you need to run a production-quality clustering framework completely free!**

The framework is designed to scale from free local development to enterprise cloud deployment, so you can start experimenting immediately without any costs.
