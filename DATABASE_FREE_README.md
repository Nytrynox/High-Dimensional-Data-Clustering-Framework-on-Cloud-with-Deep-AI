# 🆓 Database-Free High-Dimensional Clustering Framework

**Complete clustering solution using ONLY local files - no databases, no cloud dependencies!**

## 🎯 What This Is

A simplified version of the clustering framework that runs entirely on your local machine using:
- ✅ **CSV files** instead of databases
- ✅ **JSON files** for metadata storage  
- ✅ **Local directories** for all data
- ✅ **Minimal dependencies** (just core Python libraries)
- ✅ **100% Free** to use

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Minimal Requirements
```bash
# Create virtual environment
python -m venv clustering-env
source clustering-env/bin/activate  # Windows: clustering-env\Scripts\activate

# Install minimal requirements (all free)
pip install -r requirements_minimal.txt
```

### Step 2: Generate Sample Data & Run Clustering
```bash
# Generate test data
python main_simple.py generate-sample

# Run clustering (creates results in local files)
python main_simple.py cluster data/sample_data.csv --algorithm kmeans

# Check results
python main_simple.py experiments
```

### Step 3: Optional API Server
```bash
# Install FastAPI if you want the web API (optional)
pip install fastapi uvicorn[standard]

# Start server
python main_simple.py serve

# Visit http://127.0.0.1:8000/docs for web interface
```

## 📁 How Data is Stored (No Database!)

```
rishi-ccs/
├── data/                    # CSV data files
│   ├── sample_data.csv      # Your datasets
│   └── uploaded_file.csv    # API uploads
├── results/                 # Clustering results
│   ├── results_123.json     # Result metadata
│   ├── clustered_data_123.csv  # Data with cluster labels
│   └── labels_123.csv       # Just the cluster labels
├── experiments.json         # Experiment history
├── models_registry.json     # Model metadata
├── datasets_metadata.json   # Dataset info
└── cache/                   # Temporary cache files
```

## 🔧 Available Commands

```bash
# Basic clustering
python main_simple.py cluster data/file.csv --algorithm kmeans --n-clusters 5
python main_simple.py cluster data/file.csv --algorithm dbscan --eps 0.3
python main_simple.py cluster data/file.csv --algorithm hierarchical

# Quick clustering (one-line)
python main_simple.py quick data/file.csv --algorithm kmeans

# Management
python main_simple.py experiments      # List all experiments
python main_simple.py results <id>     # View specific results
python main_simple.py status          # System status
python main_simple.py cleanup         # Clean old files

# Sample data
python main_simple.py generate-sample  # Create test data

# API server (optional)
python main_simple.py serve           # Start web API
```

## 🌐 Web API (Optional)

If you install FastAPI, you get a web interface:

```bash
pip install fastapi uvicorn[standard]
python main_simple.py serve
```

**API Endpoints:**
- `POST /upload` - Upload CSV files
- `POST /cluster` - Run clustering
- `GET /experiments` - List experiments  
- `GET /results/{id}` - Get results
- `GET /files` - List uploaded files
- `GET /download-results/{id}` - Download clustered data

## 📊 Example Workflow

```bash
# 1. Setup (one time)
python -m venv clustering-env
source clustering-env/bin/activate
pip install numpy pandas scikit-learn click

# 2. Generate test data
python main_simple.py generate-sample

# 3. Run different algorithms
python main_simple.py cluster data/sample_data.csv --algorithm kmeans --n-clusters 3
python main_simple.py cluster data/sample_data.csv --algorithm dbscan --eps 0.5
python main_simple.py cluster data/sample_data.csv --algorithm hierarchical --n-clusters 4

# 4. Check results
python main_simple.py experiments

# 5. View specific results
python main_simple.py results <experiment-id>

# 6. Your clustered data is saved as CSV files in results/
```

## 🎯 What You Get

**Free Clustering Algorithms:**
- K-Means clustering
- DBSCAN (density-based)  
- Hierarchical clustering
- Automatic preprocessing and scaling
- PCA dimensionality reduction
- Cluster quality metrics

**Free Storage System:**
- All data in CSV/JSON files
- No database setup required
- Portable - copy folder anywhere
- Human-readable results
- Full experiment tracking

**Free Analysis:**
- Silhouette scores
- Cluster centers
- Data with cluster labels
- Processing metadata
- Performance metrics

## 📈 Scaling Options

**Local Machine** (Free):
- Handle datasets up to your RAM size
- Use all CPU cores automatically  
- Cache results for fast re-access

**When You Need More** (Later):
- Add GPU support: Uncomment PyTorch/TensorFlow in requirements
- Add advanced algorithms: Uncomment UMAP/HDBSCAN
- Add cloud storage: Use full framework with Azure

## 🔍 File Structure

**Your data**: `data/*.csv`
**Results**: `results/clustered_data_*.csv` 
**Experiments**: `experiments.json`
**Metadata**: Various `.json` files
**Cache**: `cache/*.pkl`

All files are **human-readable** (CSV/JSON) or standard Python pickle files.

## 🆓 Cost Breakdown

- **Python + Libraries**: $0 (open source)
- **Local storage**: $0 (your hard drive)  
- **Compute**: $0 (your CPU)
- **Advanced features**: $0 (all included)
- **Total cost**: **$0**

## 🚨 No Dependencies On:

- ❌ Databases (PostgreSQL, MongoDB, etc.)
- ❌ Cloud services (AWS, Azure, GCP)
- ❌ Docker (works without it)
- ❌ Internet connection (after install)
- ❌ Special hardware (GPU optional)
- ❌ Paid software or licenses

## 🎉 Ready to Use!

This database-free version gives you **production-quality clustering** with **zero ongoing costs**. Perfect for:

- Learning and experimentation
- Small to medium datasets  
- Prototyping before scaling
- Environments where databases aren't allowed
- Complete offline operation

**Start clustering in 2 minutes:**
```bash
pip install numpy pandas scikit-learn click
python main_simple.py generate-sample
python main_simple.py cluster data/sample_data.csv
```
