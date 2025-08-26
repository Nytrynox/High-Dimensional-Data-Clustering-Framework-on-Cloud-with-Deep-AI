"""
Simple GUI for Database-Free Clustering Framework
Built with Tkinter (included with Python - completely free!)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import threading
import queue
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import uuid


class LocalStorage:
    """Simple local storage for GUI version"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ensure_directories()
    
    def ensure_directories(self):
        dirs = ['data', 'results', 'cache']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
    
    def save_experiment(self, experiment_data):
        experiment_id = str(uuid.uuid4())[:8]
        experiment_data['id'] = experiment_id
        experiment_data['created_at'] = datetime.now().isoformat()
        
        experiments_file = self.base_dir / "experiments.json"
        experiments = {}
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    experiments = json.load(f)
            except:
                pass
        
        experiments[experiment_id] = experiment_data
        
        with open(experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
        
        return experiment_id
    
    def load_experiments(self):
        experiments_file = self.base_dir / "experiments.json"
        if not experiments_file.exists():
            return {}
        try:
            with open(experiments_file, 'r') as f:
                return json.load(f)
        except:
            return {}


class ClusteringEngine:
    """Clustering engine for GUI"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.storage = LocalStorage()
    
    def preprocess_data(self, data):
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found")
        
        numeric_data = numeric_data.fillna(numeric_data.mean())
        scaled_data = self.scaler.fit_transform(numeric_data)
        return scaled_data
    
    def run_clustering(self, filepath, algorithm, params, progress_callback=None):
        """Run clustering with progress updates"""
        
        if progress_callback:
            progress_callback("Loading data...", 10)
        
        # Start experiment
        experiment_data = {
            'algorithm': algorithm,
            'filepath': str(filepath),
            'parameters': params,
            'started_at': datetime.now().isoformat()
        }
        
        experiment_id = self.storage.save_experiment(experiment_data)
        
        try:
            # Load data
            data = pd.read_csv(filepath)
            
            if progress_callback:
                progress_callback(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns", 25)
            
            # Preprocess
            processed_data = self.preprocess_data(data)
            
            if progress_callback:
                progress_callback("Running clustering algorithm...", 50)
            
            # Run clustering
            if algorithm == 'K-Means':
                n_clusters = params.get('n_clusters', 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(processed_data)
                
                results = {
                    'algorithm': 'K-Means',
                    'labels': labels.tolist(),
                    'cluster_centers': model.cluster_centers_.tolist() if hasattr(model, 'cluster_centers_') else [],
                    'n_clusters': n_clusters
                }
                
            elif algorithm == 'DBSCAN':
                eps = params.get('eps', 0.5)
                min_samples = params.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(processed_data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                results = {
                    'algorithm': 'DBSCAN',
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }
                
            elif algorithm == 'Hierarchical':
                n_clusters = params.get('n_clusters', 3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(processed_data)
                
                results = {
                    'algorithm': 'Hierarchical',
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters
                }
            
            if progress_callback:
                progress_callback("Evaluating results...", 75)
            
            # Evaluate clustering
            labels_array = np.array(results['labels'])
            unique_labels = set(labels_array)
            
            metrics = {}
            if len(unique_labels) > 1 and not all(label == -1 for label in unique_labels):
                try:
                    metrics['silhouette_score'] = silhouette_score(processed_data, labels_array)
                except:
                    metrics['silhouette_score'] = 0.0
                    
                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(processed_data, labels_array)
                except:
                    metrics['calinski_harabasz_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
            
            results['metrics'] = metrics
            results['data_shape'] = data.shape
            results['experiment_id'] = experiment_id
            
            if progress_callback:
                progress_callback("Saving results...", 90)
            
            # Save clustered data
            clustered_data = data.copy()
            clustered_data['cluster'] = labels_array
            
            results_id = str(uuid.uuid4())[:8]
            output_file = Path("results") / f"clustered_data_{results_id}.csv"
            clustered_data.to_csv(output_file, index=False)
            
            results['results_id'] = results_id
            results['output_file'] = str(output_file)
            
            # Update experiment
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'results_id': results_id,
                'status': 'completed',
                'metrics': metrics
            })
            self.storage.save_experiment(experiment_data)
            
            if progress_callback:
                progress_callback("Completed successfully!", 100)
            
            return results
            
        except Exception as e:
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            self.storage.save_experiment(experiment_data)
            raise


class ClusteringGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("High-Dimensional Clustering Framework - Database Free")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.engine = ClusteringEngine()
        self.current_data = None
        self.current_results = None
        
        # Queue for thread communication
        self.result_queue = queue.Queue()
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
        # Start checking for results
        self.check_results()
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔬 High-Dimensional Clustering Framework", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Data & Clustering
        self.create_clustering_tab()
        
        # Tab 2: Results & Visualization
        self.create_results_tab()
        
        # Tab 3: Experiments History
        self.create_history_tab()
    
    def create_clustering_tab(self):
        """Create the main clustering tab"""
        
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="📊 Clustering")
        
        # Left panel - Data and Parameters
        left_panel = ttk.Frame(tab1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Data section
        data_frame = ttk.LabelFrame(left_panel, text="📂 Data", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        ttk.Button(data_frame, text="📁 Load CSV File", 
                  command=self.load_file, width=20).pack(pady=5)
        
        ttk.Button(data_frame, text="🎲 Generate Sample Data", 
                  command=self.generate_sample, width=20).pack(pady=5)
        
        # Data info display
        self.data_info = ttk.Label(data_frame, text="No data loaded")
        self.data_info.pack(pady=5)
        
        # Algorithm section
        algo_frame = ttk.LabelFrame(left_panel, text="🔧 Algorithm", padding="10")
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Algorithm selection
        ttk.Label(algo_frame, text="Algorithm:").pack(anchor=tk.W)
        self.algorithm_var = tk.StringVar(value="K-Means")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                 values=["K-Means", "DBSCAN", "Hierarchical"],
                                 state="readonly", width=18)
        algo_combo.pack(pady=(5, 10), fill=tk.X)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # Parameters frame (dynamic based on algorithm)
        self.params_frame = ttk.Frame(algo_frame)
        self.params_frame.pack(fill=tk.X)
        
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(algo_frame, text="🚀 Run Clustering", 
                                    command=self.run_clustering, width=20)
        self.run_button.pack(pady=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(left_panel, text="📈 Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.progress_bar.pack(pady=5, fill=tk.X)
        
        # Right panel - Data Preview
        right_panel = ttk.Frame(tab1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Data preview
        preview_frame = ttk.LabelFrame(right_panel, text="👁️ Data Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for data display
        columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_results_tab(self):
        """Create the results and visualization tab"""
        
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="📊 Results")
        
        # Left panel - Results info
        left_panel = ttk.Frame(tab2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Results summary
        results_frame = ttk.LabelFrame(left_panel, text="📋 Results Summary", padding="10")
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=40, height=15,
                                                     wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack()
        
        # Export buttons
        export_frame = ttk.LabelFrame(left_panel, text="💾 Export", padding="10")
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="📁 Open Results Folder", 
                  command=self.open_results_folder, width=20).pack(pady=2)
        
        ttk.Button(export_frame, text="💾 Export CSV", 
                  command=self.export_csv, width=20).pack(pady=2)
        
        # Right panel - Visualization
        right_panel = ttk.Frame(tab2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        viz_frame = ttk.LabelFrame(right_panel, text="📈 Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Run clustering to see visualization', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
        self.canvas.draw()
    
    def create_history_tab(self):
        """Create the experiments history tab"""
        
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="📚 History")
        
        # History frame
        history_frame = ttk.LabelFrame(tab3, text="🕐 Experiment History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for experiments
        columns = ('ID', 'Algorithm', 'Status', 'Clusters', 'Score', 'Date')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            if col == 'ID':
                self.history_tree.column(col, width=80)
            elif col in ['Clusters', 'Score']:
                self.history_tree.column(col, width=80)
            else:
                self.history_tree.column(col, width=120)
        
        # Scrollbar for history
        hist_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=hist_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        ttk.Button(history_frame, text="🔄 Refresh History", 
                  command=self.refresh_history).pack(pady=10)
        
        # Load initial history
        self.refresh_history()
    
    def create_parameter_widgets(self):
        """Create parameter input widgets based on selected algorithm"""
        
        # Clear existing widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        algorithm = self.algorithm_var.get()
        
        if algorithm == "K-Means":
            ttk.Label(self.params_frame, text="Number of Clusters:").pack(anchor=tk.W)
            self.n_clusters_var = tk.IntVar(value=3)
            n_clusters_spin = ttk.Spinbox(self.params_frame, from_=2, to=20, 
                                         textvariable=self.n_clusters_var, width=18)
            n_clusters_spin.pack(pady=(2, 10), fill=tk.X)
            
        elif algorithm == "DBSCAN":
            ttk.Label(self.params_frame, text="Epsilon (eps):").pack(anchor=tk.W)
            self.eps_var = tk.DoubleVar(value=0.5)
            eps_spin = ttk.Spinbox(self.params_frame, from_=0.1, to=2.0, increment=0.1,
                                  textvariable=self.eps_var, width=18)
            eps_spin.pack(pady=(2, 5), fill=tk.X)
            
            ttk.Label(self.params_frame, text="Min Samples:").pack(anchor=tk.W)
            self.min_samples_var = tk.IntVar(value=5)
            min_samples_spin = ttk.Spinbox(self.params_frame, from_=2, to=20,
                                          textvariable=self.min_samples_var, width=18)
            min_samples_spin.pack(pady=(2, 10), fill=tk.X)
            
        elif algorithm == "Hierarchical":
            ttk.Label(self.params_frame, text="Number of Clusters:").pack(anchor=tk.W)
            self.n_clusters_var = tk.IntVar(value=3)
            n_clusters_spin = ttk.Spinbox(self.params_frame, from_=2, to=20,
                                         textvariable=self.n_clusters_var, width=18)
            n_clusters_spin.pack(pady=(2, 10), fill=tk.X)
    
    def on_algorithm_change(self, event=None):
        """Handle algorithm selection change"""
        self.create_parameter_widgets()
    
    def load_file(self):
        """Load a CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_data = pd.read_csv(filename)
                self.filepath = filename
                self.update_data_display()
                self.data_info.config(text=f"Loaded: {Path(filename).name}\n"
                                          f"Shape: {self.current_data.shape}")
                messagebox.showinfo("Success", f"Loaded {self.current_data.shape[0]} rows, "
                                              f"{self.current_data.shape[1]} columns")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def generate_sample(self):
        """Generate sample data for testing"""
        try:
            # Generate sample data with 3 clusters
            np.random.seed(42)
            
            cluster1 = np.random.normal([2, 2], 0.8, (100, 2))
            cluster2 = np.random.normal([6, 6], 0.8, (100, 2))
            cluster3 = np.random.normal([2, 6], 0.8, (100, 2))
            
            data = np.vstack([cluster1, cluster2, cluster3])
            
            # Add additional features
            feature3 = data[:, 0] + data[:, 1] + np.random.normal(0, 0.1, 300)
            feature4 = data[:, 0] * data[:, 1] + np.random.normal(0, 0.5, 300)
            feature5 = np.random.normal(0, 1, 300)
            
            self.current_data = pd.DataFrame({
                'feature_1': data[:, 0],
                'feature_2': data[:, 1],
                'feature_3': feature3,
                'feature_4': feature4,
                'feature_5': feature5
            })
            
            # Save to file
            Path("data").mkdir(exist_ok=True)
            self.filepath = "data/sample_data_gui.csv"
            self.current_data.to_csv(self.filepath, index=False)
            
            self.update_data_display()
            self.data_info.config(text=f"Generated sample data\n"
                                      f"Shape: {self.current_data.shape}\n"
                                      f"3 natural clusters")
            
            messagebox.showinfo("Success", "Sample data generated!\n"
                                          "This data contains 3 natural clusters.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample data:\n{str(e)}")
    
    def update_data_display(self):
        """Update the data preview treeview"""
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.current_data is not None:
            # Update column headers
            columns = list(self.current_data.columns)[:5]  # Show first 5 columns
            self.data_tree['columns'] = columns
            
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # Add data rows (first 100 rows)
            for i, row in self.current_data.head(100).iterrows():
                values = [str(row[col])[:10] + "..." if len(str(row[col])) > 10 else str(row[col]) 
                         for col in columns]
                self.data_tree.insert('', 'end', values=values)
    
    def run_clustering(self):
        """Run clustering in a separate thread"""
        
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        # Disable run button
        self.run_button.config(state='disabled')
        
        # Get parameters
        algorithm = self.algorithm_var.get()
        params = {}
        
        if algorithm in ["K-Means", "Hierarchical"]:
            params['n_clusters'] = self.n_clusters_var.get()
        elif algorithm == "DBSCAN":
            params['eps'] = self.eps_var.get()
            params['min_samples'] = self.min_samples_var.get()
        
        # Start clustering in separate thread
        thread = threading.Thread(target=self._run_clustering_thread,
                                args=(self.filepath, algorithm, params))
        thread.daemon = True
        thread.start()
    
    def _run_clustering_thread(self, filepath, algorithm, params):
        """Run clustering in separate thread"""
        try:
            def progress_callback(message, progress):
                self.result_queue.put(('progress', message, progress))
            
            results = self.engine.run_clustering(filepath, algorithm, params, progress_callback)
            self.result_queue.put(('success', results))
            
        except Exception as e:
            self.result_queue.put(('error', str(e)))
    
    def check_results(self):
        """Check for results from clustering thread"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                
                if result[0] == 'progress':
                    _, message, progress = result
                    self.progress_var.set(message)
                    self.progress_bar['value'] = progress
                    
                elif result[0] == 'success':
                    _, results = result
                    self.current_results = results
                    self.display_results(results)
                    self.create_visualization(results)
                    self.progress_var.set("✅ Clustering completed successfully!")
                    self.progress_bar['value'] = 100
                    self.run_button.config(state='normal')
                    self.refresh_history()
                    
                elif result[0] == 'error':
                    _, error_msg = result
                    messagebox.showerror("Clustering Error", f"Clustering failed:\n{error_msg}")
                    self.progress_var.set("❌ Clustering failed")
                    self.progress_bar['value'] = 0
                    self.run_button.config(state='normal')
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_results)
    
    def display_results(self, results):
        """Display clustering results"""
        
        # Enable text widget for updating
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        text = f"🔬 Clustering Results\n\n"
        text += f"Algorithm: {results['algorithm']}\n"
        text += f"Clusters Found: {results['n_clusters']}\n"
        
        if 'n_noise' in results:
            text += f"Noise Points: {results['n_noise']}\n"
        
        text += f"\n📊 Quality Metrics:\n"
        metrics = results.get('metrics', {})
        text += f"Silhouette Score: {metrics.get('silhouette_score', 0):.3f}\n"
        text += f"Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.1f}\n"
        
        text += f"\n📁 Data Info:\n"
        text += f"Original Shape: {results['data_shape']}\n"
        text += f"Results ID: {results['results_id']}\n"
        text += f"Output File: {results['output_file']}\n"
        
        text += f"\n🎯 Cluster Distribution:\n"
        labels = results['labels']
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label == -1:
                text += f"Noise: {count} points\n"
            else:
                text += f"Cluster {label}: {count} points\n"
        
        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)
        
        # Switch to results tab
        self.notebook.select(1)
    
    def create_visualization(self, results):
        """Create visualization of clustering results"""
        
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get data and labels
            data = self.current_data
            labels = np.array(results['labels'])
            
            # Use first two numeric columns for 2D plot
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                x_data, y_data = data[x_col], data[y_col]
                
                # Create scatter plot with different colors for each cluster
                unique_labels = set(labels)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = labels == label
                    if label == -1:
                        # Noise points
                        self.ax.scatter(x_data[mask], y_data[mask], c='black', marker='x', 
                                       s=50, alpha=0.7, label='Noise')
                    else:
                        self.ax.scatter(x_data[mask], y_data[mask], c=[color], 
                                       s=50, alpha=0.7, label=f'Cluster {label}')
                
                # Plot cluster centers if available
                if 'cluster_centers' in results and results['cluster_centers']:
                    centers = np.array(results['cluster_centers'])
                    if centers.shape[1] >= 2:
                        self.ax.scatter(centers[:, 0], centers[:, 1], c='red', 
                                       marker='*', s=200, label='Centers')
                
                self.ax.set_xlabel(x_col)
                self.ax.set_ylabel(y_col)
                self.ax.set_title(f"{results['algorithm']} Clustering Results")
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                
            else:
                self.ax.text(0.5, 0.5, 'Need at least 2 numeric columns for visualization', 
                            ha='center', va='center', transform=self.ax.transAxes)
            
            self.canvas.draw()
            
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Visualization error:\n{str(e)}', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
    
    def refresh_history(self):
        """Refresh the experiments history"""
        
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Load experiments
        experiments = self.engine.storage.load_experiments()
        
        for exp_id, exp in experiments.items():
            status = exp.get('status', 'unknown')
            algorithm = exp.get('algorithm', 'unknown')
            
            # Get metrics if completed
            clusters = ''
            score = ''
            if status == 'completed':
                metrics = exp.get('metrics', {})
                clusters = str(metrics.get('n_clusters', ''))
                score = f"{metrics.get('silhouette_score', 0):.3f}"
            
            # Format date
            date_str = exp.get('started_at', '')
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', ''))
                    date_str = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            # Insert into tree
            self.history_tree.insert('', 'end', values=(
                exp_id, algorithm, status, clusters, score, date_str
            ))
    
    def open_results_folder(self):
        """Open the results folder in file manager"""
        results_path = Path("results").absolute()
        
        try:
            import subprocess
            import sys
            
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(results_path)])
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", str(results_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(results_path)])
                
        except Exception as e:
            messagebox.showinfo("Results Folder", f"Results are saved in:\n{results_path}")
    
    def export_csv(self):
        """Export current results as CSV"""
        if self.current_results is None:
            messagebox.showwarning("No Results", "No clustering results to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save clustering results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Load the clustered data and save to new location
                results_file = self.current_results['output_file']
                data = pd.read_csv(results_file)
                data.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")


def main():
    """Main function to run the GUI"""
    
    # Create main window
    root = tk.Tk()
    
    # Create application
    app = ClusteringGUI(root)
    
    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()
