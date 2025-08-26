#!/usr/bin/env python3
"""
Quick Launcher for Clustering GUI
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Launching High-Dimensional Clustering Framework GUI...")
    print()
    
    # Check if required packages are available
    required = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'tkinter']
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print()
        print("📦 Install them with:")
        print("pip install numpy pandas scikit-learn matplotlib")
        print()
        print("Note: tkinter comes with Python by default")
        return
    
    print("✅ All required packages found!")
    print("🎨 Starting GUI...")
    print()
    
    # Launch the GUI
    try:
        from clustering_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        print()
        print("💡 Try running directly:")
        print("python clustering_gui.py")

if __name__ == "__main__":
    main()
