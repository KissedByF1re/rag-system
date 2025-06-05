#!/usr/bin/env python3
"""Download the Russian RAG test dataset."""

import os
import sys
from pathlib import Path
import urllib.request
import pickle

def download_dataset():
    """Download the Russian RAG test dataset from GitHub."""
    dataset_url = "https://github.com/slivka83/ru_rag_test_dataset/raw/main/ru_rag_test_dataset.pkl"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "ru_rag_test_dataset.pkl"
    
    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
        return
    
    print(f"Downloading Russian RAG test dataset...")
    print(f"URL: {dataset_url}")
    print(f"Destination: {output_path}")
    
    try:
        urllib.request.urlretrieve(dataset_url, output_path)
        print("✅ Dataset downloaded successfully!")
        
        # Verify the dataset
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
            print(f"✅ Dataset verified: {len(data)} samples loaded")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()