import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Reuse existing loader logic
from src.models.pytorch.train_gated_fusion import FusionDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_tsne(data, labels, title, ax):
    logger.info(f"Computing t-SNE for {title} (shape {data.shape})...")
    # Subsample for speed/clarity if needed
    if data.shape[0] > 5000:
        idx = np.random.choice(data.shape[0], 5000, replace=False)
        data = data[idx]
        labels = labels[idx]
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(data)
    
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1], 
        hue=labels, palette="coolwarm", alpha=0.6, s=15, ax=ax
    )
    ax.set_title(title)
    ax.legend(title='Readmission')

def main():
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    loader = FusionDataLoader()
    # Load with PCA=0 to get raw embeddings first
    ehr_data, txt_data, y, splits, _ = loader.load_data(pca_components=0)
    
    # We want to visualize the Test set primarily, or a mix. 
    # Let's visualize the Validation set to be clean.
    mask = splits == 'val'
    ehr_subset = ehr_data[mask]
    txt_subset = txt_data[mask]
    y_subset = y[mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    plot_tsne(ehr_subset, y_subset, "Structured EHR (Transformer)", axes[0])
    plot_tsne(txt_subset, y_subset, "Clinical Notes (ModernBERT)", axes[1])
    
    plt.tight_layout()
    out_path = save_dir / "embeddings_tsne.png"
    plt.savefig(out_path, dpi=300)
    logger.info(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
