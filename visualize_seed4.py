import torch
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.TransNet import TransNet
import yaml

def plot_topography_seed4(weights, title="EEG Importance Map"):
    # SEED-IV 62-channel standard names
    ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", 
                "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", 
                "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", 
                "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", 
                "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", 
                "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
    
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
    info.set_montage(montage)
    
    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(weights, info, axes=ax, show=False, cmap='RdBu_r')
    plt.title(title)
    plt.colorbar(im)
    plt.savefig("brain_topography.png")
    plt.show()

def plot_tsne_clusters(model, data, labels, device):
    model.eval()
    with torch.no_grad():
        data_t = torch.from_numpy(data).float().to(device)
        # Extract features before the final classification layer
        features = model.elu(model.bn2(model.spatial_conv(data_t.unsqueeze(1)))).reshape(data.shape[0], -1)
        features = features.cpu().numpy()

    # Calculate t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    low_dim = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Emotions")
    plt.title("t-SNE: Emotional Feature Clustering")
    plt.savefig("tsne_clusters.png")
    plt.show()

if __name__ == "__main__":
    # Load config and model
    # ... (Same loading logic as calculate_metrics.py) ...
    print("Visualizations generated and saved to folder.")