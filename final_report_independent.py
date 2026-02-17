import torch
import numpy as np
import os, yaml, mne
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from model.TransNet import TransNet
from data.dataset import eegDataset

def generate_independent_report(root_dir, data_path, config):
    ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", 
                "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", 
                "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", 
                "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", 
                "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", 
                "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_accuracies = []
    folds = []

    for sub_id in range(1, 16):
        fold_folder = os.path.join(root_dir, f'test_sub{sub_id}')
        if not os.path.exists(fold_folder): continue
        
        model_path = os.path.join(fold_folder, 'model.pth')
        # Batch loading to prevent OOM on 6GB VRAM
        test_x = np.concatenate([np.load(os.path.join(data_path, f"sub{sub_id}_sess{s}_data.npy")) for s in [1,2,3]]).astype('float32')
        test_y = np.concatenate([np.load(os.path.join(data_path, f"sub{sub_id}_sess{s}_label.npy")) for s in [1,2,3]]).astype('int64')
        loader = DataLoader(eegDataset(test_x, test_y), batch_size=64, shuffle=False)

        model = TransNet(**config['network_args']).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        batch_preds, batch_features = [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    # FIX: Correct feature extraction
                    t_out = torch.cat((model.temp_conv1(x.unsqueeze(1)), model.temp_conv2(x.unsqueeze(1)), 
                                      model.temp_conv3(x.unsqueeze(1)), model.temp_conv4(x.unsqueeze(1))), 1)
                    t_out = model.bn1(t_out)
                    s_out = model.elu(model.bn2(model.spatial_conv(t_out)))
                    
                    logits = model(x)
                    batch_preds.extend(torch.max(logits, 1)[1].cpu().numpy())
                    batch_features.append(s_out.reshape(x.shape[0], -1).cpu().numpy())

        acc = accuracy_score(test_y, batch_preds)
        all_accuracies.append(acc * 100)
        folds.append(f"Fold {sub_id}")

        # Visualization for Best Fold (Subject 7)
        if sub_id == 7:
            full_features = np.concatenate(batch_features, axis=0)
            tsne = TSNE(n_components=2, random_state=42).fit_transform(full_features)
            plt.figure(figsize=(10, 7))
            plt.scatter(tsne[:, 0], tsne[:, 1], c=test_y, cmap='viridis', alpha=0.5)
            plt.title("Subject 7: Independent (LOSO) t-SNE")
            plt.savefig("subject_independent_tsne_sub7.png")

            weights = model.spatial_conv.weight.abs().mean(dim=(0,1,3)).cpu().numpy()
            info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
            info.set_montage(mne.channels.make_standard_montage('standard_1020'))
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(weights, info, axes=ax, cmap='RdBu_r', show=False)
            plt.title("Subject 7: Spatial Importance")
            plt.savefig("subject_independent_topo_sub7.png")

    # Final Bar Chart Report
    plt.figure(figsize=(12, 6))
    plt.bar(folds, all_accuracies, color='salmon', edgecolor='darkred')
    plt.axhline(np.mean(all_accuracies), color='darkgreen', linestyle='--', label=f'Mean: {np.mean(all_accuracies):.2f}%')
    plt.axhline(25, color='gray', linestyle=':', label='Chance (25%)')
    plt.title("SEED-IV Subject-Independent (LOSO) Report")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("subject_independent_performance_report.png")
    print("Independent reports and visualizations saved.")

if __name__ == "__main__":
    with open('config/seed4_transnet.yaml', 'r') as f:
        conf = yaml.safe_load(f)
    generate_independent_report('output/seed4_results/Independent', conf['data_path'], conf)