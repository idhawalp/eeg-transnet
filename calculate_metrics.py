import torch
import numpy as np
import os
import yaml
from model.TransNet import TransNet
from data.dataset import eegDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def generate_full_report(model_path, test_data_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = TransNet(**config['network_args']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Load Test Data
    test_x = np.load(test_data_path.replace('label', 'data')).astype('float32')
    test_y = np.load(test_data_path).astype('int64')
    dataloader = DataLoader(eegDataset(test_x, test_y), batch_size=config['batch_size'], shuffle=False)

    all_preds, all_labels = [], []
    
    # 3. Inference with Mixed Precision for Speed
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for x, y in dataloader:
                x = x.to(device)
                outputs = model(x)
                preds = torch.max(outputs, 1)[1]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

    # 4. Calculate Stats
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n--- Final Performance Report ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Kappa:    {kappa:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neutral', 'Sad', 'Fear', 'Happy'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.savefig("final_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    with open('config/seed4_transnet.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Example: Analyze your best subject (Sub 7 for Independent)
    best_model = 'output/seed4_results/Independent/test_sub7/model.pth'
    test_labels = 'dataset/seed4_processed/sub7_sess3_label.npy' 
    generate_full_report(best_model, test_labels, config)