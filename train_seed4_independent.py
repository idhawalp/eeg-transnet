import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
from model.TransNet import TransNet
from model.baseModel import baseModel
from data.dataset import eegDataset

def load_loso_data(data_path, test_sub_id):
    """
    Loads data from all subjects except the test_sub_id.
    Ensures data is cast to float32 to prevent Double vs Half errors.
    """
    train_x, train_y, test_x, test_y = [], [], [], []
    for sub in range(1, 16):
        for sess in range(1, 4):
            x_path = os.path.join(data_path, f"sub{sub}_sess{sess}_data.npy")
            y_path = os.path.join(data_path, f"sub{sub}_sess{sess}_label.npy")
            
            if os.path.exists(x_path):
                x, y = np.load(x_path), np.load(y_path)
                if sub == test_sub_id:
                    test_x.append(x)
                    test_y.append(y)
                else:
                    train_x.append(x)
                    train_y.append(y)
    
    # Casting to float32 is critical for Mixed Precision stability
    return (np.concatenate(train_x).astype('float32'), np.concatenate(train_y).astype('int64'), 
            np.concatenate(test_x).astype('float32'), np.concatenate(test_y).astype('int64'))

def main(config):
    # Hardware acceleration for RTX 4050
    torch.backends.cudnn.benchmark = True 
    torch.set_num_threads(14) 
    
    data_path = config['data_path']
    out_folder = config['out_folder']
    
    for test_sub in range(1, 16):
        print(f"\n--- Starting LOSO Fold: Testing on Subject {test_sub} ---")
        train_x, train_y, test_x, test_y = load_loso_data(data_path, test_sub)
        
        train_dataset = eegDataset(train_x, train_y)
        test_dataset = eegDataset(test_x, test_y)

        out_path = os.path.join(out_folder, 'Independent', f'test_sub{test_sub}')
        if not os.path.exists(out_path): os.makedirs(out_path)

        net = TransNet(**config['network_args'])
        optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))
        
        # Uses baseModel with num_workers=0 and FP16 Scaler
        model = baseModel(net, config, optimizer, nn.CrossEntropyLoss(), result_savepath=out_path)
        model.train_test(train_dataset, test_dataset)

if __name__ == '__main__':
    with open('config/seed4_transnet.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)