import numpy as np
import torch
torch.set_num_threads(14) # Utilize your 14 cores
import torch.nn as nn
import torch.optim as optim
import os
import yaml
from model.TransNet import TransNet
from model.baseModel import baseModel
from data.dataset import eegDataset
import time

def load_subject_data(data_path, sub_id, train_sessions, test_session):
    train_x, train_y = [], []
    for sess in train_sessions:
        x = np.load(os.path.join(data_path, f"sub{sub_id}_sess{sess}_data.npy"))
        y = np.load(os.path.join(data_path, f"sub{sub_id}_sess{sess}_label.npy"))
        train_x.append(x)
        train_y.append(y)
    
    test_x = np.load(os.path.join(data_path, f"sub{sub_id}_sess{test_session}_data.npy"))
    test_y = np.load(os.path.join(data_path, f"sub{sub_id}_sess{test_session}_label.npy"))
    
    return np.concatenate(train_x), np.concatenate(train_y), test_x, test_y

def main(config):
    # CRITICAL SPEED BOOST FOR RTX 4050
    torch.backends.cudnn.benchmark = True 
    
    data_path = config['data_path']
    out_folder = config['out_folder']
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    
    for subId in range(1, 16):
        print(f"\n--- Starting Subject {subId} (Dependent) ---")
        train_x, train_y, test_x, test_y = load_subject_data(data_path, subId, [1, 2], 3)

        train_dataset = eegDataset(train_x, train_y)
        test_dataset = eegDataset(test_x, test_y)

        out_path = os.path.join(out_folder, 'Dependent', f'sub{subId}', random_folder)
        if not os.path.exists(out_path): os.makedirs(out_path)

        net = TransNet(**config['network_args'])
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])
        loss_func = nn.CrossEntropyLoss()

        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)
        model.train_test(train_dataset, test_dataset)

if __name__ == '__main__':
    with open('config/seed4_transnet.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)