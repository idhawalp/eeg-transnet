import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
import copy

class baseModel():
    def __init__(self, net, config, optimizer, loss_func, scheduler=None, result_savepath=None):        
        self.batchsize = config['batch_size']
        self.epochs = config['epochs']
        self.patience = config.get('patience', 20) 
        self.preferred_device = config['preferred_device']
        self.num_classes = config['num_classes']
        self.num_segs = config['num_segs']

        self.device = None
        self.set_device(config['nGPU'])
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.loss_func = torch.nn.CrossEntropyLoss() 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        self.scaler = torch.cuda.amp.GradScaler()

        self.result_savepath = result_savepath
        if self.result_savepath is not None:
            self.log_write = open(os.path.join(self.result_savepath, 'log_result.txt'), 'w')

    def set_device(self, nGPU):
        self.device = torch.device(f'cuda:{nGPU}' if torch.cuda.is_available() else 'cpu')
        print("Code running on: ", self.device)

    def data_augmentation(self, data, label):
        aug_data, aug_label = [], []
        label_np = label.cpu().numpy() if torch.is_tensor(label) else label
        N, C, T = data.shape
        seg_size = T // self.num_segs
        aug_data_size = self.batchsize // self.num_classes
        
        for cls in range(self.num_classes):
            cls_idx = np.where(label_np == cls)[0]
            if len(cls_idx) <= 1: continue 
            
            cls_data = data[cls_idx]
            temp_aug_data = np.zeros((aug_data_size, C, T))
            for i in range(aug_data_size):
                rand_idx = np.random.randint(0, len(cls_idx), self.num_segs)
                for j in range(self.num_segs):
                    temp_aug_data[i, :, j*seg_size:(j+1)*seg_size] = cls_data[rand_idx[j], :, j*seg_size:(j+1)*seg_size]
            aug_data.append(temp_aug_data)
            aug_label.extend([cls]*aug_data_size)

        if len(aug_data) == 0:
            return torch.tensor([]), torch.tensor([])

        aug_data = torch.from_numpy(np.concatenate(aug_data, axis=0))
        aug_label = torch.from_numpy(np.array(aug_label))
        aug_shuffle = np.random.permutation(len(aug_data))
        return aug_data[aug_shuffle], aug_label[aug_shuffle]

    def train_test(self, train_dataset, test_dataset):
        # STABILITY: num_workers=0 and drop_last=True for Windows
        train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, 
                                      num_workers=0, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batchsize, num_workers=0, 
                                     pin_memory=True, drop_last=False)

        best_acc, counter = 0, 0
        best_model = None

        for epoch in range(self.epochs):
            self.net.train()
            train_predicted, train_actual = [], []
            for train_data, train_label in train_dataloader:
                aug_data, aug_label = self.data_augmentation(train_data, train_label)
                
                # Combine original and augmented data
                if aug_data.nelement() > 0:
                    train_data = torch.cat((train_data.float(), aug_data.float()), 0)
                    train_label = torch.cat((train_label.long(), aug_label.long()), 0)
                
                # CRITICAL FIX: Explicitly cast to .float() to resolve Double vs Half error
                train_data = train_data.float().to(self.device, non_blocking=True)
                train_label = train_label.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    train_output = self.net(train_data)
                    loss = self.loss_func(train_output, train_label)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_predicted.extend(torch.max(train_output, 1)[1].cpu().tolist())
                train_actual.extend(train_label.cpu().tolist())

            if self.scheduler: self.scheduler.step()

            # Test Loop
            self.net.eval()
            test_predicted, test_actual = [], []
            with torch.no_grad():
                for test_data, test_label in test_dataloader:
                    test_data = test_data.float().to(self.device, non_blocking=True)
                    test_label = test_label.long().to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        test_output = self.net(test_data)
                    test_predicted.extend(torch.max(test_output, 1)[1].cpu().tolist())
                    test_actual.extend(test_label.cpu().tolist())

            test_acc = accuracy_score(test_actual, test_predicted)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(self.net.state_dict())
                counter = 0
                print(f"--> Best Accuracy Improved to {best_acc:.4f}. Saving weights.")
                if self.log_write: self.log_write.write(f"Best Accuracy: {best_acc:.4f}\n")
            else:
                counter += 1
            print(f'Epoch [{epoch+1}] | Train Acc: {accuracy_score(train_actual, train_predicted):.4f} | Test Acc: {test_acc:.4f} | LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            if counter >= self.patience: break

        torch.save(best_model, os.path.join(self.result_savepath, 'model.pth'))