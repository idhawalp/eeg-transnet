import os
import numpy as np

def aggregate_dependent_results(root_dir):
    accuracies = []
    print(f"{'Subject':<12} | {'Best Accuracy':<15}")
    print("-" * 30)
    
    # Iterate through subject folders 1-15
    for sub in range(1, 16):
        sub_path = os.path.join(root_dir, f'sub{sub}')
        if not os.path.exists(sub_path): continue
        
        # Get the latest timestamp folder
        time_folders = sorted(os.listdir(sub_path))
        if not time_folders: continue
        
        log_path = os.path.join(sub_path, time_folders[-1], 'log_result.txt')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                # Find all lines containing "Best Accuracy" and take the last one
                best_acc_lines = [l for l in lines if "Best Accuracy" in l]
                if best_acc_lines:
                    acc = float(best_acc_lines[-1].split()[-1])
                    accuracies.append(acc)
                    print(f"Subject {sub:<3} | {acc*100:.2f}%")
                
    if accuracies:
        print("-" * 30)
        print(f"Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"Std Dev:       {np.std(accuracies)*100:.2f}%")

if __name__ == "__main__":
    # Path to your dependent results
    aggregate_dependent_results('output/seed4_results/Dependent')