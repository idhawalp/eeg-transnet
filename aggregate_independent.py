import os
import numpy as np

def aggregate_independent_results(root_dir):
    accuracies = []
    print(f"{'Test Subject':<15} | {'Best Accuracy':<15}")
    print("-" * 35)
    
    # Iterate through subject folds 1-15
    for sub in range(1, 16):
        # Folder naming matches your train_seed4_independent.py logic
        sub_folder = f'test_sub{sub}'
        log_path = os.path.join(root_dir, sub_folder, 'log_result.txt')
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                # Find the final "Best Accuracy" record for this fold
                best_acc_lines = [l for l in lines if "Best Accuracy" in l]
                if best_acc_lines:
                    acc = float(best_acc_lines[-1].split()[-1])
                    accuracies.append(acc)
                    print(f"Subject {sub:<8} | {acc*100:.2f}%")
                
    if accuracies:
        print("-" * 35)
        # Final Mean LOSO Accuracy across all folds
        print(f"Mean LOSO Accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"LOSO Std Dev:       {np.std(accuracies)*100:.2f}%")
    else:
        print("No independent results found. Ensure training has completed.")

if __name__ == "__main__":
    # Path to your independent results folder
    results_path = 'output/seed4_results/Independent'
    aggregate_independent_results(results_path)