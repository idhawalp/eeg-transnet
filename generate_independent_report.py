import os
import numpy as np
import matplotlib.pyplot as plt

def create_independent_visual_report(root_dir):
    folds = []
    accuracies = []
    
    # LOSO folds traverse test_sub1 to test_sub15
    for sub_id in range(1, 16):
        fold_folder = f'test_sub{sub_id}'
        log_path = os.path.join(root_dir, fold_folder, 'log_result.txt')
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                acc_lines = [l for l in lines if "Best Accuracy" in l]
                if acc_lines:
                    acc = float(acc_lines[-1].split()[-1]) * 100
                    folds.append(f"Fold {sub_id}")
                    accuracies.append(acc)

    # Statistics
    mean_loso = np.mean(accuracies)
    std_loso = np.std(accuracies)

    # Visualization
    plt.figure(figsize=(12, 7))
    bars = plt.bar(folds, accuracies, color='salmon', edgecolor='darkred', alpha=0.8)
    plt.axhline(mean_loso, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean LOSO: {mean_loso:.2f}%')
    plt.axhline(25, color='gray', linestyle=':', label='Chance Level (25%)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.title("Detailed Generalization Report: Subject-Independent (LOSO) Paradigm", fontsize=15)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.xlabel("Leave-One-Out Test Subject", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 80)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Save to Root
    plt.savefig("subject_independent_report.png", dpi=300, bbox_inches='tight')
    print("Subject-Independent Report saved as 'subject_independent_report.png'")

if __name__ == "__main__":
    create_independent_visual_report('output/seed4_results/Independent')