import os
import numpy as np
import matplotlib.pyplot as plt

def create_dependent_visual_report(root_dir):
    subjects = []
    accuracies = []
    
    # Traverse through sub1 to sub15
    for sub_id in range(1, 16):
        sub_folder = f'sub{sub_id}'
        sub_path = os.path.join(root_dir, sub_folder)
        if not os.path.exists(sub_path): continue
        
        # Get the latest training run timestamp
        time_folders = sorted(os.listdir(sub_path))
        if not time_folders: continue
        
        log_path = os.path.join(sub_path, time_folders[-1], 'log_result.txt')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                # Extract the last recorded best accuracy
                acc_lines = [l for l in lines if "Best Accuracy" in l]
                if acc_lines:
                    acc = float(acc_lines[-1].split()[-1]) * 100
                    subjects.append(f"S{sub_id}")
                    accuracies.append(acc)

    # Statistical Calculation
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # Visualization Implementation
    plt.figure(figsize=(12, 7))
    bars = plt.bar(subjects, accuracies, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2f}%')
    plt.fill_between([-0.5, 14.5], mean_acc-std_acc, mean_acc+std_acc, color='red', alpha=0.1, label=f'Std Dev: ±{std_acc:.2f}%')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.title("Detailed Performance Report: Subject-Dependent Paradigm (SEED-IV)", fontsize=15)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.xlabel("Subject ID", fontsize=12)
    plt.ylim(0, 100)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Save to Root
    plt.savefig("subject_dependent_report.png", dpi=300, bbox_inches='tight')
    print("Subject-Dependent Report saved as 'subject_dependent_report.png'")

if __name__ == "__main__":
    create_dependent_visual_report('output/seed4_results/Dependent')