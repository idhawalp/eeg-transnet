import scipy.io as sio
import numpy as np
import os
import glob

def preprocess_seed4(raw_path, save_path):
    # Labels from ReadMe.txt
    session_labels = [
        [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3], # Session 1
        [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1], # Session 2
        [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]  # Session 3
    ]
    
    window_size = 800 # 4 seconds at 200Hz
    
    if not os.path.exists(save_path): os.makedirs(save_path)

    for session_idx in range(1, 4):
        session_dir = os.path.join(raw_path, str(session_idx))
        files = glob.glob(os.path.join(session_dir, "*.mat"))
        
        for file in files:
            data = sio.loadmat(file)
            sub_id = os.path.basename(file).split('_')[0]
            
            all_x = []
            all_y = []
            
            # Extract trials (usually named 'variable_eeg1' to 'variable_eeg24')
            trial_keys = [k for k in data.keys() if 'eeg' in k]
            
            for i, key in enumerate(trial_keys):
                trial_data = data[key] # Shape: (62, points)
                label = session_labels[session_idx-1][i]
                
                # Sliding window segmenting
                steps = trial_data.shape[1] // window_size
                for s in range(steps):
                    start = s * window_size
                    end = start + window_size
                    segment = trial_data[:, start:end]
                    
                    # Normalization (Z-score)
                    segment = (segment - np.mean(segment)) / np.std(segment)
                    
                    all_x.append(segment)
                    all_y.append(label)
            
            # Save per subject per session
            np.save(f"{save_path}/sub{sub_id}_sess{session_idx}_data.npy", np.array(all_x))
            np.save(f"{save_path}/sub{sub_id}_sess{session_idx}_label.npy", np.array(all_y))
            print(f"Processed Subject {sub_id} Session {session_idx}")

if __name__ == "__main__":
    preprocess_seed4("./data/seed4_raw", "./dataset/seed4_processed")