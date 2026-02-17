import os
import numpy as np

def calculate_seed_iv_samples(data_path):
    """
    Calculate total number of samples from preprocessed SEED-IV dataset.
    """
    total_samples = 0
    sample_details = []

    print("=" * 70)
    print("Calculating total samples from SEED-IV preprocessed data")
    print("=" * 70)

    subjects = range(1, 16)  # 15 subjects
    sessions = range(1, 4)   # 3 sessions

    for subject in subjects:
        subject_samples = 0
        for session in sessions:
            data_filename = f"sub{subject}_sess{session}_data.npy"
            data_filepath = os.path.join(data_path, data_filename)

            if os.path.exists(data_filepath):
                data = np.load(data_filepath)
                num_samples = data.shape[0]
                total_samples += num_samples
                subject_samples += num_samples
                sample_details.append({
                    'subject': subject,})